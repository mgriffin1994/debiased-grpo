"""Unified GRPO / Debiased GRPO training script.

Runs one experiment cell (see notes/experiments.md) via CLI flags. Every cell
trains a QLoRA-adapted Qwen2-0.5B on GSM8K with identical model/data/step
budget --- only the loss configuration varies.
(The original plan called for Qwen2-1.5B; two 1.5B-class models in 4-bit
do not fit on 8 GB alongside generation activations, so the configs ship
with Qwen2-0.5B and rely on a shared PEFT model with adapters toggled off
for the reference forward / generate pass.)

Flags map directly to ``debiased_grpo.config.LossConfig``:

    --baseline           mean_with_self | loo | independent
    --reward-assignment  broadcast | reward_to_go
    --is-weighting       full_sequence | per_token
    --clipping           none | log_ratio_token | ppo_classical
    --length-norm        per_response | token_level | fixed_constant
    --std-norm           / --no-std-norm
    --ema-baseline       / --no-ema-baseline

Cell presets (copied into the Makefile):

    G0 paper GRPO            mean_with_self + full_sequence + ppo_classical + per_response + std_norm
    G0a paper GRPO (per-tok) mean_with_self + per_token + ppo_classical + per_response + std_norm
    Debiased GRPO            independent + full_sequence + none + fixed_constant + no_std + log_w_clamp=5

Usage:
    python scripts/train_grpo.py --config configs/gsm8k_qwen05b.yaml \\
        --baseline independent --is-weighting full_sequence \\
        --clipping none --length-norm fixed_constant --no-std-norm \\
        --log-w-clamp 5.0 --output-dir outputs/debiased_grpo
"""

import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks import ModelCheckpoint


# ─── Dataset ────────────────────────────────────────────────────────────────

class GSM8KDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GSM8KDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        max_samples = self.cfg.get("max_train_samples", len(ds))
        data = [
            {"question": row["question"], "answer": row["answer"]}
            for row in ds.select(range(min(max_samples, len(ds))))
        ]
        split = max(1, len(data) // 10)
        self.train_data = data[split:]
        self.val_data = data[:split]

    def train_dataloader(self):
        return DataLoader(
            GSM8KDataset(self.train_data),
            batch_size=1, shuffle=True,
            collate_fn=lambda x: x[0], num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            GSM8KDataset(self.val_data),
            batch_size=1, shuffle=False,
            collate_fn=lambda x: x[0], num_workers=0,
        )


# ─── Reward ─────────────────────────────────────────────────────────────────

def _extract_number(text: str):
    if "####" in text:
        text = text.split("####")[-1]
    text = text.replace(",", "").strip()
    matches = re.findall(r"-?\d+\.?\d*", text)
    return matches[-1] if matches else None


def compute_reward(completion_tokens: torch.Tensor, ground_truth: str, tokenizer) -> float:
    completion_str = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    pred = _extract_number(completion_str)
    gt = _extract_number(ground_truth)
    if pred is None or gt is None:
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(gt)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


# ─── Lightning Module ────────────────────────────────────────────────────────

class UnifiedGRPOModule(L.LightningModule):
    def __init__(self, cfg, loss_config):
        super().__init__()
        self.cfg = cfg
        self.loss_config = loss_config  # debiased_grpo.config.LossConfig
        self.automatic_optimization = False
        self.inner_loop_mu = int(cfg.get("inner_loop_mu", 4))

        # Stateful EMA shared across steps; trainer updates past-only.
        from debiased_grpo.utils import EMABaseline
        self.ema = EMABaseline(decay=loss_config.ema_decay) if loss_config.ema_baseline else None

    def configure_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from debiased_grpo.sampling import GroupRolloutSampler, IndependentBaselineSampler

        model_name = self.cfg["model_name"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            trust_remote_code=True,
        )
        base = prepare_model_for_kbit_training(base)
        lora_cfg = LoraConfig(
            r=self.cfg.get("lora_r", 16), lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora_cfg)
        self.model.print_trainable_parameters()

        # No separate reference model — the PEFT base IS the reference. Both
        # samplers receive ``self.model`` and disable adapters when running
        # the reference forward / generate. Saves the second model's worth of
        # GPU memory (~0.5 GB for Qwen2-0.5B in 4-bit; ~1.5 GB for Qwen2-1.5B),
        # which is what made this fit on 8 GB at all.
        self.ref_model = self.model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sampler = GroupRolloutSampler(
            model=self.model, tokenizer=self.tokenizer, ref_model=None,
            group_size=self.cfg.get("group_size", 8),
            max_new_tokens=self.cfg.get("max_new_tokens", 256),
            temperature=self.cfg.get("temperature", 0.9),
        )

        if self.loss_config.baseline == "independent":
            self.baseline_sampler = IndependentBaselineSampler(
                ref_model=self.model, tokenizer=self.tokenizer,
                num_baseline=self.cfg.get("num_baseline_rollouts", 4),
                max_new_tokens=self.cfg.get("max_new_tokens", 256),
                temperature=1.0,
                adapter_disable=True,
            )
        else:
            self.baseline_sampler = None

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=self.cfg.get("learning_rate", 1e-5))

    def load_state_dict(self, state_dict, strict=True):
        # QLoRA resume: a Lightning checkpoint stores the frozen 4-bit base
        # model's bitsandbytes quant buffers (base_layer.weight.absmax,
        # quant_state.bitsandbytes__nf4, ...). A freshly-built quantized model
        # does not register those as expected keys, so a strict load raises
        # "Unexpected key(s)". They are the frozen base (identical after a fresh
        # build), so it is safe to load non-strict — only the trainable LoRA
        # params need to match, and those do.
        return super().load_state_dict(state_dict, strict=False)

    def training_step(self, batch, batch_idx):
        from debiased_grpo.config import build_components
        from debiased_grpo.losses import compute_loss
        from debiased_grpo.strategies import FullSequenceIS
        from debiased_grpo.utils import compute_ess

        opt = self.optimizers()
        item = batch
        ground_truth = item["answer"]

        prompt = f"Question: {item['question']}\nAnswer:"
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        prompt_ids = enc["input_ids"][0].to("cuda:0")

        # Sample once per outer step. ``behavior_log_probs`` snapshots the
        # sampling-time policy and is used as the IS denominator across the
        # inner loop. ``ref_log_probs`` (frozen base) is the KL anchor.
        raw_completions, completions_padded, behavior_log_probs, ref_log_probs, mask = (
            self.sampler.sample(prompt_ids, with_grad_log_probs=False)
        )
        device = behavior_log_probs.device
        mask = mask.to(device)
        behavior_log_probs = behavior_log_probs.detach()
        ref_log_probs = ref_log_probs.detach()

        rewards = torch.tensor(
            [compute_reward(comp.cpu(), ground_truth, self.tokenizer)
             for comp in raw_completions],
            dtype=torch.float32,
            device=device,
        )

        baseline_rewards = None
        if self.baseline_sampler is not None:
            token_reward_fn = lambda tokens: compute_reward(tokens, ground_truth, self.tokenizer)
            baseline_rewards = self.baseline_sampler.sample_baseline_rewards(
                prompt_ids, token_reward_fn
            ).to(device)

        components = build_components(self.loss_config, ema=self.ema)
        kl_ref_coef = float(getattr(self.loss_config, "kl_ref_coef", 0.0))
        kl_behavior_coef = float(getattr(self.loss_config, "kl_behavior_coef", 0.0))

        # Inner loop: μ gradient steps per resample, with the IS ratio
        # ``π_θ_current / π_behavior``. At inner step k=0 the ratio is
        # identically 1; for k>=1 the ratio reflects the within-loop drift.
        last_loss = None
        last_log_probs_detached = None
        for k in range(self.inner_loop_mu):
            log_probs = self.sampler.compute_log_probs(prompt_ids, completions_padded)
            loss = compute_loss(
                log_probs=log_probs,
                ref_log_probs=ref_log_probs,
                rewards=rewards, mask=mask,
                baseline_rewards=baseline_rewards,
                behavior_log_probs=behavior_log_probs,
                kl_ref_coef=kl_ref_coef,
                kl_behavior_coef=kl_behavior_coef,
                **components,
            )

            opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=1.0)
            opt.step()

            last_loss = loss.detach()
            last_log_probs_detached = log_probs.detach()

        # Past-only EMA update: must happen AFTER all gradient steps so the
        # value used at step t is independent of step t's actions.
        if self.ema is not None:
            with torch.no_grad():
                primary_baseline = components["baseline"].compute(rewards, baseline_rewards)
                batch_mean_adv = (rewards - primary_baseline).mean().item()
            self.ema.update(batch_mean_adv)
            self.log("train/ema_value", self.ema.value, on_step=True)

        # Diagnostics (no grad). Computed against the FINAL inner-step policy
        # to capture the post-resample state of the IS ratio and KL.
        with torch.no_grad():
            mask_f = mask.float()
            n_tok = mask_f.sum().clamp(min=1.0)

            # IS-correction stability: ratio = π_θ_current / π_behavior.
            is_log_ratio = (last_log_probs_detached - behavior_log_probs) * mask_f
            is_weights = FullSequenceIS().compute(
                last_log_probs_detached - behavior_log_probs, mask,
            )  # (N, 1): one full-sequence weight per rollout
            # ESS over the batch of per-sequence weights (not the singleton token
            # axis) — measures weight concentration across rollouts.
            ess = compute_ess(is_weights.reshape(1, -1))
            mean_is_log_ratio = is_log_ratio.sum() / n_tok
            mean_sq_is = is_log_ratio.pow(2).sum() / n_tok
            is_log_ratio_std = (mean_sq_is - mean_is_log_ratio.pow(2)).clamp(min=0.0).sqrt()

            # Clamp activity: the summed (per-sequence) log-IS-weight is what the
            # one-sided ``log_w_clamp`` caps. Log its max and the fraction of
            # sequences whose weight the clamp actually bounds — this is the direct
            # measure of whether the overflow guard is firing (vs being latent).
            logw_seq = is_log_ratio.sum(dim=1)                       # (N,) summed log-weight
            clamp_val = getattr(components["is_weighter"], "log_w_clamp", None)

            # KL to frozen base π_ref (the trust-region anchor).
            log_ratio_to_ref = (last_log_probs_detached - ref_log_probs) * mask_f
            mean_log_ratio_to_ref = log_ratio_to_ref.sum() / n_tok
            kl_to_ref = mean_log_ratio_to_ref

            reward_std = rewards.std() if rewards.numel() > 1 else torch.tensor(0.0)

        bs = behavior_log_probs.shape[0]
        self.log("train/loss", last_loss, prog_bar=True, on_step=True, batch_size=bs)
        self.log("train/mean_reward", rewards.mean(), prog_bar=True, on_step=True, batch_size=bs)
        self.log("train/reward_std", reward_std, on_step=True, batch_size=bs)
        self.log("train/ess", ess, on_step=True, batch_size=bs)
        self.log("train/kl_to_ref", kl_to_ref, on_step=True, batch_size=bs)
        self.log("train/is_log_ratio_std", is_log_ratio_std, on_step=True, batch_size=bs)
        self.log("train/logw_seq_max", logw_seq.max(), on_step=True, batch_size=bs)
        if clamp_val is not None:
            self.log("train/clamp_fire_frac",
                     (logw_seq > clamp_val).float().mean(), on_step=True, batch_size=bs)
        self.log("train/mean_response_length", mask_f.sum(dim=1).mean(),
                 on_step=True, batch_size=bs)
        return last_loss

    def validation_step(self, batch, batch_idx):
        if batch_idx >= 100:
            return
        item = batch
        prompt = f"Question: {item['question']}\nAnswer:"
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"][0].to("cuda:0")

        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                input_ids.unsqueeze(0), max_new_tokens=128, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        self.model.train()
        completion = out[0, input_ids.shape[0]:]
        r = compute_reward(completion.cpu(), item["answer"], self.tokenizer)
        self.log("val/acc", r, on_step=False, on_epoch=True, batch_size=1)

    def on_train_end(self):
        out_dir = Path(self.cfg.get("output_dir", "outputs/grpo_run"))
        self.model.save_pretrained(str(out_dir / "adapter_final"))
        print(f"Saved final adapter to {out_dir / 'adapter_final'}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified GRPO / Debiased GRPO training")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default="outputs/grpo_run")

    p.add_argument("--baseline", choices=["mean_with_self", "loo", "independent"],
                   required=True)
    p.add_argument("--reward-assignment", choices=["broadcast", "reward_to_go"],
                   default="broadcast")
    p.add_argument("--is-weighting",
                   choices=["full_sequence", "per_token"],
                   required=True)
    p.add_argument("--clipping", choices=["none", "log_ratio_token", "ppo_classical"],
                   required=True)
    p.add_argument("--length-norm",
                   choices=["per_response", "token_level", "fixed_constant"],
                   required=True)
    p.add_argument("--fixed-divisor", type=float, default=0.0,
                   help="Constant divisor for length_norm=fixed_constant. "
                        "If 0, defaults to group_size · max_new_tokens from the config.")

    std_group = p.add_mutually_exclusive_group(required=True)
    std_group.add_argument("--std-norm", dest="std_norm", action="store_true",
                           help="Divide advantage by group reward std (paper GRPO).")
    std_group.add_argument("--no-std-norm", dest="std_norm", action="store_false",
                           help="Use raw centred reward (Dr. GRPO / Debiased GRPO).")

    ema_group = p.add_mutually_exclusive_group()
    ema_group.add_argument("--ema-baseline", dest="ema_baseline", action="store_true",
                           help="Subtract a past-only EMA of mean advantage as a second baseline.")
    ema_group.add_argument("--no-ema-baseline", dest="ema_baseline", action="store_false")
    p.set_defaults(ema_baseline=False)

    p.add_argument("--ema-decay", type=float, default=0.95)
    p.add_argument("--log-w-clamp", type=float, default=None,
                   help="Upper cap on the full-sequence log IS-weight before exp "
                        "for is_weighting=full_sequence. None = strictly unbiased "
                        "(default); a finite cap (e.g. 5.0) bounds the weight to "
                        "exp(cap) at the cost of small bias.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override seed from config (used for multi-seed sweeps).")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning_rate from config (used to slow drift "
                        "in unbiased full-sequence IS where the log-ratio sum can blow up).")
    p.add_argument("--log-ratio-clip-c", type=float, default=math.log(1.5),
                   help="Symmetric clamp on per-token log-ratio for clipping=log_ratio_token.")
    p.add_argument("--inner-loop-mu", type=int, default=4,
                   help="Number of gradient steps per resample (PPO/GRPO inner-loop μ). "
                        "Default 4 — gives the IS ratio room to deviate from 1, exercising "
                        "full-sequence IS / PPO clipping. μ=1 reduces to REINFORCE + KL.")
    p.add_argument("--kl-ref-coef", type=float, default=0.04,
                   help="Coefficient β on KL(π_θ || π_ref), the RLHF anchor to the "
                        "frozen base. Default 0.04 matches DeepSeekMath GRPO; 0 disables.")
    p.add_argument("--kl-behavior-coef", type=float, default=0.0,
                   help="Coefficient on KL(π_θ || π_behavior), an unbiased soft trust "
                        "region on the IS ratio (alternative/complement to PPO clip). "
                        "Default 0 (off).")
    p.add_argument("--log-every", type=int, default=10,
                   help="Lightning log_every_n_steps. Set to 1 for per-step "
                        "diagnostics on short probe runs.")
    p.add_argument("--resume-ckpt", type=str, default=None,
                   help="Path to a Lightning .ckpt to resume training from "
                        "(restores model/optimizer/global_step; continues to num_steps).")

    return p.parse_args()


def main():
    from debiased_grpo.config import LossConfig

    torch.set_float32_matmul_precision("high")
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["output_dir"] = args.output_dir
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.seed is not None:
        cfg["seed"] = args.seed

    fixed_divisor = args.fixed_divisor or float(
        cfg.get("group_size", 8) * cfg.get("max_new_tokens", 256)
    )
    cfg["inner_loop_mu"] = int(args.inner_loop_mu)
    loss_config = LossConfig(
        baseline=args.baseline,
        reward_assignment=args.reward_assignment,
        is_weighting=args.is_weighting,
        clipping=args.clipping,
        length_norm=args.length_norm,
        std_norm=args.std_norm,
        ema_baseline=args.ema_baseline,
        clip_eps=cfg.get("clip_eps", 0.2),
        log_ratio_clip_c=args.log_ratio_clip_c,
        ema_decay=args.ema_decay,
        fixed_divisor=fixed_divisor,
        log_w_clamp=args.log_w_clamp,
        kl_ref_coef=float(args.kl_ref_coef),
        kl_behavior_coef=float(args.kl_behavior_coef),
    )
    print(f"[train_grpo] loss config: {loss_config}")

    L.seed_everything(cfg.get("seed", 42))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = GSM8KDataModule(cfg)
    module = UnifiedGRPOModule(cfg, loss_config)

    from lightning.pytorch.loggers import CSVLogger
    logger = CSVLogger(save_dir=str(out_dir), name="logs")

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        every_n_train_steps=cfg.get("eval_every", 100),
        save_top_k=-1,
    )

    trainer = L.Trainer(
        max_steps=cfg.get("num_steps", 1000),
        accelerator="gpu", devices=1,
        logger=logger, callbacks=[checkpoint_cb],
        log_every_n_steps=getattr(args, "log_every", 10),
        val_check_interval=cfg.get("eval_every", 100),
        limit_val_batches=100,
        enable_progress_bar=True,
        # Gradients are clipped manually inside training_step (the Trainer's own
        # gradient_clip_val is ignored under manual optimization).
    )
    trainer.fit(module, dm, ckpt_path=getattr(args, "resume_ckpt", None) or None)


if __name__ == "__main__":
    main()
