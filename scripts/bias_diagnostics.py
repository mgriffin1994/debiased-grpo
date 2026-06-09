"""Inference-only diagnostics for trained Debiased GRPO checkpoints.

For each (cell, training_step) we sample R rollouts on a fixed 50-prompt
val subset of GSM8K and record per-rollout
``{prompt_id, reward, length, log_prob_theta, log_prob_ref}`` to a JSONL
file. We also sample M reference rollouts (adapter disabled) per prompt
once, since pi_ref is the frozen base and identical across all cells.

The dumped data supports four diagnostics counterfactually:
  1. full-sequence IS variance vs per-token IS
  2. per-response length-norm bias (response length x correctness)
  3. std-norm bias (per-prompt sigma(reward), implied 1/sigma amplification)
  4. LOO baseline bias (mean-with-self vs independent reference baseline)

Reuses ``GroupRolloutSampler``/``IndependentBaselineSampler`` and the
GSM8K reward function from ``scripts/train_grpo.py``.

Run: ``make diagnose`` (queues exclusive GPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse the trainer's reward + extract function.
from train_grpo import compute_reward  # noqa: E402


CELLS = [
    ("g0_paper_grpo", "paper GRPO (full-seq IS)"),
    ("g0a_paper_grpo", "paper GRPO (per-token IS)"),
    ("debiased_grpo", "debiased GRPO (ours)"),
]
STEPS = [200, 500, 1000]
N_PROMPTS = 50
N_THETA_ROLLOUTS = 4       # matches trainer's group_size=4; 8GB-safe at T=192
N_REF_ROLLOUTS = 8         # for independent baseline mean
DEVICE = "cuda:0"


def load_val_prompts(cfg: dict) -> list[dict]:
    """Match the trainer's val split exactly: first 10 % of max_train_samples."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    max_samples = cfg.get("max_train_samples", len(ds))
    data = [
        {"question": row["question"], "answer": row["answer"]}
        for row in ds.select(range(min(max_samples, len(ds))))
    ]
    split = max(1, len(data) // 10)
    val = data[:split][:N_PROMPTS]
    print(f"[diagnostics] using {len(val)} val prompts")
    return val


def build_base_model(cfg: dict):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], quantization_config=bnb_config, trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=cfg.get("lora_r", 16), lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_lightning_ckpt(model, ckpt_path: Path) -> None:
    """Load LoRA-A/B weights from a Lightning checkpoint into ``model`` in-place.

    Avoids ``map_location='cuda'`` on the full state_dict — that would pull the
    base 4-bit weights (~600 MB) onto GPU alongside the model's own copy and
    blow the 8 GB budget on the second iteration. Instead we load to CPU,
    keep only the ``lora_*`` parameters, and copy each one to the device
    individually before discarding.
    """
    print(f"[diagnostics] loading {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["state_dict"]
    target = dict(model.named_parameters())
    n_lora = 0
    with torch.no_grad():
        for k, v in sd.items():
            if "lora_" not in k.lower():
                continue
            stripped = k.removeprefix("model.")
            if stripped not in target:
                continue
            target[stripped].copy_(v.to(DEVICE, dtype=target[stripped].dtype))
            n_lora += 1
    # Sanity: the trainer saves 384 LoRA tensors (16 layers × 4 projections × 2 matrices × 3 dim variants),
    # we expect ≥ 100 actually-used ones to be copied.
    if n_lora < 100:
        raise RuntimeError(f"only {n_lora} LoRA tensors loaded — ckpt format may have shifted")
    print(f"[diagnostics] copied {n_lora} LoRA tensors")
    del raw, sd
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def make_samplers(model, tokenizer, cfg, group_size: int):
    from debiased_grpo.sampling import GroupRolloutSampler, IndependentBaselineSampler
    sampler = GroupRolloutSampler(
        model=model, tokenizer=tokenizer, ref_model=None,
        group_size=group_size,
        max_new_tokens=cfg.get("max_new_tokens", 192),
        temperature=cfg.get("temperature", 0.9),
    )
    ref_sampler = IndependentBaselineSampler(
        ref_model=model, tokenizer=tokenizer,
        num_baseline=N_REF_ROLLOUTS,
        max_new_tokens=cfg.get("max_new_tokens", 192),
        temperature=1.0,
        adapter_disable=True,
    )
    return sampler, ref_sampler


def rollouts_for_prompt(sampler, tokenizer, item: dict) -> list[dict]:
    """Sample ``sampler.group_size`` theta rollouts; return per-rollout records."""
    prompt = f"Question: {item['question']}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    prompt_ids = enc["input_ids"][0].to(DEVICE)
    with torch.no_grad():
        # Unpack 5-tuple; ``completions_padded`` is not needed for inference-only diagnostics.
        raw_completions, _completions_padded, log_probs, ref_log_probs, mask = sampler.sample(
            prompt_ids, with_grad_log_probs=False,
        )
    out: list[dict] = []
    for i, comp in enumerate(raw_completions):
        m = mask[i].cpu().bool()
        L = int(m.sum().item())
        out.append({
            "reward": compute_reward(comp.cpu(), item["answer"], tokenizer),
            "length": L,
            "log_prob_theta": log_probs[i, :L].detach().cpu().tolist(),
            "log_prob_ref": ref_log_probs[i, :L].detach().cpu().tolist(),
        })
    return out


def ref_rewards_for_prompt(ref_sampler, tokenizer, item: dict) -> list[float]:
    prompt = f"Question: {item['question']}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    prompt_ids = enc["input_ids"][0].to(DEVICE)

    def reward_fn(tokens):
        return compute_reward(tokens, item["answer"], tokenizer)

    rewards = ref_sampler.sample_baseline_rewards(prompt_ids, reward_fn)
    return rewards.tolist()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[diagnostics] wrote {len(rows)} rows -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gsm8k_qwen05b.yaml"))
    parser.add_argument(
        "--cell", choices=[c[0] for c in CELLS] + ["ref-only", "all"],
        default="all",
        help="Cell to diagnose. 'ref-only' samples only the pi_ref baselines; "
             "'all' processes ref + every cell sequentially.",
    )
    parser.add_argument(
        "--out-root", default=str(REPO_ROOT / "outputs" / "diagnostics"),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    val = load_val_prompts(cfg)
    out_root = Path(args.out_root)

    # ─── Pass 1: pi_ref baseline rollouts (cell-independent) ──────────────
    ref_path = out_root / "_ref_baselines.jsonl"
    if args.cell in ("ref-only", "all") and not ref_path.exists():
        print(f"[diagnostics] sampling {N_REF_ROLLOUTS} ref rollouts x {len(val)} prompts")
        model, tokenizer = build_base_model(cfg)
        model.eval()
        _, ref_sampler = make_samplers(model, tokenizer, cfg, group_size=N_THETA_ROLLOUTS)
        rows: list[dict] = []
        for pid, item in enumerate(val):
            rewards = ref_rewards_for_prompt(ref_sampler, tokenizer, item)
            rows.append({"prompt_id": pid, "rewards": rewards})
            if (pid + 1) % 10 == 0:
                print(f"  prompt {pid+1}/{len(val)}")
        write_jsonl(ref_path, rows)
        del model
        torch.cuda.empty_cache()

    if args.cell == "ref-only":
        return

    # ─── Pass 2: per-cell, per-step theta rollouts ────────────────────────
    # Build the base model + LoRA wrapper ONCE per cell (or even globally)
    # and swap LoRA weights in-place between training steps. Rebuilding the
    # 4-bit-quantised base on each iteration was OOMing before this change.
    cells = CELLS if args.cell == "all" else [c for c in CELLS if c[0] == args.cell]
    model, tokenizer = build_base_model(cfg)
    sampler, _ = make_samplers(model, tokenizer, cfg, group_size=N_THETA_ROLLOUTS)

    for cell_dir, cell_label in cells:
        for step in STEPS:
            out_path = out_root / cell_dir / f"step_{step:04d}.jsonl"
            if out_path.exists():
                print(f"[diagnostics] skip {out_path} (exists)")
                continue
            ckpt_path = REPO_ROOT / "outputs" / cell_dir / "checkpoints" / f"epoch=0-step={step}.ckpt"
            if not ckpt_path.exists():
                print(f"[diagnostics] WARN missing ckpt: {ckpt_path}")
                continue

            load_lightning_ckpt(model, ckpt_path)
            model.eval()
            print(f"[diagnostics] {cell_label} @ step {step}")
            rows: list[dict] = []
            for pid, item in enumerate(val):
                rollouts = rollouts_for_prompt(sampler, tokenizer, item)
                for j, r in enumerate(rollouts):
                    rows.append({"prompt_id": pid, "rollout_id": j, **r})
                if (pid + 1) % 10 == 0:
                    print(f"  prompt {pid+1}/{len(val)}")
                # Periodic cleanup to keep GPU pool healthy across long runs.
                if (pid + 1) % 5 == 0:
                    torch.cuda.empty_cache()
            write_jsonl(out_path, rows)


if __name__ == "__main__":
    main()
