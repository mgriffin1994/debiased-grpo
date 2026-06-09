"""
DebiasedGRPOTrainer — single training-step logic for Debiased GRPO.

This class encapsulates one gradient step: sample rollouts, compute the
debiased loss, and return a metrics dict. It deliberately does NOT implement the outer
training loop (epoch counting, logging, checkpointing) so that training scripts
remain in full control of that logic and can customise it freely.

NOTE: this is a reference single-step trainer. The runs reported in the README /
notes are produced by the multi-inner-step Lightning module in
``scripts/train_grpo.py`` (which calls ``losses.compute_loss`` directly); this
class is exercised by the tests and kept as a minimal reusable surface.

Typical usage in a training script::

    trainer = DebiasedGRPOTrainer(model, ref_model, tokenizer, reward_fn, config)
    for step, batch in enumerate(dataloader):
        metrics = trainer.train_step(batch)
        optimizer.step()
        optimizer.zero_grad()
        # `metrics` (loss, ESS, mean reward, ...) can be logged however you like.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Any, Callable, Dict, List

from debiased_grpo.losses import debiased_loss, grpo_loss
from debiased_grpo.sampling import GroupRolloutSampler, IndependentBaselineSampler
from debiased_grpo.utils import compute_ess

try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
except ImportError:  # pragma: no cover
    PreTrainedModel = object  # type: ignore[assignment,misc]
    PreTrainedTokenizerBase = object  # type: ignore[assignment,misc]


def _pad_and_mask(
    log_probs_list: List[Tensor],
    ref_log_probs_list: List[Tensor],
    mask_list: List[Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Stack a list of (G, T_i) tensors into a single (B*G, T_max) batch.

    Two invariants matter for correctness:
      - log_probs are copied WITHOUT detaching, so the computation graph back to
        the policy parameters is preserved and ``loss.backward()`` populates
        their gradients.
      - the per-group padding masks from GroupRolloutSampler are stacked
        alongside, so padding tokens added by _pad_sequences are marked as
        padding rather than real.

    Args:
        log_probs_list:     List of (G, T_i) policy log-prob tensors, one per prompt.
                            Must retain their grad_fn (not detached).
        ref_log_probs_list: Matching list of (G, T_i) reference log-prob tensors.
        mask_list:          List of (G, T_i) bool mask tensors from _pad_sequences,
                            True where tokens are real (non-padding).

    Returns:
        log_probs:     (N, T_max) where N = B * G.  Grad graph is preserved.
        ref_log_probs: (N, T_max).
        mask:          (N, T_max) bool — True for non-padding positions.
    """
    # Find the global max length across all prompts and all group members.
    T_max = max(lp.shape[1] for lp in log_probs_list)
    N = sum(lp.shape[0] for lp in log_probs_list)

    device = log_probs_list[0].device
    log_probs_out = torch.zeros(N, T_max, device=device)
    ref_log_probs_out = torch.zeros(N, T_max, device=device)
    mask_out = torch.zeros(N, T_max, dtype=torch.bool, device=device)

    row = 0
    for lp, rlp, mk in zip(log_probs_list, ref_log_probs_list, mask_list):
        G, T = lp.shape
        # lp is assigned WITHOUT .detach(): the grad_fn must be preserved so
        # that loss.backward() can flow gradients back to model parameters.
        log_probs_out[row : row + G, :T] = lp
        ref_log_probs_out[row : row + G, :T] = rlp
        mask_out[row : row + G, :T] = mk
        row += G

    return log_probs_out, ref_log_probs_out, mask_out


class DebiasedGRPOTrainer:
    """Encapsulates the per-step logic for Debiased GRPO training.

    Args:
        model:      The LoRA-adapted policy model (π_θ). Must be in training mode.
        ref_model:  Frozen reference model (π_ref). eval() mode, no grad.
        tokenizer:  Shared tokenizer for both models.
        reward_fn:  Callable mapping a completion tensor (1-D token ids) → float.
                    Should be deterministic and fast (rule-based, not a NN).
        config:     Dict or namespace with the following keys:
                        group_size         (int, default 8)
                        num_baseline_rollouts (int, default 4)
                        max_new_tokens     (int, default 256)
                        temperature        (float, default 0.9)
                        clip_eps           (float, default 0.2)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        reward_fn: Callable[[Tensor], float],
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config

        # Convenience: support both dict and namespace-style configs.
        def _get(key: str, default: Any) -> Any:
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        self.group_size: int = _get("group_size", 8)
        self.num_baseline: int = _get("num_baseline_rollouts", 4)
        self.clip_eps: float = _get("clip_eps", 0.2)
        # Fixed denominator for unbiased length normalisation in debiased loss.
        # Defaults to ``B · max_new_tokens`` where B = group_size + num_baseline.
        max_new = int(_get("max_new_tokens", 256))
        rollouts_per_step = self.group_size  # only policy rollouts enter the loss
        self.fixed_divisor: float = float(
            _get("fixed_divisor", rollouts_per_step * max_new)
        )

        self.rollout_sampler = GroupRolloutSampler(
            model=model,
            tokenizer=tokenizer,
            ref_model=ref_model,
            group_size=self.group_size,
            max_new_tokens=_get("max_new_tokens", 256),
            temperature=_get("temperature", 0.9),
        )
        self.baseline_sampler = IndependentBaselineSampler(
            ref_model=ref_model,
            tokenizer=tokenizer,
            num_baseline=self.num_baseline,
            max_new_tokens=_get("max_new_tokens", 256),
            temperature=1.0,  # higher temperature for diversity in baseline
        )

    def train_step(self, batch: List[str]) -> Dict[str, float]:
        """Execute one Debiased GRPO gradient step.

        This method performs the full forward pass through sampling and loss
        computation. The caller is responsible for calling ``loss.backward()``
        and the optimiser step — those are intentionally left outside so that
        gradient accumulation and mixed-precision can be handled by the script.

        Args:
            batch: List of B prompt strings. Each will receive ``group_size``
                   policy rollouts and ``num_baseline_rollouts`` reference rollouts.

        Returns:
            Dict with the following float-valued keys:
                loss            — the debiased loss (scalar, gradient-attached).
                grpo_loss_value — GRPO loss on the same data (diagnostic only,
                                  no gradient, detached).
                ess             — batch-averaged effective sample size of IS weights.
                mean_reward     — mean reward across all policy rollouts.

        Note: ``loss`` in the returned dict is a Tensor with requires_grad=True.
        All other values are plain Python floats for easy logging.
        """
        device = next(self.model.parameters()).device

        all_log_probs: List[Tensor] = []
        all_ref_log_probs: List[Tensor] = []
        all_masks: List[Tensor] = []
        all_rewards: List[float] = []
        all_baseline_rewards: List[Tensor] = []

        for prompt_str in batch:
            # Tokenize the prompt on the correct device.
            enc = self.tokenizer(
                prompt_str, return_tensors="pt", truncation=True, max_length=512
            )
            prompt_ids = enc["input_ids"][0].to(device)

            # --- Policy rollouts ---
            # ``sample`` now returns 5 values; ``completions_padded`` is unused
            # in this single-step trainer (no inner-loop) but is needed by the
            # Lightning script's μ>1 path.
            completions, _completions_padded, log_probs, ref_log_probs, rollout_mask = (
                self.rollout_sampler.sample(prompt_ids)
            )
            # log_probs, ref_log_probs, rollout_mask: (G, T_i)

            # Compute rewards for each completion.
            for completion in completions:
                r = self.reward_fn(completion)
                all_rewards.append(float(r))

            all_log_probs.append(log_probs)
            all_ref_log_probs.append(ref_log_probs)
            all_masks.append(rollout_mask)

            # --- Baseline rollouts (no grad) ---
            baseline_rewards = self.baseline_sampler.sample_baseline_rewards(
                prompt_ids, self.reward_fn
            )
            all_baseline_rewards.append(baseline_rewards)

        # Stack policy rollouts into a single (N, T_max) batch. Masks are passed
        # through and log_probs are NOT detached, preserving the grad graph.
        log_probs_batch, ref_log_probs_batch, mask = _pad_and_mask(
            all_log_probs, all_ref_log_probs, all_masks
        )

        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
        baseline_rewards_tensor = torch.cat(all_baseline_rewards, dim=0)

        # log_probs_batch retains its grad_fn from the policy forward pass —
        # no requires_grad_() hack needed.

        # --- Debiased loss (primary) ---
        loss = debiased_loss(
            log_probs_batch,
            ref_log_probs_batch,
            rewards_tensor,
            baseline_rewards_tensor,
            mask,
            fixed_divisor=self.fixed_divisor,
        )

        # --- GRPO loss (diagnostic comparison, no grad) ---
        with torch.no_grad():
            grpo_loss_value = grpo_loss(
                log_probs_batch,
                ref_log_probs_batch,
                rewards_tensor,
                mask,
                clip_eps=self.clip_eps,
            )

        # --- ESS of IS weights (diagnostic) ---
        with torch.no_grad():
            from debiased_grpo.strategies import FullSequenceIS
            weights = FullSequenceIS().compute(log_probs_batch - ref_log_probs_batch, mask)
            # ESS over the batch of per-sequence weights (the (N,1) column), not
            # the singleton token axis.
            ess = compute_ess(weights.reshape(1, -1))

        return {
            "loss": loss,  # Tensor, caller must call .backward()
            "grpo_loss_value": grpo_loss_value.item(),
            "ess": ess.item(),
            "mean_reward": rewards_tensor.mean().item(),
        }
