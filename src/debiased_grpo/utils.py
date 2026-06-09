"""
Utility functions for Debiased GRPO training.

Pure tensor operations and small stateful helpers with no model dependencies,
straightforward to unit-test on CPU with synthetic data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor


def compute_ess(weights: Tensor) -> Tensor:
    """Estimate the effective sample size (ESS) of IS weights.

    Uses the standard estimator:

        ESS = (sum_t w_t)^2 / sum_t w_t^2

    computed per sequence (row), then averaged over the batch.  When all
    weights are equal the ESS equals T (the number of tokens), so the return
    value is on the scale of sequence length — not normalised to [0, 1].

    Args:
        weights: Per-token IS weights. Shape (B, T).

    Returns:
        Scalar tensor containing the batch-averaged ESS.
    """
    sum_w = weights.sum(dim=1)         # (B,)
    sum_w_sq = (weights ** 2).sum(dim=1)  # (B,)
    ess_per_seq = (sum_w ** 2) / (sum_w_sq + 1e-8)  # (B,)
    return ess_per_seq.mean()


def mask_padding(tensor: Tensor, mask: Tensor, fill_value: float = 0.0) -> Tensor:
    """Zero out (or fill) positions where mask is False.

    A thin convenience wrapper so callers don't have to remember the
    torch.where argument order.

    Args:
        tensor:     Any float tensor to mask. Shape (B, T) or broadcastable.
        mask:       Boolean mask; True = keep, False = fill. Shape (B, T).
        fill_value: Value to write at masked-out positions. Defaults to 0.0.

    Returns:
        Tensor of the same shape as *tensor* with padding positions replaced by
        *fill_value*.
    """
    return torch.where(mask, tensor, torch.full_like(tensor, fill_value))


def normalize_rewards(rewards: Tensor) -> Tensor:
    """Standardize a batch of scalar rewards to zero mean and unit variance.

    Adds a small epsilon to the standard deviation to avoid division by zero
    when all rewards in the batch are identical (e.g. all-correct or all-wrong).

    The std uses ``correction=0`` (population std), not ``correction=1``
    (Bessel's correction / ddof=1).  With ddof=1 a single element (B=1) makes
    the denominator zero before epsilon is added, returning NaN.  Population std
    is correct here because we are normalising the observed rewards, not
    estimating a population standard deviation, and it keeps the epsilon guard
    effective for B=1: std=0 → return all-zeros instead of NaN.

    Args:
        rewards: Per-sequence scalar rewards. Shape (B,).

    Returns:
        Tensor of the same shape with mean ≈ 0 and std ≈ 1.
    """
    mean = rewards.mean()
    std = rewards.std(correction=0)  # population std; safe for B=1
    return (rewards - mean) / (std + 1e-8)


def reverse_cumsum(tensor: Tensor, dim: int = 1) -> Tensor:
    """Cumulative sum from the right.

    For a 1-D slice ``[x_0, x_1, ..., x_{T-1}]`` returns
    ``[x_0 + x_1 + ... + x_{T-1}, x_1 + ... + x_{T-1}, ..., x_{T-1}]`` — i.e.
    position ``t`` holds the sum of all elements at positions ``>= t``.

    Implemented as ``flip → cumsum → flip`` so that PyTorch autograd handles
    the backward pass correctly. No in-place operations.

    Args:
        tensor: Any float tensor.
        dim: Axis along which to accumulate. Default 1 (token axis).

    Returns:
        Tensor of the same shape as ``tensor`` with the reverse cumulative sum.
    """
    return torch.flip(torch.cumsum(torch.flip(tensor, dims=[dim]), dim=dim), dims=[dim])


def make_token_rewards(rewards: Tensor, mask: Tensor) -> Tensor:
    """Convert per-rollout scalar rewards into a per-token reward tensor.

    Places the scalar reward at the last non-padding position of each rollout
    and leaves all other positions at zero. This is the natural representation
    of a sparse terminal reward (e.g., the GSM8K binary correctness verifier):
    the reward is realised exactly when the response ends.

    Under this representation, reward-to-go ``Σ_{k>=t} r_{i,k}`` equals the
    scalar ``rewards[i]`` for every non-padding ``t`` (because only the
    terminal position contributes). The reward-to-go gradient is therefore
    sample-by-sample identical to the reward-broadcast gradient — there is no
    variance reduction. Per-token reward shaping (KL-per-token, length penalty,
    or a process reward model) must be supplied directly as a ``token_rewards``
    tensor of shape ``(B, T)`` to make reward-to-go non-trivial.

    Args:
        rewards: Per-rollout scalar rewards. Shape (B,).
        mask:    Boolean mask; True at non-padding positions. Shape (B, T).

    Returns:
        Float tensor of shape (B, T) with ``rewards[i]`` at the last True
        position of row ``i`` and zeros elsewhere.
    """
    B, T = mask.shape
    if rewards.shape != (B,):
        raise ValueError(
            f"rewards must have shape (B,) = ({B},), got {tuple(rewards.shape)}"
        )

    mask_int = mask.int()
    lengths = mask_int.sum(dim=1)                          # (B,)
    last_idx = (lengths - 1).clamp(min=0)                  # (B,) — index of last real token
    token_rewards = torch.zeros(B, T, dtype=rewards.dtype, device=rewards.device)
    # Rows with at least one real token receive the scalar at last_idx.
    rows_with_tokens = lengths > 0
    if rows_with_tokens.any():
        row_ids = torch.arange(B, device=rewards.device)[rows_with_tokens]
        col_ids = last_idx[rows_with_tokens]
        token_rewards[row_ids, col_ids] = rewards[rows_with_tokens]
    return token_rewards


class EMABaseline:
    """Scalar exponential moving average of per-batch mean advantage.

    Used by ``EMAShift`` (in ``debiased_grpo.strategies``) as a second baseline
    subtracted from the advantage. To preserve unbiasedness, the EMA value
    used at training step ``t`` must be a function of batches ``< t`` only —
    i.e. the caller MUST update the EMA *after* the gradient step using the
    current batch's mean advantage:

        ema_shift = ema.value if cfg.ema_baseline else 0.0
        loss = compute_loss(..., baseline_shifter=EMAShift(ema))
        loss.backward(); optimizer.step()
        if cfg.ema_baseline:
            ema.update(batch_mean_advantage)

    With this update timing, the EMA shift is conditionally independent of
    the current batch's actions and therefore acts as a state-independent
    constant offset — unbiased by the standard baseline-subtraction identity
    (see ``notes/derivation.md`` §7).

    For the first step (no past history), the EMA is uninitialised and
    ``value`` returns the supplied ``init`` (default 0.0). The first
    ``update()`` call sets the EMA to the supplied value directly rather than
    blending with ``init``, so warm-up does not bias the EMA toward 0.

    Args:
        decay: EMA smoothing factor in [0, 1). ``new = decay*old + (1-decay)*x``.
               Default 0.95 (≈ 20-step effective horizon).
        init:  Initial value used before the first ``update()`` call. Default 0.0.

    Attributes:
        value: Current EMA value (float). Read this when applying the shift.
        initialized: Whether ``update()`` has been called at least once.
    """

    def __init__(self, decay: float = 0.95, init: float = 0.0) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.value = float(init)
        self.initialized = False

    def update(self, x: float) -> None:
        """Fold a new observation into the EMA.

        First call sets ``value = x`` directly (warm start). Subsequent calls
        apply the standard EMA recursion.
        """
        x = float(x)
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * x

    def state_dict(self) -> Dict[str, Any]:
        """Serialisable state for checkpoint round-trip."""
        return {
            "decay": self.decay,
            "value": self.value,
            "initialized": self.initialized,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore from a previous ``state_dict()``."""
        self.decay = float(state["decay"])
        self.value = float(state["value"])
        self.initialized = bool(state["initialized"])

    def __repr__(self) -> str:
        return (
            f"EMABaseline(decay={self.decay}, value={self.value:.6f}, "
            f"initialized={self.initialized})"
        )
