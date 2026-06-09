"""
Strategy classes for the GRPO / Debiased GRPO loss pipeline.

The loss in ``debiased_grpo.losses.compute_loss`` is a thin orchestrator that wires
seven small strategies together; every experiment cell is expressed as a
different combination of strategy instances rather than as conditional branches
inside the loss function.

The strategies, in the order they execute inside the orchestrator:

    1. ``Baseline``              — per-rollout scalar baseline b_i.
    2. ``RewardAssigner``        — map (rewards, baseline) to per-token advantage.
    3. ``AdvantageNormalizer``   — optional per-prompt scaling.
    4. ``BaselineShifter``       — optional second-baseline shift (e.g. EMA).
    5. ``Clipper.pre_product``   — optional per-token log-ratio clamp.
    6. ``ISWeighter``            — full-sequence or per-token IS weight.
    7. ``Clipper.compose_loss``  — assemble per-token loss (with optional PPO min-form).
    8. ``LengthAggregator``      — reduce per-token loss to scalar.

Adding a new variant means writing a new class that implements the relevant
``Protocol`` — the orchestrator does not change. ``validate_combination`` is the
config-build-time hook for any cross-strategy invariants (currently none are
required: each field is validated independently by its registry).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor

from debiased_grpo.utils import EMABaseline, make_token_rewards, reverse_cumsum


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class Baseline(Protocol):
    """Compute a per-rollout scalar baseline b_i.

    The unbiasedness identity (see ``notes/derivation.md`` §2) requires b_i to
    be conditionally independent of the action a_i being differentiated. Each
    concrete implementation either satisfies this directly (LOO, independent
    pi_ref) or accepts the resulting O(1/N) bias (mean-with-self).
    """

    def compute(self, rewards: Tensor, baseline_rewards: Optional[Tensor]) -> Tensor:
        """Return shape (B,) baseline values, one per rollout."""


@runtime_checkable
class RewardAssigner(Protocol):
    """Map per-rollout rewards and baselines into a per-token advantage.

    Output shape is (B, T). Reward-broadcast assigners place the same
    (rewards - baseline) value at every non-padding position; reward-to-go
    assigners place ``(sum_{k>=t} r_{i,k}) - b_i`` at position t.
    """

    def assign(
        self,
        rewards: Tensor,
        baseline: Tensor,
        token_rewards: Optional[Tensor],
        mask: Tensor,
    ) -> Tensor: ...


@runtime_checkable
class AdvantageNormalizer(Protocol):
    """Optional per-prompt scaling of the per-token advantage tensor.

    ``StdNormalizer`` divides by the group reward standard deviation (paper
    GRPO); ``IdentityNormalizer`` is a no-op. Std normalisation is a changed
    objective rather than a bias fix --- see ``notes/derivation.md`` §6c.
    """

    def apply(self, advantage: Tensor, rewards: Tensor) -> Tensor: ...


@runtime_checkable
class BaselineShifter(Protocol):
    """Optional second-baseline subtraction applied to the per-token advantage.

    ``EMAShift`` subtracts the value of an ``EMABaseline`` updated past-only
    by the trainer. ``NoShift`` is a no-op. The EMA value must be conditionally
    independent of the current batch's actions for the gradient to remain
    unbiased --- see ``notes/derivation.md`` §7.
    """

    def shift(self, advantage: Tensor) -> Tensor: ...


@runtime_checkable
class Clipper(Protocol):
    """Two-stage clipping interface.

    ``pre_product`` clamps per-token log-ratios *before* the IS weight is
    formed. ``compose_loss`` builds the per-token loss tensor from the
    already-computed IS weight and advantage. Most clippers act in only one
    stage and pass the other through unchanged.
    """

    def pre_product(self, log_ratio: Tensor) -> Tensor: ...

    def compose_loss(self, ratio: Tensor, advantage: Tensor, mask: Tensor) -> Tensor: ...


@runtime_checkable
class ISWeighter(Protocol):
    """Convert masked per-token log-ratios into per-token IS weights.

    ``FullSequenceIS`` returns a ``(B, 1)`` tensor (the same full-sequence
    ratio for every token in a rollout); ``PerTokenIS`` returns a ``(B, T)``
    tensor of per-token ratios with no cumulation.
    """

    def compute(self, log_ratio: Tensor, mask: Tensor) -> Tensor: ...


@runtime_checkable
class LengthAggregator(Protocol):
    """Reduce a (B, T) per-token loss tensor to a scalar."""

    def aggregate(self, token_loss: Tensor, mask: Tensor) -> Tensor: ...


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class MeanWithSelfBaseline:
    """Group mean including the current rollout's reward (paper GRPO).

    Biased: b_i is a function of a_i through r_i, breaking the
    baseline-subtraction identity. The bias is O(1/N) and persists at any
    sample size; see ``notes/derivation.md`` §3.
    """

    def compute(self, rewards: Tensor, baseline_rewards: Optional[Tensor]) -> Tensor:
        return rewards.mean().expand_as(rewards)


class LOOBaseline:
    """Leave-one-out group mean (RLOO; Ahmadian et al. 2024).

    Unbiased: each b_i depends only on the other rollouts a_{j != i}, which
    are conditionally independent of a_i under i.i.d. group sampling. For
    B == 1 the implementation falls back to a zero baseline.
    """

    def compute(self, rewards: Tensor, baseline_rewards: Optional[Tensor]) -> Tensor:
        B = rewards.shape[0]
        if B <= 1:
            return torch.zeros_like(rewards)
        total = rewards.sum()
        return (total - rewards) / (B - 1)


class IndependentBaseline:
    """Mean of M reference-policy rollouts collected independently of the gradient rollouts.

    Unbiased: the baseline rollouts share no random bits with the gradient
    rollouts, so they are conditionally independent of a_i. Variance is
    typically slightly higher than LOO in late training (LOO estimates
    V^pi_theta whereas independent pi_ref estimates V^pi_ref) but composes
    naturally with the off-policy IS correction, which already samples from
    pi_ref.
    """

    def compute(self, rewards: Tensor, baseline_rewards: Optional[Tensor]) -> Tensor:
        if baseline_rewards is None:
            raise ValueError(
                "IndependentBaseline requires baseline_rewards "
                "(M rewards from pi_ref rollouts)."
            )
        return baseline_rewards.mean().expand_as(rewards)


# ---------------------------------------------------------------------------
# Reward assigners
# ---------------------------------------------------------------------------

class BroadcastAssigner:
    """Broadcast the scalar advantage (r_i - b_i) across every non-padding token.

    Mathematically equivalent to placing a single terminal reward and using
    reward-to-go. Default for binary terminal-reward settings such as GSM8K.
    """

    def assign(
        self,
        rewards: Tensor,
        baseline: Tensor,
        token_rewards: Optional[Tensor],
        mask: Tensor,
    ) -> Tensor:
        # token_rewards is ignored: total reward is the scalar `rewards[i]`.
        advantage_scalar = (rewards - baseline).unsqueeze(1)               # (B, 1)
        return advantage_scalar.expand(rewards.shape[0], mask.shape[1])    # (B, T)


class RewardToGoAssigner:
    """Reward-to-go: at token t, advantage uses sum_{k>=t} r_{i,k} - b_i.

    When ``token_rewards`` is None, the per-token reward tensor is built from
    the scalar ``rewards`` by placing each scalar at the rollout's last
    non-padding position (sparse terminal representation). Under that
    construction the reward-to-go at every non-padding t equals the scalar
    reward, so the gradient is sample-by-sample identical to
    ``BroadcastAssigner`` --- there is no variance reduction unless callers
    supply a non-trivial ``token_rewards`` (KL-per-token, length penalty, or a
    process reward model). See ``notes/derivation.md`` §6.
    """

    def assign(
        self,
        rewards: Tensor,
        baseline: Tensor,
        token_rewards: Optional[Tensor],
        mask: Tensor,
    ) -> Tensor:
        if token_rewards is None:
            token_rewards = make_token_rewards(rewards, mask)
        if token_rewards.shape != mask.shape:
            raise ValueError(
                f"token_rewards shape {tuple(token_rewards.shape)} must match "
                f"mask shape {tuple(mask.shape)}"
            )
        masked_rewards = token_rewards * mask.float()
        rtg = reverse_cumsum(masked_rewards, dim=1)                        # (B, T)
        return rtg - baseline.unsqueeze(1)


# ---------------------------------------------------------------------------
# Advantage normalizers
# ---------------------------------------------------------------------------

class IdentityNormalizer:
    """Pass the advantage through unchanged."""

    def apply(self, advantage: Tensor, rewards: Tensor) -> Tensor:
        return advantage


@dataclass
class StdNormalizer:
    """Divide the advantage by the group reward standard deviation.

    Paper GRPO uses this; Dr. GRPO drops it because the per-prompt 1/sigma(x)
    factor sits inside the expectation and changes the optimised objective to
    a variance-weighted version of expected reward.
    """

    eps: float = 1e-8

    def apply(self, advantage: Tensor, rewards: Tensor) -> Tensor:
        std = rewards.std(unbiased=False).clamp(min=self.eps)
        return advantage / std


# ---------------------------------------------------------------------------
# Baseline shifters (second baseline)
# ---------------------------------------------------------------------------

class NoShift:
    """No second-baseline subtraction (default)."""

    def shift(self, advantage: Tensor) -> Tensor:
        return advantage


@dataclass
class EMAShift:
    """Subtract a scalar EMA-of-mean-advantage from the per-token advantage.

    The EMA must be updated by the trainer *after* the gradient step using the
    current batch's mean advantage, so that the value applied at step t depends
    only on batches < t. Past-only updating makes the shift conditionally
    independent of the current batch's actions and therefore unbiased.
    """

    ema: EMABaseline

    def shift(self, advantage: Tensor) -> Tensor:
        return advantage - self.ema.value


# ---------------------------------------------------------------------------
# Clippers
# ---------------------------------------------------------------------------

class NoClip:
    """No clipping at any stage.

    Numerical stability of ``FullSequenceIS`` comes from its optional
    ``log_w_clamp`` cap on the summed log-weight; the loss expression is the
    unbiased full-sequence (or per-token) IS estimator up to a uniform
    Adam-absorbed rescale.
    """

    def pre_product(self, log_ratio: Tensor) -> Tensor:
        return log_ratio

    def compose_loss(self, ratio: Tensor, advantage: Tensor, mask: Tensor) -> Tensor:
        return -(ratio * advantage) * mask.float()


@dataclass
class LogRatioTokenClip:
    """Clamp each per-token log-ratio to [-c, +c] BEFORE the IS product is formed.

    Bounds the per-token contribution of the log-ratio so that the
    full-sequence sum cannot grow faster than ``c`` per token.
    Biased: per-token truncation breaks the unbiasedness of the IS weight.
    Different objective from PPO-classical, which clips the *full-sequence*
    ratio after the product. See ``notes/derivation.md`` §8.

    Args:
        c: Symmetric clamp bound on log_pi_theta - log_pi_ref. A reasonable
           starting value is ``log(1.5) ~= 0.405``, mirroring the standard
           PPO ratio bound of [1/1.5, 1.5].
    """

    c: float = math.log(1.5)

    def pre_product(self, log_ratio: Tensor) -> Tensor:
        return log_ratio.clamp(min=-self.c, max=self.c)

    def compose_loss(self, ratio: Tensor, advantage: Tensor, mask: Tensor) -> Tensor:
        return -(ratio * advantage) * mask.float()


@dataclass
class PPOClassicalClip:
    """Standard PPO-Clip min-form applied to an IS ratio.

    Clamps the IS ratio to ``[1 - eps, 1 + eps]`` and returns
    ``-min(ratio*A, clipped_ratio*A)`` per token. The clamp is applied via the
    ratio directly (not in log space) to match the published PPO objective
    semantics; numerically equivalent to ``log_ratio.clamp(log(1-eps), log(1+eps))``
    followed by ``exp`` since ``clamp`` commutes with the monotonic ``exp``.

    Works with either:
      * ``FullSequenceIS`` (ratio shape ``(B, 1)``): the same clip decision
        applies to all tokens in a sequence — a single token's drift can
        push the whole-sequence ratio outside the window and clip everything.
      * ``PerTokenIS`` (ratio shape ``(B, T)``): per-token clipping — each
        token's ratio is clipped independently. This is the paper-faithful
        DeepSeekMath GRPO formulation (Eq. 21).
    """

    eps: float = 0.2

    def pre_product(self, log_ratio: Tensor) -> Tensor:
        return log_ratio

    def compose_loss(self, ratio: Tensor, advantage: Tensor, mask: Tensor) -> Tensor:
        clipped_ratio = ratio.clamp(1.0 - self.eps, 1.0 + self.eps)
        unclipped = ratio * advantage
        clipped = clipped_ratio * advantage
        return -torch.min(unclipped, clipped) * mask.float()


# ---------------------------------------------------------------------------
# IS weighters
# ---------------------------------------------------------------------------

class FullSequenceIS:
    """Full-trajectory IS ratio: rho(y) = exp(sum_t log rho_t).

    This is the **unbiased** off-policy correction for a terminal /
    sequence-level reward: the reward depends on the whole completion, so each
    token's score must be reweighted by the importance ratio of the entire
    sequence ``π_θ(y) / π_behavior(y) = exp(Σ_t log ρ_t)``, not by any
    per-token or cumulative-prefix ratio. Returns a ``(B, 1)`` tensor that
    broadcasts across the token dimension when multiplied by the advantage.

    The optional ``log_w_clamp`` caps the summed log-weight at ``log_w_clamp``
    *before* the exponential, bounding the weight to ``exp(log_w_clamp)``. This
    is the one residual approximation: a numerical guard against the log-normal
    variance of ``exp(Σ log ρ_t)`` blowing up for long completions with policy
    drift. ``None`` (the default) keeps the estimator strictly unbiased.

    Args:
        log_w_clamp: Upper cap on the summed log-weight before exp, or ``None``
                     for the strictly-unbiased estimator.
    """

    def __init__(self, log_w_clamp: Optional[float] = None) -> None:
        self.log_w_clamp = log_w_clamp

    def compute(self, log_ratio: Tensor, mask: Tensor) -> Tensor:
        # Caller has already masked log_ratio (zeros at padding positions),
        # but multiply by the mask defensively in case a custom clipper added
        # non-zero values back at padding positions.
        log_w = (log_ratio * mask.float()).sum(dim=1, keepdim=True)  # (B, 1)
        if self.log_w_clamp is not None:
            log_w = log_w.clamp(max=self.log_w_clamp)
        return torch.exp(log_w)


class PerTokenIS:
    """Per-token IS ratio: w_{i,t} = exp(log_ratio_{i,t}).  No cumulation.

    Returns ``(B, T)``.  Each token's gradient is weighted by its own ratio
    ``π_θ(a_t | ctx_t) / π_behavior(a_t | ctx_t)``.  This is the IS estimator
    used in DeepSeekMath GRPO Eq. 21 (and HuggingFace TRL's GRPOTrainer):
    when paired with ``PPOClassicalClip``, every token is clipped
    independently in ``[1−ε, 1+ε]``, so individual outlier tokens get clipped
    without killing the gradient on the rest of the sequence — unlike
    ``FullSequenceIS`` whose product over T tokens lands far outside the clip
    window after even small per-token drift.

    Compared to ``FullSequenceIS`` this estimator is **biased** for a
    terminal / sequence-level reward (the full-sequence ratio needed for the
    proper off-policy correction is replaced by per-token ratios), but the
    bias is small in practice when the inner-loop drift between sampling and
    gradient is small — which is the regime PPO/GRPO operate in.
    """

    def compute(self, log_ratio: Tensor, mask: Tensor) -> Tensor:
        # Multiply by mask before exp so padding positions get exp(0) = 1
        # (a no-op weight); the final ``* mask`` in the loss zeros their
        # contribution anyway, but keeping the weight sane avoids surprising
        # downstream consumers (e.g. ESS / log-ratio diagnostics) with
        # non-trivial padding values.
        return torch.exp(log_ratio * mask.float())


# ---------------------------------------------------------------------------
# Length aggregators
# ---------------------------------------------------------------------------

class PerResponseAggregator:
    """Average per-token loss within each rollout, then mean across rollouts.

    Paper GRPO. The per-rollout 1/T_i factor sits inside the expectation and
    introduces a length-dependent reweighting of the gradient. Short
    sequences get larger per-token gradient.
    """

    def aggregate(self, token_loss: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.float()
        tokens_per_seq = mask_f.sum(dim=1).clamp(min=1.0)
        per_seq = token_loss.sum(dim=1) / tokens_per_seq
        return per_seq.mean()


class TokenLevelAggregator:
    """Sum per-token loss and divide by total non-padding token count (Dr. GRPO / DAPO).

    Every token contributes equally to the gradient. The denominator
    ``Σᵢ Tᵢ`` is itself a function of the batch's actions, so this estimator
    has O(1/B) bias from finite-batch stochasticity — small in practice but
    not strictly zero. Use ``FixedConstantAggregator`` for exact unbiasedness.
    """

    def aggregate(self, token_loss: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.float()
        total_tokens = mask_f.sum().clamp(min=1.0)
        return token_loss.sum() / total_tokens


@dataclass
class FixedConstantAggregator:
    """Sum per-token loss and divide by a fixed scalar set at config time.

    The divisor is a constant (typically ``B · T_max``), so it commutes with
    the expectation: this is the exact-unbiased length-normalisation form.
    Adam absorbs the constant scale; the choice of divisor only affects the
    effective learning rate.
    """

    divisor: float

    def aggregate(self, token_loss: Tensor, mask: Tensor) -> Tensor:
        return token_loss.sum() / self.divisor


# ---------------------------------------------------------------------------
# Combination validation
# ---------------------------------------------------------------------------

def validate_combination(
    *,
    baseline: Baseline,
    reward_assigner: RewardAssigner,
    advantage_normalizer: AdvantageNormalizer,
    baseline_shifter: BaselineShifter,
    clipper: Clipper,
    is_weighter: ISWeighter,
    length_aggregator: LengthAggregator,
) -> None:
    """Hook for cross-strategy invariants, called once at config-build time.

    Every individual field is already validated by its registry lookup; this is
    where invariants that span *more than one* strategy would go, so an
    incompatible combination would surface as a single ValueError before training
    rather than deep inside the loss. No such invariant is currently required —
    the full component set is accepted; the signature is the documented extension
    surface.
    """
    return None
