"""Loss functions for GRPO, Debiased GRPO, and the experiment-grid ablations.

The canonical entry point is ``compute_loss``: a thin orchestrator that wires
seven strategy objects from ``debiased_grpo.strategies`` into a single per-step
gradient. Every experiment cell is expressed as a different
combination of strategy instances --- the orchestrator itself contains zero
conditional branches over which axis is on or off.

Three preset wrappers are provided for the most common configurations:
    grpo_loss     --- paper-GRPO (DeepSeekMath, all four failure modes on)
    debiased_loss --- full Debiased GRPO (unbiased baseline + length norm + full-sequence IS)
    rloo_loss     --- REINFORCE-LOO baseline, no IS correction or clipping

Use ``debiased_grpo.config.build_components`` to translate a ``LossConfig`` (or
the flat YAML mapping) into the dict of strategies expected by
``compute_loss``.

A legacy ``grpo_loss_unified`` function is preserved as a translator from the
flat string flags used by older callers. New code should call ``compute_loss``
directly with strategy objects.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from debiased_grpo.strategies import (
    AdvantageNormalizer,
    Baseline,
    BaselineShifter,
    Clipper,
    ISWeighter,
    LengthAggregator,
    RewardAssigner,
)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_loss(
    log_probs: Tensor,
    ref_log_probs: Tensor,
    rewards: Tensor,
    mask: Tensor,
    *,
    baseline: Baseline,
    reward_assigner: RewardAssigner,
    advantage_normalizer: AdvantageNormalizer,
    baseline_shifter: BaselineShifter,
    clipper: Clipper,
    is_weighter: ISWeighter,
    length_aggregator: LengthAggregator,
    baseline_rewards: Optional[Tensor] = None,
    token_rewards: Optional[Tensor] = None,
    behavior_log_probs: Optional[Tensor] = None,
    kl_ref_coef: float = 0.0,
    kl_behavior_coef: float = 0.0,
) -> Tensor:
    """GRPO / Debiased GRPO loss with every component injected.

    The pipeline executes in a fixed order:

        1. baseline.compute             ->  b_i, shape (B,)
        2. reward_assigner.assign       ->  per-token advantage A_{i,t}, shape (B, T)
        3. advantage_normalizer.apply   ->  optional 1/sigma scaling
        4. baseline_shifter.shift       ->  optional second-baseline subtraction
        5. clipper.pre_product          ->  optional per-token log-ratio clamp
        6. is_weighter.compute          ->  per-token IS weight, shape (B, T) or (B, 1)
        7. clipper.compose_loss         ->  per-token loss, shape (B, T)
        8. length_aggregator.aggregate  ->  scalar loss
        9. (optional) + kl_ref_coef·KL(π_θ||π_ref) + kl_behavior_coef·KL(π_θ||π_behavior)

    No conditional branching on which strategy is in use lives inside this
    function; every variation is expressed by swapping a strategy instance.

    Args:
        log_probs:           Per-token log-probs from the policy being trained, (B, T).
                             Must retain its grad_fn.
        ref_log_probs:       Per-token log-probs from the frozen reference policy,
                             (B, T). Used as the KL anchor when ``kl_ref_coef > 0``.
                             If ``behavior_log_probs`` is None, ``ref_log_probs``
                             is also used as the IS denominator (the on-policy
                             fallback path).
        rewards:             Per-rollout scalar rewards, (B,).
        mask:                Boolean mask, True at non-padding positions, (B, T).
        baseline_rewards:    Optional rewards from independent reference rollouts,
                             required by ``IndependentBaseline``.
        token_rewards:       Optional per-token reward tensor, (B, T). Required for
                             non-trivial reward-to-go; otherwise a sparse terminal
                             representation is built internally.
        behavior_log_probs:  Per-token log-probs from the **sampling-time policy**
                             (the behavior policy that generated the rollouts),
                             (B, T). When provided, this is the IS denominator
                             ``log pi_behavior(a)`` and the IS log-ratio becomes
                             ``log pi_theta - log pi_behavior``. When None, the
                             function uses ``ref_log_probs`` as the IS denominator
                             (the on-policy fallback path, where the IS ratio is 1
                             at the sampling step).
        kl_ref_coef:         Scalar weight β on KL(π_θ || π_ref), the RLHF anchor
                             to the frozen base (GRPO's KL term). k3 estimator,
                             unbiased and non-negative; aggregated with the same
                             length aggregator as the policy loss.
        kl_behavior_coef:    Scalar weight on KL(π_θ || π_behavior), an unbiased
                             soft trust region on the IS ratio (complement to /
                             substitute for PPO clipping). No-op when
                             ``behavior_log_probs`` is None.

    Returns:
        Scalar loss tensor with grad_fn attached.
    """
    b = baseline.compute(rewards, baseline_rewards)
    advantage = reward_assigner.assign(rewards, b, token_rewards, mask)
    advantage = advantage_normalizer.apply(advantage, rewards)
    advantage = baseline_shifter.shift(advantage)

    # IS denominator. With multiple inner gradient steps per rollout batch the
    # importance ratio is taken against the sampling-time ("behavior") policy,
    # so ``behavior_log_probs`` is the denominator. When it is not supplied (the
    # single-update preset helpers below, and unit tests) the ratio is taken
    # against the reference policy instead.
    is_denominator = behavior_log_probs if behavior_log_probs is not None else ref_log_probs

    log_ratio = (log_probs - is_denominator) * mask.float()
    log_ratio = clipper.pre_product(log_ratio)
    ratio = is_weighter.compute(log_ratio, mask)

    token_loss = clipper.compose_loss(ratio, advantage, mask)
    main_loss = length_aggregator.aggregate(token_loss, mask)

    # Two independent KL anchors, each added with its own coefficient:
    #   * kl_ref_coef      → KL(π_θ ∥ π_ref):       the RLHF anchor to the frozen
    #                        base (GRPO's KL term). Keeps the policy near the base.
    #   * kl_behavior_coef → KL(π_θ ∥ π_behavior):  an unbiased soft trust region
    #                        on the IS ratio (alternative/complement to PPO clip).
    # Both use Schulman's k3 estimator (http://joschu.net/blog/kl-approx.html):
    #   KL(π_θ ∥ π_anchor) ~ E_{a~π_θ}[ exp(δ) − δ − 1 ],  δ = log π_anchor − log π_θ
    # which is unbiased and non-negative. δ is masked to 0 at padding BEFORE exp()
    # so padding can never produce exp(junk)=inf; the result is masked again.
    def _kl_to(anchor_log_probs: Tensor) -> Tensor:
        delta = (anchor_log_probs - log_probs) * mask.float()
        kl_per_token = (delta.exp() - delta - 1.0) * mask.float()
        return length_aggregator.aggregate(kl_per_token, mask)

    loss = main_loss
    if kl_ref_coef > 0.0:
        loss = loss + kl_ref_coef * _kl_to(ref_log_probs)
    if kl_behavior_coef > 0.0 and behavior_log_probs is not None:
        loss = loss + kl_behavior_coef * _kl_to(behavior_log_probs)
    return loss


# ---------------------------------------------------------------------------
# Preset wrappers (back-compat)
# ---------------------------------------------------------------------------

def grpo_loss(
    log_probs: Tensor,
    ref_log_probs: Tensor,
    rewards: Tensor,
    mask: Tensor,
    clip_eps: float = 0.2,
) -> Tensor:
    """G0 paper preset --- mean-with-self baseline, full-sequence IS + PPO clip,
    per-response averaging, std normalisation on.
    """
    from debiased_grpo.config import build_components, grpo_paper_config

    cfg = grpo_paper_config()
    cfg.clip_eps = clip_eps
    components = build_components(cfg)
    return compute_loss(log_probs, ref_log_probs, rewards, mask, **components)


def debiased_loss(
    log_probs: Tensor,
    ref_log_probs: Tensor,
    rewards: Tensor,
    baseline_rewards: Tensor,
    mask: Tensor,
    fixed_divisor: float = 1.0,
) -> Tensor:
    """G1 — full Debiased GRPO.

    Independent π_ref baseline, full-sequence IS (the unbiased correction for
    a terminal reward) with a log-weight clamp at 5.0 for numerical stability,
    no clipping, fixed-constant length aggregation, no std normalisation.

    This preset is single-update: it takes the importance ratio against the
    reference policy. The multi-inner-step regime (ratio against the
    sampling-time policy) is exercised by the trainer, which passes
    ``behavior_log_probs`` to :func:`compute_loss` directly.
    """
    from debiased_grpo.config import build_components, debiased_grpo_config

    cfg = debiased_grpo_config(fixed_divisor=fixed_divisor)
    components = build_components(cfg)
    return compute_loss(
        log_probs, ref_log_probs, rewards, mask,
        baseline_rewards=baseline_rewards, **components,
    )


def rloo_loss(
    log_probs: Tensor,
    ref_log_probs: Tensor,
    rewards: Tensor,
    mask: Tensor,
) -> Tensor:
    """REINFORCE Leave-One-Out (Ahmadian et al. 2024, arXiv:2402.14740).

    LOO baseline applied directly to the log-prob gradient --- no IS
    correction, no PPO clipping. ``ref_log_probs`` is accepted for signature
    symmetry with the other losses but does not enter the gradient.

    Kept as a separate function rather than a strategy combination because
    REINFORCE's surrogate is ``-A * log_pi_theta`` (autograd differentiates
    through ``log_pi_theta`` directly), whereas the strategy pipeline's
    surrogate is ``-ratio * A`` (autograd differentiates through ``ratio``).
    The two surrogates produce different gradients.
    """
    B = rewards.shape[0]
    if B > 1:
        total = rewards.sum()
        loo = (total - rewards) / (B - 1)
    else:
        loo = torch.zeros_like(rewards)
    advantage = (rewards - loo).unsqueeze(1)                # (B, 1)
    token_loss = -(log_probs * advantage) * mask.float()
    total_tokens = mask.float().sum().clamp(min=1.0)
    return token_loss.sum() / total_tokens


# ---------------------------------------------------------------------------
# Legacy flat-flag API (back-compat for callers built before the refactor)
# ---------------------------------------------------------------------------

def grpo_loss_unified(
    log_probs: Tensor,
    ref_log_probs: Tensor,
    rewards: Tensor,
    mask: Tensor,
    baseline: str = "mean_with_self",
    is_weighting: str = "full_sequence_ppo",
    length_norm: str = "per_response",
    std_norm: bool = True,
    baseline_rewards: Optional[Tensor] = None,
    clip_eps: float = 0.2,
    fixed_divisor: float = 1.0,
) -> Tensor:
    """Flat-flag back-compat wrapper. Translates legacy strings into a
    ``LossConfig`` and calls ``compute_loss``.
    """
    from debiased_grpo.config import LossConfig, build_components

    if is_weighting == "full_sequence_ppo":
        new_is, new_clip = "full_sequence", "ppo_classical"
    elif is_weighting == "full_sequence":
        new_is, new_clip = "full_sequence", "none"
    else:
        raise ValueError(
            f"Unknown legacy is_weighting={is_weighting!r}; expected one of "
            "{'full_sequence_ppo', 'full_sequence'}"
        )

    cfg = LossConfig(
        baseline=baseline,
        reward_assignment="broadcast",
        is_weighting=new_is,
        clipping=new_clip,
        length_norm=length_norm,
        std_norm=std_norm,
        clip_eps=clip_eps,
        fixed_divisor=fixed_divisor,
    )
    components = build_components(cfg)
    return compute_loss(
        log_probs, ref_log_probs, rewards, mask,
        baseline_rewards=baseline_rewards, **components,
    )
