"""
Config dataclass and strategy factory for the GRPO / Debiased GRPO loss pipeline.

This module decouples user-facing configuration (flat string-valued YAML
fields) from the strategy objects consumed by ``debiased_grpo.losses.compute_loss``.
Adding a new strategy means registering it in one of the registries below;
the orchestrator and trainer code do not need to change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from debiased_grpo.strategies import (
    AdvantageNormalizer,
    Baseline,
    BaselineShifter,
    BroadcastAssigner,
    Clipper,
    EMAShift,
    FullSequenceIS,
    IdentityNormalizer,
    IndependentBaseline,
    ISWeighter,
    LengthAggregator,
    LOOBaseline,
    LogRatioTokenClip,
    MeanWithSelfBaseline,
    FixedConstantAggregator,
    NoClip,
    NoShift,
    PerResponseAggregator,
    PerTokenIS,
    PPOClassicalClip,
    RewardAssigner,
    RewardToGoAssigner,
    StdNormalizer,
    TokenLevelAggregator,
    validate_combination,
)
from debiased_grpo.utils import EMABaseline


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

BASELINES: Dict[str, Callable[[], Baseline]] = {
    "mean_with_self": MeanWithSelfBaseline,
    "loo": LOOBaseline,
    "independent": IndependentBaseline,
}

REWARD_ASSIGNERS: Dict[str, Callable[[], RewardAssigner]] = {
    "broadcast": BroadcastAssigner,
    "reward_to_go": RewardToGoAssigner,
}

NORMALIZERS: Dict[str, Callable[[], AdvantageNormalizer]] = {
    "identity": IdentityNormalizer,
    "std": StdNormalizer,
}

IS_WEIGHTERS: Dict[str, Callable[..., ISWeighter]] = {
    "full_sequence": lambda **kw: FullSequenceIS(log_w_clamp=kw.get("log_w_clamp")),
    "per_token": lambda **kw: PerTokenIS(),
}

CLIPPERS: Dict[str, Callable[..., Clipper]] = {
    "none": lambda **kw: NoClip(),
    "log_ratio_token": lambda **kw: LogRatioTokenClip(
        c=kw.get("log_ratio_clip_c", math.log(1.5))
    ),
    "ppo_classical": lambda **kw: PPOClassicalClip(eps=kw.get("clip_eps", 0.2)),
}

AGGREGATORS: Dict[str, Callable[..., LengthAggregator]] = {
    "per_response": lambda **kw: PerResponseAggregator(),
    "token_level": lambda **kw: TokenLevelAggregator(),
    "fixed_constant": lambda **kw: FixedConstantAggregator(
        divisor=float(kw["fixed_divisor"])
    ),
}


# ---------------------------------------------------------------------------
# Loss configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    """Flat string-valued config describing a single cell of the experiment grid.

    Maps to strategy objects via ``build_components``. Fields are kept as
    primitives (strings, bools, floats) so the same dataclass can be loaded
    from YAML, command-line flags, or constructed directly in tests.
    """

    baseline: str = "mean_with_self"
    reward_assignment: str = "broadcast"
    is_weighting: str = "full_sequence"
    clipping: str = "ppo_classical"
    length_norm: str = "per_response"
    std_norm: bool = True
    ema_baseline: bool = False

    # Hyperparameters
    clip_eps: float = 0.2
    log_ratio_clip_c: float = math.log(1.5)
    ema_decay: float = 0.95

    # Constant divisor for length_norm="fixed_constant" (typically B · T_max).
    # Required when length_norm == "fixed_constant"; ignored otherwise.
    fixed_divisor: float = 1.0

    # Clamp on the full-sequence log IS-weight before exp; None = strictly
    # unbiased, a finite cap bounds the weight at the cost of a small bias
    # (used by the debiased config).
    log_w_clamp: float | None = None

    # Two independent KL anchors (Schulman k3 estimator), each with its own β:
    #   kl_ref_coef      → β·KL(π_θ ∥ π_ref): the RLHF anchor to the frozen base.
    #                      Default 0.04 matches DeepSeekMath GRPO. 0.0 disables it.
    #   kl_behavior_coef → β·KL(π_θ ∥ π_behavior): an unbiased soft trust region on
    #                      the IS ratio (alternative/complement to PPO clipping that
    #                      does not bias the importance weight). Default 0.0; no-op
    #                      when behavior log-probs are unavailable (single-update presets).
    kl_ref_coef: float = 0.04
    kl_behavior_coef: float = 0.0


# ---------------------------------------------------------------------------
# Preset configs for grid cells
# ---------------------------------------------------------------------------

def grpo_paper_config() -> LossConfig:
    """G0 --- paper GRPO with all four failure modes active.

    Uses ``full_sequence`` IS for backward-compat with the original codebase;
    paper-faithful per-token IS lives in ``grpo_paper_token_config`` (G0a).
    """
    return LossConfig(
        baseline="mean_with_self",
        reward_assignment="broadcast",
        is_weighting="full_sequence",
        clipping="ppo_classical",
        length_norm="per_response",
        std_norm=True,
    )


def grpo_paper_token_config() -> LossConfig:
    """G0a --- paper-faithful GRPO with per-token IS (DeepSeekMath Eq. 21).

    Identical to ``grpo_paper_config`` except for ``is_weighting="per_token"``,
    which makes the PPO clip apply per-token (each token's ``r_t`` clipped to
    ``[1-eps, 1+eps]``) rather than to the full-sequence product.
    """
    return LossConfig(
        baseline="mean_with_self",
        reward_assignment="broadcast",
        is_weighting="per_token",
        clipping="ppo_classical",
        length_norm="per_response",
        std_norm=True,
    )


def debiased_grpo_config(fixed_divisor: float = 1.0) -> LossConfig:
    """G1 — Debiased GRPO.

    Independent π_ref baseline; full-sequence IS (the unbiased choice for
    terminal rewards) with a log-weight clamp at 5.0 for numerical stability;
    no clipping; fixed-constant length aggregation; no std-norm. The only
    residual bias is the clamp.

    ``fixed_divisor`` is the constant denominator for the length-normalisation
    term. Set it to ``B · T_max`` from the training config so the divisor is
    a constant of the data, not of the actions. Adam absorbs the constant
    scale; the choice only sets the effective LR.
    """
    return LossConfig(
        baseline="independent",
        reward_assignment="broadcast",
        is_weighting="full_sequence",
        clipping="none",
        length_norm="fixed_constant",
        std_norm=False,
        fixed_divisor=fixed_divisor,
        log_w_clamp=5.0,
    )


def rloo_config() -> LossConfig:
    """RLOO baseline --- LOO advantage with no IS correction or clipping."""
    return LossConfig(
        baseline="loo",
        reward_assignment="broadcast",
        is_weighting="per_token",  # IS weights are 1.0 when policy == ref
        clipping="none",
        length_norm="token_level",
        std_norm=False,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_components(
    cfg: LossConfig | Mapping[str, Any],
    ema: Optional[EMABaseline] = None,
) -> Dict[str, Any]:
    """Translate a ``LossConfig`` (or mapping) into a dict of strategy instances.

    The returned dict can be splatted directly into
    ``debiased_grpo.losses.compute_loss(...)`` as keyword arguments.

    Args:
        cfg: A ``LossConfig`` or any mapping with the same field names.
        ema: An ``EMABaseline`` instance, required iff ``cfg.ema_baseline`` is
             True. The factory does not own the EMA --- the trainer must
             update it past-only after each gradient step.

    Returns:
        Dict with keys ``baseline``, ``reward_assigner``,
        ``advantage_normalizer``, ``baseline_shifter``, ``clipper``,
        ``is_weighter``, ``length_aggregator``.

    Raises:
        ValueError: when an unknown registry key appears, when
            ``ema_baseline=True`` but no EMA is supplied, or when the resulting
            strategy combination is incoherent.
    """
    cfg = _as_config(cfg)

    components = {
        "baseline": _lookup(BASELINES, cfg.baseline, "baseline")(),
        "reward_assigner": _lookup(REWARD_ASSIGNERS, cfg.reward_assignment, "reward_assignment")(),
        "advantage_normalizer": _lookup(
            NORMALIZERS, "std" if cfg.std_norm else "identity", "std_norm"
        )(),
        "baseline_shifter": _build_shifter(cfg.ema_baseline, ema),
        "clipper": _lookup(CLIPPERS, cfg.clipping, "clipping")(
            clip_eps=cfg.clip_eps,
            log_ratio_clip_c=cfg.log_ratio_clip_c,
        ),
        "is_weighter": _lookup(IS_WEIGHTERS, cfg.is_weighting, "is_weighting")(
            log_w_clamp=cfg.log_w_clamp,
        ),
        "length_aggregator": _lookup(AGGREGATORS, cfg.length_norm, "length_norm")(
            fixed_divisor=cfg.fixed_divisor,
        ),
    }
    validate_combination(**components)
    return components


def _as_config(cfg: LossConfig | Mapping[str, Any]) -> LossConfig:
    if isinstance(cfg, LossConfig):
        return cfg
    fields = LossConfig.__dataclass_fields__.keys()
    kwargs = {k: cfg[k] for k in fields if k in cfg}
    return LossConfig(**kwargs)


def _lookup(registry: Dict[str, Callable], key: str, field_name: str) -> Callable:
    if key not in registry:
        raise ValueError(
            f"Unknown {field_name}={key!r}; valid options: {sorted(registry.keys())}"
        )
    return registry[key]


def _build_shifter(use_ema: bool, ema: Optional[EMABaseline]) -> BaselineShifter:
    if not use_ema:
        return NoShift()
    if ema is None:
        raise ValueError(
            "ema_baseline=True requires an EMABaseline instance to be passed "
            "to build_components(...)."
        )
    return EMAShift(ema)
