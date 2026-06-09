"""Unit tests for the strategy classes and the strategy-based loss orchestrator.

Each strategy class is exercised in isolation; behavioural integration tests
that combine multiple strategies live further down. All tests run on CPU with
synthetic data --- no model downloads required.
"""

from __future__ import annotations

import math

import pytest
import torch

from debiased_grpo.config import (
    LossConfig,
    build_components,
    grpo_paper_config,
    grpo_paper_token_config,
    debiased_grpo_config,
)
from debiased_grpo.losses import compute_loss
from debiased_grpo.strategies import (
    BroadcastAssigner,
    EMAShift,
    FullSequenceIS,
    IdentityNormalizer,
    IndependentBaseline,
    LOOBaseline,
    LogRatioTokenClip,
    MeanWithSelfBaseline,
    NoClip,
    NoShift,
    PPOClassicalClip,
    PerResponseAggregator,
    PerTokenIS,
    RewardToGoAssigner,
    StdNormalizer,
    TokenLevelAggregator,
    validate_combination,
)
from debiased_grpo.utils import EMABaseline, make_token_rewards, reverse_cumsum


# ---------------------------------------------------------------------------
# Baseline strategies
# ---------------------------------------------------------------------------

def test_mean_with_self_baseline_equals_group_mean():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = MeanWithSelfBaseline().compute(rewards, baseline_rewards=None)
    assert b.shape == rewards.shape
    assert torch.allclose(b, torch.full_like(rewards, rewards.mean().item()))


def test_loo_baseline_excludes_self():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = LOOBaseline().compute(rewards, baseline_rewards=None)
    expected = torch.tensor([
        (2.0 + 3.0 + 4.0) / 3,
        (1.0 + 3.0 + 4.0) / 3,
        (1.0 + 2.0 + 4.0) / 3,
        (1.0 + 2.0 + 3.0) / 3,
    ])
    assert torch.allclose(b, expected, atol=1e-6)


def test_loo_baseline_b1_returns_zero():
    rewards = torch.tensor([7.0])
    b = LOOBaseline().compute(rewards, baseline_rewards=None)
    assert torch.allclose(b, torch.zeros(1))


def test_independent_baseline_uses_baseline_rewards_mean():
    rewards = torch.tensor([1.0, 2.0])
    baseline_rewards = torch.tensor([0.25, 0.5, 0.75])
    b = IndependentBaseline().compute(rewards, baseline_rewards)
    assert torch.allclose(b, torch.full_like(rewards, 0.5))


def test_independent_baseline_requires_baseline_rewards():
    with pytest.raises(ValueError, match="IndependentBaseline"):
        IndependentBaseline().compute(torch.zeros(3), baseline_rewards=None)


# ---------------------------------------------------------------------------
# Reward assigners
# ---------------------------------------------------------------------------

def test_broadcast_assigner_repeats_scalar_advantage():
    rewards = torch.tensor([1.0, 0.0])
    baseline = torch.tensor([0.5, 0.5])
    mask = torch.ones(2, 4, dtype=torch.bool)
    advantage = BroadcastAssigner().assign(rewards, baseline, token_rewards=None, mask=mask)
    assert advantage.shape == (2, 4)
    assert torch.allclose(advantage[0], torch.full((4,), 0.5))
    assert torch.allclose(advantage[1], torch.full((4,), -0.5))


def test_reward_to_go_terminal_only_matches_broadcast():
    """Under sparse terminal reward (RtG via make_token_rewards), the output
    must equal the broadcast assigner sample-by-sample at every non-padding t.
    """
    rewards = torch.tensor([1.0, 0.0, 0.5])
    baseline = torch.tensor([0.25, 0.25, 0.25])
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[:, -2:] = False                                 # last 2 positions padding

    advantage_rtg = RewardToGoAssigner().assign(rewards, baseline, None, mask)
    advantage_bc = BroadcastAssigner().assign(rewards, baseline, None, mask)

    # At non-padding positions, the two are identical.
    assert torch.allclose(advantage_rtg[mask], advantage_bc[mask], atol=1e-6)


def test_reward_to_go_with_per_token_shaping_diverges_from_broadcast():
    """When token_rewards has non-zero intermediate rewards, RtG and broadcast
    must differ predictably: RtG_t = sum_{k>=t} r_{i,k} - b_i.
    """
    B, T = 2, 4
    rewards = torch.tensor([1.0, 1.0])                  # ignored when token_rewards supplied
    baseline = torch.tensor([0.0, 0.0])
    mask = torch.ones(B, T, dtype=torch.bool)
    # Constant per-token shaping reward of 0.25, plus terminal of 0.5.
    token_rewards = torch.full((B, T), 0.25)
    token_rewards[:, -1] += 0.5

    advantage = RewardToGoAssigner().assign(rewards, baseline, token_rewards, mask)

    # RtG at t = sum_{k>=t} r_k.
    # Per-row totals: t=0 -> 0.25*4 + 0.5 = 1.5
    #                 t=1 -> 0.25*3 + 0.5 = 1.25
    #                 t=2 -> 0.25*2 + 0.5 = 1.0
    #                 t=3 -> 0.25*1 + 0.5 = 0.75
    expected = torch.tensor([[1.5, 1.25, 1.0, 0.75], [1.5, 1.25, 1.0, 0.75]])
    assert torch.allclose(advantage, expected, atol=1e-6)


def test_reward_to_go_subtracts_baseline_per_rollout():
    B, T = 2, 3
    rewards = torch.tensor([1.0, 1.0])
    baseline = torch.tensor([0.5, -0.5])
    mask = torch.ones(B, T, dtype=torch.bool)
    token_rewards = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
    ])
    advantage = RewardToGoAssigner().assign(rewards, baseline, token_rewards, mask)
    # RtG row 0: [1, 1, 1] - 0.5 = [0.5, 0.5, 0.5]
    # RtG row 1: [1, 0.5, 0] - (-0.5) = [1.5, 1.0, 0.5]
    expected = torch.tensor([[0.5, 0.5, 0.5], [1.5, 1.0, 0.5]])
    assert torch.allclose(advantage, expected, atol=1e-6)


def test_reward_to_go_padding_does_not_contribute():
    B, T = 1, 5
    rewards = torch.tensor([0.0])
    baseline = torch.tensor([0.0])
    mask = torch.tensor([[True, True, True, False, False]])
    token_rewards = torch.tensor([[1.0, 1.0, 1.0, 99.0, 99.0]])  # junk in padding
    advantage = RewardToGoAssigner().assign(rewards, baseline, token_rewards, mask)
    # Padded positions should not contribute to the cumsum at real positions:
    # at t=0, rtg = 1+1+1 = 3; at t=1, rtg = 1+1 = 2; at t=2, rtg = 1.
    assert torch.allclose(advantage[0, :3], torch.tensor([3.0, 2.0, 1.0]), atol=1e-6)


def test_reward_to_go_shape_check():
    with pytest.raises(ValueError, match="token_rewards shape"):
        RewardToGoAssigner().assign(
            rewards=torch.tensor([0.0]),
            baseline=torch.tensor([0.0]),
            token_rewards=torch.zeros(1, 4),
            mask=torch.ones(1, 5, dtype=torch.bool),
        )


# ---------------------------------------------------------------------------
# Advantage normalizers
# ---------------------------------------------------------------------------

def test_identity_normalizer_passthrough():
    advantage = torch.randn(3, 5)
    rewards = torch.randn(3)
    out = IdentityNormalizer().apply(advantage, rewards)
    assert torch.equal(out, advantage)


def test_std_normalizer_divides_by_population_std():
    advantage = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rewards = torch.tensor([1.0, 2.0])
    expected_std = rewards.std(unbiased=False)
    out = StdNormalizer().apply(advantage, rewards)
    assert torch.allclose(out, advantage / expected_std, atol=1e-6)


def test_std_normalizer_eps_guards_constant_rewards():
    """Zero reward-std must not produce NaN/inf via division by zero."""
    rewards = torch.tensor([5.0, 5.0])  # std = 0
    out = StdNormalizer().apply(torch.ones(2, 3), rewards)
    assert out.isfinite().all()
    # In real use the advantages are already ~0 when group rewards are equal
    # (the baseline subtracts the group mean), so the normalised output is 0 —
    # the eps guard never has to rescue a nonzero numerator.
    out_zero = StdNormalizer().apply(torch.zeros(2, 3), rewards)
    assert torch.allclose(out_zero, torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# Baseline shifters
# ---------------------------------------------------------------------------

def test_no_shift_is_identity():
    advantage = torch.randn(2, 3)
    out = NoShift().shift(advantage)
    assert torch.equal(out, advantage)


def test_ema_shift_subtracts_current_value():
    ema = EMABaseline(decay=0.9, init=0.0)
    ema.update(2.0)
    advantage = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    out = EMAShift(ema).shift(advantage)
    assert torch.allclose(out, advantage - 2.0)


# ---------------------------------------------------------------------------
# Clippers
# ---------------------------------------------------------------------------

def test_no_clip_pre_product_is_identity():
    log_ratio = torch.randn(2, 5)
    out = NoClip().pre_product(log_ratio)
    assert torch.equal(out, log_ratio)


def test_no_clip_compose_loss_is_negative_ratio_advantage():
    ratio = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    advantage = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    out = NoClip().compose_loss(ratio, advantage, mask)
    expected = -(ratio * advantage)
    assert torch.allclose(out, expected, atol=1e-6)


def test_log_ratio_token_clip_bounds_per_token():
    c = 0.5
    log_ratio = torch.tensor([[-2.0, -0.3, 0.1, 1.5]])
    out = LogRatioTokenClip(c=c).pre_product(log_ratio)
    assert torch.all(out >= -c - 1e-6)
    assert torch.all(out <= c + 1e-6)
    # Values inside the bound must pass through unchanged.
    assert out[0, 1].item() == pytest.approx(-0.3)
    assert out[0, 2].item() == pytest.approx(0.1)


def test_log_ratio_token_clip_bounds_cumulative_full_sequence():
    """After per-token log-ratio clipping at c, the full-sequence ratio is
    bounded by exp(c * T_real).
    """
    B, T = 2, 10
    c = 0.2
    log_pi = torch.full((B, T), 1.0)         # per-token log-ratio of 1.0 -> clamped to c
    log_ref = torch.zeros(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)

    log_ratio = (log_pi - log_ref) * mask.float()
    log_ratio = LogRatioTokenClip(c=c).pre_product(log_ratio)
    ratio = FullSequenceIS().compute(log_ratio, mask)

    expected_max = math.exp(c * T)
    assert torch.all(ratio <= expected_max + 1e-4)


def test_ppo_classical_clip_min_form_with_negative_advantage():
    """When advantage > 0 and ratio > 1 + eps, PPO uses the smaller
    (clipped * adv); when advantage < 0 and ratio > 1 + eps, PPO uses the
    larger unclipped value (i.e. min(unclipped, clipped) is unclipped).
    """
    eps = 0.2
    ratio = torch.tensor([[1.5], [1.5]])               # both above clip ceiling
    advantage = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    out = PPOClassicalClip(eps=eps).compose_loss(ratio, advantage, mask)
    # Row 0 (positive adv): use clipped (1.2), so token loss = -1.2
    # Row 1 (negative adv): use unclipped (1.5), so token loss = -(1.5 * -1) = +1.5
    assert torch.allclose(out, torch.tensor([[-1.2, -1.2], [1.5, 1.5]]), atol=1e-5)


# ---------------------------------------------------------------------------
# IS weighters
# ---------------------------------------------------------------------------

def test_full_sequence_is_returns_broadcastable_shape():
    B, T = 3, 7
    log_ratio = torch.zeros(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = FullSequenceIS().compute(log_ratio, mask)
    assert out.shape == (B, 1)
    assert torch.allclose(out, torch.ones(B, 1))


def test_full_sequence_is_equals_exp_sum_log_ratio():
    B, T = 2, 4
    log_ratio = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]])
    mask = torch.ones(B, T, dtype=torch.bool)
    out = FullSequenceIS().compute(log_ratio, mask)
    expected = torch.exp(log_ratio.sum(dim=1, keepdim=True))
    assert torch.allclose(out, expected, atol=1e-6)


def test_full_sequence_is_log_w_clamp_caps_weight():
    """FullSequenceIS(log_w_clamp=5.0) caps the weight at exp(5); with
    log_w_clamp=None the same large drift produces a weight far above exp(5)."""
    B, T = 1, 10
    # Summed log-weight = 10 * 1.0 = 10.0, well above the clamp of 5.0.
    log_ratio = torch.full((B, T), 1.0)
    mask = torch.ones(B, T, dtype=torch.bool)

    clamped = FullSequenceIS(log_w_clamp=5.0).compute(log_ratio, mask)
    unclamped = FullSequenceIS(log_w_clamp=None).compute(log_ratio, mask)

    assert torch.allclose(clamped, torch.full((B, 1), math.exp(5.0)), atol=1e-3)
    assert torch.allclose(unclamped, torch.full((B, 1), math.exp(10.0)), rtol=1e-4)
    assert unclamped.item() > clamped.item()


def test_full_sequence_is_log_w_clamp_noop_below_cap():
    """When the summed log-weight is below the cap, clamping is a no-op."""
    B, T = 1, 4
    log_ratio = torch.full((B, T), 0.1)  # sum = 0.4 < 5.0
    mask = torch.ones(B, T, dtype=torch.bool)
    clamped = FullSequenceIS(log_w_clamp=5.0).compute(log_ratio, mask)
    unclamped = FullSequenceIS(log_w_clamp=None).compute(log_ratio, mask)
    assert torch.allclose(clamped, unclamped, atol=1e-6)


def test_per_token_is_returns_per_token_shape_and_identity():
    """PerTokenIS output is (B, T) and equals exp(log_ratio) element-wise.

    Unlike FullSequenceIS (which sums first), per-token IS leaves each token's
    ratio independent — this is the DeepSeekMath GRPO Eq. 21 formulation.
    """
    B, T = 3, 5
    log_ratio = torch.tensor([
        [0.0, 0.1, -0.2, 0.3, -0.4],
        [0.5, -0.5, 0.5, -0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    mask = torch.ones(B, T, dtype=torch.bool)
    out = PerTokenIS().compute(log_ratio, mask)
    assert out.shape == (B, T)
    expected = torch.exp(log_ratio)
    assert torch.allclose(out, expected, atol=1e-6)
    # All-zero log-ratio row is exactly 1.
    assert torch.allclose(out[2], torch.ones(T), atol=1e-6)


def test_per_token_is_padding_returns_one():
    """PerTokenIS returns exp(0) = 1 at padding positions, ensuring downstream
    consumers (ESS, log-ratio std) don't see surprising zero values.
    The final ``* mask`` in compose_loss zeros their gradient contribution.
    """
    B, T = 2, 6
    log_ratio = torch.tensor([
        [0.1, 0.1, 0.1, 99.0, 99.0, 99.0],   # last 3 positions padding
        [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
    ])
    mask = torch.tensor([
        [True, True, True, False, False, False],
        [True, True, True, True, True, True],
    ])
    out = PerTokenIS().compute(log_ratio, mask)
    # Real positions: exp(log_ratio).
    assert torch.allclose(out[0, :3], torch.exp(torch.tensor([0.1, 0.1, 0.1])), atol=1e-6)
    # Padding positions: exp(log_ratio * 0) = exp(0) = 1, NOT exp(99).
    assert torch.allclose(out[0, 3:], torch.ones(3), atol=1e-6)
    # Row 1 fully real: all exp(-0.2).
    assert torch.allclose(out[1], torch.full((T,), torch.exp(torch.tensor(-0.2)).item()), atol=1e-6)


def test_per_token_is_with_ppo_clip_clips_per_token_independently():
    """Integration: PerTokenIS + PPOClassicalClip clips each token's ratio
    independently, unlike FullSequenceIS which clips the whole sequence
    together. We construct a sequence where one token has large drift but
    the rest are near 1, and verify only that one gets clipped.
    """
    eps = 0.2
    B, T = 1, 5
    # Token 2 has log_ratio = 1.5 → ratio = exp(1.5) ≈ 4.48 (way above 1.2).
    # All others have log_ratio = 0.05 → ratio ≈ 1.05 (inside [0.8, 1.2]).
    log_ratio = torch.tensor([[0.05, 0.05, 1.5, 0.05, 0.05]])
    mask = torch.ones(B, T, dtype=torch.bool)

    ratio = PerTokenIS().compute(log_ratio, mask)              # (1, 5)
    advantage = torch.full((B, T), 1.0)                        # positive A
    out = PPOClassicalClip(eps=eps).compose_loss(ratio, advantage, mask)
    # With positive advantage and ratio > 1+eps, PPO uses the smaller
    # (clipped * A); with ratio inside the window, unclipped == clipped.
    # Token loss = -min(ratio*A, clipped_ratio*A) * mask.
    expected_inside = -torch.exp(torch.tensor(0.05)).item()
    expected_clipped_token = -(1.0 + eps)
    assert abs(out[0, 0].item() - expected_inside) < 1e-5
    assert abs(out[0, 2].item() - expected_clipped_token) < 1e-5  # the one outlier
    assert abs(out[0, 4].item() - expected_inside) < 1e-5


# ---------------------------------------------------------------------------
# Length aggregators
# ---------------------------------------------------------------------------

def test_per_response_aggregator_averages_per_sequence_then_mean():
    token_loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    out = PerResponseAggregator().aggregate(token_loss, mask)
    expected = ((1 + 2 + 3) / 3 + (4 + 5 + 6) / 3) / 2
    assert out.item() == pytest.approx(expected)


def test_token_level_aggregator_normalises_by_total_token_count():
    token_loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    out = TokenLevelAggregator().aggregate(token_loss, mask)
    assert out.item() == pytest.approx(21.0 / 6)


def test_aggregators_handle_padding():
    token_loss = torch.tensor([[1.0, 2.0, 99.0], [3.0, 99.0, 99.0]])  # 99s in padding
    mask = torch.tensor([[True, True, False], [True, False, False]])
    # token-level aggregator divides masked token sum by 3 real tokens
    masked_loss = token_loss * mask.float()
    out_token = TokenLevelAggregator().aggregate(masked_loss, mask)
    assert out_token.item() == pytest.approx((1 + 2 + 3) / 3)
    # per-response averages each row by its real-token count then means
    out_per = PerResponseAggregator().aggregate(masked_loss, mask)
    expected_per = (((1 + 2) / 2) + (3 / 1)) / 2
    assert out_per.item() == pytest.approx(expected_per)


# ---------------------------------------------------------------------------
# validate_combination
# ---------------------------------------------------------------------------

def test_validate_accepts_full_sequence_plus_ppo_classical():
    # Should not raise
    validate_combination(
        baseline=LOOBaseline(),
        reward_assigner=BroadcastAssigner(),
        advantage_normalizer=IdentityNormalizer(),
        baseline_shifter=NoShift(),
        clipper=PPOClassicalClip(eps=0.2),
        is_weighter=FullSequenceIS(),
        length_aggregator=TokenLevelAggregator(),
    )


# ---------------------------------------------------------------------------
# build_components factory
# ---------------------------------------------------------------------------

def test_build_components_paper_preset_returns_expected_classes():
    components = build_components(grpo_paper_config())
    assert isinstance(components["baseline"], MeanWithSelfBaseline)
    assert isinstance(components["reward_assigner"], BroadcastAssigner)
    assert isinstance(components["advantage_normalizer"], StdNormalizer)
    assert isinstance(components["baseline_shifter"], NoShift)
    assert isinstance(components["clipper"], PPOClassicalClip)
    assert isinstance(components["is_weighter"], FullSequenceIS)
    assert isinstance(components["length_aggregator"], PerResponseAggregator)


def test_build_components_paper_token_preset_uses_per_token_is():
    """G0a preset must produce PerTokenIS + PPOClassicalClip — paper-faithful
    DeepSeekMath GRPO Eq. 21.
    """
    components = build_components(grpo_paper_token_config())
    assert isinstance(components["is_weighter"], PerTokenIS)
    assert isinstance(components["clipper"], PPOClassicalClip)
    assert isinstance(components["baseline"], MeanWithSelfBaseline)
    assert isinstance(components["advantage_normalizer"], StdNormalizer)
    assert isinstance(components["length_aggregator"], PerResponseAggregator)


def test_build_components_debiased_preset_returns_expected_classes():
    from debiased_grpo.strategies import FixedConstantAggregator
    components = build_components(debiased_grpo_config(fixed_divisor=64.0))
    assert isinstance(components["baseline"], IndependentBaseline)
    assert isinstance(components["clipper"], NoClip)
    assert isinstance(components["is_weighter"], FullSequenceIS)
    assert components["is_weighter"].log_w_clamp == 5.0
    assert isinstance(components["length_aggregator"], FixedConstantAggregator)
    assert components["length_aggregator"].divisor == 64.0
    assert isinstance(components["advantage_normalizer"], IdentityNormalizer)


def test_build_components_ema_requires_instance():
    cfg = debiased_grpo_config()
    cfg.ema_baseline = True
    with pytest.raises(ValueError, match="EMABaseline instance"):
        build_components(cfg)


def test_build_components_ema_supplied():
    cfg = debiased_grpo_config()
    cfg.ema_baseline = True
    ema = EMABaseline(decay=0.9)
    components = build_components(cfg, ema=ema)
    assert isinstance(components["baseline_shifter"], EMAShift)


def test_build_components_unknown_key_raises():
    cfg = LossConfig(baseline="bogus_baseline")
    with pytest.raises(ValueError, match="Unknown baseline"):
        build_components(cfg)


# ---------------------------------------------------------------------------
# Behavioural integration tests using compute_loss
# ---------------------------------------------------------------------------

def _tiny_batch():
    torch.manual_seed(0)
    B, T = 4, 8
    log_probs = (torch.randn(B, T) - 1.0).requires_grad_(True)
    ref_log_probs = torch.randn(B, T) - 1.0
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -2:] = False
    return log_probs, ref_log_probs, rewards, mask


def test_compute_loss_is_finite_scalar():
    lp, rlp, r, m = _tiny_batch()
    components = build_components(grpo_paper_config())
    loss = compute_loss(lp, rlp, r, m, **components)
    assert loss.shape == ()
    assert loss.isfinite()
    loss.backward()
    assert lp.grad is not None
    assert lp.grad.norm() > 0


def test_compute_loss_debiased_preset_matches_debiased_loss_wrapper():
    """Calling compute_loss with the debiased preset must equal debiased_loss(...)."""
    from debiased_grpo.losses import debiased_loss

    lp, rlp, r, m = _tiny_batch()
    baseline_rewards = torch.tensor([0.25, 0.5, 0.5])
    components = build_components(debiased_grpo_config())
    loss_orchestrator = compute_loss(
        lp, rlp, r, m, baseline_rewards=baseline_rewards, **components,
    )
    loss_wrapper = debiased_loss(lp, rlp, r, baseline_rewards, m)
    assert torch.allclose(loss_orchestrator, loss_wrapper, atol=1e-6)


def test_compute_loss_grpo_preset_matches_grpo_loss_wrapper():
    lp, rlp, r, m = _tiny_batch()
    components = build_components(grpo_paper_config())
    loss_orchestrator = compute_loss(lp, rlp, r, m, **components)
    from debiased_grpo.losses import grpo_loss
    loss_wrapper = grpo_loss(lp, rlp, r, m)
    assert torch.allclose(loss_orchestrator, loss_wrapper, atol=1e-6)


# ---------------------------------------------------------------------------
# Reward-to-go behavioural tests via compute_loss
# ---------------------------------------------------------------------------

def test_rtg_terminal_equivalence_via_compute_loss():
    """Under sparse terminal token_rewards (the make_token_rewards
    representation), RtG and broadcast must produce identical losses
    sample-by-sample.
    """
    lp, rlp, r, m = _tiny_batch()
    baseline_rewards = torch.tensor([0.5, 0.5, 0.5])

    cfg_bc = debiased_grpo_config()
    cfg_rtg = debiased_grpo_config()
    cfg_rtg.reward_assignment = "reward_to_go"

    loss_bc = compute_loss(
        lp, rlp, r, m, baseline_rewards=baseline_rewards,
        **build_components(cfg_bc),
    )
    loss_rtg = compute_loss(
        lp, rlp, r, m, baseline_rewards=baseline_rewards,
        **build_components(cfg_rtg),
    )
    assert torch.allclose(loss_bc, loss_rtg, atol=1e-5)


def test_rtg_with_per_token_shaping_diverges_from_broadcast():
    """When a non-trivial token_rewards tensor is passed, RtG must give a
    different loss than broadcast (the broadcast assigner ignores
    token_rewards entirely).
    """
    lp, rlp, r, m = _tiny_batch()
    baseline_rewards = torch.tensor([0.5, 0.5, 0.5])
    B, T = lp.shape
    # Constant per-token shaping: 0.1 at every position, plus the scalar reward
    # as a terminal bonus.
    token_rewards = torch.full((B, T), 0.1)

    cfg_bc = debiased_grpo_config()
    cfg_rtg = debiased_grpo_config()
    cfg_rtg.reward_assignment = "reward_to_go"

    loss_bc = compute_loss(
        lp, rlp, r, m, baseline_rewards=baseline_rewards,
        token_rewards=token_rewards, **build_components(cfg_bc),
    )
    loss_rtg = compute_loss(
        lp, rlp, r, m, baseline_rewards=baseline_rewards,
        token_rewards=token_rewards, **build_components(cfg_rtg),
    )
    # Broadcast ignores token_rewards, RtG uses it -> losses differ.
    assert not torch.allclose(loss_bc, loss_rtg, atol=1e-3)


# ---------------------------------------------------------------------------
# EMA-of-advantages behavioural tests
# ---------------------------------------------------------------------------

def test_ema_shift_unbiased_when_value_is_zero():
    """An uninitialised EMA (value=0) must produce the same loss as NoShift."""
    lp, rlp, r, m = _tiny_batch()
    baseline_rewards = torch.tensor([0.25, 0.5, 0.5])

    ema = EMABaseline(decay=0.9, init=0.0)  # not yet initialized
    cfg_ema = debiased_grpo_config()
    cfg_ema.ema_baseline = True

    cfg_no = debiased_grpo_config()

    loss_with_ema = compute_loss(
        lp, rlp, r, m, baseline_rewards=baseline_rewards,
        **build_components(cfg_ema, ema=ema),
    )
    loss_no_ema = compute_loss(
        lp, rlp, r, m, baseline_rewards=baseline_rewards,
        **build_components(cfg_no),
    )
    assert torch.allclose(loss_with_ema, loss_no_ema, atol=1e-6)


def test_ema_shift_changes_loss_when_value_nonzero():
    lp, rlp, r, m = _tiny_batch()
    baseline_rewards = torch.tensor([0.25, 0.5, 0.5])
    ema = EMABaseline(decay=0.9, init=0.0)
    ema.update(0.5)                               # value now 0.5
    assert ema.value == pytest.approx(0.5)

    cfg = debiased_grpo_config()
    cfg.ema_baseline = True
    components = build_components(cfg, ema=ema)
    loss_shifted = compute_loss(
        lp, rlp, r, m, baseline_rewards=baseline_rewards, **components,
    )

    cfg_no = debiased_grpo_config()
    loss_unshifted = compute_loss(
        lp, rlp, r, m, baseline_rewards=baseline_rewards,
        **build_components(cfg_no),
    )
    assert not torch.allclose(loss_shifted, loss_unshifted, atol=1e-4)


# ---------------------------------------------------------------------------
# Clipping mode behavioural tests
# ---------------------------------------------------------------------------

def test_no_clip_full_sequence_equals_unbiased_estimator():
    """With clipping=none and is_weighting=full_sequence, the loss is the
    unbiased full-sequence IS estimator: -mean(exp(sum log_ratio) * adv)."""
    B, T = 2, 4
    torch.manual_seed(1)
    lp = torch.randn(B, T) - 1.0
    rlp = torch.randn(B, T) - 1.0
    r = torch.tensor([1.0, 0.0])
    m = torch.ones(B, T, dtype=torch.bool)

    cfg = LossConfig(
        baseline="loo", reward_assignment="broadcast",
        is_weighting="full_sequence", clipping="none",
        length_norm="token_level", std_norm=False,
    )
    components = build_components(cfg)
    loss = compute_loss(lp, rlp, r, m, **components)

    log_ratio_sum = (lp - rlp).sum(dim=1)
    ratio = torch.exp(log_ratio_sum)
    loo = (r.sum() - r) / (B - 1)
    advantage = r - loo
    expected = -(ratio * advantage).sum() / m.float().sum()  # token-level normalisation
    # Each token in a row carries the same advantage, so total contribution
    # per row is T * (ratio_i * adv_i); divided by total tokens (B * T) gives:
    expected_per_token_sum = -(ratio * advantage).repeat_interleave(T) / (B * T)
    expected = expected_per_token_sum.sum()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_log_ratio_token_clip_changes_loss_under_drift():
    B, T = 4, 4
    # Different per-row drift so the full-sequence ratios are not identical
    # across rows; otherwise the LOO antisymmetric advantage causes them to
    # cancel and the clipping has nothing to act on.
    lp = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [-0.5, -0.5, -0.5, -0.5],
        [0.2, 0.2, 0.2, 0.2],
    ])
    rlp = torch.zeros(B, T)
    r = torch.tensor([1.0, 0.0, 1.0, 0.0])
    m = torch.ones(B, T, dtype=torch.bool)

    cfg_none = LossConfig(
        baseline="loo", reward_assignment="broadcast",
        is_weighting="full_sequence", clipping="none",
        length_norm="token_level", std_norm=False,
    )
    cfg_clip = LossConfig(
        baseline="loo", reward_assignment="broadcast",
        is_weighting="full_sequence", clipping="log_ratio_token",
        length_norm="token_level", std_norm=False,
        log_ratio_clip_c=0.1,
    )
    loss_none = compute_loss(lp, rlp, r, m, **build_components(cfg_none))
    loss_clip = compute_loss(lp, rlp, r, m, **build_components(cfg_clip))

    # Loss magnitudes must differ — clipping bounds the per-token contribution.
    assert not torch.allclose(loss_none, loss_clip, atol=1e-3)


# ---------------------------------------------------------------------------
# IS denominator semantics (behavior_log_probs vs ref_log_probs)
# ---------------------------------------------------------------------------

def test_compute_loss_behavior_log_probs_overrides_is_denominator():
    """When ``behavior_log_probs`` is supplied, the IS log-ratio uses it as
    denominator instead of ``ref_log_probs``. With ``behavior == log_probs``,
    the IS ratio is identically 1 — the loss reduces to the policy-gradient
    surrogate ``-mean(advantage)`` (modulo aggregator normalisation).
    """
    B, T = 3, 4
    torch.manual_seed(0)
    log_probs = (torch.randn(B, T) - 1.0).requires_grad_(True)
    behavior_log_probs = log_probs.detach().clone()  # ratio ≡ 1
    ref_log_probs = torch.randn(B, T) - 1.0          # different from behavior
    rewards = torch.tensor([1.0, 0.0, 0.5])
    mask = torch.ones(B, T, dtype=torch.bool)

    cfg = LossConfig(
        baseline="loo", reward_assignment="broadcast",
        is_weighting="full_sequence", clipping="none",
        length_norm="token_level", std_norm=False,
    )
    components = build_components(cfg)
    loss_with_behavior = compute_loss(
        log_probs, ref_log_probs, rewards, mask,
        behavior_log_probs=behavior_log_probs, **components,
    )
    # Hand-computed expectation: ratio = 1, so token loss = -advantage; mean
    # over all tokens equals -mean(advantage) (same advantage at every token).
    loo = (rewards.sum() - rewards) / (B - 1)
    advantage = rewards - loo
    expected = -advantage.mean()
    assert torch.allclose(loss_with_behavior, expected, atol=1e-5)


def test_compute_loss_kl_ref_coef_adds_nonneg_term():
    """The KL term added by ``kl_ref_coef > 0`` is non-negative (k3 estimator).
    Setting both KL coefs to 0 reproduces the loss without any KL term.
    """
    B, T = 4, 6
    torch.manual_seed(1)
    log_probs = (torch.randn(B, T) - 1.0).requires_grad_(True)
    behavior_log_probs = log_probs.detach().clone()
    ref_log_probs = torch.randn(B, T) - 1.5
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(B, T, dtype=torch.bool)

    cfg = LossConfig(
        baseline="loo", reward_assignment="broadcast",
        is_weighting="full_sequence", clipping="none",
        length_norm="token_level", std_norm=False,
    )
    components = build_components(cfg)
    loss_no_kl = compute_loss(
        log_probs, ref_log_probs, rewards, mask,
        behavior_log_probs=behavior_log_probs, kl_ref_coef=0.0, **components,
    )
    loss_with_kl = compute_loss(
        log_probs, ref_log_probs, rewards, mask,
        behavior_log_probs=behavior_log_probs, kl_ref_coef=0.1, **components,
    )
    # kl_ref_coef > 0 must add a non-negative term (k3 estimator is non-negative).
    assert (loss_with_kl - loss_no_kl).item() >= -1e-6


def test_compute_loss_kl_ref_vs_behavior_anchors():
    """The two KL coefficients anchor to different policies, additively.

    With behavior == theta (the inner-step-0 state), KL(π_θ ∥ π_behavior) = 0, so
    ``kl_behavior_coef`` alone reproduces the no-KL loss, while ``kl_ref_coef``
    adds a strictly positive term (ref ≠ theta). Setting both equals ref-only here.
    """
    B, T = 4, 6
    torch.manual_seed(2)
    log_probs = (torch.randn(B, T) - 1.0).requires_grad_(True)
    behavior_log_probs = log_probs.detach().clone()      # behavior == theta
    ref_log_probs = torch.randn(B, T) - 1.5              # ref != theta
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(B, T, dtype=torch.bool)

    cfg = LossConfig(
        baseline="independent", reward_assignment="broadcast",
        is_weighting="full_sequence", clipping="none",
        length_norm="token_level", std_norm=False,
    )
    components = build_components(cfg)
    baseline_rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])
    base = dict(behavior_log_probs=behavior_log_probs,
                baseline_rewards=baseline_rewards)
    loss_no_kl = compute_loss(log_probs, ref_log_probs, rewards, mask,
                              kl_ref_coef=0.0, kl_behavior_coef=0.0,
                              **base, **components)
    loss_beh = compute_loss(log_probs, ref_log_probs, rewards, mask,
                            kl_ref_coef=0.0, kl_behavior_coef=0.1, **base, **components)
    loss_ref = compute_loss(log_probs, ref_log_probs, rewards, mask,
                            kl_ref_coef=0.1, kl_behavior_coef=0.0, **base, **components)
    loss_both = compute_loss(log_probs, ref_log_probs, rewards, mask,
                             kl_ref_coef=0.1, kl_behavior_coef=0.1, **base, **components)
    # behavior anchor with behavior==theta contributes ~0 KL.
    assert abs((loss_beh - loss_no_kl).item()) < 1e-6
    # ref anchor adds a strictly positive KL term.
    assert (loss_ref - loss_no_kl).item() > 1e-4
    # both == ref-only here (behavior term is ~0).
    assert abs((loss_both - loss_ref).item()) < 1e-6


# ---------------------------------------------------------------------------
# Legacy back-compat
# ---------------------------------------------------------------------------

def test_legacy_grpo_loss_unified_full_sequence_ppo():
    """The legacy flag value is_weighting='full_sequence_ppo' must continue
    to work and be equivalent to the new is_weighting='full_sequence' plus
    clipping='ppo_classical' combination.
    """
    from debiased_grpo.losses import grpo_loss_unified

    lp, rlp, r, m = _tiny_batch()
    legacy_loss = grpo_loss_unified(
        lp, rlp, r, m,
        baseline="loo",
        is_weighting="full_sequence_ppo",
        length_norm="token_level",
        std_norm=False,
        clip_eps=0.2,
    )
    cfg = LossConfig(
        baseline="loo", reward_assignment="broadcast",
        is_weighting="full_sequence", clipping="ppo_classical",
        length_norm="token_level", std_norm=False,
        clip_eps=0.2,
    )
    new_loss = compute_loss(lp, rlp, r, m, **build_components(cfg))
    assert torch.allclose(legacy_loss, new_loss, atol=1e-6)


def test_legacy_grpo_loss_unified_full_sequence_unclipped():
    """is_weighting='full_sequence' in the legacy API must map to full-sequence
    IS with no clipping.
    """
    from debiased_grpo.losses import grpo_loss_unified

    lp, rlp, r, m = _tiny_batch()
    baseline_rewards = torch.tensor([0.25, 0.5, 0.5])
    legacy_loss = grpo_loss_unified(
        lp, rlp, r, m,
        baseline="independent",
        is_weighting="full_sequence",
        length_norm="token_level",
        std_norm=False,
        baseline_rewards=baseline_rewards,
    )
    cfg = LossConfig(
        baseline="independent", reward_assignment="broadcast",
        is_weighting="full_sequence", clipping="none",
        length_norm="token_level", std_norm=False,
    )
    new_loss = compute_loss(
        lp, rlp, r, m, baseline_rewards=baseline_rewards, **build_components(cfg),
    )
    assert torch.allclose(legacy_loss, new_loss, atol=1e-6)
