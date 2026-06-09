"""
Unit tests for debiased_grpo.utils.

All tests run on CPU with synthetic data — no model downloads required.

One property is pinned down here:
  normalize_rewards stays finite for B=1.
    The std uses correction=0 (population std); correction=1 (ddof=1) would
    return NaN for a single-element batch.
    Exercised by: test_normalize_rewards_single_element
"""

import torch
import pytest
from debiased_grpo.utils import (
    EMABaseline,
    compute_ess,
    make_token_rewards,
    mask_padding,
    normalize_rewards,
    reverse_cumsum,
)


def test_ess_is_n_when_weights_equal():
    """ESS of uniform weights should equal T (sequence length)."""
    B, T = 4, 16
    weights = torch.ones(B, T)
    ess = compute_ess(weights)
    # ESS = (sum w)^2 / sum(w^2) = T^2 / T = T per row, averaged = T
    assert abs(ess.item() - T) < 1e-3


def test_ess_is_lower_for_unequal_weights():
    """Non-uniform weights should yield a lower ESS than uniform weights."""
    B, T = 4, 16
    uniform_weights = torch.ones(B, T)
    # Make one weight very large — concentrates all mass on one token.
    peaked_weights = torch.ones(B, T)
    peaked_weights[:, 0] = 100.0
    ess_uniform = compute_ess(uniform_weights)
    ess_peaked = compute_ess(peaked_weights)
    assert ess_peaked.item() < ess_uniform.item()


def test_ess_is_positive():
    """ESS must be strictly positive for any non-zero weight tensor."""
    B, T = 4, 16
    weights = torch.rand(B, T).abs() + 0.01
    ess = compute_ess(weights)
    assert ess.item() > 0.0


def test_mask_padding_zeros_out_padding():
    """mask_padding should replace False positions with the fill_value."""
    B, T = 3, 8
    tensor = torch.ones(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -2:] = False
    out = mask_padding(tensor, mask)
    assert out.shape == (B, T)
    assert torch.all(out[:, -2:] == 0.0)
    assert torch.all(out[:, :-2] == 1.0)


def test_mask_padding_custom_fill():
    """mask_padding should use the custom fill_value when provided."""
    B, T = 2, 6
    tensor = torch.zeros(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 3:] = False
    out = mask_padding(tensor, mask, fill_value=-999.0)
    assert torch.all(out[:, 3:] == -999.0)


def test_normalize_rewards_zero_mean():
    """Normalised rewards must have mean ≈ 0."""
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normed = normalize_rewards(rewards)
    assert abs(normed.mean().item()) < 1e-5


def test_normalize_rewards_unit_std():
    """Normalised rewards must have population std ≈ 1.

    normalize_rewards uses correction=0 (population std) so that the output
    is well-defined for B=1.  Checking with correction=0 here to match the
    implementation; a larger tolerance accounts for floating-point epsilon.
    """
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normed = normalize_rewards(rewards)
    # Population std of the normalised output must be ≈ 1.
    assert abs(normed.std(correction=0).item() - 1.0) < 1e-4


def test_normalize_rewards_constant_input():
    """Constant rewards (std=0) must not produce NaN — epsilon guards against division by zero."""
    rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
    normed = normalize_rewards(rewards)
    assert not torch.any(torch.isnan(normed))
    assert torch.all(normed == 0.0)


# ---------------------------------------------------------------------------
# normalize_rewards stays finite for B=1
# ---------------------------------------------------------------------------

def test_normalize_rewards_single_element():
    """normalize_rewards with a single reward (B=1) must return 0.0, not NaN.

    Population std (correction=0) is 0 for a single element, so the epsilon
    guard makes the result 0.0; ddof=1 would be undefined and return NaN.
    """
    rewards = torch.tensor([7.5])
    normed = normalize_rewards(rewards)
    assert not torch.isnan(normed).any(), (
        "normalize_rewards(B=1) returned NaN — population std (correction=0) required"
    )
    assert torch.allclose(normed, torch.tensor([0.0]), atol=1e-6), (
        f"normalize_rewards(B=1) should return 0.0 but got {normed.item()}"
    )


def test_normalize_rewards_b2_constant():
    """normalize_rewards with B=2 and identical rewards must return zeros, not NaN."""
    rewards = torch.tensor([3.0, 3.0])
    normed = normalize_rewards(rewards)
    assert not torch.isnan(normed).any()
    assert torch.allclose(normed, torch.zeros(2), atol=1e-6)


# ---------------------------------------------------------------------------
# ESS bounds: must be in [1, T] per sequence
# ---------------------------------------------------------------------------

def test_ess_lower_bound_is_one():
    """ESS with all weight concentrated on one token per row must be ≈ 1."""
    B, T = 4, 32
    weights = torch.zeros(B, T)
    weights[:, 0] = 1.0  # single nonzero weight per row
    ess = compute_ess(weights)
    assert abs(ess.item() - 1.0) < 0.01, (
        f"ESS with one dominant weight should be ≈1, got {ess.item():.4f}"
    )


def test_ess_upper_bound_is_t():
    """ESS with uniform weights of shape (B, T) must equal T (the sequence length)."""
    B, T = 4, 20
    weights = torch.ones(B, T)
    ess = compute_ess(weights)
    assert abs(ess.item() - T) < 0.01, (
        f"ESS with uniform weights should equal T={T}, got {ess.item():.4f}"
    )


def test_ess_between_bounds_for_mixed_weights():
    """ESS for non-trivial weights must be strictly between 1 and T."""
    B, T = 4, 16
    torch.manual_seed(0)
    weights = torch.rand(B, T).abs() + 0.1  # all positive, varied
    ess = compute_ess(weights)
    assert 1.0 < ess.item() < T, (
        f"ESS {ess.item():.4f} must be strictly between 1 and T={T} for mixed weights"
    )


def test_ess_monotone_in_concentration():
    """More concentrated weights must produce lower ESS."""
    B, T = 4, 16
    # Low concentration: weights are uniform.
    w_uniform = torch.ones(B, T)
    # High concentration: all weight on one token.
    w_peaked = torch.zeros(B, T)
    w_peaked[:, 0] = float(T)  # same total weight, but concentrated
    # Medium:
    w_medium = torch.ones(B, T)
    w_medium[:, 0] = T / 2.0

    ess_uniform = compute_ess(w_uniform).item()
    ess_medium = compute_ess(w_medium).item()
    ess_peaked = compute_ess(w_peaked).item()

    assert ess_peaked < ess_medium < ess_uniform, (
        f"ESS should decrease with concentration: peaked={ess_peaked:.2f}, "
        f"medium={ess_medium:.2f}, uniform={ess_uniform:.2f}"
    )


# ---------------------------------------------------------------------------
# reverse_cumsum
# ---------------------------------------------------------------------------

def test_reverse_cumsum_basic():
    """reverse_cumsum at position t must equal the sum of all elements >= t."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = reverse_cumsum(x, dim=1)
    expected = torch.tensor([[10.0, 9.0, 7.0, 4.0]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_reverse_cumsum_negatives():
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    out = reverse_cumsum(x, dim=1)
    expected = torch.tensor([[-2.0, -3.0, -1.0, -4.0]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_reverse_cumsum_2d_batch():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = reverse_cumsum(x, dim=1)
    expected = torch.tensor([[6.0, 5.0, 3.0], [15.0, 11.0, 6.0]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_reverse_cumsum_gradient_flows():
    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    out = reverse_cumsum(x, dim=1)
    out.sum().backward()
    # d(sum of reverse cumsum)/dx_t = number of positions <= t = (t+1).
    expected_grad = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(x.grad, expected_grad, atol=1e-6)


# ---------------------------------------------------------------------------
# make_token_rewards
# ---------------------------------------------------------------------------

def test_make_token_rewards_places_scalar_at_last_real_position():
    rewards = torch.tensor([1.0, 0.5, 0.0])
    mask = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
        [True, False, False, False],
    ])
    out = make_token_rewards(rewards, mask)
    expected = torch.tensor([
        [0.0, 0.0, 1.0, 0.0],     # last real position is index 2
        [0.0, 0.5, 0.0, 0.0],     # last real position is index 1
        [0.0, 0.0, 0.0, 0.0],     # only index 0 is real, scalar = 0 -> still 0
    ])
    expected[2, 0] = 0.0           # rewards[2] is 0, so position 0 stays 0
    assert torch.allclose(out, expected, atol=1e-6)


def test_make_token_rewards_zero_length_row_safe():
    """Rows with no real tokens (mask all False) must produce all zeros."""
    rewards = torch.tensor([1.0, 1.0])
    mask = torch.tensor([
        [True, True, False],
        [False, False, False],
    ])
    out = make_token_rewards(rewards, mask)
    assert torch.allclose(out[1], torch.zeros(3))


def test_make_token_rewards_shape_check():
    with pytest.raises(ValueError, match="rewards must have shape"):
        make_token_rewards(torch.tensor([1.0, 2.0]), torch.ones(3, 4, dtype=torch.bool))


def test_make_token_rewards_sums_to_scalar_under_full_mask():
    """Sum of token rewards along the token axis must equal the scalar reward
    when there is at least one real token per row.
    """
    rewards = torch.tensor([3.0, -1.5])
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = make_token_rewards(rewards, mask)
    assert torch.allclose(out.sum(dim=1), rewards, atol=1e-6)


# ---------------------------------------------------------------------------
# EMABaseline
# ---------------------------------------------------------------------------

def test_ema_baseline_initial_value_is_init():
    ema = EMABaseline(decay=0.9, init=1.5)
    assert ema.value == 1.5
    assert ema.initialized is False


def test_ema_baseline_first_update_warm_starts_to_observed():
    ema = EMABaseline(decay=0.9, init=0.0)
    ema.update(2.0)
    assert ema.value == pytest.approx(2.0)
    assert ema.initialized is True


def test_ema_baseline_subsequent_updates_blend():
    ema = EMABaseline(decay=0.9, init=0.0)
    ema.update(2.0)
    ema.update(0.0)
    # 0.9 * 2.0 + 0.1 * 0.0 = 1.8
    assert ema.value == pytest.approx(1.8)


def test_ema_baseline_converges_to_constant_signal():
    ema = EMABaseline(decay=0.9)
    for _ in range(200):
        ema.update(0.7)
    assert ema.value == pytest.approx(0.7, abs=1e-4)


def test_ema_baseline_invalid_decay_raises():
    with pytest.raises(ValueError, match="decay must be in"):
        EMABaseline(decay=1.0)
    with pytest.raises(ValueError, match="decay must be in"):
        EMABaseline(decay=-0.1)


def test_ema_baseline_state_dict_round_trip():
    ema = EMABaseline(decay=0.95, init=0.0)
    ema.update(1.0)
    ema.update(0.5)
    state = ema.state_dict()

    restored = EMABaseline(decay=0.5, init=99.0)
    restored.load_state_dict(state)
    assert restored.decay == ema.decay
    assert restored.value == ema.value
    assert restored.initialized == ema.initialized


def test_ema_baseline_repr_contains_value():
    ema = EMABaseline(decay=0.9, init=0.0)
    ema.update(0.42)
    assert "0.42" in repr(ema)


# ---------------------------------------------------------------------------
# Pre-existing tests continue below
# ---------------------------------------------------------------------------

def test_mask_padding_tokens_zero_in_loss():
    """When mask excludes padding tokens, debiased_loss must not include them in the sum.

    Construct a batch where the padded version and the un-padded version should
    give the same loss (the extra padding positions contribute zero).
    """
    from debiased_grpo.losses import debiased_loss
    B, T_real, T_padded = 4, 8, 12

    torch.manual_seed(42)
    log_probs_real = torch.randn(B, T_real)
    ref_log_probs_real = torch.randn(B, T_real)
    rewards = torch.randn(B)
    baseline_rewards = torch.randn(3)
    mask_real = torch.ones(B, T_real, dtype=torch.bool)

    # Padded version: append random junk tokens but mark them as padding.
    junk = torch.randn(B, T_padded - T_real) * 10.0  # large junk values
    log_probs_padded = torch.cat([log_probs_real, junk], dim=1)
    ref_log_probs_padded = torch.cat([ref_log_probs_real, junk * 0.5], dim=1)
    mask_padded = torch.zeros(B, T_padded, dtype=torch.bool)
    mask_padded[:, :T_real] = True

    loss_real = debiased_loss(log_probs_real, ref_log_probs_real, rewards, baseline_rewards, mask_real)
    loss_padded = debiased_loss(log_probs_padded, ref_log_probs_padded, rewards, baseline_rewards, mask_padded)

    assert torch.allclose(loss_real, loss_padded, atol=1e-5), (
        f"Padding tokens affect loss: real={loss_real.item():.6f}, "
        f"padded={loss_padded.item():.6f}"
    )
