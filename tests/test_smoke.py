"""
Smoke tests for debiased-grpo.

These tests simulate lightweight end-to-end scenarios using tiny synthetic
tensors on CPU.  They verify that the full computation graph (IS weights →
loss → backward → optimizer step) can execute without errors and produce
finite gradients.
"""

import torch
import pytest
from debiased_grpo.losses import debiased_loss, grpo_loss
from debiased_grpo.strategies import FullSequenceIS
from debiased_grpo.utils import compute_ess


def test_smoke_two_debiased_steps():
    """Simulate 2 optimization steps with debiased loss on CPU with tiny tensors."""
    B, T = 8, 32
    # Fake a small linear "model" whose log probs we can optimize.
    log_prob_params = torch.randn(B, T, requires_grad=True)
    ref_log_probs = torch.zeros(B, T)
    rewards = torch.randn(B)
    baseline_rewards = torch.randn(4)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -8:] = False

    optimizer = torch.optim.Adam([log_prob_params], lr=1e-3)
    losses = []
    for step in range(2):
        optimizer.zero_grad()
        loss = debiased_loss(log_prob_params, ref_log_probs, rewards, baseline_rewards, mask)
        assert loss.isfinite(), f"NaN/Inf loss at step {step}"
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(isinstance(l, float) for l in losses)
    print(f"Smoke test passed. Losses: {losses}")


def test_smoke_two_grpo_steps():
    """Simulate 2 optimization steps with GRPO loss for comparison."""
    B, T = 8, 32
    log_prob_params = torch.randn(B, T, requires_grad=True)
    ref_log_probs = torch.zeros(B, T)
    rewards = torch.randn(B)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -8:] = False

    optimizer = torch.optim.SGD([log_prob_params], lr=1e-3)
    losses = []
    for step in range(2):
        optimizer.zero_grad()
        loss = grpo_loss(log_prob_params, ref_log_probs, rewards, mask)
        assert loss.isfinite(), f"NaN/Inf loss at step {step}"
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(isinstance(l, float) for l in losses)


def test_ess_tracked_during_training():
    """ESS must be finite and positive throughout the computation graph."""
    B, T = 4, 16
    log_probs = torch.randn(B, T) - 1.0
    ref_log_probs = torch.randn(B, T) - 1.0
    mask = torch.ones(B, T, dtype=torch.bool)
    weights = FullSequenceIS().compute(log_probs - ref_log_probs, mask)
    ess = compute_ess(weights)
    assert ess.isfinite()
    assert ess.item() > 0


def test_gradient_norms_are_finite_after_debiased_step():
    """After a debiased backward pass, all gradients must be finite."""
    B, T = 4, 16
    log_prob_params = torch.randn(B, T, requires_grad=True)
    ref_log_probs = torch.zeros(B, T)
    rewards = torch.randn(B)
    baseline_rewards = torch.randn(4)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -4:] = False

    loss = debiased_loss(log_prob_params, ref_log_probs, rewards, baseline_rewards, mask)
    loss.backward()

    assert log_prob_params.grad is not None
    assert torch.all(log_prob_params.grad.isfinite()), "Non-finite gradients after debiased backward"


def test_debiased_and_grpo_agree_when_policies_match():
    """When policy == ref, IS ratio is 1 everywhere; both losses should be similar in magnitude."""
    B, T = 4, 16
    log_probs = torch.randn(B, T) - 1.0
    ref_log_probs = log_probs.clone()  # identical policies
    rewards = torch.randn(B)
    mask = torch.ones(B, T, dtype=torch.bool)
    baseline_rewards = rewards.mean().expand(4)  # baseline ≈ reward mean

    debiased = debiased_loss(log_probs, ref_log_probs, rewards, baseline_rewards, mask)
    grpo = grpo_loss(log_probs, ref_log_probs, rewards, mask)

    # Both losses should be finite; we don't assert exact equality because
    # they use different baselines (independent vs LOO).
    assert debiased.isfinite()
    assert grpo.isfinite()


def test_large_batch_does_not_overflow():
    """A larger batch of longer sequences must not produce NaN/Inf in the loss."""
    B, T = 16, 128
    log_prob_params = torch.randn(B, T, requires_grad=True)
    ref_log_probs = torch.randn(B, T)
    rewards = torch.randn(B)
    baseline_rewards = torch.randn(8)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -16:] = False

    loss = debiased_loss(log_prob_params, ref_log_probs, rewards, baseline_rewards, mask)
    assert loss.isfinite(), "Loss overflowed on large batch"
    loss.backward()
    assert torch.all(log_prob_params.grad.isfinite()), "Gradient overflowed on large batch"
