"""
Shared pytest fixtures for debiased-grpo unit tests.

All tests run on CPU.  No model downloads, no API keys, no GPU required.
"""

import torch
import pytest


@pytest.fixture
def device() -> str:
    """Always return 'cpu' so tests run without a GPU."""
    return "cpu"


@pytest.fixture
def tiny_batch() -> dict:
    """Return a small synthetic batch for testing loss functions.

    Shapes match a realistic training batch with B=4 sequences, each padded
    to T=16 tokens.  The last 4 tokens of each sequence are padding.

    Returns a dict with keys:
        log_probs      (B, T) float — log-probabilities from the current policy.
        ref_log_probs  (B, T) float — log-probabilities from the reference policy.
        rewards        (B,)   float — scalar reward per sequence.
        mask           (B, T) bool  — True for real (non-padding) tokens.
    """
    B, T = 4, 16
    torch.manual_seed(0)

    log_probs = torch.randn(B, T) - 1.0      # negative, as real log-probs are
    ref_log_probs = torch.randn(B, T) - 1.0
    rewards = torch.randn(B)

    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -4:] = False  # last 4 tokens are padding

    return {
        "log_probs": log_probs,
        "ref_log_probs": ref_log_probs,
        "rewards": rewards,
        "mask": mask,
    }
