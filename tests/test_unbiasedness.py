"""Unbiasedness of the full-sequence IS off-policy policy gradient.

This is the load-bearing math test for the refactor: for a *terminal*
(sequence-level) reward, the unbiased off-policy correction is the
**full-sequence** importance ratio ``w(y) = exp(Σ_t log π_θ(a_t)/μ(a_t))``,
NOT a per-token ratio. We verify this on a tiny exact toy where every quantity
can be enumerated.

Setup (deterministic, no Monte-Carlo noise):
  * Horizon T = 2 timesteps, vocabulary {0, 1} → exactly 4 trajectories.
  * A behavior policy μ and a target policy π_θ that differ.
  * A terminal reward R(a_1, a_2) that depends on the whole trajectory.
  * An INDEPENDENT baseline b (a constant, not a function of the sampled
    actions) — independence is what keeps the baseline subtraction unbiased.

The off-policy estimator we test is
    g_hat = E_{y ~ μ}[ w(y) · (R(y) − b) · Σ_t ∇ log π_θ(a_t) ]
and the on-policy ground truth is
    g     = E_{y ~ π_θ}[ R(y) · Σ_t ∇ log π_θ(a_t) ].
Because the baseline is independent, E_{π_θ}[b · Σ_t ∇ log π_θ] = 0, so the
two expressions are equal in expectation iff the IS weight is the full-sequence
ratio. We enumerate all 4 trajectories and weight by μ, so both sides are
computed exactly.

We also confirm a per-token-IS surrogate (weight ρ_t on token t only) does NOT
match the exact gradient — documenting why full-sequence is required.
"""

import math

import torch

from debiased_grpo.strategies import FullSequenceIS, PerTokenIS


VOCAB = 2
T = 2


def _policy_logits(seed: int) -> torch.Tensor:
    """A (T, VOCAB) logit tensor describing a time-homogeneous-ish policy.

    Each timestep has its own categorical distribution over the vocabulary.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, VOCAB, generator=g)


def _log_probs(logits: torch.Tensor) -> torch.Tensor:
    """Log-softmax over the vocab dimension. Shape (T, VOCAB)."""
    return torch.log_softmax(logits, dim=-1)


def _reward(a1: int, a2: int) -> float:
    """A terminal reward that genuinely depends on the whole trajectory."""
    table = {
        (0, 0): 0.0,
        (0, 1): 1.0,
        (1, 0): 2.0,
        (1, 1): -1.0,
    }
    return table[(a1, a2)]


def _grad_log_pi_sum(theta_logits: torch.Tensor, a1: int, a2: int) -> torch.Tensor:
    """Σ_t ∇_θ log π_θ(a_t) for one trajectory, returned as a flat vector."""
    theta = theta_logits.clone().detach().requires_grad_(True)
    logp = torch.log_softmax(theta, dim=-1)
    score = logp[0, a1] + logp[1, a2]
    (grad,) = torch.autograd.grad(score, theta)
    return grad.reshape(-1)


def _exact_on_policy_gradient(theta_logits: torch.Tensor) -> torch.Tensor:
    """g = Σ_y π_θ(y) R(y) Σ_t ∇ log π_θ(a_t), enumerated over all trajectories."""
    logp = _log_probs(theta_logits)
    g = torch.zeros(theta_logits.numel())
    for a1 in range(VOCAB):
        for a2 in range(VOCAB):
            p = math.exp(logp[0, a1].item() + logp[1, a2].item())
            r = _reward(a1, a2)
            g += p * r * _grad_log_pi_sum(theta_logits, a1, a2)
    return g


def _exact_full_sequence_is_gradient(
    theta_logits: torch.Tensor,
    mu_logits: torch.Tensor,
    baseline: float,
) -> torch.Tensor:
    """g_hat = Σ_y μ(y) w(y) (R(y) − b) Σ_t ∇ log π_θ(a_t), enumerated exactly.

    ``w(y)`` is computed with the production ``FullSequenceIS`` strategy so the
    test exercises the exact code path used in training.
    """
    logp_theta = _log_probs(theta_logits)
    logp_mu = _log_probs(mu_logits)
    weighter = FullSequenceIS()  # strictly unbiased (no clamp)

    g = torch.zeros(theta_logits.numel())
    for a1 in range(VOCAB):
        for a2 in range(VOCAB):
            mu_p = math.exp(logp_mu[0, a1].item() + logp_mu[1, a2].item())
            # Per-token log-ratio for this trajectory, shape (1, T).
            log_ratio = torch.tensor(
                [[
                    logp_theta[0, a1].item() - logp_mu[0, a1].item(),
                    logp_theta[1, a2].item() - logp_mu[1, a2].item(),
                ]]
            )
            mask = torch.ones(1, T, dtype=torch.bool)
            w = weighter.compute(log_ratio, mask).item()  # (1, 1) → scalar
            r = _reward(a1, a2)
            g += mu_p * w * (r - baseline) * _grad_log_pi_sum(theta_logits, a1, a2)
    return g


def _exact_per_token_is_gradient(
    theta_logits: torch.Tensor,
    mu_logits: torch.Tensor,
    baseline: float,
) -> torch.Tensor:
    """Biased surrogate: weight token t by ρ_t = π_θ(a_t)/μ(a_t) only.

    The per-token contribution at timestep t is ρ_t · (R − b) · ∇ log π_θ(a_t).
    Summing over t and taking E_μ does NOT recover the on-policy gradient for a
    terminal reward, because the full trajectory ratio is dropped.
    """
    logp_theta = _log_probs(theta_logits)
    logp_mu = _log_probs(mu_logits)
    weighter = PerTokenIS()

    g = torch.zeros(theta_logits.numel())
    for a1 in range(VOCAB):
        for a2 in range(VOCAB):
            mu_p = math.exp(logp_mu[0, a1].item() + logp_mu[1, a2].item())
            log_ratio = torch.tensor(
                [[
                    logp_theta[0, a1].item() - logp_mu[0, a1].item(),
                    logp_theta[1, a2].item() - logp_mu[1, a2].item(),
                ]]
            )
            mask = torch.ones(1, T, dtype=torch.bool)
            rho = weighter.compute(log_ratio, mask)[0]  # (T,) per-token ratios
            r = _reward(a1, a2)
            # Per-token score gradients (∇ log π_θ(a_t)) for each timestep.
            theta = theta_logits.clone().detach().requires_grad_(True)
            logp = torch.log_softmax(theta, dim=-1)
            (g1,) = torch.autograd.grad(logp[0, a1], theta, retain_graph=True)
            (g2,) = torch.autograd.grad(logp[1, a2], theta)
            per_token = rho[0].item() * g1.reshape(-1) + rho[1].item() * g2.reshape(-1)
            g += mu_p * (r - baseline) * per_token
    return g


def test_full_sequence_is_matches_on_policy_gradient_exactly():
    """Full-sequence IS off-policy estimator equals the exact on-policy gradient.

    Enumerated over all 4 trajectories, so this is exact (no MC tolerance
    needed). The baseline is an arbitrary constant — independence makes it drop
    out in expectation, leaving the two gradients equal.
    """
    theta_logits = _policy_logits(seed=1)
    mu_logits = _policy_logits(seed=2)  # genuinely different behavior policy
    baseline = 0.7  # independent of the sampled actions

    g_exact = _exact_on_policy_gradient(theta_logits)
    g_is = _exact_full_sequence_is_gradient(theta_logits, mu_logits, baseline)

    assert torch.allclose(g_exact, g_is, atol=1e-6), (
        f"Full-sequence IS gradient does not match on-policy gradient:\n"
        f"  on-policy = {g_exact}\n  full-seq IS = {g_is}"
    )


def test_full_sequence_is_unbiased_for_several_baselines():
    """The independent baseline must not change the (full-sequence IS) gradient."""
    theta_logits = _policy_logits(seed=3)
    mu_logits = _policy_logits(seed=4)
    g_exact = _exact_on_policy_gradient(theta_logits)
    for baseline in (-2.0, 0.0, 0.5, 5.0):
        g_is = _exact_full_sequence_is_gradient(theta_logits, mu_logits, baseline)
        assert torch.allclose(g_exact, g_is, atol=1e-6), (
            f"Baseline {baseline} perturbed the unbiased gradient: {g_is} vs {g_exact}"
        )


def test_per_token_is_is_biased_for_terminal_reward():
    """Per-token IS does NOT match the exact gradient — it is biased.

    This documents why full-sequence IS is the required correction for a
    terminal reward. We assert the per-token estimator differs meaningfully from
    the on-policy ground truth when μ and π_θ differ.
    """
    theta_logits = _policy_logits(seed=1)
    mu_logits = _policy_logits(seed=2)
    baseline = 0.7

    g_exact = _exact_on_policy_gradient(theta_logits)
    g_per_token = _exact_per_token_is_gradient(theta_logits, mu_logits, baseline)

    assert not torch.allclose(g_exact, g_per_token, atol=1e-4), (
        "Per-token IS unexpectedly matched the exact gradient; with differing "
        "policies it must be biased for a terminal reward."
    )


def test_per_token_bias_diagnostic_grows_full_seq_stays_zero():
    """The per-token-bias diagnostic: full-seq IS unbiased at all drift/length,
    per-token bias positive and monotone in drift. Locks scripts/per_token_bias.py.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from per_token_bias import sweep_drift  # noqa: E402

    rows = sweep_drift(T=3, alphas=[0.0, 0.5, 1.5])
    # Full-sequence IS is unbiased at every drift level (exact enumeration).
    assert all(r["full_seq_bias"] < 1e-5 for r in rows)
    # Per-token bias starts ~0 at zero drift and is strictly positive once drifted.
    assert rows[0]["per_token_bias"] < 1e-5
    assert rows[-1]["per_token_bias"] > 0.05
    # Monotone non-decreasing in drift.
    pt = [r["per_token_bias"] for r in rows]
    assert pt[0] <= pt[1] <= pt[2] + 1e-9
