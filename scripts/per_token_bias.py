"""Quantify the bias of the per-token IS surrogate for a terminal reward.

`tests/test_unbiasedness.py` proves (exact enumeration) that for a sequence-level
reward the full-sequence importance ratio recovers the on-policy gradient while
the per-token surrogate (GRPO's choice, DeepSeekMath Eq. 3) does not. This script
turns that binary proof into a *diagnostic*: it measures the gradient bias of each
estimator as a function of (a) how far the target policy has drifted from the
behavior policy, and (b) the sequence length T — the two knobs that control the
bias. Everything is enumerated over all V^T trajectories, so the numbers are exact
(no Monte-Carlo noise), and the IS weights use the production `FullSequenceIS` /
`PerTokenIS` strategies.

Bias metric: relative L2 deviation of the estimator's expected gradient from the
exact on-policy gradient, ‖E[ĝ] − ∇J‖ / ‖∇J‖, averaged over several random
(behavior policy, drift direction, reward) draws. Full-sequence IS is ≈ 0 at every
drift/length (unbiased); the per-token surrogate grows with both.

Output: ``outputs/diagnostics/per_token_bias.json`` and the figure
``notes/figures/diagnostics/per_token_bias.png``. Run: ``make per-token-bias``
(CPU only, no GPU).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from debiased_grpo.strategies import FullSequenceIS, PerTokenIS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
VOCAB = 2
N_DRAWS = 8                       # random (mu, direction, reward) setups to average


def _logp(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits, dim=-1)            # (T, V)


def _kl_theta_mu(theta: torch.Tensor, mu: torch.Tensor) -> float:
    """KL(pi_theta || mu) summed over the T independent per-timestep categoricals."""
    lt, lm = _logp(theta), _logp(mu)
    return float((lt.exp() * (lt - lm)).sum())


def _grad_sum_logpi(theta: torch.Tensor, traj) -> torch.Tensor:
    """∇_θ Σ_t log π_θ(a_t) for one trajectory (flat)."""
    th = theta.clone().detach().requires_grad_(True)
    lp = torch.log_softmax(th, dim=-1)
    score = sum(lp[t, a] for t, a in enumerate(traj))
    (g,) = torch.autograd.grad(score, th)
    return g.reshape(-1)


def _grad_each_logpi(theta: torch.Tensor, traj):
    """List of ∇_θ log π_θ(a_t), one flat tensor per timestep."""
    th = theta.clone().detach().requires_grad_(True)
    lp = torch.log_softmax(th, dim=-1)
    grads = []
    for t, a in enumerate(traj):
        (g,) = torch.autograd.grad(lp[t, a], th, retain_graph=True)
        grads.append(g.reshape(-1))
    return grads


def _gradients(theta, mu, reward, baseline):
    """Exact on-policy, full-sequence-IS, and per-token-IS gradients (enumerated)."""
    lt, lm = _logp(theta), _logp(mu)
    T = theta.shape[0]
    fs, pt = FullSequenceIS(), PerTokenIS()
    g_on = torch.zeros(theta.numel())
    g_fs = torch.zeros(theta.numel())
    g_pt = torch.zeros(theta.numel())
    for traj in itertools.product(range(VOCAB), repeat=T):
        theta_p = float(np.exp(sum(lt[t, a].item() for t, a in enumerate(traj))))
        mu_p = float(np.exp(sum(lm[t, a].item() for t, a in enumerate(traj))))
        r = reward[traj]
        log_ratio = torch.tensor([[lt[t, a].item() - lm[t, a].item()
                                   for t, a in enumerate(traj)]])      # (1,T)
        mask = torch.ones(1, T, dtype=torch.bool)
        gsum = _grad_sum_logpi(theta, traj)
        # On-policy ground truth: E_{pi_theta}[ R · Σ_t ∇log π ] (independent
        # baseline drops out, so use the raw reward).
        g_on += theta_p * r * gsum
        # Full-sequence IS: weight whole trajectory by w(y)=exp(Σ log ρ_t).
        w = fs.compute(log_ratio, mask).item()
        g_fs += mu_p * w * (r - baseline) * gsum
        # Per-token IS: weight token t by ρ_t only.
        rho = pt.compute(log_ratio, mask)[0]                          # (T,)
        each = _grad_each_logpi(theta, traj)
        per = sum(rho[t].item() * each[t] for t in range(T))
        g_pt += mu_p * (r - baseline) * per
    return g_on, g_fs, g_pt


def _rel_bias(g_est, g_on) -> float:
    denom = float(g_on.norm())
    return float((g_est - g_on).norm() / denom) if denom > 1e-9 else 0.0


def _draw(seed: int, T: int):
    gA = torch.Generator().manual_seed(seed)
    gB = torch.Generator().manual_seed(seed + 10_000)
    mu = torch.randn(T, VOCAB, generator=gA)
    direction = torch.randn(T, VOCAB, generator=gB)
    reward = {traj: float(torch.randn(1, generator=gA).item())
              for traj in itertools.product(range(VOCAB), repeat=T)}
    return mu, direction, reward


def sweep_drift(T: int = 4, alphas=None):
    """Bias vs policy drift KL(pi_theta||mu) at fixed sequence length T."""
    if alphas is None:
        alphas = np.linspace(0.0, 2.5, 9)
    rows = []
    for alpha in alphas:
        kls, fs_bias, pt_bias = [], [], []
        for s in range(N_DRAWS):
            mu, direction, reward = _draw(s, T)
            theta = mu + float(alpha) * direction
            g_on, g_fs, g_pt = _gradients(theta, mu, reward, baseline=0.5)
            kls.append(_kl_theta_mu(theta, mu))
            fs_bias.append(_rel_bias(g_fs, g_on))
            pt_bias.append(_rel_bias(g_pt, g_on))
        rows.append({"alpha": float(alpha), "kl": float(np.mean(kls)),
                     "full_seq_bias": float(np.mean(fs_bias)),
                     "per_token_bias": float(np.mean(pt_bias))})
    return rows


def sweep_length(Ts=(1, 2, 3, 4, 6, 8), alpha: float = 1.0):
    """Bias vs sequence length T at a fixed drift scale alpha."""
    rows = []
    for T in Ts:
        fs_bias, pt_bias, kls = [], [], []
        for s in range(N_DRAWS):
            mu, direction, reward = _draw(s, T)
            theta = mu + alpha * direction
            g_on, g_fs, g_pt = _gradients(theta, mu, reward, baseline=0.5)
            kls.append(_kl_theta_mu(theta, mu))
            fs_bias.append(_rel_bias(g_fs, g_on))
            pt_bias.append(_rel_bias(g_pt, g_on))
        rows.append({"T": T, "kl": float(np.mean(kls)),
                     "full_seq_bias": float(np.mean(fs_bias)),
                     "per_token_bias": float(np.mean(pt_bias))})
    return rows


def main() -> None:
    drift = sweep_drift()
    length = sweep_length()
    out = {"vocab": VOCAB, "n_draws": N_DRAWS, "metric": "rel_L2_grad_bias",
           "drift_sweep_T4": drift, "length_sweep_alpha1": length}
    dest = REPO_ROOT / "outputs" / "diagnostics" / "per_token_bias.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)
    kl = [r["kl"] for r in drift]
    axL.plot(kl, [r["per_token_bias"] for r in drift], "o-", color="tab:red",
             label="per-token IS (GRPO)")
    axL.plot(kl, [r["full_seq_bias"] for r in drift], "s-", color="tab:blue",
             label="full-sequence IS (debiased)")
    axL.set_xlabel(r"policy drift  $\mathrm{KL}(\pi_\theta \,\|\, \pi_\mathrm{behavior})$")
    axL.set_ylabel(r"relative gradient bias  $\|\mathbb{E}[\hat g]-\nabla J\|/\|\nabla J\|$")
    axL.set_title(f"Per-token IS bias grows with drift (T={4}, exact)\n"
                  "full-sequence IS stays ~0 (unbiased)")
    axL.legend(fontsize=9); axL.grid(True, alpha=0.25)

    Ts = [r["T"] for r in length]
    axR.plot(Ts, [r["per_token_bias"] for r in length], "o-", color="tab:red",
             label="per-token IS (GRPO)")
    axR.plot(Ts, [r["full_seq_bias"] for r in length], "s-", color="tab:blue",
             label="full-sequence IS (debiased)")
    axR.set_xlabel("sequence length T (terminal reward)")
    axR.set_ylabel("relative gradient bias")
    axR.set_title(r"and compounds with length (fixed drift $\alpha$=1, exact)")
    axR.legend(fontsize=9); axR.grid(True, alpha=0.25)

    figpath = REPO_ROOT / "notes" / "figures" / "diagnostics" / "per_token_bias.png"
    fig.savefig(figpath, dpi=140); plt.close(fig)
    print(json.dumps(out, indent=2))
    print(f"[per-token-bias] wrote {dest} and {figpath}")


if __name__ == "__main__":
    main()
