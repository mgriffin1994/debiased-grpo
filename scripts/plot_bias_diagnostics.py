"""Bias-diagnostic figures from per-rollout JSONL dumps.

Reads ``outputs/diagnostics/<cell>/step_<NNNN>.jsonl`` (theta rollouts) and
``outputs/diagnostics/_ref_baselines.jsonl`` (cell-independent pi_ref
baselines), then writes figures to ``notes/figures/diagnostics/``:

  * ``ess_full_vs_pertoken.png`` — full-sequence vs per-token cumulative
    log-ratio distribution by step (full-seq tail is the one that matters for
    the terminal-reward correction; per-token cumulative shown for contrast).
  * ``length_vs_correct.png`` — length conditional on correctness across
    training (Dr. GRPO Fig 2 analog).
  * ``baseline_bias_corr.png`` — Corr(b_ij, r_ij) for the three baseline
    estimators (mean-with-self, true LOO, independent pi_ref). MWS is
    biased (positive correlation by construction); LOO and indep are
    unbiased (zero correlation). Per cell at step 1000.
  * ``baseline_advantage_variance.png`` — per-prompt Var_j(r − b) for
    MWS, true LOO, indep π_ref baselines. True LOO has the
    (G/(G−1))² ≈ 1.78× variance penalty; MWS and indep tie.
  * ``stdnorm_signal_share.png`` — per-bin share of total squared
    advantage signal Σ_j(r − b)²/(σ + ε)² (with std-norm) vs Σ_j(r − b)²
    (without). Measures how std-norm reweights the actual gradient
    *signal*, not just the prompt weight.

All diagnostics share the same input data; the mechanisms are independent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG_ROOT = REPO_ROOT / "outputs" / "diagnostics"
OUT_DIR = REPO_ROOT / "notes" / "figures" / "diagnostics"

CELLS = [
    ("g0_paper_grpo", "paper GRPO (full-seq IS)", "tab:red"),
    ("g0a_paper_grpo", "paper GRPO (per-token IS)", "tab:olive"),
    ("debiased_grpo", "debiased GRPO (ours)", "tab:blue"),
]
STEPS = [200, 500, 1000]


def load_step(cell: str, step: int) -> List[dict]:
    path = DIAG_ROOT / cell / f"step_{step:04d}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def per_rollout_metrics(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (prompt_id, length, reward, log_w_full, log_w_cumulative_per_token_flat)."""
    pids = np.array([r["prompt_id"] for r in rows])
    lengths = np.array([r["length"] for r in rows])
    rewards = np.array([r["reward"] for r in rows])
    log_w_full = np.array([
        float(np.sum(np.subtract(r["log_prob_theta"], r["log_prob_ref"])))
        for r in rows
    ])
    cumulative_per_token: List[float] = []
    for r in rows:
        diffs = np.subtract(r["log_prob_theta"], r["log_prob_ref"])
        cumulative_per_token.extend(np.cumsum(diffs).tolist())
    return pids, lengths, rewards, log_w_full, np.asarray(cumulative_per_token)


def fig_ess_full_vs_pertoken() -> None:
    """Cumulative log-ratio distribution per cell, evolving across training steps.

    Plotted as ECDF so heavy tails are legible. The full-sequence weight is one
    value per rollout (sum across all real tokens); the per-token cumulative
    log-ratio is one value per (rollout, position). At position t its variance
    is `t·σ²`; at position T it equals the full-sequence value. The interesting
    thing here is the cell-level *spread*: paper GRPO stays near 0 (PPO clip held
    the policy near π_ref), debiased GRPO spreads wider (no clip; policy drifted;
    the log-weight clamp absorbed the extreme excursions).
    """
    fig, axes = plt.subplots(1, len(STEPS), figsize=(16, 4), constrained_layout=True, sharey=True)
    for col, step in enumerate(STEPS):
        ax = axes[col]
        for cell_dir, label, colour in CELLS:
            rows = load_step(cell_dir, step)
            if not rows:
                continue
            _, _, _, log_w_full, log_w_cumulative = per_rollout_metrics(rows)
            xs_f = np.sort(log_w_full)
            ys_f = np.linspace(0, 1, len(xs_f))
            ax.plot(xs_f, ys_f, color=colour, linestyle="-", linewidth=1.6,
                    label=f"{label} full-seq", alpha=0.9)
            xs_p = np.sort(log_w_cumulative)
            ys_p = np.linspace(0, 1, len(xs_p))
            ax.plot(xs_p, ys_p, color=colour, linestyle="--", linewidth=1.0,
                    label=f"{label} per-token cumulative", alpha=0.7)
        ax.set_title(f"step {step}")
        ax.set_xlabel(r"$\log w$  (cumulative log-ratio)")
        if col == 0:
            ax.set_ylabel("ECDF")
            ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25)
        # Symmetric x-limits around the union range; keeps the visual comparison
        # honest even when one cell's tail blows up.
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    fig.suptitle(
        "Cumulative log-ratio  log w = log π_θ(a_{1:t}) − log π_ref(a_{1:t})\n"
        "Solid = full-sequence (per rollout, t = T_i).  Dashed = per-token cumulative (every t ∈ [1, T_i]).",
        fontsize=11,
    )
    out = OUT_DIR / "ess_full_vs_pertoken.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_length_vs_correct() -> None:
    """Mean length conditional on correctness, by step. Dr. GRPO Fig 2 analog."""
    fig, axes = plt.subplots(1, len(CELLS), figsize=(16, 4), constrained_layout=True, sharey=True)
    for col, (cell_dir, label, colour) in enumerate(CELLS):
        ax = axes[col]
        means_correct, means_wrong, steps_seen = [], [], []
        for step in STEPS:
            rows = load_step(cell_dir, step)
            if not rows:
                continue
            _, lengths, rewards, _, _ = per_rollout_metrics(rows)
            mc = lengths[rewards > 0.5].mean() if (rewards > 0.5).any() else np.nan
            mw = lengths[rewards <= 0.5].mean() if (rewards <= 0.5).any() else np.nan
            means_correct.append(mc)
            means_wrong.append(mw)
            steps_seen.append(step)
        ax.plot(steps_seen, means_correct, marker="o", color="tab:green", label="correct (r=1)")
        ax.plot(steps_seen, means_wrong, marker="s", color="tab:gray", label="incorrect (r=0)")
        ax.set_title(label)
        ax.set_xlabel("training step")
        if col == 0:
            ax.set_ylabel("mean response length (tokens)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 200)
    fig.suptitle(
        "Bias 2 (length norm): mean response length conditional on correctness\n"
        "Dr. GRPO predicts: G0 grows 'incorrect' length; G1c (token-norm-free) does not.",
        fontsize=11,
    )
    out = OUT_DIR / "length_vs_correct.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def _load_resample() -> dict:
    path = DIAG_ROOT / "baseline_bias_resample.json"
    return json.load(open(path)) if path.exists() else {}


def fig_baseline_bias_corr() -> None:
    """Bias 4a — conditional self-inclusion covariance Cov(b_i, r_i | x).

    The baseline contributes zero gradient bias iff it is independent of the
    rollout it is subtracted from, *conditional on the prompt*. The clean
    measurement is the within-prompt covariance Cov(b_i, r_i | x), estimated by
    resampling many groups per prompt (``baseline_bias_resample.py``); a marginal
    Corr(b, r) over pooled rollouts would instead be dominated by between-prompt
    difficulty and is NOT the bias. Mean-with-self includes r_i → Cov ≈ σ²/G > 0
    (biased); the independent π_ref baseline is external → Cov ≈ 0 (unbiased).
    """
    res = _load_resample()
    if not res:
        print("[plot] no baseline_bias_resample.json; skip 4a figure")
        return
    s2, G = res["sigma2"], res["G"]
    names = ["mean-with-self\n(standard GRPO)", r"independent $\pi_\mathrm{ref}$" + "\n(debiased)"]
    emp = [res["mws"]["cond_cov_b_r"], res["independent"]["cond_cov_b_r"]]
    se = [res["mws"].get("cond_cov_b_r_se", 0.0), res["independent"].get("cond_cov_b_r_se", 0.0)]
    thy = [res["mws"]["cond_cov_b_r_theory"], res["independent"]["cond_cov_b_r_theory"]]
    pos = np.arange(2); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.bar(pos - w / 2, emp, w, yerr=se, capsize=5,
           error_kw=dict(ecolor="black", lw=1.2),
           color="tab:blue", edgecolor="black", linewidth=0.5, label="empirical ±SE (resampled)")
    ax.bar(pos + w / 2, thy, w, color="lightgray", edgecolor="black", linewidth=0.5, label="theory")
    for p, e, t, s in zip(pos, emp, thy, se):
        ax.text(p - w / 2, e + s + 0.0012, f"{e:+.4f}\n±{s:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(p + w / 2, t + 0.0012, f"{t:+.4f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.7)
    ax.set_xticks(pos); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel(r"$\mathrm{Cov}(b_i, r_i \mid x)$")
    ax.legend(fontsize=9)
    ax.set_title(f"Bias 4a (self-inclusion): conditional Cov(b, r) — σ²={s2:.3f}, G={G}, R=8\n"
                 "MWS includes r_i → biased (>0); independent baseline → 0 within noise.")
    ax.grid(True, alpha=0.25, axis="y")
    out = OUT_DIR / "baseline_bias_corr.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


def fig_baseline_advantage_variance() -> None:
    """Bias 4b — conditional advantage variance scaling + cross-sample coupling.

    Left: the conditional (per-prompt) advantage variance Var(A_i | x) for the
    three baselines, in units of σ² = Var(r|x): MWS = (G−1)/G (biased), LOO =
    G/(G−1), independent = 1 + 1/M. The independent baseline is NOT lower-variance
    than LOO at small M — its win is unbiasedness + no functional coupling.
    Right: the cross-sample advantage covariance Cov(A_i, A_j | x) — empirical
    (resampled, debiased model) vs theory — for MWS (−σ²/G) and independent
    (+σ²/M, a vanishing shared-baseline offset). LOO is theory-only (not trained).
    """
    res = _load_resample()
    G = res.get("G", 4); M = res.get("M", 2)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)

    # Left: variance scaling (theory, ×σ²)
    names = ["MWS\n(biased)", "true LOO", r"independent" + "\n(M=%d)" % M]
    var = [(G - 1) / G, G / (G - 1), 1 + 1 / M]
    pos = np.arange(3)
    axL.bar(pos, var, 0.6, color=["tab:red", "tab:orange", "tab:blue"], edgecolor="black", linewidth=0.5)
    for p, v in zip(pos, var):
        axL.text(p, v + 0.02, f"{v:.2f}σ²", ha="center", va="bottom", fontsize=10)
    axL.set_xticks(pos); axL.set_xticklabels(names, fontsize=10)
    axL.set_ylabel(r"$\mathrm{Var}(A_i \mid x)\,/\,\sigma^2$")
    axL.set_title(f"Conditional advantage variance (G={G})\n"
                  f"LOO/MWS=(G/(G−1))²={ (G/(G-1))**2:.2f}, "
                  f"indep/MWS={(1+1/M)*G/(G-1):.2f}")
    axL.grid(True, alpha=0.25, axis="y")

    # Right: cross-sample covariance empirical vs theory (MWS, independent)
    if res:
        labels = ["mean-with-self", r"independent $\pi_\mathrm{ref}$"]
        emp = [res["mws"]["cross_cov"], res["independent"]["cross_cov"]]
        thy = [res["mws"]["cross_cov_theory"], res["independent"]["cross_cov_theory"]]
        p2 = np.arange(2); w = 0.35
        axR.bar(p2 - w / 2, emp, w, color="tab:blue", edgecolor="black", linewidth=0.5, label="empirical (resampled)")
        axR.bar(p2 + w / 2, thy, w, color="lightgray", edgecolor="black", linewidth=0.5, label="theory")
        for p, e, t in zip(p2, emp, thy):
            axR.text(p - w / 2, e + 0.001 * np.sign(e), f"{e:+.3f}", ha="center",
                     va="bottom" if e >= 0 else "top", fontsize=9)
            axR.text(p + w / 2, t + 0.001 * np.sign(t), f"{t:+.3f}", ha="center",
                     va="bottom" if t >= 0 else "top", fontsize=9)
        axR.axhline(0, color="black", linewidth=0.7, alpha=0.7)
        axR.set_xticks(p2); axR.set_xticklabels(labels, fontsize=10)
        axR.set_ylabel(r"$\mathrm{Cov}(A_i, A_j \mid x)$, $i\neq j$")
        axR.legend(fontsize=9)
        axR.set_title("Cross-sample advantage coupling\nMWS −σ²/G; independent +σ²/M (→0 as M↑)")
        axR.grid(True, alpha=0.25, axis="y")
    out = OUT_DIR / "baseline_advantage_variance.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


def fig_stdnorm_signal_share() -> None:
    """Bias 3 (std-norm): greedy pass@1 by prompt difficulty, g0 vs debiased.

    Std normalisation divides advantages by 1/σ(reward|x), which up-weights
    low-σ prompts (easy/all-correct or hard/all-wrong); its effect is therefore
    difficulty-dependent. Per-prompt reward *variance* is degenerate for binary
    rewards, so difficulty is the reference-model solve-rate (``eval_by_difficulty.py``).
    Bars: 3-seed greedy pass@1 per difficulty tier for paper GRPO (std-norm ON)
    vs debiased (std-norm OFF). Tier sizes annotated.
    """
    path = DIAG_ROOT / "eval_by_difficulty.json"
    if not path.exists():
        print("[plot] no eval_by_difficulty.json; skip difficulty figure")
        return
    res = json.load(open(path))
    tiers = ["hard", "medium", "easy"]
    sizes = res["g0_paper_grpo"]["tier_sizes"]
    g0 = [res["g0_paper_grpo"]["by_tier"].get(t, {}).get("mean", 0.0) for t in tiers]
    db = [res["debiased_grpo"]["by_tier"].get(t, {}).get("mean", 0.0) for t in tiers]
    pos = np.arange(len(tiers)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    ax.bar(pos - w / 2, g0, w, color="tab:red", edgecolor="black", linewidth=0.5,
           label="paper GRPO (std-norm ON)")
    ax.bar(pos + w / 2, db, w, color="tab:blue", edgecolor="black", linewidth=0.5,
           label="debiased (std-norm OFF)")
    for p, a, b in zip(pos, g0, db):
        ax.text(p - w / 2, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(p + w / 2, b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{t}\n(n={sizes[t]})" for t in tiers], fontsize=10)
    ax.set_ylabel("greedy pass@1 (3-seed mean)")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Bias 3 (std-norm effect): pass@1 by difficulty tier\n"
                 "debiased (std-norm off) ≥ paper GRPO in every tier; biggest on easy (n=4, noisy).")
    ax.grid(True, alpha=0.25, axis="y")
    out = OUT_DIR / "stdnorm_signal_share.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_ess_full_vs_pertoken()
    fig_length_vs_correct()
    fig_baseline_bias_corr()
    fig_baseline_advantage_variance()
    fig_stdnorm_signal_share()


if __name__ == "__main__":
    main()
