"""Bias 1 IS-weight diagnostics, reproducible from logged data (no GPU).

Two *distinct* importance-weight quantities are reported, because conflating them
is the easy mistake:

1. **Training-time IS weight** ρ = π_θ_current / π_behavior — the actual off-policy
   correction applied inside the μ-step inner loop. Read from each cell's
   ``metrics.csv`` (``train/is_log_ratio_std`` = per-token log-ratio std vs the
   behavior policy) together with the loss trace. This is what determines whether
   the ±5 log-weight clamp ever *binds* and whether training is stable.

2. **Drift from the frozen base** log w = Σ_t (log π_θ − log π_ref) — how far the
   policy has moved from the frozen reference π_ref over all of training. Computed
   from the per-rollout diagnostic JSONL (``outputs/diagnostics/<cell>/step_*.jsonl``).
   This is a trust-region-drift measure, NOT the IS correction (the clamp is not
   applied to it during training); we report a clamped-vs-unclamped *diagnostic*
   ESS on it only to illustrate weight concentration, clearly labelled as such.

Output: ``outputs/diagnostics/bias1_isweight.json``. Run: ``python scripts/bias1_isweight.py``.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG = REPO_ROOT / "outputs" / "diagnostics"
CELLS = ["g0_paper_grpo", "g0a_paper_grpo", "debiased_grpo"]
STEPS = [200, 500, 1000]
CLAMP = 5.0


def _ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float((w.sum() ** 2) / np.sum(w ** 2))


def _drift_from_ref(cell: str, step: int) -> dict | None:
    path = DIAG / cell / f"step_{step:04d}.jsonl"
    if not path.exists():
        return None
    logw, cumulative = [], []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        d = np.subtract(r["log_prob_theta"], r["log_prob_ref"])
        logw.append(float(d.sum()))
        cumulative.extend(np.cumsum(d).tolist())
    logw = np.asarray(logw)
    n = len(logw)
    # ESS on exp(log w): unclamped (scale-stabilised) vs ±5-clamped, as a fraction
    # of n. This is a *diagnostic* on drift-from-ref weights, not the inner-loop IS.
    w_uncl = np.exp(logw - logw.max())
    w_clmp = np.exp(np.clip(logw, None, CLAMP))
    return {
        "n": n,
        "logw_std": float(logw.std()),
        "logw_var_full": float(logw.var()),
        "logw_var_cumulative": float(np.var(cumulative)),
        "full_over_cumulative": float(logw.var() / max(np.var(cumulative), 1e-9)),
        "ess_unclamped_frac": _ess(w_uncl) / n,
        "ess_clamped_frac": _ess(w_clmp) / n,
    }


def _training_is(cell: str) -> dict | None:
    fs = glob.glob(str(REPO_ROOT / "outputs" / cell / "**" / "metrics.csv"), recursive=True)
    if not fs:
        return None
    df = pd.read_csv(fs[0])
    loss = df["train/loss"].dropna()
    islog = df["train/is_log_ratio_std"].dropna()
    return {
        "is_log_ratio_std_mean": float(islog.mean()),
        "is_log_ratio_std_max": float(islog.max()),
        "logged_steps": int(len(loss)),
        "spike_steps_abs_loss_gt_10": int((loss.abs() > 10).sum()),
        "loss_min": float(loss.min()),
        "loss_max": float(loss.max()),
    }


def main() -> None:
    out = {"clamp": CLAMP, "cells": {}}
    for cell in CELLS:
        out["cells"][cell] = {
            "training_is_weight": _training_is(cell),
            "drift_from_ref": {str(s): _drift_from_ref(cell, s) for s in STEPS},
        }
    dest = DIAG / "bias1_isweight.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"[bias1] wrote {dest}")


if __name__ == "__main__":
    main()
