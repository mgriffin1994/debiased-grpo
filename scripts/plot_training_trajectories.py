"""Unified training-trajectory plot for the trained loss-pipeline cells.

Reads each cell's metrics.csv (Lightning CSVLogger output) and overlays the
six per-step diagnostics — ESS, KL(pi_theta || pi_ref), log-ratio std, mean
response length, mean reward, val accuracy — on a single 6-panel figure.
Writes the figure to ``notes/figures/diagnostics/training_trajectories.png``.

This is a CSV-only pass: no model checkpoint loading, no GPU. The CSVs
contain interleaved train rows (every 10 steps, with ``val/acc`` empty) and
val rows (every 100 steps, with most ``train/*`` columns empty); we route
each metric to the rows that actually contain it.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "notes" / "figures" / "diagnostics"

# (label, csv path, line colour, line style)
CELLS: List[Tuple[str, Path, str, str]] = [
    (
        "G0  paper GRPO",
        REPO_ROOT / "outputs" / "g0_paper_grpo" / "logs" / "version_6" / "metrics.csv",
        "tab:red",
        "-",
    ),
    (
        "Debiased GRPO",
        REPO_ROOT / "outputs" / "debiased_grpo" / "logs" / "version_0" / "metrics.csv",
        "tab:blue",
        "-",
    ),
    (
        "Debiased GRPO (seed 43)",
        REPO_ROOT / "outputs" / "debiased_grpo_s43" / "logs" / "version_0" / "metrics.csv",
        "tab:green",
        "-",
    ),
]


# (column name, panel title, log-y axis?)
PANELS: List[Tuple[str, str, bool]] = [
    ("train/ess", "Effective sample size of IS weights", False),
    ("train/kl_to_ref", "Empirical KL(π_θ ‖ π_ref)", False),
    ("train/log_ratio_std", "Std of per-token log-ratio", False),
    ("train/mean_response_length", "Mean response length (tokens)", False),
    ("train/mean_reward", "Mean reward (group)", False),
    ("val/acc", "Val pass@1 (greedy, 100 prompts)", False),
]


def _load(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing CSV: {path}")
    with open(path) as f:
        return list(csv.DictReader(f))


def _column(rows: List[Dict[str, str]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Pull a (step, value) series, skipping rows where the column is empty."""
    xs: List[int] = []
    ys: List[float] = []
    for r in rows:
        v = r.get(key, "")
        if v == "" or v is None:
            continue
        try:
            ys.append(float(v))
            xs.append(int(r["step"]))
        except (ValueError, KeyError):
            continue
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.float64)


def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    if y.size < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax_idx, (column, title, log_y) in enumerate(PANELS):
        ax = axes_flat[ax_idx]
        for label, path, colour, style in CELLS:
            if not path.exists():
                continue
            rows = _load(path)
            xs, ys = _column(rows, column)
            if xs.size == 0:
                continue
            # Smooth dense per-step series; leave sparse val curves raw.
            if column == "val/acc":
                ax.plot(xs, ys, marker="o", linestyle=style, color=colour,
                        label=label, alpha=0.9, markersize=4)
            else:
                ax.plot(xs, _smooth(ys), linestyle=style, color=colour,
                        label=label, alpha=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(True, alpha=0.25)
        if log_y:
            ax.set_yscale("log")
        if ax_idx == 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Training trajectories — paper GRPO vs Debiased GRPO (full-sequence IS + log-clamp, 1000 steps)",
        fontsize=12,
    )
    out_path = OUT_DIR / "training_trajectories.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
