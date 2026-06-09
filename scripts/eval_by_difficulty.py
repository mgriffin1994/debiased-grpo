"""Greedy pass@1 by prompt difficulty, for the paper-GRPO and debiased models.

Bias 3 (std normalization) reweights prompts by 1/σ(reward), so its effect is
difficulty-dependent. Per-prompt reward *variance* is degenerate for binary
rewards (0 for most prompts), so difficulty is instead the **reference-model
solve-rate** over the M' samples already collected in
``outputs/diagnostics/_ref_baselines.jsonl`` (continuous in [0,1]); prompts are
split into difficulty tiers by that rate.

For each model (``g0_paper_grpo`` = std-norm ON, ``debiased_grpo`` = std-norm OFF)
we greedily evaluate every val prompt with each available seed's final
checkpoint, then report **pass@1 per difficulty tier** (seed-averaged). Run:
``make eval-by-difficulty`` (needs GPU). Output:
``outputs/diagnostics/eval_by_difficulty.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bias_diagnostics import build_base_model, load_lightning_ckpt, load_val_prompts  # noqa: E402
from train_grpo import compute_reward  # noqa: E402

STEP = 1000
MODELS = {
    "g0_paper_grpo": ["g0_paper_grpo", "g0_paper_grpo_s43", "g0_paper_grpo_s44"],
    "debiased_grpo": ["debiased_grpo", "debiased_grpo_s43", "debiased_grpo_s44"],
}
TIERS = [("hard", 0.0, 0.25), ("medium", 0.25, 0.625), ("easy", 0.625, 1.001)]
DEVICE = "cuda:0"


def difficulty() -> dict:
    path = REPO_ROOT / "outputs" / "diagnostics" / "_ref_baselines.jsonl"
    out = {}
    for line in open(path):
        row = json.loads(line)
        out[row["prompt_id"]] = float(np.mean(row["rewards"]))   # solve-rate
    return out


def tier_of(rate: float) -> str:
    for name, lo, hi in TIERS:
        if lo <= rate < hi:
            return name
    return "easy"


@torch.no_grad()
def greedy_correct(model, tokenizer, item) -> float:
    prompt = f"Question: {item['question']}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    out = model.generate(**enc, do_sample=False, max_new_tokens=192,
                         pad_token_id=tokenizer.pad_token_id)
    comp = out[0, enc["input_ids"].shape[1]:].cpu()
    return float(compute_reward(comp, item["answer"], tokenizer))


def main() -> None:
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "gsm8k_qwen05b.yaml"))
    val = load_val_prompts(cfg)
    diff = difficulty()
    tiers = {pid: tier_of(diff.get(pid, 1.0)) for pid in range(len(val))}
    counts = {t: sum(1 for v in tiers.values() if v == t) for t, _, _ in TIERS}
    print(f"[difficulty] tier sizes: {counts}")

    model, tokenizer = build_base_model(cfg)
    results = {}
    for model_type, seeds in MODELS.items():
        per_tier_acc = {t: [] for t, _, _ in TIERS}
        overall = []
        for seed_dir in seeds:
            ckpt = REPO_ROOT / "outputs" / seed_dir / "checkpoints" / f"epoch=0-step={STEP}.ckpt"
            if not ckpt.exists():
                print(f"[eval] WARN missing {ckpt}; skip"); continue
            load_lightning_ckpt(model, ckpt); model.eval()
            correct = {t: [] for t, _, _ in TIERS}
            for pid, item in enumerate(val):
                c = greedy_correct(model, tokenizer, item)
                correct[tiers[pid]].append(c)
                if (pid + 1) % 25 == 0:
                    torch.cuda.empty_cache()
            for t, _, _ in TIERS:
                if correct[t]:
                    per_tier_acc[t].append(float(np.mean(correct[t])))
            overall.append(float(np.mean([x for v in correct.values() for x in v])))
            print(f"  {seed_dir}: " + "  ".join(
                f"{t}={np.mean(correct[t]):.2f}" for t, _, _ in TIERS if correct[t]))
        results[model_type] = {
            "overall_mean": float(np.mean(overall)) if overall else None,
            "by_tier": {t: {"mean": float(np.mean(v)), "n_seeds": len(v)}
                        for t, v in per_tier_acc.items() if v},
            "tier_sizes": counts,
        }
    out = REPO_ROOT / "outputs" / "diagnostics" / "eval_by_difficulty.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"[eval] wrote {out}")
    for mt, res in results.items():
        print(f"{mt}: overall={res['overall_mean']:.3f}  "
              + "  ".join(f"{t}={d['mean']:.2f}" for t, d in res["by_tier"].items()))


if __name__ == "__main__":
    main()
