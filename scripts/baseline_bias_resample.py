"""Live (measured-on-the-model) baseline diagnostics via multi-group resampling.

The per-step diagnostics in ``bias_diagnostics.py`` sample ONE group of G
rollouts per prompt, which is enough for the length / IS-variance diagnostics but
NOT for the baseline algebra: the conditional self-inclusion covariance
``Cov(b_i, r_i | x)`` and the cross-sample advantage covariance
``Cov(A_i, A_j | x)`` are properties *across draws of the group at a fixed
prompt*. Here we resample R groups of G rollouts per prompt on the final
``debiased_grpo`` adapter, plus R draws of M reference rollouts, and estimate the
mean-with-self (MWS) and independent baselines on the SAME reward distribution
(the baseline algebra is a property of the rewards, not of which baseline the
model trained with). Per prompt, then averaged:

  - σ² (within-group reward variance, conditional on x),
  - MWS:        Cov(b_i, r_i | x) ≈ σ²/G  (self-inclusion ⇒ biased),  Cov(A_i,A_j|x) ≈ −σ²/G
  - independent: Cov(b_i, r_i | x) ≈ 0,                              Cov(A_i,A_j|x) ≈ +σ²/M

Empirical counterpart to the exact algebra in ``tests/test_baseline_algebra.py``.
Output: ``outputs/diagnostics/baseline_bias_resample.json``. Run:
``make baseline-bias-resample`` (needs GPU).
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
from debiased_grpo.sampling import GroupRolloutSampler, IndependentBaselineSampler  # noqa: E402

R, G, M = 8, 4, 2                 # group draws, group size, ref rollouts (training M)
POLICY_BATCH = 16                # completions per generate() call (8 GB-safe)
CELL = "debiased_grpo"
STEP = 1000
DEVICE = "cuda:0"


def _cond_cov_b_r(b, r):
    b = b - b.mean()
    return float(np.mean([float((b * (r[:, i] - r[:, i].mean())).mean()) for i in range(r.shape[1])]))


def _cross_cov(A):
    Ac = A - A.mean(axis=0, keepdims=True)
    C = (Ac.T @ Ac) / A.shape[0]
    g = A.shape[1]
    return float((C.sum() - np.trace(C)) / (g * (g - 1)))


@torch.no_grad()
def _policy_rewards(sampler, tokenizer, item, n):
    prompt = f"Question: {item['question']}\nAnswer:"
    pid = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"][0].to(DEVICE)
    rew = []
    while len(rew) < n:
        raw, *_ = sampler.sample(pid, with_grad_log_probs=False)
        rew.extend(float(compute_reward(c.cpu(), item["answer"], tokenizer)) for c in raw)
    return np.array(rew[:n])


@torch.no_grad()
def _ref_rewards(ref_sampler, tokenizer, item, n):
    prompt = f"Question: {item['question']}\nAnswer:"
    pid = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"][0].to(DEVICE)
    out = []
    while len(out) < n:
        out.extend(ref_sampler.sample_baseline_rewards(
            pid, lambda toks: compute_reward(toks, item["answer"], tokenizer)).tolist())
    return np.array(out[:n])


def main() -> None:
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "gsm8k_qwen05b.yaml"))
    val = load_val_prompts(cfg)
    model, tokenizer = build_base_model(cfg)
    ckpt = REPO_ROOT / "outputs" / CELL / "checkpoints" / f"epoch=0-step={STEP}.ckpt"
    load_lightning_ckpt(model, ckpt)
    model.eval()
    sampler = GroupRolloutSampler(model=model, tokenizer=tokenizer, ref_model=None,
                                  group_size=POLICY_BATCH,
                                  max_new_tokens=cfg.get("max_new_tokens", 192),
                                  temperature=cfg.get("temperature", 0.9))
    ref_sampler = IndependentBaselineSampler(ref_model=model, tokenizer=tokenizer,
                                             num_baseline=POLICY_BATCH,
                                             max_new_tokens=cfg.get("max_new_tokens", 192),
                                             temperature=1.0, adapter_disable=True)

    sig2, cbr_mws, cbr_ind, cross_mws, cross_ind = [], [], [], [], []
    bind_var, bmws_var = [], []          # sample-variance of each baseline across the R draws
    for pid, item in enumerate(val):
        r = _policy_rewards(sampler, tokenizer, item, R * G).reshape(R, G)     # (R,G)
        refs = _ref_rewards(ref_sampler, tokenizer, item, R * M).reshape(R, M)  # (R,M)
        sig2.append(float(r.var(axis=1, ddof=0).mean()))
        b_mws = r.mean(axis=1)                       # (R,) incl self
        b_ind = refs.mean(axis=1)                    # (R,) external, ⟂ r
        bind_var.append(float(b_ind.var())); bmws_var.append(float(b_mws.var()))
        cbr_mws.append(_cond_cov_b_r(b_mws, r));  cbr_ind.append(_cond_cov_b_r(b_ind, r))
        cross_mws.append(_cross_cov(r - b_mws[:, None]))
        cross_ind.append(_cross_cov(r - b_ind[:, None]))
        if (pid + 1) % 10 == 0:
            print(f"  prompt {pid+1}/{len(val)}"); torch.cuda.empty_cache()

    s2 = float(np.mean(sig2))
    # Bootstrap a noise band on the across-prompt mean cond-cov (the empirical
    # estimate is a mean over n_prompts noisy per-prompt covariances; "exactly 0"
    # would only be credible if b_ind had no across-draw variance — which the
    # b_ind_var_mean / n_prompts_bind_varying fields let a reader check).
    def _se(x):
        x = np.asarray(x); return float(x.std(ddof=1) / np.sqrt(len(x)))
    res = {
        "cell": CELL, "sigma2": s2, "G": G, "M": M, "R": R, "n_prompts": len(val),
        "mws": {"cond_cov_b_r": float(np.mean(cbr_mws)), "cond_cov_b_r_se": _se(cbr_mws),
                "cond_cov_b_r_theory": s2 / G,
                "cross_cov": float(np.mean(cross_mws)), "cross_cov_theory": -s2 / G,
                "b_var_mean": float(np.mean(bmws_var))},
        "independent": {"cond_cov_b_r": float(np.mean(cbr_ind)), "cond_cov_b_r_se": _se(cbr_ind),
                        "cond_cov_b_r_theory": 0.0,
                        "cross_cov": float(np.mean(cross_ind)), "cross_cov_theory": s2 / M,
                        "b_var_mean": float(np.mean(bind_var)),
                        "n_prompts_bind_varying": int(sum(1 for v in bind_var if v > 1e-9))},
    }
    out = REPO_ROOT / "outputs" / "diagnostics" / "baseline_bias_resample.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"[resample] wrote {out}")


if __name__ == "__main__":
    main()
