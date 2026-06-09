# Experiments: debiased-grpo

Results and reproduction for the cells in this release. All numbers are GSM8K
validation pass@1; see the README for the figures.

The corrected method cell is `debiased_grpo` (independent baseline + full-sequence
IS with a log-weight clamp + fixed-constant length norm + no std-norm). It has
now been **trained on 3 seeds (42, 43, 44) and diagnosed**; its numbers are filled
in below. The GRPO baselines — Standard GRPO `g0a` (per-token, Eq. 3) and the `g0`
full-seq-IS ablation — are real, already-trained, and unaffected by the corrected
estimator.

## Setup

| Field | Value |
|-------|-------|
| Model | Qwen2-0.5B (4-bit NF4 QLoRA, r=16, target q/k/v/o projections) |
| Dataset | GSM8K (val split, N=100 batches for eval); terminal/sequence-level reward |
| Rollouts per prompt | 4 gradient + 2 independent baseline |
| Inner loop | μ=4 gradient steps per rollout batch |
| Training budget | 1000 gradient updates (= 250 outer steps × μ=4) |
| max_new_tokens | 192 |
| Seed | 42 (seeds 43, 44 for the 3-seed pass@1 where noted) |
| Hardware | single RTX 3060 Ti (8 GB VRAM) |

The training script `scripts/train_grpo.py` is flag-driven; every cell is one
combination of `--baseline`, `--is-weighting`, `--clipping`, `--length-norm`,
`--std-norm` / `--no-std-norm`, `--log-w-clamp`, `--kl-ref-coef`,
`--kl-behavior-coef`, `--ema-baseline`.
`val/acc` is logged inside the Lightning loop every 100 outer steps (≈ every 400
gradient updates at μ=4); there is no separate eval driver.

## Cells and make targets

| Cell | Description | Make target |
|------|-------------|-------------|
| `g0a` | **Standard GRPO (the paper baseline)**: per-token clipped IS (DeepSeekMath Eq. 3), mean-with-self, per-response length, std-norm | `make train-g0a` / `-s43` / `-s44` |
| `g0` | *Ablation:* GRPO with full-sequence IS swapped in (IS-granularity control; not a published algorithm) | `make train-g0` / `-s43` / `-s44` |
| `debiased_grpo` | Independent π_ref baseline, full-sequence IS, `--log-w-clamp 5.0`, fixed-constant length norm, no std-norm, no clipping | `make train-debiased` / `-s43` / `-s44` |

All cells run at 3 seeds (42/43/44). `g0a` is the faithful paper algorithm (the
GRPO baseline); `g0` swaps in full-sequence IS so `debiased` vs `g0` isolates the
non-IS fixes and `g0` vs `g0a` isolates the IS-granularity axis.

The flag axes (`--baseline`, `--is-weighting`, `--clipping`, `--length-norm`,
`--std-norm`, `--log-w-clamp`) support other combinations for ablation, but only
the cells above are wired as make targets in this release.

## Final GSM8K val pass@1

| Cell | s=42 | s=43 | s=44 | mean | std | 95% CI (t, n=3) | spike% |
|------|------|------|------|------|-----|-----|-----|
| `g0a` — **Standard GRPO** (per-token clipped IS, Eq. 3; MWS, per-resp length, std-norm) | 0.13 | 0.18 | 0.13 | **0.147** | 0.029 | ±0.072 | 0% |
| `debiased_grpo` — indep baseline + full-seq IS + log-clamp + fixed-const length + no std-norm | 0.15 | 0.20 | 0.21 | **0.187** | 0.032 | ±0.080 | 23% |
| `g0` — *ablation:* GRPO + full-sequence IS (IS-granularity control) | 0.17 | 0.15 | 0.19 | 0.170 | 0.020 | ±0.050 | 0% |

Headline: **debiased (0.187) vs Standard GRPO `g0a` (0.147)** — +0.040, overlapping
95% CIs (suggestive, not significant). Standard GRPO and the `g0` ablation are
clip-stable (0% loss spikes); debiased trades that for unbiasedness (23% spikes,
recovered — see Bias 1′).

(Baseline val/acc values are at step 199 — the last val checkpoint before
training ended at gradient step 1000.)

The absolute 0.13–0.18 pass@1 is dominated by the 0.5B model and the 1000-step
budget, not the algorithm — full-scale GRPO systems (e.g. DeepSeek-R1) reach far
higher GSM8K accuracy with much larger models and far longer training. Every
comparison here holds model, data, and budget fixed across cells.

## Key findings

1. **Per-token vs full-sequence IS is within noise on the GRPO baselines.**
   Standard GRPO `g0a` (per-token, Eq. 3, 3 seeds) **0.147** and the `g0` full-seq
   ablation **0.170** are within ~1 prompt of each other, both PPO-clip-stable (0%
   spikes). At μ=4 the per-token drift between sampling and gradient is small enough
   that the full-sequence ratio rarely escapes the clip window, so the IS-granularity
   axis is near-null on GSM8K/0.5B on its own — whether the *unbiased* full-sequence
   correction is worth its higher variance is the question the debiased cell answers.

2. **Debiased GRPO beats the paper baseline on the mean, but not decisively.**
   `debiased_grpo` posts the higher 3-seed pass@1 mean (**0.187 vs Standard GRPO
   `g0a` 0.147**, +0.040; and +0.017 over the `g0` ablation), but the 95% CIs
   (±0.080 / ±0.072, n=3) overlap, so the honest reading is "consistent with ≥ GRPO"
   — no decisive win from N=100 eval on a single 0.5B model. The real cost is
   **stability**, not seed-variance: standard GRPO's per-token clip gives 0% loss
   spikes, while the unclipped debiased estimator spikes `|loss|>10` on ~24% of steps
   (−37→+37), bounded by the clamp and recovered (Bias 1′). The robust, noise-free
   results live in the bias diagnostics: (a) per-token IS is exactly biased and
   full-seq is not (Bias 1); (b) the length gap stays flat at ≈ +8 tokens vs GRPO's
   ≈ +18 (Bias 2); (c) the conditional self-inclusion bias is removed (Bias 4).
   Unbiasedness is proven by exact enumeration in `tests/test_unbiasedness.py`, not
   by the pass@1 number.

## Open question the corrected cell is designed to test

Full-sequence IS is the **unbiased** off-policy correction for a terminal reward
(proven by exact enumeration in `tests/test_unbiasedness.py`), but its weight is
log-normal with variance growing exponentially in sequence length. The
`--log-w-clamp 5.0` guard bounds that variance at the cost of a small tail bias.
The empirical question is whether this **unbiased-but-higher-variance** estimator
(plus the independent baseline and the Dr. GRPO normalization fixes) beats the
**biased-but-stable** per-token-clipped paper GRPO at this scale. The baseline
`g0` vs `g0a` result above suggests the IS axis alone is near-null on GSM8K/0.5B;
the corrected run tested the full bundle. The answer: the unbiased bundle is
**consistent with ≥ paper GRPO on pass@1 (within seed noise)** while delivering
the noise-free structural wins (length bias removed, conditional self-inclusion
bias removed, log-normal IS growth confirmed). The cost is real within-run
instability: dropping PPO clip gives heavy-tailed IS weights, and the ±5 clamp is
the load-bearing NaN guard that caps them (e²⁴→e⁵ on the spike steps) so the run
recovers instead of diverging. Not a decisive headline win, but no regression and a
cleaner, unbiased estimator.

## Trust-region ablation (KL anchor) — 3 seeds

Debiased GRPO drops PPO clipping (which biases the IS weight) and keeps only the
GRPO KL→π_ref penalty (β=0.04) as its trust region. But that anchor is to the
*frozen base*, which π_behavior drifts away from over training, so it is a weak
*per-batch* trust region — consistent with Bias 1, where the unclipped estimator
spikes (`|loss|>10` on ~24% of steps, clamp firing on the tail). This ablation
asks: what controls that instability best? We vary **only** the KL anchor on the
debiased config (two independent coefficients, `--kl-ref-coef` / `--kl-behavior-coef`):

| Cell (ref β / beh β) | what it tests | make target |
|---|---|---|
| `refcur` (0.04 / 0) | current debiased (= GRPO's KL→ref) | `train-debiased` |
| `klbehavior` (0 / 0.04) | KL→π_behavior: an **unbiased soft trust region** on the IS ratio (a clip alternative that does not bias the weight) | `train-debiased-klbehavior` |
| `both` (0.04 / 0.04) | base anchor **and** IS trust region together | `train-debiased-both` |

(`nokl` and `highbeta`=0.2→ref were also run as short probes; `highbeta`
over-anchored and collapsed train reward, so it was dropped from the full runs.)

**Full runs, 3 seeds (42/43/44), 250 steps each** (`clamp_fire_frac` /
`logw_seq_max` added as training diagnostics):

| Cell | pass@1 s42/43/44 | mean ± 95% CI | spike % mean (range) | clamp-fire mean |
|---|---|---|---|---|
| `refcur` (.04/0) | 0.15 / 0.20 / 0.21 | **0.187** ± 0.080 | 23% (16–28) | n/a † |
| `klbehavior` (0/.04) | 0.16 / 0.17 / 0.19 | 0.173 ± **0.038** | 20% (12–32) | 0.170 |
| `both` (.04/.04) | 0.16 / 0.18 / 0.13 | 0.157 ± 0.063 | 24% (16–36) | 0.147 |

† `refcur` (the existing `debiased_grpo` run) predates the clamp-fire
instrumentation; **spike % and pass@1 are the cross-comparable metrics**.

**Findings — the single-seed signal did not survive.** An earlier single-seed
(s42) snapshot showed behavior-KL *halving* the spike rate (24%→12%) and nudging
pass@1 up (0.15→0.16). **Both effects were seed-42 noise.** Across 3 seeds:
- **No robust pass@1 difference.** Means are 0.187 / 0.173 / 0.157 with 95% CIs of
  ±0.04–0.08 that all overlap — statistically indistinguishable at n=3, N=100. If
  anything the *current* config (`refcur`, KL→ref) has the highest mean; the
  single-seed "klbehavior nudges pass@1 up" reversed.
- **No robust stability gain.** Spike rates are ~equal across cells (23 / 20 / 24%);
  `klbehavior`'s spike rate ranges 12–32% across seeds, so the s42 "12%" that drove
  the halving claim was the low tail of a wide distribution, not a real reduction.
- **One mild, weak signal:** `klbehavior` has the tightest pass@1 seed-variance
  (CI ±0.038 vs `refcur` ±0.080) — possibly more *consistent* across seeds — but at
  n=3 the variance estimate is itself noisy, so this is suggestive at best.

**Conclusion.** At this scale (0.5B, 250 steps, N=100) the **KL-anchor choice does
not robustly change pass@1 or training stability** — the apparent single-seed win
was an artifact. The current KL→π_ref config is retained as the headline; behavior-KL
is kept as a *documented, available* option (`--kl-behavior-coef`), not a promoted
default. This is exactly the failure mode a single-seed ablation invites, and the
reason the headline cells are reported at 3 seeds with CIs. The two-coefficient KL
interface, the per-step clamp-fire diagnostics, and `--resume-ckpt` are committed;
every cell reproduces via the make targets above.

## Caveats

- **Headline pass@1 win is within noise.** `debiased_grpo`'s 0.187 3-seed mean
  beats Standard GRPO `g0a`'s 0.147 (+0.040; +0.017 over the `g0` ablation's 0.170)
  but the 95% CIs overlap (±0.080 / ±0.072, n=3), on N=100 eval with a single 0.5B
  model. Read it as "consistent with ≥ GRPO," not a decisive win. The clamp and the k3-KL-from-behavior anchor are residual
  approximations, so strict unbiasedness is not claimed for the trained estimator
  (it is proven for the unclamped full-sequence correction in
  `tests/test_unbiasedness.py`).
- **Single 0.5B model, single benchmark.** GSM8K is a sparse-terminal-reward
  task; the failure modes the corrected cell fixes may matter more or less on
  dense-reward or longer-sequence settings.
- **μ=4 deviates from DeepSeekMath GRPO's μ=1.** At μ=1 the IS ratio is
  identically 1 and there is no within-loop drift for any IS estimator to
  correct. The same μ=4 is held across all cells for an apples-to-apples
  comparison.

## Bias diagnostics

The four bias sources are diagnosed individually in `notes/bias_diagnostics.md`
using inference-only runs on each (cell, training step) checkpoint. The
diagnostics on the `g0` / `g0a` baselines and on `debiased_grpo` are all filled
in (200 rollouts/cell at training steps 200/500/1000). Reproduce with
`make diagnose && make plot-diagnostics`.
