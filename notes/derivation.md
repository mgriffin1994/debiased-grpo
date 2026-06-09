# Mathematical Derivation of debiased-grpo

This document presents the full mathematical derivation of the debiased-grpo
algorithm: the identification of the four bias sources in standard GRPO, formal
definitions of the corrected estimators, and proof sketches of their
unbiasedness — together with an honest account of the two residual
approximations that keep the method from being *strictly* unbiased.

The thesis is **an (almost-)unbiased GRPO**. We remove four named bias sources
from standard GRPO and replace each with the unbiased choice. Two
approximations remain — a numerical clamp on the importance weight and a KL
penalty estimated from behavior-policy samples — and we state both plainly. We
do **not** claim strict unbiasedness, and we do **not** claim the method is the
only unbiased formulation; the central tradeoff (the unbiased IS correction is
higher-variance) is laid out in §4.

---

## 1. Standard GRPO: Setup and Gradient Estimator

Let π_θ be the policy being trained (the language model parametrized by θ), and
let π_ref be a fixed reference policy (typically the SFT-initialized policy).
Let x be a prompt drawn from a distribution D, and let y = (a_1, a_2, ..., a_T)
be a response of length T, where each a_t is a token drawn autoregressively from
the policy.

The policy gradient objective for RL fine-tuning is:

```
J(θ) = E_{x ~ D, y ~ π_θ(·|x)} [r(y)]
```

where r(y) is a scalar **terminal** reward for the full response (this matters —
see §4). The policy gradient theorem gives:

```
∇_θ J(θ) = E_{x, y ~ π_θ} [ r(y) · ∇_θ log π_θ(y|x) ]
           = E_{x, y ~ π_θ} [ r(y) · Σ_t ∇_θ log π_θ(a_t | x, a_{<t}) ]
```

The second equality uses the chain rule: log π_θ(y|x) = Σ_t log π_θ(a_t | x, a_{<t}).

In practice, the policy at sampling time (the **behavior policy** π_behavior,
which equals the policy state that produced the rollouts) differs from the
policy at gradient time (π_θ) because rollouts are collected once and then
reused for μ inner gradient steps. The standard approach uses importance
sampling to correct for this distributional shift:

```
∇_θ J(θ) = E_{x, y ~ π_behavior} [ (π_θ(y|x) / π_behavior(y|x)) · r(y) · Σ_t ∇_θ log π_θ(a_t | x, a_{<t}) ]
```

The full-sequence IS ratio is:

```
ρ(y) = π_θ(y|x) / π_behavior(y|x)
     = ∏_{t=1}^{T} [π_θ(a_t | x, a_{<t}) / π_behavior(a_t | x, a_{<t})]
     = ∏_{t=1}^T ρ_t
```

> **IS denominator vs KL anchor.** Two distinct policies appear in the loss and
> they play different roles. The IS *denominator* is the **behavior /
> sampling-time policy** π_behavior — the ratio is `log π_θ − log π_behavior`.
> The frozen reference π_ref is used **only as the KL anchor** in the
> KL(π_θ ‖ π_ref) regulariser (§7); it is **not** the IS denominator. This
> matches the code: `compute_loss(...)` takes the ratio against
> `behavior_log_probs` whenever it is supplied (the multi-inner-step trainer
> path) and falls back to `ref_log_probs` only on the single-update preset
> helpers and unit tests, where the two coincide at the sampling step. At inner
> step k=0 the policy equals the behavior policy, so ρ(y) ≡ 1; for k ≥ 1 the
> ratio reflects the within-loop drift. See `notes/implementation_notes.md`
> §11.

**Standard GRPO (DeepSeekMath paper formulation).** Collect a group of N
responses {y_1, ..., y_N} from π_behavior, compute rewards {r_1, ..., r_N}.
Paper-GRPO uses the group **mean** — *including* r_i — as baseline, then
standardises by the group std:

```
b_mean = (1/N) · Σ_j r_j          (note: includes r_i itself)
A_i = (r_i - b_mean) / std(r)
```

GRPO then clips the (per-token) IS ratio and minimises a per-response
token-averaged loss:

```
L_GRPO = -1/N · Σ_i (1/T_i) · Σ_t min(ρ_{i,t}·A_i, clip(ρ_{i,t}, 1±ε)·A_i) · ∇_θ log π_θ(a_{i,t})
```

This composes **four orthogonal failure modes**. It is important to keep them
distinct — some are formal biases, some are variance problems, and some are
changed objectives. Conflating them is where much of the literature confusion
comes from. The four, and the corrected choice debiased-grpo adopts:

| # | Failure mode | Category | Corrected choice |
|---|---|---|---|
| 1 | Self-inclusion / correlated baseline | genuine gradient bias | independent π_ref baseline (§2–§3, §5) |
| 2 | Biased IS surrogate (per-token under-correction for a terminal reward) | genuine gradient bias | full-sequence IS (§4) |
| 3 | Per-response length normalisation | changed objective | fixed-constant (group-level) length norm (§6b) |
| 4 | Std normalisation | changed objective | std-norm off (§6c) |

> **Implementation note.** `grpo_loss()` in `debiased_grpo/losses.py` is a thin
> preset over the strategy pipeline. The G0 paper cell uses
> `baseline=mean_with_self`, `is_weighting=full_sequence`,
> `clipping=ppo_classical`, `length_norm=per_response`, `std_norm=on`; the G0a
> paper-faithful cell swaps `is_weighting=per_token` so the PPO clip fires
> per-token (DeepSeekMath Eq. 3). Both are real baselines and unaffected by the
> corrected estimator.

---

## 2. The Statistical Grounding: Why Baselines Depend on State, Not Action

Before diagnosing GRPO's baseline issue, it's worth deriving carefully why
subtracting a baseline is unbiased. The policy gradient theorem gives

```
∇_θ J(θ) = E_{s, a ~ π_θ}[ Q(s, a) · ∇_θ log π_θ(a|s) ].
```

In Monte-Carlo form, Q(s, a) is replaced by a sample estimate G = r(y). To
reduce variance without changing the expectation, we subtract a baseline b to
get (G - b) · ∇log π_θ(a). The condition for unbiasedness is that **b is
independent of the action a being differentiated**, conditional on the state.

**The identity (in one line):**

```
E_{a ~ π_θ(·|s)}[ b · ∇_θ log π_θ(a|s) ]
  = b · Σ_a π_θ(a|s) · ∇_θ log π_θ(a|s)
  = b · Σ_a ∇_θ π_θ(a|s)                  [since π · ∇log π = ∇π]
  = b · ∇_θ Σ_a π_θ(a|s)                  [swap sum and gradient]
  = b · ∇_θ 1
  = 0.
```

The identity used `π · ∇log π = ∇π` to pull b outside the sum, then moved the
sum through the gradient because the distribution normalises to 1 independent of
θ. This breaks **if and only if b depends on a** — once b = b(a), it cannot be
pulled out of the inner sum.

**b can be a random variable**, not just a deterministic function b(s), as long
as it is *conditionally independent* of a given s. If b is a function of other
samples drawn from a distribution that is conditionally independent of a_i given
s (e.g. other samples from π_θ, or samples from any other policy π'), the
identity applies and the baseline is unbiased.

This is why:
- A deterministic b(s) (e.g. a learned value network output) is unbiased.
- An LOO baseline b = mean of {r_j : j ≠ i} drawn from the behavior policy is
  unbiased (other samples are conditionally independent of a_i).
- An independent baseline b = mean of rewards from separate π_ref rollouts is
  unbiased (different samples, different random draws).
- A self-inclusion baseline b = mean of {r_j : j} including r_i is **biased** (b
  is a function of a_i through r_i).

---

## 3. Failure Mode 1: Self-Inclusion Baseline Bias — *genuine gradient bias*

Paper-GRPO's baseline is b_mean = (1/N)·Σ_j r_j including r_i. Since r_i is a
function of a_i, b_mean depends on a_i through the (1/N)·r_i term. The identity
in §2 breaks exactly here. Expanding:

```
E[(r_i - b_mean) · ∇_θ log π_θ(a_i)]
  = (1 - 1/N) · [Q(s_i, a_i) - V(s_i)]
  - (1/N) · Σ_{j ≠ i} E[r_j · ∇_θ log π_θ(a_i)].
```

The first term is (1 - 1/N) times the correct advantage — a small multiplicative
shrinkage. The second term is zero when the r_j are drawn with actions
conditionally independent of a_i (which holds under i.i.d. group sampling). The
residual bias is O(1/N) — small but persistent, and it interacts with the
std-normalisation step discussed in §6c.

**Two fixes that both restore unbiasedness** (by making b conditionally
independent of a_i):
- **LOO baseline** (RLOO-style; Ahmadian et al. 2024, arXiv:2402.14740):
  b_LOO_i = (1/(N-1))·Σ_{j ≠ i} r_j drops r_i from the sum. The remaining
  samples a_j (j ≠ i) are i.i.d. draws with independent random bits —
  conditionally independent of a_i given s.
- **Independent π_ref baseline** (the Debiased GRPO choice): draw M fresh
  rollouts from π_ref *before* the policy step, use b = (1/M)·Σ_m r(τ_m). The
  rollouts τ_m are drawn from a different distribution with independent random
  bits — conditionally independent of a_i given s.

Both are unbiased. The substantive differences are not about unbiasedness:

**What is being estimated.** LOO estimates V^{π_behavior}(s) using samples from
the current rollout policy. Independent π_ref estimates V^{π_ref}(s) — which
drifts from V^{π_θ} as π_θ moves away from π_ref during training. Since V^{π_θ}
is the variance-minimising baseline for the on-policy gradient, LOO is typically
the **lower-variance** choice as training progresses. Independent π_ref holds a
fixed reference that becomes increasingly off-target.

**Why is "independent" π_ref not particularly independent in practice?** π_ref
and π_θ share initialisation and are KL-anchored — the two policies give similar
action distributions, especially at the early tokens of a response (before drift
accumulates). The *action distributions* are close, but the *realised samples*
are independent (different random draws from close-but-not-identical
distributions). The unbiasedness argument depends only on the sample-level
independence, which holds trivially; the variance argument depends on the
closeness of the distributions, which is fine early but degrades later.

**Why pick π_ref-based over LOO?** Both are unbiased. The deciding axis is
gradient coupling, not variance — and the variance story is *not* free (see §5b):
the independent baseline's advantage variance is `σ²(1+1/M)` (1.50σ² at M=2),
**higher** than both mean-with-self (0.75σ², but biased) and true LOO (1.33σ²).
What singles it out is that it is the only one that does **not** couple the
per-sample gradients through shared rewards — MWS and LOO both make `A_i` a
function of the *other* samples' rewards (LOO most strongly), whereas the
independent baseline subtracts an external constant whose only cross-sample effect
is a `+σ²/M` shared offset that vanishes as M grows. So we trade a small,
M-tunable variance cost for unbiasedness *and* decoupled gradients. A G-LOO
ablation would separate this from any remaining engineering preference and is a
reasonable addition to the experiment grid.

---

## 4. Failure Mode 2: The IS Surrogate — Full-Sequence IS Is the Unbiased Correction for a Terminal Reward

This is the load-bearing correction of the refactor, and the one most easily
gotten wrong. **For a terminal (sequence-level) reward, the unbiased off-policy
correction weights *every* token's score by the full-sequence importance ratio
— not by a per-token ratio, and not by a cumulative-prefix (step-wise) ratio.**

### 4.1 Why the full-sequence ratio is required

Start from the off-policy identity. For any function g of the whole trajectory,

```
E_{y ~ π_behavior}[ (π_θ(y)/π_behavior(y)) · g(y) ] = E_{y ~ π_θ}[ g(y) ].   (★)
```

Take g(y) = r(y) · Σ_t ∇_θ log π_θ(a_t). Substituting into (★) and writing
w(y) = π_θ(y)/π_behavior(y) = exp(Σ_t log ρ_t):

```
∇_θ J(θ) = E_{y ~ π_behavior}[ w(y) · r(y) · Σ_t ∇_θ log π_θ(a_t) ].
```

The IS weight is the **full-sequence** ratio w(y), one scalar per response,
shared across all tokens of that response. The transition probabilities
p(s_{t+1}|s_t, a_t) drop out of the trajectory-density ratio because the
environment is the same under both policies — what remains is the product of
policy ratios, the correct IS weight.

**Why a per-token (or cumulative-prefix) ratio is biased here.** A natural but
wrong idea is to weight token t's score by only the prefix product
∏_{k≤t} ρ_k (the cumulative-prefix / step-wise weight) while still
multiplying by the full-sequence reward r(y). That estimator drops the *suffix*
ratios ∏_{k>t} ρ_k. The terminal reward r(y) depends on the **whole**
trajectory — including the suffix tokens a_{t+1:T} — so token t's contribution
to the off-policy gradient must be corrected by the importance weight of the
*entire* sequence that produced r(y), not just its prefix. Dropping the suffix
ratios under-corrects and biases the estimate.

Concretely, the per-token estimator targets, for token t,

```
E_{y ~ π_behavior}[ ρ_t · r(y) · ∇_θ log π_θ(a_t) ]   (or the prefix-product variant),
```

which is **not** equal to E_{y ~ π_θ}[ r(y) · ∇_θ log π_θ(a_t) ] unless the
suffix ratios are identically 1 (i.e. π_θ = π_behavior beyond t, the on-policy
case). With μ > 1 inner steps the policy *has* drifted, so the bias is real.

### 4.2 When the step-wise prefix weight *is* unbiased: dense rewards only

Step-wise importance sampling (Precup, Sutton & Singh 2000) replaces the
full-trajectory ratio with a prefix weight ∏_{k≤t} ρ_k. It gives **unbiased
variance reduction**, but only for a **dense** reward structure r = Σ_t r_t,
where a reward r_t accrues at step t. There, by the causality / "a reward cannot
be influenced by future actions" argument, the unbiased estimator weights r_t by
the prefix ratio ∏_{k≤t} ρ_k — the suffix actions do not affect r_t, so their
ratios correctly drop. That is the classical step-wise-IS result.

**Our setting has no step reward.** GSM8K (and verifiable-answer reasoning
generally) gives a single terminal reward r(y) = R that depends on the whole
completion: r_t = 0 for t < T and r_T = R. Under a terminal reward the prefix
weight is *not* the unbiased correction — the suffix ratios do not drop, because
the single reward depends on the suffix. The unbiased correction collapses to the
full-sequence ratio. This is exactly why the earlier "the prefix weight is
unbiased at any sequence length" framing was wrong for this repo's training
setting, and why the prefix (step-wise) IS estimator has been removed from the
method.

### 4.3 Executable proof

`tests/test_unbiasedness.py` proves both halves of this section by **exact
enumeration** on a toy (horizon T=2, vocab {0,1} → 4 trajectories, a terminal
reward table, an independent constant baseline, a behavior policy genuinely
different from π_θ):

- `test_full_sequence_is_matches_on_policy_gradient_exactly` — the
  full-sequence-IS estimator (computed with the production `FullSequenceIS`
  strategy) **equals** the exact enumerated on-policy gradient, to 1e-6.
- `test_full_sequence_is_unbiased_for_several_baselines` — the independent
  baseline value (−2, 0, 0.5, 5) does not perturb that gradient.
- `test_per_token_is_is_biased_for_terminal_reward` — the per-token-IS surrogate
  (computed with `PerTokenIS`) **does not** match the exact gradient when
  π_behavior ≠ π_θ; it is biased.

These are deterministic enumerations, not Monte-Carlo estimates, so the proof
carries no statistical tolerance.

### 4.4 The tradeoff: unbiased but higher variance

Full-sequence IS is unbiased but **higher variance** — and this is exactly why
practitioners reach for per-token clipping instead. Under mild conditions (each
ρ_t > 0, finite log-variance), the CLT gives

```
log w(y) ~ N(μ_T, σ_T²)     with     σ_T² = T · Var[log ρ_t],
```

so w(y) is log-normal with variance growing **exponentially in T**. For T = 192
and even a small per-token drift the effective sample size of the batch
collapses toward one. Per-token clipping (PPO-Clip on per-token ratios, the G0a
cell) trades a small bias for a bounded, stable weight; full-sequence IS keeps
the estimator unbiased but must control the variance some other way.

debiased-grpo controls it with a **log-space clamp** on the summed log-weight
(§9): `log w ← min(log w, c)` with c = 5.0, bounding w to exp(5) ≈ 148. The
clamp is the **first of two residual approximations** (the KL estimator in §7 is
the second) and the reason the method is *almost*- rather than strictly
unbiased. Whether the unbiased-but-higher-variance full-sequence IS actually
*beats* a biased-but-stable per-token-clipped GRPO at this scale is an **open
empirical question**. Across 3 seeds (see `notes/experiments.md`) the corrected
cell reaches pass@1 0.187 vs Standard GRPO's (`g0a`, per-token, Eq. 3) 0.147 — a
+0.040 edge with overlapping 95% CIs, i.e. *consistent with ≥ GRPO*, not a decisive
win — while the `g0`/`g0a` baselines show the per-token-vs-full-sequence IS axis is
itself within seed noise (both clip-stable) on GSM8K/0.5B.

---

## 5. Independent Baseline: Unbiasedness Proof

The independent baseline replaces the within-group mean with a sample mean
computed from a separate set of M rollouts drawn from π_ref, collected
independently of the gradient rollouts:

```
b = (1/M) · Σ_{j=1}^{M} r(τ_j)   where   τ_1, ..., τ_M ~ π_ref(·|x)
```

The gradient estimate for rollout i (with the full-sequence IS weight w(y_i) of
§4) is:

```
g_i = w(y_i) · (r(y_i) - b) · Σ_t ∇_θ log π_θ(a_{i,t} | x, a_{i,<t})
```

The bias contributed by the baseline is the term E[w(y_i) · b · Σ_t ∇_θ log
π_θ(a_{i,t})]. Because the baseline rollouts {τ_j} are independent of the
gradient rollouts {y_i}, and w(y_i) is a function of y_i only:

```
E[ w(y_i) · b · Σ_t ∇_θ log π_θ(a_{i,t}) ]
  = E[b] · E[ w(y_i) · Σ_t ∇_θ log π_θ(a_{i,t}) ]      (independence pulls E[b] out)
  = E[b] · E_{y_i ~ π_θ}[ Σ_t ∇_θ log π_θ(a_{i,t}) ]    (full-sequence IS identity (★))
  = E[b] · ∇_θ E_{y_i ~ π_θ}[ 1 ]
  = 0,
```

where the IS identity (★) of §4 converts the behavior-policy expectation to an
on-policy one, and the score function has zero mean under π_θ. The constant E[b]
= E_{y ~ π_ref}[r(y)] does not depend on θ. So the independent baseline
contributes **exactly zero** bias to the gradient — no distributional
assumptions on the reward are required. (Its *variance* relative to the
within-group baselines, and a subtler cross-sample coupling effect, are analysed
in §5b.) This is the executable content of
`test_full_sequence_is_unbiased_for_several_baselines`.

---

## 5b. Baseline variance scaling and cross-sample gradient coupling

The three baselines differ not only in bias but in (i) the variance of the
advantage they produce and (ii) whether they make one rollout's advantage depend
on the *other* rollouts in the group. We work **conditional on a fixed prompt
x**: the G group rewards `r_1..r_G` are i.i.d. with within-prompt variance
`σ² = Var(r|x)` (this removes the between-prompt difficulty, which inflates a
naive marginal correlation and is not what we care about). The independent
baseline averages M reference draws `ρ_1..ρ_M ⟂ r`. Advantage `A_i = r_i − b_i`.

**Advantage variance.**
- *Mean-with-self* `b = (1/G)Σ_k r_k` (includes `r_i`): `A_i = r_i − (1/G)Σ_k r_k`,
  so `Var(A_i) = σ²(1 − 1/G) = σ²(G−1)/G`. (Biased — `b` contains `r_i`.)
- *Leave-one-out* `b_i = (1/(G−1))Σ_{k≠i} r_k`: `r_i ⟂ b_i`, so
  `Var(A_i) = Var(r_i) + Var(b_i) = σ² + σ²/(G−1) = σ²·G/(G−1)`.
- *Independent* `b = (1/M)Σ_m ρ_m ⟂ r`: `Var(A_i) = σ² + σ²/M = σ²(1 + 1/M)`.

Ratios: `Var(LOO)/Var(MWS) = (G/(G−1))²` and
`Var(indep)/Var(MWS) = (1+1/M)·G/(G−1)`. At G=4, M=2 these are **1.78** and
**2.0**. Note the independent baseline is **not** lower-variance than LOO at small
M (1.5σ² vs 1.33σ²); its advantage is unbiasedness *and* the coupling property
below, with the variance tunable toward σ² by raising M.

**Cross-sample covariance** `Cov(A_i, A_j | x)`, i≠j (using `Cov(r_i,r_j)=0`):
- *Independent*: `A_i, A_j` share only the external `b`, so
  `Cov = Var(b) = +σ²/M`. Crucially `A_i` is **not a function of the other
  gradient rollouts** — the coupling is only this shared external offset, which
  → 0 as M grows.
- *MWS*: `Cov(r_i − m, r_j − m) = −σ²/G` (m is the group mean).
- *LOO*: expanding `Cov(r_i − (1/(G−1))Σ_{k≠i}r_k, · )` gives
  `−G·σ²/(G−1)²` — the **largest-magnitude** coupling. Here `A_i` depends
  directly on the other samples' rewards.

At G=4: MWS `−0.25σ²`, LOO `−0.44σ²`, independent `+σ²/M = +0.5σ²` (M=2).

**Conclusion (the honest statement).** MWS and LOO both make `A_i` a function of
the *other* rollouts' rewards — LOO most strongly — so the per-sample policy-
gradient terms within a group are statistically coupled; MWS additionally
self-includes (`b` contains `r_i`), which is the source of its gradient bias
(§3). The independent baseline keeps `A_i` a function of only its own reward plus
an external shared baseline: it is unbiased and introduces no functional coupling
to the gradient rollouts (only the vanishing `+σ²/M` offset). All six closed
forms above are verified numerically in `tests/test_baseline_algebra.py`, and the
empirical estimates on the trained model (via multi-group resampling) are in
`notes/bias_diagnostics.md`.

---

## 6. Per-Token Control Variates (optional, not in the default cell)

For additional variance reduction one can introduce per-token baselines
b_t(x, a_{<t}) that depend on the context up to step t but not on the current
action a_t. The unbiasedness argument is the §2 identity applied per token:
since b_t does not depend on a_t, E[ w · b_t · ∇_θ log π_θ(a_t) ] = 0 still
holds. In practice computing per-token baselines requires a learned value
network (reintroducing the PPO critic) or extra rollouts. For the QLoRA / 8 GB
setting the default cell uses the single scalar independent baseline of §5,
accepting slightly higher variance in exchange for not needing a critic.

---

## 6b. Failure Mode 3: Length Normalisation — *changed objective*

GRPO's per-response token averaging is

```
L_GRPO = -1/N · Σ_i (1/T_i) · Σ_t [(...)_{i,t}]
```

The per-response factor (1/T_i) appears *inside* the outer expectation over
rollouts. Unlike a constant scalar that would commute with ∇_θ, T_i is
**response-dependent** and therefore reshapes the gradient direction: each token
in a correct short response contributes 1/T_short · advantage to the gradient,
while each token in a correct long response contributes only 1/T_long ·
advantage. Since T_short < T_long, short sequences are systematically preferred
over long sequences of equal quality. The estimator no longer optimises E[r(y)]
— it optimises a length-reweighted version of it.

This is a **changed objective**, not a variance or bias issue on the original
objective — no amount of data or rollouts fixes it.

**Fix: fixed-constant (group-level) aggregation.** Sum (not average) over tokens
within each rollout, then normalise by a fixed constant divisor (e.g. D = G ·
T_max) that is the same for every rollout, preserving the unbiased gradient
direction:

```
L_token = -(1 / D) · Σ_i Σ_t [(...)_{i,t}]
```

Now every token contributes equally and the gradient direction matches the true
policy gradient. This is the Dr. GRPO fix (Liu et al., arXiv:2503.20783); the
same token-level aggregation is adopted by DAPO (Yu et al. 2025,
arXiv:2503.14476). In the code this is `length_norm=fixed_constant`
(`FixedConstantAggregator`).

---

## 6c. Failure Mode 4: Std Normalisation — *changed objective, same category as length normalisation*

GRPO's advantage is A_i = (r_i - b_mean) / σ(r_{1..N}), where σ is the group
reward std for the prompt x. Split the advantage into two terms:

```
A_i = r_i / σ(x)  -  b_mean / σ(x)
```

The second term b_mean / σ(x) is a state-dependent constant (function of the
prompt x, not of a_i) — so by the §2 identity, it does not bias the gradient.
But the *first* term shows the reward itself has been scaled by a prompt-specific
factor 1/σ(x) that sits *inside* the expectation. What the estimator actually
targets is:

```
E_x [ E_{a ~ π_θ}[ r(a, x) / σ(x) ] ] = E_x [ (1/σ(x)) · J(θ|x) ]
```

— a *prompt-weighted* version of expected reward, where prompts with low reward
variance (σ → 0) get up-weighted and high-variance prompts get down-weighted.
The policy is trained to maximise "reward per unit of per-prompt reward
variance," not expected reward itself. The σ → 0 instability ("advantage blows
up when all rollouts agree") is a *symptom* of this objective distortion, not the
underlying issue.

This is **the same class of problem as length normalisation** — a data-dependent
factor inside the expectation changes what is optimised. Dr. GRPO's reason for
dropping std normalisation is exactly this, not the numerical-instability
framing: "σ(x) is a function of x, so dividing by it re-weights the objective
across prompts."

**Fix:** turn std-norm **off** — use the raw centred reward (r_i - b) with no std
scaling. Any overall reward scale is absorbed into the learning rate (a constant
that commutes with ∇_θ and doesn't affect the objective). In the code this is
`--no-std-norm` (`IdentityNormalizer`).

---

## 7. KL Penalty Formulation (the second residual approximation)

Standard GRPO uses clipped IS ratios (inherited from PPO-Clip) as a trust-region
mechanism. Hard clipping biases the estimator: the clipped weight
clip(ρ, 1−ε, 1+ε) is no longer a valid importance weight (it does not integrate
to 1 under the behavior policy), so E[clip(ρ)·g] ≠ E_{π_θ}[g] in general.

debiased-grpo instead uses a soft KL penalty added as a separate regularisation
term anchored to the **frozen reference** π_ref (the KL anchor, *not* the IS
denominator):

```
L_total = L_policy + β · KL(π_θ ‖ π_ref)
```

where L_policy is the full-sequence-IS policy loss of §4–§6c. The KL term is
estimated per token with **Schulman's k3 estimator**
(http://joschu.net/blog/kl-approx.html):

```
KL_t ≈ exp(log π_ref(a_t) − log π_θ(a_t)) − (log π_ref(a_t) − log π_θ(a_t)) − 1
```

which is unbiased and non-negative as an estimator of KL(π_θ ‖ π_ref) *when the
samples a_t are drawn from π_θ*. **This is the second residual approximation.**
In our pipeline the tokens a_t were sampled from the behavior policy π_behavior,
not from the current π_θ, so early in each inner loop (and early in training,
before π_θ has moved) the k3 estimate carries a **small bias** as an estimate of
KL(π_θ ‖ π_ref). The bias shrinks as π_θ ≈ π_behavior (k=0) and as the policy
settles. We state it honestly rather than claim the KL term is bias-free.

The k3 term is aggregated with the **same length aggregator** as the policy loss
(matching the GRPO paper objective) and scaled by `kl_ref_coef` (default 0.04, the
DeepSeekMath value).

**Why the KL penalty does not bias the *policy-gradient* identity.** The KL
gradient is a separate additive term; it does not enter the IS weight or the
advantage. The policy term remains the unbiased full-sequence-IS estimator of
§4–§5; the KL term is a deliberate trust-region regulariser whose *own* k3
estimate has the small sampling bias noted above.

---

## 8. Full Algorithm Summary

For a single prompt x, the complete debiased-grpo update:

**Step 1.** Sample N gradient rollouts {y_1, ..., y_N} from the behavior policy
π_behavior(·|x) and record their per-token behavior log-probs.

**Step 2.** Sample M baseline rollouts {τ_1, ..., τ_M} from π_ref(·|x)
independently; compute b = (1/M)·Σ_j r(τ_j).

**Step 3.** Compute rewards r_i = r(y_i) for the gradient rollouts.

**Step 4.** For each inner step k = 0..μ−1, run a with-grad policy forward to get
π_θ log-probs and form the per-token log-ratio against the **behavior** policy:

```
log ρ_{i,t} = log π_θ(a_{i,t} | x, a_{i,<t}) − log π_behavior(a_{i,t} | x, a_{i,<t})
```

**Step 5.** Form the **full-sequence** IS weight per rollout (one scalar,
broadcast across tokens), in log space with the clamp:

```
log w_i = clamp( Σ_t (log ρ_{i,t} · mask_{i,t}),  max = c=5.0 )
w_i = exp(log w_i)
```

**Step 6.** Policy loss (fixed-constant length norm D, no std-norm):

```
L_policy = -(1/D) · Σ_i Σ_t  w_i · (r_i − b) · log π_θ(a_{i,t} | x, a_{i,<t})
```

**Step 7.** KL penalty (k3, anchored to π_ref):

```
L_KL = β · (1/D) · Σ_i Σ_t [ exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1 ]_{i,t}
```

**Step 8.** Update:

```
L_total = L_policy + L_KL
θ ← θ − α · ∇_θ L_total
```

(then advance the inner loop / resample for the next outer step.)

---

## 9. PyTorch Implementation Sketch

The IS weight is computed as a sum in log space, then exponentiated once, with
the clamp applied to the summed log-weight before the exponential. This mirrors
`FullSequenceIS.compute` in `src/debiased_grpo/strategies.py`:

```python
import torch

def full_sequence_is_weight(log_probs, behavior_log_probs, mask, log_w_clamp=5.0):
    """Full-sequence IS weight w_i = exp(Σ_t log ρ_{i,t}), one scalar per rollout.

    log_probs:          (B, T) log π_θ(a_t | ctx), retains grad.
    behavior_log_probs: (B, T) log π_behavior(a_t | ctx)  ← the IS denominator.
    mask:               (B, T) bool, True at non-padding tokens.
    log_w_clamp:        upper cap on the summed log-weight before exp; the one
                        numerical-stability approximation. None ⇒ strictly unbiased.
    """
    # Per-token log IS ratio, masked at padding so it cannot leak into the sum.
    log_rho = (log_probs - behavior_log_probs) * mask.float()       # (B, T)

    # Sum over the whole sequence → one log-weight per rollout.
    log_w = log_rho.sum(dim=1, keepdim=True)                        # (B, 1)

    if log_w_clamp is not None:
        log_w = log_w.clamp(max=log_w_clamp)                        # numerical guard

    return torch.exp(log_w)                                         # (B, 1), broadcasts over T
```

Notes:

- **Exp-sum-log, not cumprod.** Summing log-ratios and exponentiating once is
  numerically stable; a direct `cumprod` of per-token ratios overflows/underflows
  for any non-trivial T.
- **One weight per rollout.** The shape is `(B, 1)` and broadcasts across the
  token dimension when multiplied by the advantage — this is the full-sequence
  ratio shared by every token of the response, as §4 requires.
- **The clamp.** Capping `log_w` at c bounds the otherwise log-normal weight to
  `exp(c)`. This is the residual approximation; `log_w_clamp=None` recovers the
  strictly-unbiased estimator (used by the exact-enumeration unbiasedness test,
  which sets no clamp).
- The KL term is computed separately with the k3 estimator against `ref_log_probs`
  and added after the policy loss (§7).

---

## 10. Reward-to-Go: Variance Reduction Under Sparse vs Shaped Rewards

The standard policy gradient with a per-token reward stream `r_{i,t}` and
per-rollout total `R_i = Σ_t r_{i,t}` admits two equivalent estimators in
expectation:

```
g_naive  = E[ Σ_t R_i · ∇_θ log π_θ(a_{i,t}) ]
g_RtG    = E[ Σ_t (Σ_{k>=t} r_{i,k}) · ∇_θ log π_θ(a_{i,t}) ]
```

The two are equal because for `k < t`, the past reward `r_k` does not depend on
the current action `a_t` once the context is fixed, and the score function has
zero mean:

```
E[ ∇_θ log π_θ(a_t) ] = 0     ⇒    E[ r_k · ∇_θ log π_θ(a_t) ] = 0    for k < t
```

`g_naive` includes those zero-mean past-reward terms; `g_RtG` drops them.
Therefore `Var[g_RtG] ≤ Var[g_naive]`, with equality iff every past-reward term
is deterministically zero.

**Sparse terminal reward (our case).** The binary-correctness reward (GSM8K) has
`r_T = R, r_{t<T} = 0`. The dropped past-reward terms are deterministically zero,
so `g_RtG = g_naive` sample-by-sample. The reward-to-go axis is a no-op in this
regime — which is also why the IS correction must be the full-sequence ratio
(§4): there is no step reward to license a step-wise prefix weight.

**Per-token shaping.** When the reward stream contains non-zero intermediate
rewards (PRM scores, KL-per-token, length penalty), the dropped terms are
non-zero and stochastic and reward-to-go provides genuine variance reduction.
Note that *if* such a dense step reward existed, the step-wise (prefix) IS weight
would become the unbiased variance-reduced correction (§4.2) — but this repo
trains on a terminal reward and ships only the full-sequence IS weighter.

**Composition with the full-sequence weight.** With a terminal reward the
reward-to-go advantage equals the broadcast advantage sample-by-sample, and the
shared full-sequence weight w_i multiplies it; the §4–§5 unbiasedness carries
through unchanged.

**Implementation.** The per-token reward tensor is reverse-cumulatively summed
along the token dimension; padding positions are masked to zero before the cumsum
to avoid leaking junk values into real positions. See
`notes/implementation_notes.md` §8.

---

## 11. EMA-of-Advantages Second Baseline: Unbiasedness and Variance

Let `b_i` be the primary baseline (§5) and let `c_t` denote the EMA-of-advantages
value used at training step `t`. The second-baseline-shifted advantage is:

```
A^shift_i = (R_i - b_i - c_t)
```

For unbiasedness of `E[A^shift · ∇_θ log π_θ]`, the term `c_t · ∇_θ log π_θ(a)`
must vanish in expectation, which holds whenever `c_t` is conditionally
independent of the current batch's actions given the state.

**Past-only update preserves independence.** The EMA at step `t` is

```
c_t = decay · c_{t-1} + (1 - decay) · m_{t-1}
```

where `m_{t-1}` is the mean advantage observed in batch `t-1`. The trainer
updates the EMA *after* the gradient step at time `t`, so `c_t` is a function of
batches `< t` only. The actions sampled at step `t` therefore have no influence
on `c_t`, and the baseline-subtraction identity applies:

```
E_{a ~ π_θ}[ c_t · ∇_θ log π_θ(a) ] = c_t · E_{a ~ π_θ}[ ∇_θ log π_θ(a) ] = 0.
```

Hence the second baseline subtraction is unbiased.

**Bias if updated before the gradient step.** Updating the EMA *before* using its
value at step `t` would make `c_t` depend on `m_t`, which depends on `a_t`. The
conditional independence breaks and a small bias enters. The implementation
enforces the past-only discipline by exposing `update()` as a separate method
the trainer calls after `optimizer.step()`.

**When does the second baseline reduce variance?** With an unbiased primary
baseline, the EMA tracks the per-batch mean advantage and subtracts it; variance
of the gradient is monotone-decreasing in `|A^shift|² = (A - c)²` for a fixed
score-function magnitude. The EMA subtraction reduces `|A^shift|²` whenever it
pulls the per-rollout advantage closer to zero. **Stratified extension:** the
same argument holds if the EMA is replaced by a per-stratum dictionary updated
past-only, provided the stratum key is a function of the prompt only.

---

## 12. Clipping vs the Log-Weight Clamp: Two Different Knobs

The pipeline exposes a per-token log-ratio clip and a PPO-classical clip via the
`Clipper` strategy, and — orthogonally — a `log_w_clamp` on the full-sequence IS
weight. They are distinct mechanisms:

| Knob | Where it acts | Bias character |
|---|---|---|
| `ppo_classical(ε)` | `compose_loss`, full-sequence min-form | biased — clipped IS not normalised; G0 reproduction only |
| per-token PPO clip (`per_token` IS + `ppo_classical`) | per token | biased per-token; the canonical DeepSeekMath Eq. 3 surrogate (G0a) |
| `log_w_clamp` (default 5.0) on `FullSequenceIS` | the summed log-weight, before exp | the residual approximation of the debiased cell — bounds the log-normal IS variance |

**Why the clamp rather than per-token clipping in the debiased cell?** Per-token
clipping is the standard *biased-but-stable* control on full-sequence IS
variance. The clamp is the *minimal* intervention that keeps the estimator
unbiased everywhere except in the heavy-drift tail (where `log w > c`): for
rollouts that have not drifted far it is exactly the unbiased full-sequence
ratio; only the extreme tail is bounded. Setting `log_w_clamp=None` recovers the
strictly-unbiased estimator at the cost of potential overflow on long
completions under heavy drift. The clamp should be treated as a last-resort
numerical guard, not a regulariser — if it fires often, raise the KL coefficient
or check that the policy is not drifting unrealistically. Whether the
unbiased-with-clamp choice beats biased-but-stable per-token clipping at scale is
the open empirical question of §4.4.

---

## 13. Decoupling Clipping from IS Weighting

The strategy refactor splits IS weighting and clipping into two independent axes:

```
is_weighting ∈ {full_sequence, per_token}
clipping     ∈ {none, log_ratio_token, ppo_classical}
```

The cells used in this release:

- `full_sequence + ppo_classical` — the paper PPO-style GRPO objective (G0).
- `per_token + ppo_classical` — the paper-faithful per-token surrogate
  (DeepSeekMath Eq. 3), where the clip fires per token (G0a).
- `full_sequence + none` (with `log_w_clamp=5.0`) — **the debiased cell**:
  unbiased full-sequence IS, numerically guarded by the clamp, paired with the
  independent baseline, fixed-constant length norm, and std-norm off.

The orchestrator (`compute_loss`) contains no conditional branching over which
axis is on — every variation is a different strategy instance, so the loss code,
sampler, and trainer are shared across all cells.
