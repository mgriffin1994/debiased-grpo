# Implementation Notes

Practical engineering notes for implementing debiased-grpo: tensor shapes,
numerical stability, memory management, and verification. These assume the
mathematical derivation in `derivation.md` and are intended as a direct coding
reference.

**The two residual approximations** (everything else in the policy term is the
unbiased choice):
- the **log-weight clamp** on the full-sequence IS weight (§3) — a numerical
  guard against the log-normal variance of `exp(Σ log ρ_t)`; and
- the **KL penalty estimated from behavior-policy samples** with Schulman's k3
  estimator (§11) — a small bias early in training.

---

## 1. Tensor Shapes for the Full-Sequence IS Weight

The IS weight in debiased-grpo is **one scalar per rollout** (the full-sequence
ratio), broadcast across the token dimension when multiplied by the advantage.
Let:

- `B` = number of rollouts in the batch (prompts × gradient rollouts per prompt)
- `M` = number of baseline rollouts per prompt (separate set)
- `T` = maximum sequence length (padded)
- `V` = vocabulary size

The key tensors and their shapes:

```
# Per-token log-probs of the sampled tokens under π_θ (retains grad), (B, T)
log_probs: Tensor[B, T]

# Per-token log-probs under the BEHAVIOR (sampling-time) policy — the IS
# denominator. (B, T)
behavior_log_probs: Tensor[B, T]

# Per-token log-probs under the frozen reference π_ref — the KL anchor only. (B, T)
ref_log_probs: Tensor[B, T]

# Attention mask: True at non-padding tokens, (B, T)
mask: Tensor[B, T]  # bool

# Per-token log IS ratio: log ρ_t = log π_θ(a_t) - log π_behavior(a_t), (B, T)
log_rho: Tensor[B, T]

# Full-sequence log-weight: log w_i = Σ_t log ρ_{i,t}, ONE per rollout, (B, 1)
log_w: Tensor[B, 1]

# Full-sequence IS weight (exponentiated), (B, 1) — broadcasts over T
w: Tensor[B, 1]

# Scalar reward per rollout, (B,)
rewards: Tensor[B]

# Scalar baseline per prompt, broadcast across the rollouts
baseline: Tensor[...]

# Advantage per rollout, (B, 1) for broadcasting
advantage: Tensor[B, 1]
```

Per-token log-probs are read out of the logits with the fused `cross_entropy`
path (cheaper than `log_softmax + gather`, which materialises a (B,T,V≈151k)
tensor that won't fit on 8 GB).

---

## 2. Handling Variable-Length Sequences

Responses have variable lengths; padding is standard, but incorrect masking will
corrupt the IS log-weight by summing padding-token ratios.

The rule is: **zero the per-token log IS ratio at padding positions before
summing.** Then the summed log-weight depends only on real tokens.

```python
# Zero padding positions in log_rho before the sum
log_rho_masked = log_rho * mask.float()           # (B, T), zero at padding

# Full-sequence log-weight: one scalar per rollout
log_w = log_rho_masked.sum(dim=1, keepdim=True)   # (B, 1)
```

Because the weight is a single sum over the whole sequence (not a per-position
cumulative quantity), there is no "cumulative value carried into padding
positions" subtlety — padding ratios are zeroed before the sum and contribute
nothing.

---

## 3. Log-Space Computation and the Log-Weight Clamp

Never compute the IS weight as a direct product of per-token ratios — for any
non-trivial `T` it overflows/underflows in float32. Sum the log-ratios and
exponentiate once:

```python
# WRONG (numerically unstable):
rho = torch.exp(log_probs - behavior_log_probs)   # per-token ratio
w   = rho.prod(dim=1, keepdim=True)               # product over T → overflow/underflow

# CORRECT (exp-sum-log):
log_rho = (log_probs - behavior_log_probs) * mask.float()
log_w   = log_rho.sum(dim=1, keepdim=True)        # (B, 1)
w       = torch.exp(log_w)                         # exponentiate once at the end
```

**The clamp (residual approximation #1).** The full-sequence weight
`w = exp(Σ log ρ_t)` is log-normal with variance growing like `T · Var[log ρ_t]`
— exponentially in sequence length. To bound it, cap the summed log-weight before
the exponential:

```python
LOG_W_CLAMP = 5.0          # bounds w to exp(5) ≈ 148
log_w = log_w.clamp(max=LOG_W_CLAMP)
w = torch.exp(log_w)
```

This matches `FullSequenceIS.compute` in `src/debiased_grpo/strategies.py`: it
sums the masked per-token log-ratios into a `(B, 1)` log-weight, clamps it at
`log_w_clamp` (default 5.0 in the debiased cell), and exponentiates once. The
clamp introduces a small bias **only** in the heavy-drift tail (rollouts whose
summed log-ratio exceeds `c`); for rollouts that have not drifted far it is
exactly the unbiased full-sequence ratio. Setting `log_w_clamp=None` recovers the
strictly-unbiased estimator (this is what the exact-enumeration unbiasedness test
uses). The clamp is a last-resort numerical guard, not a regulariser — if it fires
often, raise the KL coefficient or check for unrealistic policy drift.

---

## 4. Independent Baseline Sampling

The baseline rollouts must be generated from π_ref (the frozen reference), not
from the policy that produced the gradient rollouts, and **separately** from the
gradient rollouts — reusing the gradient rollouts would reintroduce the
correlated-baseline bias.

```python
# 1. Generate N gradient rollouts (their behavior log-probs are the IS denominator)
gradient_rollouts = model.generate(prompt_ids, num_return_sequences=N, do_sample=True, ...)

# 2. Generate M baseline rollouts from π_ref (frozen reference), forward only
with torch.no_grad():
    baseline_rollouts = ref_model.generate(prompt_ids, num_return_sequences=M, do_sample=True, ...)
    baseline_rewards = reward_fn(baseline_rollouts)             # (B*M,)
    baseline = baseline_rewards.reshape(B, M).mean(dim=1)       # (B,)

# 3. Rewards for the gradient rollouts
gradient_rewards = reward_fn(gradient_rollouts)                 # (B*N,)
```

The reference model is virtualized rather than held as a second copy: in QLoRA
the policy and reference share the frozen 4-bit base weights, so a reference
forward is a policy forward with the LoRA adapters disabled:

```python
with model.disable_adapter():
    ref_logprobs = compute_logprobs(model, baseline_rollouts)
```

This saves the memory cost of a second model entirely.

---

## 5. QLoRA Setup on an 8 GB GPU

The shipped and tested config is Qwen2-0.5B; the settings below are sized for it
on an RTX 3060 Ti (8 GB). Larger models (≈1.5B) are plausible on the same card but
untested here.

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
```

**Memory budget (Qwen2-0.5B at 4-bit):** weights ~250 MB; LoRA adapters ~20 MB;
optimizer states (AdamW on LoRA params only) ~40 MB; forward activations (N=4,
T=192) chunked over the rollout dimension; logprob tensors and IS weights small
(the weight is `(B, 1)`). Peak ~6.5–7 GB during the policy forward+backward of a
single chunk.

**Recommended hyperparameters:** LR 1e-4 to 5e-4; gradient accumulation 4
(effective batch = 4 prompts); N=4 gradient rollouts; M=2 baseline rollouts;
max_new_tokens=192; KL-to-ref coefficient `--kl-ref-coef` default 0.04 (the
DeepSeekMath value); optional KL-to-behavior trust region via `--kl-behavior-coef`.

---

## 6. Gradient Accumulation Strategy

Effective batch size is the number of prompts per gradient step, not the number
of rollouts. For `accum_steps = 4` prompts with `N = 4` rollouts each:

1. For each accumulation step (one prompt), generate rollouts, compute the loss,
   call `loss.backward()`, do not `optimizer.step()`.
2. After `accum_steps` prompts, `optimizer.step()` and `optimizer.zero_grad()`.
3. Normalize the loss by `accum_steps` before `backward()`.

The baseline `b` is computed per prompt, not per batch — each prompt has its own
difficulty; do not average baselines across prompts in an accumulation batch.

---

## 7. Verifying Unbiasedness

**The executable proof** lives in `tests/test_unbiasedness.py`. On a tiny exact
toy (T=2, vocab {0,1} → 4 trajectories, terminal reward, independent constant
baseline, a behavior policy genuinely different from π_θ) it enumerates all
trajectories and checks, with no Monte-Carlo tolerance:

- the **full-sequence-IS** estimator (using the production `FullSequenceIS`)
  equals the exact on-policy gradient to 1e-6;
- the independent baseline value does not perturb that gradient;
- the **per-token-IS** surrogate (`PerTokenIS`) does **not** match the exact
  gradient — it is biased for a terminal reward.

**Training-time diagnostics to log** (sanity, not proofs):

```python
metrics = {
    "grad_norm": gradient_norm,
    "kl_to_ref": kl_penalty_value,
    "ess": (w.sum()**2) / (w**2).sum(),                 # ESS of the IS weights
    "baseline_reward_mean": baseline.mean(),
    "gradient_reward_mean": rewards.mean(),
    "log_ratio_std": log_rho[mask].std(),               # drift diagnostic
    "baseline_gradient_correlation": pearsonr(baseline, rewards),
}
```

ESS near 1 means the full-sequence IS weight has collapsed onto one sample (the
log-normal-variance regime the clamp is meant to bound); ESS near B means healthy
IS estimation. For the independent baseline, `baseline_gradient_correlation`
should be near zero — a significantly non-zero value indicates the rollout
separation logic is broken.

---

## 8. Reward-to-Go: Per-Token Rewards and Reverse Cumsum

The reward-to-go assigner replaces the broadcast scalar advantage with a per-token
sum `Σ_{k>=t} r_{i,k}` via a reverse cumulative sum:

```python
def reverse_cumsum(x, dim=1):
    return torch.flip(torch.cumsum(torch.flip(x, dims=[dim]), dim=dim), dims=[dim])
```

This `flip → cumsum → flip` is autograd-safe (no in-place ops). **Mask the
per-token reward tensor BEFORE the reverse cumsum**, or junk padding values leak
into real positions:

```python
masked_rewards = token_rewards * mask.float()        # zero at padding
rtg = reverse_cumsum(masked_rewards, dim=1)
advantage = rtg - baseline.unsqueeze(1)
```

**Sparse terminal representation gotcha.** When the caller passes only a scalar
`rewards` tensor (no per-token shaping), the assigner builds `token_rewards` via
`make_token_rewards`, placing the scalar at the rollout's last non-padding
position. Then `Σ_{k>=t} r_{i,k}` equals the scalar reward for every
`t <= last_real_t`, so the reward-to-go advantage equals the broadcast advantage
sample-by-sample. The reward-to-go axis is meaningful only with a non-trivial
`token_rewards` tensor (PRM output, per-token KL, length penalty). This is also
why the IS correction here is the full-sequence ratio: with a single terminal
reward there is no step reward to license a step-wise prefix weight (see
`derivation.md` §4 and §10).

---

## 9. EMA-of-Advantages: Past-Only Update Timing

The EMA second baseline subtracts a scalar EMA of the per-batch mean advantage.
To preserve unbiasedness, the EMA value used at step `t` must be a function of
batches `< t` only — i.e. **updated after the gradient step**, not before. The
`EMABaseline` helper enforces this with a separate `update()` method.

```python
loss = compute_loss(..., baseline_shifter=EMAShift(ema))
loss.backward()
optimizer.step()

if cfg.ema_baseline:
    primary_baseline = components["baseline"].compute(rewards, baseline_rewards)
    batch_mean_adv = (rewards - primary_baseline).mean().item()
    ema.update(batch_mean_adv)        # past-only: feeds into NEXT step's shift
```

Reversing the order — `ema.update()` before `compute_loss` — makes the shift a
function of the current batch's actions, breaking conditional independence and
reintroducing a small bias.

**Warm-start gotcha.** The first `update()` sets `value = x` directly rather than
blending with the `init=0.0` default, avoiding a toward-zero bias for the first
~`1/(1-decay)` steps. The first gradient step uses `ema_shift = 0.0`.

**Checkpoint round-trip.** `EMABaseline.state_dict()` returns decay, value, and
the initialised flag; include the EMA state in checkpoint saves so warm restarts
do not silently reset the second baseline to 0.

**Diagnostic.** With an unbiased primary baseline, the EMA value should hover near
0; substantial drift indicates a biased primary baseline or non-stationary
expected reward as the policy improves. Log `ema.value` alongside `mean_reward`
and `mean_response_length`.

---

## 10. Clipping Modes and the Log-Weight Clamp: Where Each Knob Lives

The pipeline exposes clipping via the `Clipper` strategy and — orthogonally — the
`log_w_clamp` on the full-sequence IS weight. They are different mechanisms:

| Mode / knob | Stage | Effect on bias | When to use |
|------|-------|----------------|-------------|
| `none` (Clipper) | nowhere | unbiased | the debiased cell's clipper (variance controlled by the clamp instead) |
| `log_ratio_token(c)` | `pre_product` (per-token) | biased — per-token truncation | numerical stability without committing to PPO semantics |
| `ppo_classical(eps)` | `compose_loss` (full-sequence min-form) | biased — clipped IS not normalised | `g0` reproduction, paired with `FullSequenceIS` |
| per-token IS + `ppo_classical` | per token | biased per token | `g0a` — the canonical DeepSeekMath Eq. 3 surrogate |
| `log_w_clamp` (default 5.0) | the summed log-weight, before exp | tail-only bias | the debiased cell's numerical guard on full-sequence IS |

**The debiased cell uses `clipping=none` plus `log_w_clamp=5.0`.** The clamp,
not per-token clipping, is the variance control: it leaves the estimator unbiased
everywhere except in the heavy-drift tail (where `log w > c`). `log_w_clamp=None`
recovers the strictly-unbiased estimator at the cost of potential overflow on long
completions.

---

## 11. IS denominator vs KL anchor (the inner loop)

Two distinct policies appear in the loss with different roles:

- **IS denominator — the sampling-time (behavior) policy.** The IS ratio is
  `log_probs − behavior_log_probs`, where `behavior_log_probs` are the log-probs
  under the policy state that *produced the rollouts*. `GroupRolloutSampler.sample()`
  returns these (optionally without grad), and `compute_log_probs(prompt_ids,
  completions_padded)` runs a fresh with-grad policy forward at each inner step.
  `compute_loss(...)` uses `behavior_log_probs` as the denominator whenever it is
  provided.
- **KL anchor — the frozen base π_ref.** The frozen base enters **only** through
  the separate KL(π_θ ‖ π_ref) regulariser, never the IS ratio. The term uses
  **Schulman's k3 estimator** `exp(log_ref − log_θ) − (log_ref − log_θ) − 1`,
  aggregated with the same length aggregator as the policy loss and scaled by
  `kl_ref_coef` (`LossConfig.kl_ref_coef`, default 0.04; override with
  `--kl-ref-coef`). A second coefficient `kl_behavior_coef` (`--kl-behavior-coef`,
  default 0) anchors instead/also to π_behavior — an unbiased soft trust region on
  the IS ratio (see the trust-region ablation in `experiments.md`).

> **Residual approximation #2.** The k3 estimator is unbiased and non-negative as
> an estimator of KL(π_θ ‖ π_ref) **when the samples are drawn from π_θ**. In this
> pipeline the tokens were sampled from the behavior policy, so early in each
> inner loop (and early in training, before π_θ ≈ π_behavior holds) the k3 value
> carries a **small bias**. It shrinks as π_θ approaches π_behavior at k=0 and as
> training settles. We log it rather than claim the KL term is bias-free.

The trainer runs μ (`--inner-loop-mu`, default 4) inner gradient steps per rollout
batch:

```python
behavior_log_probs, ref_log_probs = sampler.sample(prompt, with_grad=False)
for k in range(μ):
    log_probs = sampler.compute_log_probs(prompt, completions_padded)  # with grad
    loss = compute_loss(log_probs, ref_log_probs, ...,
                        behavior_log_probs=behavior_log_probs,
                        kl_ref_coef=β_ref, kl_behavior_coef=β_beh)
    backward(); opt.step()
```

At inner step `k=0`, `log_probs == behavior_log_probs`, so the IS ratio is
identically 1 (single-step REINFORCE + KL). For `k ≥ 1` the ratio reflects the
within-loop drift from the sampling-time policy to the current policy — the regime
in which the full-sequence IS correction (and the clamp) matter. The default μ=4
deviates from DeepSeekMath GRPO's μ=1 (where there is no within-loop drift and no
IS to speak of) specifically to exercise the IS estimator; the same μ is held
across all cells for an apples-to-apples comparison.
