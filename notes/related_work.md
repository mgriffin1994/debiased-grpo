# Related Work: Annotated Survey

This document surveys the key papers informing the design of debiased-grpo. The papers are grouped thematically: first the five papers directly studied during the algorithm derivation, then foundational RL and RLHF baselines, then closely related GRPO variants and the off-policy-evaluation prior work found during literature search.

---

## Papers Directly Studied in the Derivation

### 1. Your Group-Relative Advantage Is Biased

- **Authors:** Fengkai Yang, Zherui Chen, Xiaohan Wang, Xiaodong Lu, Jiajun Chai, Guojun Yin, Wei Lin, Shuai Ma, Fuzhen Zhuang, Deqing Wang, Yaodong Yang, Jianxin Li, Yikun Ban
- **Year:** 2026
- **arXiv:** [2601.08521](https://arxiv.org/abs/2601.08521)
- **Venue:** Preprint

**Contribution:** This paper provides a formal characterization of the bias in group-relative advantage estimation, showing that the leave-one-out (LOO) estimator used in standard GRPO systematically underestimates the advantage for hard prompts (where all sampled responses tend to be poor) and overestimates it for easy prompts (where all responses tend to succeed). The bias is structural and inherent to finite-sample within-group baselines. The authors propose History-Aware Adaptive Difficulty Weighting (HA-DW) as a corrective mechanism that adjusts advantage estimates using an evolving difficulty anchor computed from training history.

**Relevance to debiased-grpo:** This paper is the primary motivation for replacing the LOO baseline with an independent baseline drawn from separate π_ref rollouts. The bias analysis here directly supports the claim that the within-group mean is not a valid unbiased estimator of the true value function under realistic conditions.

---

### 2. Geometric-Mean Policy Optimization (GMPO)

- **Authors:** Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, Fang Wan, Furu Wei
- **Year:** 2025
- **arXiv:** [2507.20673](https://arxiv.org/abs/2507.20673)
- **Venue:** Preprint (July 2025)

**Contribution:** GMPO replaces GRPO's arithmetic mean aggregation of token-level rewards with a geometric mean. The geometric mean is inherently less sensitive to outlier tokens and keeps the effective importance sampling ratio within a narrower numerical range, improving training stability for long-sequence tasks. The paper provides both theoretical motivation (geometric mean as a log-space average) and empirical results showing reduced variance and more stable training curves compared to GRPO.

**Relevance to debiased-grpo:** GMPO highlights that the token aggregation strategy in GRPO is a primary source of instability and numerical problems — the same log-normal variance of the full-sequence importance ratio that debiased-grpo controls with a log-space clamp on the summed log-weight. GMPO's geometric-mean aggregation and debiased-grpo's clamp are two different numerical controls on the same exploding-ratio problem; both compute in log space (sum-then-exp) rather than forming the product directly.

---

### 3. Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)

- **Authors:** Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin
- **Year:** 2025
- **arXiv:** [2503.20783](https://arxiv.org/abs/2503.20783)
- **Venue:** COLM 2025

**Contribution:** This paper critically examines the R1-Zero training paradigm and identifies two optimization biases in standard GRPO: (1) a length bias introduced by per-response normalization, wherein shorter correct responses receive disproportionately large gradient updates while longer incorrect responses are penalized less severely, and (2) a standard deviation normalization bias. The authors introduce Dr. GRPO (GRPO Done Right), which replaces per-response token averaging with group-level scaling, eliminating both biases. Dr. GRPO achieves 43.3% on AIME 2024 with a 7B base model in only 27 hours of compute on 8× A100 GPUs.

**Relevance to debiased-grpo:** Dr. GRPO supplies two of debiased-grpo's four fixes directly — fixed-constant (group-level) length normalization in place of per-response 1/T_i averaging, and turning std-normalization off (both framed by Dr. GRPO as changed-objective distortions, not numerical issues). debiased-grpo adopts these unchanged and adds the unbiased IS correction (full-sequence IS for a terminal reward) and the unbiased independent baseline.

---

### 4. Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (RLOO)

- **Authors:** Arash Ahmadian et al.
- **Year:** 2024
- **arXiv:** [2402.14740](https://arxiv.org/abs/2402.14740)
- **Venue:** ICLR 2025 (published as conference paper)

**Contribution:** This paper argues that most of the complexity of PPO is unnecessary in the RLHF setting and that a simpler REINFORCE-style estimator with a leave-one-out baseline (RLOO) matches or exceeds PPO performance at substantially lower computational cost. The RLOO estimator uses k sampled responses per prompt and computes each response's advantage as the reward minus the average reward of the remaining k−1 responses, eliminating the need for a learned value network entirely.

**Relevance to debiased-grpo:** RLOO is the direct predecessor to GRPO and shares its LOO baseline structure. The theoretical analysis of when LOO is and is not unbiased, central to the debiased-grpo derivation, applies equally to RLOO. Note that LOO removes the self-inclusion bias but pays a (G/(G−1))² advantage-variance penalty (verified in `notes/bias_diagnostics.md` §4b); debiased-grpo instead uses an independent π_ref baseline, which removes the self-inclusion bias and — unlike LOO — does not couple the per-sample gradients, at a small M-tunable advantage-variance cost (σ²(1+1/M); see `notes/bias_diagnostics.md` §4b), and pairs it with the full-sequence IS off-policy correction and the Dr. GRPO normalization fixes.

---

### 5. DAPO: An Open-Source LLM Reinforcement Learning System at Scale

- **Authors:** Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, and many others (ByteDance Seed and Tsinghua AIR)
- **Year:** 2025
- **arXiv:** [2503.14476](https://arxiv.org/abs/2503.14476)
- **Venue:** Preprint (March 2025)

**Contribution:** DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) introduces four engineering-level improvements over GRPO that enable stable large-scale RL training: (1) Clip-Higher, which uses asymmetric clipping bounds to promote policy diversity and prevent entropy collapse; (2) Dynamic Sampling, which filters out prompts where all rollouts succeed or all fail, improving training signal quality; (3) a token-level policy gradient loss critical for long chain-of-thought scenarios; and (4) Overlong Reward Shaping to reduce noise from excessively long outputs. DAPO achieves 50 points on AIME 2024 with Qwen2.5-32B.

**Relevance to debiased-grpo:** DAPO's token-level policy gradient loss is the same token-level aggregation debiased-grpo adopts (summing token contributions and normalizing by a fixed constant rather than a per-response 1/T_i). DAPO's dynamic sampling addresses the same uninformative-batch problem that motivates the independent baseline: when all rollouts get the same reward, the within-group advantage is zero and the gradient carries no signal regardless of IS weights. DAPO keeps per-token clipping as its trust region, where debiased-grpo uses an unbiased full-sequence IS weight with a KL penalty instead.

---

## Foundational Papers

### 6. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

- **Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxian He, Yichen Zhu, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo (DeepSeek-AI)
- **Year:** 2024
- **arXiv:** [2402.03300](https://arxiv.org/abs/2402.03300)
- **Venue:** Preprint (February 2024)

**Contribution:** This paper introduces the DeepSeekMath series of models and, crucially, first proposes GRPO (Group Relative Policy Optimization) as a memory-efficient alternative to PPO for mathematical reasoning fine-tuning. GRPO eliminates the value network by computing advantages using within-group reward normalization: the advantage of a response is its reward minus the mean reward of responses in the same group, divided by the group standard deviation. The paper demonstrates strong results on competition-level math benchmarks.

**Relevance to debiased-grpo:** This is the origin paper for GRPO and the baseline algorithm that debiased-grpo directly improves upon (the `g0` / `g0a` cells reproduce it). All four bias sources debiased-grpo targets — correlated/self-inclusion baseline, the per-token IS surrogate that under-corrects for a terminal reward, per-response length normalization, and std normalization — are traceable to design choices made here. The per-token clipped IS surrogate reproduced by the `g0a` cell is in the GRPO objective, DeepSeekMath **Eq. 3** (Eq. 21 is the separate gradient-coefficient form in the unified-paradigm appendix).

---

### 7. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

- **Authors:** DeepSeek-AI (multiple authors)
- **Year:** 2025
- **arXiv:** [2501.12948](https://arxiv.org/abs/2501.12948)
- **Venue:** Nature (2025); preprint January 2025

**Contribution:** DeepSeek-R1 demonstrates that large-scale GRPO-based RL fine-tuning can elicit sophisticated chain-of-thought reasoning in LLMs without supervised fine-tuning on reasoning traces. The paper introduces two variants: R1-Zero (pure RL from base model) and R1 (warm-started from supervised data). The work shows that self-play RL with a verifiable reward signal can produce models that rival or exceed OpenAI's o1 on a range of reasoning benchmarks.

**Relevance to debiased-grpo:** DeepSeek-R1 is the primary demonstration that GRPO-style training works at scale and has strong real-world impact. It establishes the benchmark targets (AIME 2024, MATH) and the training recipe (verifiable rewards, long chain-of-thought) that debiased-grpo aims to improve upon with a theoretically sounder gradient estimator.

---

### 8. Proximal Policy Optimization Algorithms (PPO)

- **Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
- **Year:** 2017
- **arXiv:** [1707.06347](https://arxiv.org/abs/1707.06347)
- **Venue:** Preprint (July 2017)

**Contribution:** PPO introduces two practical policy gradient algorithms: PPO-Penalty, which augments the objective with a KL divergence penalty between the new and old policy, and PPO-Clip, which clips the importance sampling ratio to lie within [1−ε, 1+ε], preventing large policy updates. PPO became the dominant RL algorithm for RLHF due to its stability and ease of tuning relative to earlier trust region methods (TRPO).

**Relevance to debiased-grpo:** PPO's clipped IS ratio is the mechanism that GRPO inherited in spirit but discarded the value network from. debiased-grpo uses an unclipped KL penalty (rather than clipped IS) as the trust-region mechanism, specifically because hard clipping makes the IS weight a non-normalized ratio and biases the off-policy correction. The tradeoff is honest: dropping the clip restores the unbiased full-sequence IS estimator but exposes its higher (log-normal) variance, which the log-weight clamp then bounds. Whether unbiased-higher-variance beats PPO/GRPO's biased-but-stable clipping at scale is an open empirical question.

---

### 9. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model (DPO)

- **Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **Year:** 2023
- **arXiv:** [2305.18290](https://arxiv.org/abs/2305.18290)
- **Venue:** NeurIPS 2023

**Contribution:** DPO reformulates RLHF as a supervised contrastive learning problem by deriving a closed-form mapping from the reward model to the optimal policy under a KL-regularized objective. This eliminates the need for explicit reward modeling and RL optimization, instead fine-tuning the LLM directly on preference pairs using a binary cross-entropy-style loss. DPO achieved strong empirical results and became widely adopted due to its simplicity.

**Relevance to debiased-grpo:** DPO represents the major alternative paradigm to online RL for LLM alignment. Understanding DPO clarifies what online RL methods like GRPO and debiased-grpo offer that offline contrastive methods cannot: the ability to generate new rollouts during training, explore the output space, and optimize for verifiable scalar rewards without a preference dataset.

---

## Closely Related GRPO Variants (Found During Literature Search)

### 10. Group Sequence Policy Optimization (GSPO)

- **Authors:** Chujie Zheng, Shixuan Liu, and others (Qwen team)
- **Year:** 2025
- **arXiv:** [2507.18071](https://arxiv.org/abs/2507.18071)
- **Venue:** Preprint (July 2025)

**Contribution:** GSPO replaces GRPO's token-level importance sampling with a sequence-level formulation: the importance ratio is computed as the ratio of full-sequence log-likelihoods under the current and old policy, with length normalization applied to prevent the ratio from exploding for long sequences. GSPO demonstrates superior training stability and performance relative to GRPO, particularly for large Mixture-of-Experts models, and has been incorporated into the Qwen3 model family training pipeline.

**Relevance to debiased-grpo:** GSPO and debiased-grpo both move the importance ratio to the **sequence level** — for a terminal reward this is the correct, unbiased choice (the reward depends on the whole trajectory). The difference is the variance control: GSPO applies an explicit **length normalization** to the sequence-level ratio (which trades exact unbiasedness for a bounded ratio), while debiased-grpo keeps the unmodified full-sequence ratio and bounds only its heavy-drift tail with a log-space clamp. Both are answers to the same exploding-full-sequence-product problem; the comparison between length-normalized and clamped sequence-level IS is a natural axis for any empirical evaluation. (Per-token IS, by contrast, is the biased under-correction for a terminal reward — see `derivation.md` §4.)

---

### 11. Eligibility Traces for Off-Policy Policy Evaluation (step-wise importance sampling)

- **Authors:** Doina Precup, Richard S. Sutton, Satinder Singh
- **Year:** 2000
- **Venue:** ICML 2000

**Contribution:** Introduces step-wise importance sampling for off-policy
evaluation. For a **dense** reward structure r = Σ_t r_t (reward accrues at every
step), the unbiased off-policy correction weights each step's reward r_t by the
*prefix* importance ratio ∏_{k≤t} ρ_k — future actions cannot influence a reward
already received, so their ratios correctly drop. This step-wise decomposition
gives unbiased variance reduction relative to weighting every reward by the
full-trajectory ratio.

**Relevance to debiased-grpo:** This is **cited prior work, not this repo's
method.** The step-wise unbiased-variance-reduction result holds specifically for
**dense** rewards. debiased-grpo trains on a **terminal** (sequence-level) reward
(GSM8K), for which the prefix step-wise weight is *not* the unbiased correction —
the single reward depends on the whole trajectory, so the suffix ratios do not
drop, and the unbiased correction is the **full-sequence** importance ratio. See
`derivation.md` §4.2 for the dense-vs-terminal distinction and
`tests/test_unbiasedness.py` for the exact-enumeration proof that the per-token
ratio is biased here while full-sequence IS is exact.

---

## Summary Table

| Paper | arXiv | Year | Key Contribution | Relationship to debiased-grpo |
|-------|-------|------|-----------------|--------------------------|
| Your Group-Relative Advantage Is Biased | 2601.08521 | 2026 | Formal bias proof for LOO estimator | Motivation for the independent baseline |
| Geometric-Mean Policy Optimization | 2507.20673 | 2025 | Geometric mean aggregation for stable IS | Alternative numerical control on the same exploding-ratio problem the clamp bounds |
| Understanding R1-Zero (Dr. GRPO) | 2503.20783 | 2025 | Length/std-dev bias in GRPO normalization | Supplies the fixed-constant length norm + std-norm-off fixes, adopted directly |
| Back to Basics (RLOO) | 2402.14740 | 2024 | LOO baseline without value network | Predecessor; LOO removes self-incl. bias but couples per-sample gradients — we use indep π_ref (decoupled; M-tunable variance) |
| DAPO | 2503.14476 | 2025 | Asymmetric clipping, dynamic sampling, token-level loss | Token-level aggregation matches ours; keeps per-token clipping |
| DeepSeekMath (GRPO) | 2402.03300 | 2024 | Original GRPO algorithm (Eq. 3 objective; per-token clipped IS) | Baseline being improved (`g0a` paper-faithful; `g0` full-seq-IS ablation) |
| DeepSeek-R1 | 2501.12948 | 2025 | GRPO at scale for reasoning | Benchmark setter |
| PPO | 1707.06347 | 2017 | Clipped IS policy gradient | Foundational; we drop the clip (unbiased but higher variance) |
| DPO | 2305.18290 | 2023 | Offline preference optimization | Alternative paradigm |
| GSPO | 2507.18071 | 2025 | Sequence-level IS with length norm | Also sequence-level IS; length-norm vs our clamp as the variance control |
| Precup–Sutton–Singh | (ICML 2000) | 2000 | Step-wise IS for off-policy evaluation | Cited prior work; unbiased only for dense rewards, not our terminal reward |
