REPO := debiased-grpo
ENV  := debiased-grpo
PYTHON := conda run --no-capture-output -n $(ENV) python
GPU_QUEUE := $(HOME)/gpu-queue.sh
CONFIG := configs/gsm8k_qwen05b.yaml

.PHONY: env install test \
        train-g0 train-g0a train-debiased train-debiased-s43 \
        train-g0-s43 train-g0-s44 train-g0a-s43 train-g0a-s44 \
        train-debiased-s44 train-all \
        train-debiased-klbehavior train-debiased-both \
        train-debiased-klbehavior-s43 train-debiased-klbehavior-s44 \
        train-debiased-both-s43 train-debiased-both-s44 \
        diagnose baseline-bias-resample eval-by-difficulty bias1-isweight \
        per-token-bias plot-diagnostics clean

env:
	conda create -n $(ENV) python=3.11 -y

install:
	conda run --no-capture-output -n $(ENV) pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
	conda run --no-capture-output -n $(ENV) pip install transformers==4.44.0 trl==0.10.1 bitsandbytes==0.43.3 datasets accelerate==0.33.0 peft==0.12.0
	conda run --no-capture-output -n $(ENV) pip install lightning einops pyyaml pytest matplotlib
	conda run --no-capture-output -n $(ENV) pip install -e ".[dev]"

test:
	conda run --no-capture-output -n $(ENV) pytest tests/ -v --tb=short

# ---------------------------------------------------------------------------
# Training cells: G0 / G0a paper-GRPO baselines, and the Debiased GRPO run
# (independent baseline + full-sequence IS with log-weight clamp + fixed-constant
# length norm + no std-norm). See notes/experiments.md.
# ---------------------------------------------------------------------------

# G0: paper GRPO (all four problematic components on)
train-g0:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/g0_paper_grpo \
	    --baseline mean_with_self --is-weighting full_sequence \
	    --clipping ppo_classical \
	    --length-norm per_response --std-norm

# G0a: paper-faithful GRPO with per-token IS (DeepSeekMath Eq. 21).
# Identical to G0 except IS is per-token rather than full-sequence —
# the PPO clip then fires per-token, matching the canonical algorithm.
train-g0a:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/g0a_paper_grpo \
	    --baseline mean_with_self --is-weighting per_token \
	    --clipping ppo_classical \
	    --length-norm per_response --std-norm

# g0a (true paper GRPO: per-token clipped IS, DeepSeekMath Eq. 3) at seeds 43/44
# for the 3-seed headline baseline.
train-g0a-s43:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/g0a_paper_grpo_s43 --seed 43 \
	    --baseline mean_with_self --is-weighting per_token \
	    --clipping ppo_classical \
	    --length-norm per_response --std-norm

train-g0a-s44:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/g0a_paper_grpo_s44 --seed 44 \
	    --baseline mean_with_self --is-weighting per_token \
	    --clipping ppo_classical \
	    --length-norm per_response --std-norm

# Debiased GRPO: independent baseline + full-sequence IS (the unbiased
# correction for a terminal reward) with a log-weight clamp at 5.0 for
# numerical stability, no clipping, fixed-constant length norm, no std-norm.
train-debiased:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/debiased_grpo \
	    --baseline independent --is-weighting full_sequence --clipping none \
	    --length-norm fixed_constant --no-std-norm --log-w-clamp 5.0

# Debiased GRPO, second seed (43) for variance estimation.
train-debiased-s43:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/debiased_grpo_s43 --seed 43 \
	    --baseline independent --is-weighting full_sequence --clipping none \
	    --length-norm fixed_constant --no-std-norm --log-w-clamp 5.0

train-g0-s43:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/g0_paper_grpo_s43 --seed 43 \
	    --baseline mean_with_self --is-weighting full_sequence \
	    --clipping ppo_classical \
	    --length-norm per_response --std-norm

train-g0-s44:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/g0_paper_grpo_s44 --seed 44 \
	    --baseline mean_with_self --is-weighting full_sequence \
	    --clipping ppo_classical \
	    --length-norm per_response --std-norm

train-debiased-s44:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/debiased_grpo_s44 --seed 44 \
	    --baseline independent --is-weighting full_sequence --clipping none \
	    --length-norm fixed_constant --no-std-norm --log-w-clamp 5.0

# Trust-region ablation: debiased with the KL anchored to the per-batch behavior
# policy (an unbiased soft trust region on the IS ratio) instead of / in addition
# to the frozen ref. Baseline debiased uses --kl-ref-coef 0.04 (GRPO default).
train-debiased-klbehavior:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/debiased_klbehavior \
	    --baseline independent --is-weighting full_sequence --clipping none \
	    --length-norm fixed_constant --no-std-norm --log-w-clamp 5.0 \
	    --kl-ref-coef 0.0 --kl-behavior-coef 0.04

train-debiased-both:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/debiased_both \
	    --baseline independent --is-weighting full_sequence --clipping none \
	    --length-norm fixed_constant --no-std-norm --log-w-clamp 5.0 \
	    --kl-ref-coef 0.04 --kl-behavior-coef 0.04

# Seeds 43/44 for the trust-region ablation (3-seed pass@1 with CIs, matching the
# headline cells). Seed 42 is the base target above.
train-debiased-klbehavior-s43:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/debiased_klbehavior_s43 --seed 43 \
	    --baseline independent --is-weighting full_sequence --clipping none \
	    --length-norm fixed_constant --no-std-norm --log-w-clamp 5.0 \
	    --kl-ref-coef 0.0 --kl-behavior-coef 0.04

train-debiased-klbehavior-s44:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/debiased_klbehavior_s44 --seed 44 \
	    --baseline independent --is-weighting full_sequence --clipping none \
	    --length-norm fixed_constant --no-std-norm --log-w-clamp 5.0 \
	    --kl-ref-coef 0.0 --kl-behavior-coef 0.04

train-debiased-both-s43:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/debiased_both_s43 --seed 43 \
	    --baseline independent --is-weighting full_sequence --clipping none \
	    --length-norm fixed_constant --no-std-norm --log-w-clamp 5.0 \
	    --kl-ref-coef 0.04 --kl-behavior-coef 0.04

train-debiased-both-s44:
	$(GPU_QUEUE) $(PYTHON) scripts/train_grpo.py --config $(CONFIG) \
	    --output-dir outputs/debiased_both_s44 --seed 44 \
	    --baseline independent --is-weighting full_sequence --clipping none \
	    --length-norm fixed_constant --no-std-norm --log-w-clamp 5.0 \
	    --kl-ref-coef 0.04 --kl-behavior-coef 0.04

# Full 3-seed headline set (g0 + debiased across seeds 42/43/44, plus the g0a
# per-token-IS reference). Each cell is ~5 h on an RTX 3060 Ti.
train-all: train-g0 train-g0-s43 train-g0-s44 train-g0a \
           train-debiased train-debiased-s43 train-debiased-s44
	@echo "All headline GRPO cells (3 seeds) submitted to GPU queue."

# ---------------------------------------------------------------------------
# Bias diagnostics on trained checkpoints (no retraining; ~2 GPU hr).
# Produces per-rollout JSONL under outputs/diagnostics/ and figures under
# notes/figures/diagnostics/. See notes/bias_diagnostics.md.
# ---------------------------------------------------------------------------

diagnose:
	$(GPU_QUEUE) $(PYTHON) scripts/bias_diagnostics.py --cell all

baseline-bias-resample:
	$(GPU_QUEUE) $(PYTHON) scripts/baseline_bias_resample.py

eval-by-difficulty:
	$(GPU_QUEUE) $(PYTHON) scripts/eval_by_difficulty.py

# Bias-1 IS-weight table (reproducible from logged CSV + diagnostic JSONL; no GPU).
bias1-isweight:
	$(PYTHON) scripts/bias1_isweight.py

# Per-token-IS bias diagnostic (exact enumeration; no GPU). Shows GRPO's per-token
# surrogate is biased for a terminal reward and the bias grows with drift / length.
per-token-bias:
	$(PYTHON) scripts/per_token_bias.py

plot-diagnostics:
	$(PYTHON) scripts/plot_training_trajectories.py
	$(PYTHON) scripts/plot_bias_diagnostics.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
