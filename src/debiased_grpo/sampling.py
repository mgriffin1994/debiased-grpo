"""
Rollout samplers for GRPO and Debiased GRPO training.

Two classes are provided:

    GroupRolloutSampler   — samples G completions from the current policy π_θ
                            and simultaneously records reference log-probs from
                            the frozen π_ref.  Used for gradient rollouts.

    IndependentBaselineSampler — samples M completions from π_ref only, applies
                            a reward function, and returns the reward scalars.
                            All computation runs inside torch.no_grad() because
                            no gradients flow through baseline rollouts.

Both classes deliberately keep their generate() logic minimal so that callers
(trainer.py and the training scripts) control batching, device placement, and
any memory-saving tricks such as gradient checkpointing.
"""

from __future__ import annotations

import contextlib

import torch
from torch import Tensor
from typing import Callable, List, Tuple

# We import these lazily inside methods so that the module itself can be
# imported in CPU-only test environments where the full stack may not be
# installed.
try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
except ImportError:  # pragma: no cover
    PreTrainedModel = object  # type: ignore[assignment,misc]
    PreTrainedTokenizerBase = object  # type: ignore[assignment,misc]


def _gather_log_probs(logits: Tensor, input_ids: Tensor) -> Tensor:
    """Extract per-token log probabilities for the tokens that were actually sampled.

    Args:
        logits:   Raw model logits. Shape (batch, seq_len, vocab_size).
        input_ids: Sampled token ids. Shape (batch, seq_len).

    Returns:
        Per-token log probabilities. Shape (batch, seq_len).
    """
    # Use F.cross_entropy's fused kernel: log_prob = -CE(logits, ids).
    # Avoids materialising any (B, T, V) softmax / exp intermediate that
    # log_softmax or logsumexp would otherwise allocate — critical for
    # Qwen2's V≈151k on 8 GB GPUs.
    B, T, V = logits.shape
    flat_logits = logits.reshape(B * T, V)
    flat_ids = input_ids.reshape(B * T)
    nll = torch.nn.functional.cross_entropy(flat_logits, flat_ids, reduction="none")
    return (-nll).reshape(B, T)


def _pad_sequences(sequences: List[Tensor], pad_id: int) -> Tuple[Tensor, Tensor]:
    """Right-pad a list of variable-length 1-D token tensors.

    Args:
        sequences: List of 1-D tensors of varying length.
        pad_id:    Token id to use for padding.

    Returns:
        padded: (N, max_len) int64 tensor.
        mask:   (N, max_len) bool tensor; True = real token.
    """
    max_len = max(s.shape[0] for s in sequences)
    N = len(sequences)
    device = sequences[0].device if sequences else torch.device("cpu")
    padded = torch.full((N, max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((N, max_len), dtype=torch.bool, device=device)
    for i, seq in enumerate(sequences):
        L = seq.shape[0]
        padded[i, :L] = seq
        mask[i, :L] = True
    return padded, mask


class GroupRolloutSampler:
    """Sample a group of G completions from the current policy and record log-probs.

    For each prompt, this class:
        1. Generates G independent completions from ``model`` (the LoRA-adapted
           policy π_θ).
        2. Runs a second forward pass with ``torch.no_grad()`` through
           ``ref_model`` (or ``model`` with adapters disabled) to obtain
           reference log-probabilities π_ref(a_t | ctx_t) for the same tokens.

    The caller is responsible for:
        - Setting ``model.train()`` / ``model.eval()`` as appropriate.
        - Moving tensors to the correct device before calling ``sample()``.
        - Gradient accumulation and the optimiser step.

    Args:
        model:          The policy model (π_θ). Usually LoRA-adapted.
        tokenizer:      Tokenizer for the model.
        ref_model:      Frozen reference model (π_ref). If None, ``model``
                        is used with ``torch.no_grad()`` (only correct for
                        the first step when LoRA weights are still zero).
        group_size:     Number of completions G to generate per prompt.
        max_new_tokens: Maximum number of tokens to generate per completion.
        temperature:    Sampling temperature. Values < 1 sharpen the distribution.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        ref_model: PreTrainedModel | None = None,
        group_size: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model if ref_model is not None else model
        self._ref_shared = self.ref_model is model
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Use the tokenizer's pad token; fall back to eos if not set.
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    @torch.no_grad()
    def _generate_completions(self, prompt_ids: Tensor) -> List[Tensor]:
        """Generate group_size completions for a single prompt (no grad)."""
        expanded = prompt_ids.unsqueeze(0).expand(self.group_size, -1)
        attn_mask = torch.ones_like(expanded)

        # Disable gradient checkpointing during generation: the GC-wrapped
        # forward path interacts badly with bnb 4-bit + bf16 + KV cache and
        # produces nan logits after deep generation, which crashes
        # torch.multinomial. Generation needs no backward, so GC is purely
        # harmful here.
        was_training = self.model.training
        gc_enabled = getattr(self.model, "is_gradient_checkpointing", False)
        if gc_enabled:
            self.model.gradient_checkpointing_disable()
        self.model.eval()
        try:
            output_ids = self.model.generate(
                expanded,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                pad_token_id=self.pad_id,
                renormalize_logits=True,
            )
        finally:
            if was_training:
                self.model.train()
            if gc_enabled:
                self.model.gradient_checkpointing_enable()

        prompt_len = prompt_ids.shape[0]
        # Strip the prompt prefix; return only the generated completion tokens.
        return [output_ids[i, prompt_len:] for i in range(self.group_size)]

    def _compute_log_probs(
        self,
        model: PreTrainedModel,
        prompt_ids: Tensor,
        completions: Tensor,
        no_grad: bool = False,
        chunk_size: int = 1,
    ) -> Tensor:
        """Forward pass to get per-token log-probs for given completions.

        Args:
            model:       Model to use for the forward pass.
            prompt_ids:  Prompt token ids. Shape (L_prompt,).
            completions: Completion token ids. Shape (G, L_completion).
            no_grad:     Whether to wrap in torch.no_grad().
            chunk_size:  Number of rollouts to forward at once. Lowering this
                         shrinks the (chunk, T, V) logits allocation, which
                         dominates peak memory on small GPUs.

        Returns:
            Per-token log-probs for the completion portion. Shape (G, L_completion).
        """
        G, L_comp = completions.shape
        L_prompt = prompt_ids.shape[0]

        prompt_expanded = prompt_ids.unsqueeze(0).expand(G, -1)
        full_ids = torch.cat([prompt_expanded, completions], dim=1)

        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        out_chunks: List[Tensor] = []
        with ctx:
            for start in range(0, G, chunk_size):
                end = min(start + chunk_size, G)
                chunk_ids = full_ids[start:end]
                logits = model(chunk_ids).logits
                completion_logits = logits[:, L_prompt - 1 : L_prompt + L_comp - 1, :]
                lp = _gather_log_probs(completion_logits, completions[start:end])
                out_chunks.append(lp)
                del logits, completion_logits
        return torch.cat(out_chunks, dim=0)

    def sample(
        self, prompt_ids: Tensor, *, with_grad_log_probs: bool = True,
    ) -> Tuple[List[Tensor], Tensor, Tensor, Tensor, Tensor]:
        """Sample G completions and return both behavior- and reference-policy log-probs.

        Args:
            prompt_ids: Tokenized prompt. Shape (L_prompt,). On the model device.
            with_grad_log_probs: When True (default), the returned ``log_probs``
                tensor carries grad-fn through the policy (suitable for
                single-step training and for diagnostics that prefer a single
                forward). When False, all log-probs are computed under
                ``torch.no_grad()`` — used by the inner-loop trainer which
                snapshots behavior-policy log-probs once per resample and
                re-runs the with-grad forward via ``compute_log_probs`` at
                each inner step.

        Returns:
            completions:        List of G token-id tensors (variable length, pre-padding).
            completions_padded: (G, T) right-padded token ids. Returned so callers
                                can re-run the forward under an updated policy
                                via ``compute_log_probs``.
            log_probs:          Per-token log-probs from the **sampling-time
                                policy**. With grad iff ``with_grad_log_probs``.
                                Shape (G, T). This is the IS denominator
                                (behavior policy) for the inner-loop off-policy
                                correction (PPO / full-sequence IS).
            ref_log_probs:      Per-token log-probs from the frozen base π_ref.
                                Shape (G, T). No grad. Used as the KL anchor.
            mask:               Boolean mask; True for real (non-padding) tokens.
                                Shape (G, T).

        Where T is the length of the longest completion in the group.

        The mask from _pad_sequences is returned (not discarded) so padding is
        excluded downstream. The returned ``log_probs`` are explicitly the
        sampling-time (behavior) policy. For inner-loop training, this is the
        IS *denominator*; the trainer re-runs ``compute_log_probs`` at each
        inner step to obtain the IS *numerator* under the updated policy.
        """
        # Step 1: generate completions (no grad needed for sampling itself).
        raw_completions = self._generate_completions(prompt_ids)

        # Step 2: pad to uniform length within the group.
        completions_padded, mask = _pad_sequences(raw_completions, self.pad_id)

        # Free generate()'s KV cache before allocating activations for the
        # gradient forward pass — the cache is no longer needed and otherwise
        # sits in the allocator pool, leaving no room for backward activations
        # on memory-tight GPUs.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 3: compute log-probs under the sampling-time policy. Optionally
        # with grad (single-step training paths) or no-grad (inner-loop path,
        # where these are used only as the IS denominator anchor).
        log_probs = self._compute_log_probs(
            self.model, prompt_ids, completions_padded,
            no_grad=not with_grad_log_probs,
        )

        # Step 4: compute π_ref log-probs (no grad — reference policy is frozen).
        # When ref shares the policy model (PEFT setup), disable adapters so
        # the forward runs as the underlying base/reference policy.
        ref_ctx = (
            self.model.disable_adapter()
            if self._ref_shared and hasattr(self.model, "disable_adapter")
            else contextlib.nullcontext()
        )
        with ref_ctx:
            ref_log_probs = self._compute_log_probs(
                self.ref_model, prompt_ids, completions_padded, no_grad=True
            )

        return raw_completions, completions_padded, log_probs, ref_log_probs, mask

    def compute_log_probs(
        self,
        prompt_ids: Tensor,
        completions_padded: Tensor,
    ) -> Tensor:
        """Re-run the policy forward to obtain log-probs WITH grad.

        Used by the inner-loop trainer at each gradient step k > 0 to evaluate
        the IS numerator ``log π_θ_current(a | ctx)`` against the cached
        sampling-time ``behavior_log_probs`` returned by ``sample(...)``.

        Args:
            prompt_ids:         (L_prompt,) prompt tokens, on the model device.
            completions_padded: (G, T) right-padded completion tokens, as
                                returned by ``sample(...)``.

        Returns:
            (G, T) per-token log-probs under the current policy state, with grad.
        """
        return self._compute_log_probs(
            self.model, prompt_ids, completions_padded, no_grad=False,
        )


class IndependentBaselineSampler:
    """Sample M completions from π_ref and return their rewards as a baseline.

    All operations run inside torch.no_grad() because no gradients flow
    through the baseline computation.

    Args:
        ref_model:      The frozen reference model (π_ref).
        tokenizer:      Tokenizer for ref_model.
        num_baseline:   Number of independent baseline rollouts M.
        max_new_tokens: Maximum completion length.
        temperature:    Sampling temperature for baseline rollouts.
    """

    def __init__(
        self,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        num_baseline: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        adapter_disable: bool = False,
    ) -> None:
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.num_baseline = num_baseline
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Set ``adapter_disable=True`` when the supplied "ref" model is actually
        # the policy PEFT model — generation runs with adapters disabled so the
        # baseline samples come from the underlying base/reference policy.
        self._adapter_disable = adapter_disable

        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    @torch.no_grad()
    def sample_baseline_rewards(
        self,
        prompt_ids: Tensor,
        reward_fn: Callable[[Tensor], float],
    ) -> Tensor:
        """Generate M baseline completions and compute their rewards.

        The returned tensor carries no gradient — it is only used as a scalar
        baseline in the debiased_loss computation.

        Args:
            prompt_ids: Tokenised prompt. Shape (L_prompt,). On the ref_model device.
            reward_fn:  Callable that takes a 1-D token tensor and returns a
                        float reward. Called M times, once per completion.

        Returns:
            Reward tensor of shape (M,) on CPU, wrapped in stop-gradient
            (requires_grad=False).
        """
        expanded = prompt_ids.unsqueeze(0).expand(self.num_baseline, -1)

        ctx = (
            self.ref_model.disable_adapter()
            if self._adapter_disable and hasattr(self.ref_model, "disable_adapter")
            else contextlib.nullcontext()
        )
        was_training = self.ref_model.training
        gc_enabled = getattr(self.ref_model, "is_gradient_checkpointing", False)
        if gc_enabled:
            self.ref_model.gradient_checkpointing_disable()
        self.ref_model.eval()
        try:
            with ctx:
                attn_mask = torch.ones_like(expanded)
                output_ids = self.ref_model.generate(
                    expanded,
                    attention_mask=attn_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                    pad_token_id=self.pad_id,
                    renormalize_logits=True,
                )
        finally:
            if was_training:
                self.ref_model.train()
            if gc_enabled:
                self.ref_model.gradient_checkpointing_enable()

        prompt_len = prompt_ids.shape[0]
        rewards = []
        for i in range(self.num_baseline):
            completion = output_ids[i, prompt_len:]
            r = reward_fn(completion)
            rewards.append(float(r))

        # Return as a detached CPU tensor so it can be safely used as a scalar
        # in any device context.
        return torch.tensor(rewards, dtype=torch.float32, requires_grad=False)
