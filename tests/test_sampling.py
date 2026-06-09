"""
Unit tests for debiased_grpo.sampling.

Uses tiny mock models (nn.Linear-based stubs) so that no real LLM weights
are required.  All tests run on CPU.

Two invariants of the rollout-assembly path are verified here:

  sample() returns the padding mask; padding is not marked real.
    _pad_sequences produces (padded_ids, mask), and sample() must propagate the
    mask as the 4th element of (completions, log_probs, ref_log_probs, mask) so
    the trainer honours per-group completion lengths rather than marking all T
    columns real.
    Exercised by: test_sample_returns_four_tuple,
                  test_sample_mask_shape_matches_log_probs,
                  test_sample_mask_marks_padding_correctly,
                  test_pad_and_mask_uses_per_group_mask_not_all_true

  _pad_and_mask keeps the log_probs grad graph.
    log_probs must not be detached when copied into the padded output, so
    backward() reaches the model parameters.
    Exercised by: test_pad_and_mask_preserves_grad_fn,
                  test_pad_and_mask_gradient_flows_to_source
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
import pytest

from debiased_grpo.sampling import (
    GroupRolloutSampler,
    IndependentBaselineSampler,
    _gather_log_probs,
    _pad_sequences,
)


# ---------------------------------------------------------------------------
# Helpers: deterministic mock tokenizer and model
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Minimal tokenizer stub for testing."""
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, *, return_tensors: str = "pt",
                 truncation: bool = False, max_length: int = 512):
        # Always return a fixed 4-token prompt.
        return {"input_ids": torch.tensor([[2, 3, 4, 5]])}


class _MockModelOutputs:
    def __init__(self, logits: Tensor):
        self.logits = logits


class _MockModel(nn.Module):
    """
    Minimal model stub that:
      - Has a single linear layer as a parameter (so we can check gradients).
      - generate() returns completions with variable actual content lengths,
        then right-pads them to a uniform tensor so that the prompt-stripping
        logic in _generate_completions produces variable-length raw completions.
      - forward() returns small random logits of shape (B, L, V).

    Variable-length behaviour: row i produces a completion of length
    COMPLETION_LEN - (i % 2), padded to COMPLETION_LEN with pad_token_id.
    After _generate_completions strips the prompt prefix and returns
    output_ids[i, prompt_len:], each row is exactly COMPLETION_LEN tokens
    BUT rows with shorter real content have trailing pad_token_id tokens.
    _pad_sequences detects the overall max length (which equals COMPLETION_LEN
    for all rows here), so to create genuine variable-length raw completions
    we instead return rows of DIFFERENT physical lengths (not padded).
    """
    VOCAB_SIZE = 10
    COMPLETION_LEN = 6  # maximum tokens in any generated completion

    def __init__(self, seed: int = 0):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))  # single parameter to check grad flow
        self._seed = seed

    def forward(self, input_ids: Tensor, **kwargs) -> _MockModelOutputs:
        B, L = input_ids.shape
        # Deterministic but dependent on param so gradient can flow.
        logits = torch.randn(B, L, self.VOCAB_SIZE, generator=torch.Generator().manual_seed(self._seed))
        # Add param so that grad flows through logits → log_probs → loss.
        logits = logits + self.param
        return _MockModelOutputs(logits)

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 8,
                 do_sample: bool = True, temperature: float = 1.0,
                 pad_token_id: int = 0, **kwargs) -> Tensor:
        B, L_prompt = input_ids.shape
        # Return rows with DIFFERENT completion lengths so that after
        # _generate_completions strips the prompt prefix, the raw completions
        # have genuinely different lengths.  Row i has length
        # COMPLETION_LEN - (i % 2), un-padded.  To keep a ragged tensor as a
        # single stacked output we pad to the maximum length with pad_token_id.
        max_len = self.COMPLETION_LEN
        rows = []
        for i in range(B):
            comp_len = max_len - (i % 2)  # alternates max_len and max_len - 1
            comp = torch.arange(2, 2 + comp_len, dtype=torch.long)
            # Right-pad each completion to max_len so we can stack.
            padded_comp = torch.full((max_len,), pad_token_id, dtype=torch.long)
            padded_comp[:comp_len] = comp
            rows.append(torch.cat([input_ids[i], padded_comp], dim=0))
        full = torch.stack(rows, dim=0)  # (B, L_prompt + max_len)
        return full


# ---------------------------------------------------------------------------
# _pad_sequences tests
# ---------------------------------------------------------------------------

def test_pad_sequences_returns_mask():
    """_pad_sequences must return a boolean mask as its second element."""
    seqs = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
        torch.tensor([6, 7, 8, 9]),
    ]
    padded, mask = _pad_sequences(seqs, pad_id=0)
    assert padded.dtype == torch.long
    assert mask.dtype == torch.bool
    assert padded.shape == mask.shape == (3, 4)


def test_pad_sequences_mask_true_for_real_tokens():
    """Mask must be True exactly where tokens are real, False for padding."""
    seqs = [
        torch.tensor([10, 20, 30]),      # length 3 → positions 0,1,2 real
        torch.tensor([40]),              # length 1 → position 0 real
        torch.tensor([50, 60]),          # length 2 → positions 0,1 real
    ]
    _, mask = _pad_sequences(seqs, pad_id=0)
    expected = torch.tensor([
        [True, True, True],
        [True, False, False],
        [True, True, False],
    ])
    assert torch.all(mask == expected)


def test_pad_sequences_correct_padding():
    """Padded tensor must have pad_id at padding positions."""
    seqs = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    padded, _ = _pad_sequences(seqs, pad_id=99)
    assert padded[0, 2].item() == 99  # third position is padding
    assert padded[1, 2].item() == 5   # third position is real


# ---------------------------------------------------------------------------
# _gather_log_probs tests
# ---------------------------------------------------------------------------

def test_gather_log_probs_shape():
    """_gather_log_probs must return (batch, seq_len) from (batch, seq_len, vocab)."""
    B, L, V = 3, 8, 10
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    ids = torch.randint(0, V, (B, L))
    lp = _gather_log_probs(logits, ids)
    assert lp.shape == (B, L)


def test_gather_log_probs_selects_correct_token():
    """_gather_log_probs must return the log-softmax probability of the given token id."""
    # Single batch, single timestep, 4-token vocabulary.
    logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4)
    ids = torch.tensor([[2]])                          # select vocab index 2
    expected = torch.nn.functional.log_softmax(logits[0, 0], dim=-1)[2]
    lp = _gather_log_probs(logits, ids)
    assert torch.allclose(lp, expected.unsqueeze(0).unsqueeze(0), atol=1e-5)


def test_gather_log_probs_all_log_probs_negative():
    """All returned log-probabilities must be <= 0 (they are log of values in [0,1])."""
    B, L, V = 4, 16, 20
    logits = torch.randn(B, L, V)
    ids = torch.randint(0, V, (B, L))
    lp = _gather_log_probs(logits, ids)
    assert (lp <= 0).all(), "Log-probabilities must be non-positive"


# ---------------------------------------------------------------------------
# GroupRolloutSampler.sample() return value and mask tests
# ---------------------------------------------------------------------------

def test_sample_returns_five_tuple():
    """sample() must return a 5-tuple:
    (completions, completions_padded, log_probs, ref_log_probs, mask).

    The ``completions_padded`` (G, T_max) tensor is exposed so that callers
    can re-run the policy forward under an updated model state via
    ``compute_log_probs``, supporting the inner-loop μ>1 trainer.
    """
    model = _MockModel(seed=0)
    ref_model = _MockModel(seed=1)
    tokenizer = _MockTokenizer()

    sampler = GroupRolloutSampler(
        model=model,
        tokenizer=tokenizer,
        ref_model=ref_model,
        group_size=3,
        max_new_tokens=6,
    )

    prompt_ids = torch.tensor([2, 3, 4, 5])
    result = sampler.sample(prompt_ids)

    assert len(result) == 5, (
        f"sample() must return 5 values (completions, completions_padded, "
        f"log_probs, ref_log_probs, mask) but returned {len(result)}."
    )


def test_sample_mask_shape_matches_log_probs():
    """The mask returned by sample() must have the same shape as log_probs."""
    model = _MockModel(seed=0)
    tokenizer = _MockTokenizer()
    sampler = GroupRolloutSampler(
        model=model, tokenizer=tokenizer, group_size=4, max_new_tokens=6
    )
    prompt_ids = torch.tensor([2, 3])
    completions, completions_padded, log_probs, ref_log_probs, mask = sampler.sample(prompt_ids)
    assert mask.shape == log_probs.shape, (
        f"Mask shape {mask.shape} must match log_probs shape {log_probs.shape}"
    )
    assert mask.dtype == torch.bool
    assert completions_padded.shape == log_probs.shape, (
        "completions_padded must have the same (G, T) shape as log_probs"
    )


def test_sample_mask_marks_padding_correctly():
    """Mask from sample() must reflect real vs padding positions from _pad_sequences.

    We directly inject variable-length raw completions into GroupRolloutSampler
    by patching _generate_completions so that the first half of the group
    receives 6-token completions and the second half receives 4-token completions.
    After _pad_sequences, the shorter completions are right-padded to 6 tokens;
    the mask must be False for those last 2 positions.
    """
    from unittest.mock import patch

    model = _MockModel(seed=42)
    tokenizer = _MockTokenizer()
    G = 4
    sampler = GroupRolloutSampler(
        model=model, tokenizer=tokenizer, group_size=G, max_new_tokens=6
    )
    prompt_ids = torch.tensor([2, 3])

    # Variable-length completions: rows 0-1 have 6 tokens, rows 2-3 have 4 tokens.
    fake_completions = [
        torch.tensor([2, 3, 4, 5, 6, 7]),  # length 6
        torch.tensor([2, 3, 4, 5, 6, 7]),  # length 6
        torch.tensor([2, 3, 4, 5]),         # length 4
        torch.tensor([2, 3, 4, 5]),         # length 4
    ]

    with patch.object(sampler, '_generate_completions', return_value=fake_completions):
        completions, completions_padded, log_probs, ref_log_probs, mask = sampler.sample(prompt_ids)

    assert mask.shape == (G, 6), f"Expected (G=4, T=6), got {mask.shape}"
    # Rows 0-1: all 6 tokens real → mask is all True.
    assert mask[:2, :].all(), "First 2 rows (length-6 completions) must have all-True mask"
    # Rows 2-3: only first 4 tokens real → last 2 positions must be False.
    assert mask[2:, :4].all(), "Rows 2-3 real tokens (positions 0-3) must be True"
    assert not mask[2:, 4:].any(), (
        "Rows 2-3 padding positions (4-5) must be False — "
        "sample() must propagate the mask from _pad_sequences, not mark padding real."
    )


def test_sample_ref_log_probs_no_grad():
    """ref_log_probs returned by sample() must not require gradients.

    The reference policy is frozen; gradients through ref_log_probs would
    incorrectly update the reference model.
    """
    model = _MockModel(seed=0)
    ref_model = _MockModel(seed=1)
    tokenizer = _MockTokenizer()
    sampler = GroupRolloutSampler(
        model=model, tokenizer=tokenizer, ref_model=ref_model, group_size=2
    )
    prompt_ids = torch.tensor([2, 3])
    _, _, log_probs, ref_log_probs, _ = sampler.sample(prompt_ids)

    assert not ref_log_probs.requires_grad, (
        "ref_log_probs must not require gradients — reference model is frozen"
    )


def test_sample_log_probs_shape():
    """log_probs and ref_log_probs must have shape (G, T)."""
    G = 3
    model = _MockModel(seed=0)
    tokenizer = _MockTokenizer()
    sampler = GroupRolloutSampler(
        model=model, tokenizer=tokenizer, group_size=G, max_new_tokens=6
    )
    prompt_ids = torch.tensor([2, 3, 4])
    _, _, log_probs, ref_log_probs, mask = sampler.sample(prompt_ids)

    assert log_probs.shape[0] == G, f"Expected G={G} rows, got {log_probs.shape[0]}"
    assert ref_log_probs.shape == log_probs.shape


def test_compute_log_probs_grad_flows():
    """compute_log_probs must return log-probs WITH grad through model params."""
    model = _MockModel(seed=0)
    tokenizer = _MockTokenizer()
    sampler = GroupRolloutSampler(
        model=model, tokenizer=tokenizer, group_size=2, max_new_tokens=6
    )
    prompt_ids = torch.tensor([2, 3])
    _, completions_padded, _, _, _ = sampler.sample(prompt_ids)

    log_probs = sampler.compute_log_probs(prompt_ids, completions_padded)
    assert log_probs.requires_grad, (
        "compute_log_probs must return a tensor with grad attached"
    )
    log_probs.sum().backward()
    assert model.param.grad is not None, (
        "Gradient through compute_log_probs did not reach model parameters"
    )


def test_sample_no_grad_flag_returns_detached_log_probs():
    """sample(with_grad_log_probs=False) returns no-grad behavior_log_probs.

    The inner-loop trainer relies on this to snapshot the sampling-time policy
    once per resample, then re-runs compute_log_probs(...) at each inner step.
    """
    model = _MockModel(seed=0)
    tokenizer = _MockTokenizer()
    sampler = GroupRolloutSampler(
        model=model, tokenizer=tokenizer, group_size=2, max_new_tokens=6
    )
    prompt_ids = torch.tensor([2, 3])
    _, _, log_probs, _, _ = sampler.sample(prompt_ids, with_grad_log_probs=False)
    assert not log_probs.requires_grad, (
        "with_grad_log_probs=False must produce a detached log_probs tensor"
    )


# ---------------------------------------------------------------------------
# _pad_and_mask from trainer: grad-graph and per-group-mask invariants
# ---------------------------------------------------------------------------

def test_pad_and_mask_preserves_grad_fn():
    """_pad_and_mask must preserve the grad_fn of log_probs (no detach).

    Detaching log_probs when copying into the padded output would create a leaf
    disconnected from the model parameters; backward() would then silently
    produce zero parameter gradients.
    """
    from debiased_grpo.trainer import _pad_and_mask

    B, T = 3, 8
    # Create tensors with a grad_fn (not leaf).
    lp_source = torch.randn(B, T, requires_grad=True)
    lp_with_fn = lp_source * 1.0  # gives lp_with_fn a grad_fn
    rlp = torch.randn(B, T)
    mk = torch.ones(B, T, dtype=torch.bool)

    lp_out, _, _ = _pad_and_mask([lp_with_fn], [rlp], [mk])

    # The output must stay connected to the graph: a backward pass must
    # propagate to lp_source.grad.
    loss = lp_out.sum()
    loss.backward()

    assert lp_source.grad is not None, (
        "lp_source.grad is None after backward through _pad_and_mask output — "
        "log_probs must not be detached in _pad_and_mask."
    )
    assert lp_source.grad.norm().item() > 0, (
        "lp_source.grad is all-zero — the grad_fn chain must stay intact."
    )


def test_pad_and_mask_gradient_flows_to_source():
    """Gradient from loss computed with _pad_and_mask output must reach model params."""
    from debiased_grpo.trainer import _pad_and_mask
    from debiased_grpo.losses import debiased_loss

    model = nn.Linear(4, 4, bias=False)
    x = torch.randn(2, 4)
    out = model(x)  # (2, 4), has grad_fn connected to model.weight

    # Build a fake (B=2, T=4) log_probs from model output.
    lp = out  # shape (2, 4), has grad_fn
    rlp = torch.zeros(2, 4)
    mk = torch.ones(2, 4, dtype=torch.bool)

    lp_batched, rlp_batched, mask_batched = _pad_and_mask([lp], [rlp], [mk])

    rewards = torch.tensor([1.0, -1.0])
    baseline_rewards = torch.tensor([0.0, 0.5])
    loss = debiased_loss(lp_batched, rlp_batched, rewards, baseline_rewards, mask_batched)
    loss.backward()

    assert model.weight.grad is not None, (
        "model.weight.grad is None after backward — the gradient chain must reach model params"
    )
    assert model.weight.grad.norm().item() > 0, (
        "model.weight.grad is all-zero — the gradient chain must reach model params"
    )


def test_pad_and_mask_uses_per_group_mask_not_all_true():
    """_pad_and_mask must use the provided masks, not set mask=True for all positions.

    Marking all T columns real regardless of per-group sequence lengths would
    include padding tokens in the loss. Only real token positions must be True.
    """
    from debiased_grpo.trainer import _pad_and_mask

    G, T = 3, 8
    lp = torch.randn(G, T)
    rlp = torch.randn(G, T)
    # Only the first 5 tokens are real; positions 5-7 are padding.
    mk = torch.zeros(G, T, dtype=torch.bool)
    mk[:, :5] = True

    _, _, mask_out = _pad_and_mask([lp], [rlp], [mk])

    # The output mask must reflect the per-group mask, not all-True.
    assert not mask_out[:, 5:].any(), (
        "Padding positions (indices 5-7) should be False in the output mask, "
        "but they are True — _pad_and_mask must honour the per-group mask, not force all-True"
    )
    assert mask_out[:, :5].all(), "Real token positions must be True in the output mask"


def test_pad_and_mask_stacks_multiple_prompts():
    """_pad_and_mask must stack multiple prompt groups into a single (N, T_max) batch."""
    from debiased_grpo.trainer import _pad_and_mask

    # Two prompts, each with G=2 rollouts of different lengths.
    G, T1, T2 = 2, 6, 10

    lp1 = torch.randn(G, T1)
    rlp1 = torch.randn(G, T1)
    mk1 = torch.ones(G, T1, dtype=torch.bool)

    lp2 = torch.randn(G, T2)
    rlp2 = torch.randn(G, T2)
    mk2 = torch.ones(G, T2, dtype=torch.bool)
    mk2[:, -2:] = False  # last 2 tokens of prompt 2 are padding

    lp_out, rlp_out, mask_out = _pad_and_mask([lp1, lp2], [rlp1, rlp2], [mk1, mk2])

    N = G * 2
    assert lp_out.shape == (N, T2), f"Expected ({N}, {T2}), got {lp_out.shape}"
    # Prompt 1 rows are zero-padded from T1 to T2; those positions must be False.
    assert not mask_out[:G, T1:].any(), (
        f"Positions {T1}-{T2-1} for prompt 1 rows must be False (padded)"
    )
    # Prompt 2 last 2 positions must be False (from mk2).
    assert not mask_out[G:, -2:].any(), (
        "Last 2 positions of prompt 2 rows must be False (from group mask)"
    )


# ---------------------------------------------------------------------------
# IndependentBaselineSampler tests
# ---------------------------------------------------------------------------

def test_independent_baseline_sampler_no_grad():
    """IndependentBaselineSampler.sample_baseline_rewards must run in no_grad."""
    ref_model = _MockModel(seed=5)
    tokenizer = _MockTokenizer()
    sampler = IndependentBaselineSampler(
        ref_model=ref_model,
        tokenizer=tokenizer,
        num_baseline=4,
        max_new_tokens=6,
    )
    prompt_ids = torch.tensor([2, 3, 4])

    def reward_fn(completion: Tensor) -> float:
        return float(completion.sum())

    rewards = sampler.sample_baseline_rewards(prompt_ids, reward_fn)

    assert rewards.shape == (4,), f"Expected (M=4,), got {rewards.shape}"
    assert not rewards.requires_grad, (
        "baseline_rewards must not require grad — no gradient flows through baseline"
    )
    assert rewards.isfinite().all(), "Baseline rewards must all be finite"


def test_independent_baseline_sampler_float_rewards():
    """sample_baseline_rewards must convert reward_fn outputs to float32 tensor."""
    ref_model = _MockModel(seed=7)
    tokenizer = _MockTokenizer()
    sampler = IndependentBaselineSampler(
        ref_model=ref_model, tokenizer=tokenizer, num_baseline=3
    )
    prompt_ids = torch.tensor([2, 3])

    # reward_fn returns an integer — must be converted to float.
    rewards = sampler.sample_baseline_rewards(prompt_ids, lambda c: 1)
    assert rewards.dtype == torch.float32
    assert torch.all(rewards == 1.0)


def test_independent_baseline_sampler_m_rollouts():
    """sample_baseline_rewards must return exactly M reward values."""
    M = 6
    ref_model = _MockModel(seed=3)
    tokenizer = _MockTokenizer()
    sampler = IndependentBaselineSampler(
        ref_model=ref_model, tokenizer=tokenizer, num_baseline=M
    )
    rewards = sampler.sample_baseline_rewards(
        torch.tensor([2, 3]),
        lambda c: 0.5,
    )
    assert rewards.shape == (M,), f"Expected (M={M},), got {rewards.shape}"
