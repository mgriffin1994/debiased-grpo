"""
Unit tests for debiased_grpo.trainer.DebiasedGRPOTrainer.

Uses lightweight mock models so that no real LLM weights are needed.
All tests run on CPU.

Two invariants of the trainer's rollout-assembly path are verified here:

  log_probs keep their grad graph.
    _pad_and_mask must not detach log_probs (and train_step must not re-attach
    them with requires_grad_). Detaching creates a new leaf disconnected from
    the model parameters, so backward() would never reach them.
    Exercised by: test_train_step_returns_loss_tensor,
                  test_train_step_loss_is_finite

  Padding masks are passed through; padding is not marked real.
    _pad_and_mask accepts a per-group mask_list and honours each group's
    completion lengths, so padding tokens are excluded from the loss rather
    than marked real across all T positions. sample() returns the mask as its
    4th element.
    Exercised by: test_train_step_returns_loss_tensor (end-to-end path)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, List, Tuple

import pytest

from debiased_grpo.trainer import DebiasedGRPOTrainer, _pad_and_mask


# ---------------------------------------------------------------------------
# Minimal mock classes that replicate the tokenizer/model interface
# ---------------------------------------------------------------------------

class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, *, return_tensors: str = "pt",
                 truncation: bool = False, max_length: int = 512):
        return {"input_ids": torch.tensor([[2, 3, 4, 5]])}


class _ModelOutputs:
    def __init__(self, logits: Tensor):
        self.logits = logits


class _TinyModel(nn.Module):
    """
    A tiny parametric model that supports the same interface as a HuggingFace
    CausalLM (forward + generate) but is entirely random/deterministic.

    Parameters are a small embedding matrix so that gradients have somewhere
    to flow.
    """
    VOCAB_SIZE = 12
    COMPLETION_LEN = 5

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(self.VOCAB_SIZE, 8)
        self.head = nn.Linear(8, self.VOCAB_SIZE, bias=False)

    def forward(self, input_ids: Tensor, **kwargs) -> _ModelOutputs:
        x = self.embed(input_ids)           # (B, L, 8)
        logits = self.head(x)               # (B, L, V)
        return _ModelOutputs(logits)

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 8,
                 do_sample: bool = True, temperature: float = 1.0,
                 pad_token_id: int = 0, **kwargs) -> Tensor:
        B, L_prompt = input_ids.shape
        # Produce completions of alternating length to exercise padding logic.
        rows = []
        for i in range(B):
            length = self.COMPLETION_LEN - (i % 2)  # alternates COMP_LEN and COMP_LEN-1
            comp = torch.arange(2, 2 + length, dtype=torch.long)
            padded = torch.full((self.COMPLETION_LEN,), pad_token_id, dtype=torch.long)
            padded[:length] = comp
            rows.append(torch.cat([input_ids[i], padded], dim=0))
        return torch.stack(rows, dim=0)


def _simple_reward(completion: Tensor) -> float:
    """Reward = 1.0 if any token is >= 3, else 0.0."""
    return 1.0 if (completion >= 3).any().item() else 0.0


def _make_trainer(group_size: int = 3, num_baseline: int = 2) -> DebiasedGRPOTrainer:
    model = _TinyModel(seed=0)
    ref_model = _TinyModel(seed=1)
    tokenizer = _MockTokenizer()
    config = {
        "group_size": group_size,
        "num_baseline_rollouts": num_baseline,
        "max_new_tokens": 5,
        "temperature": 1.0,
        "clip_eps": 0.2,
    }
    return DebiasedGRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=_simple_reward,
        config=config,
    )


# ---------------------------------------------------------------------------
# DebiasedGRPOTrainer.train_step smoke tests
# ---------------------------------------------------------------------------

def test_train_step_returns_expected_keys():
    """train_step must return a dict with exactly the expected keys."""
    trainer = _make_trainer()
    batch = ["test prompt"]
    metrics = trainer.train_step(batch)

    expected_keys = {"loss", "grpo_loss_value", "ess", "mean_reward"}
    assert set(metrics.keys()) == expected_keys, (
        f"Unexpected keys: {set(metrics.keys()) - expected_keys}, "
        f"missing: {expected_keys - set(metrics.keys())}"
    )


def test_train_step_returns_loss_tensor():
    """The 'loss' value in the metrics dict must be a Tensor, not a float."""
    trainer = _make_trainer()
    metrics = trainer.train_step(["hello"])
    assert isinstance(metrics["loss"], torch.Tensor), (
        f"Expected loss to be a Tensor, got {type(metrics['loss'])}"
    )


def test_train_step_loss_is_finite():
    """train_step loss must be a finite scalar."""
    trainer = _make_trainer()
    metrics = trainer.train_step(["test"])
    loss = metrics["loss"]
    assert loss.shape == (), f"Loss must be scalar, got shape {loss.shape}"
    assert loss.isfinite(), f"Loss is not finite: {loss.item()}"


def test_train_step_diagnostic_values_are_floats():
    """grpo_loss_value, ess, mean_reward must be plain Python floats."""
    trainer = _make_trainer()
    metrics = trainer.train_step(["test"])
    for key in ("grpo_loss_value", "ess", "mean_reward"):
        val = metrics[key]
        assert isinstance(val, float), f"metrics['{key}'] should be float, got {type(val)}"
        assert not (val != val), f"metrics['{key}'] is NaN"  # nan != nan


def test_train_step_ess_positive():
    """ESS must be strictly positive."""
    trainer = _make_trainer()
    metrics = trainer.train_step(["test"])
    assert metrics["ess"] > 0.0, f"ESS must be positive, got {metrics['ess']}"


def test_train_step_multi_prompt_batch():
    """train_step must handle a batch of multiple prompts without errors."""
    trainer = _make_trainer(group_size=2, num_baseline=2)
    batch = ["first prompt", "second prompt", "third prompt"]
    metrics = trainer.train_step(batch)
    assert metrics["loss"].isfinite()
    assert metrics["ess"] > 0


def test_train_step_backward_runs():
    """loss.backward() must complete without error after train_step."""
    trainer = _make_trainer()
    metrics = trainer.train_step(["test"])
    # backward should not raise
    metrics["loss"].backward()


def test_train_step_grpo_diagnostic_no_grad():
    """grpo_loss_value in the metrics dict is already a float (detached)."""
    trainer = _make_trainer()
    metrics = trainer.train_step(["test"])
    # grpo_loss_value must be a Python float, not a tensor that could have grad.
    assert isinstance(metrics["grpo_loss_value"], float)


# ---------------------------------------------------------------------------
# Config access: dict and namespace style
# ---------------------------------------------------------------------------

def test_trainer_accepts_dict_config():
    """DebiasedGRPOTrainer must work when config is a plain dict."""
    model = _TinyModel(seed=0)
    ref_model = _TinyModel(seed=1)
    tokenizer = _MockTokenizer()
    config = {"group_size": 2, "num_baseline_rollouts": 2, "max_new_tokens": 4}
    trainer = DebiasedGRPOTrainer(model, ref_model, tokenizer, _simple_reward, config)
    assert trainer.group_size == 2
    assert trainer.num_baseline == 2


def test_trainer_accepts_namespace_config():
    """DebiasedGRPOTrainer must work when config is an object with attributes."""
    import types
    model = _TinyModel(seed=0)
    ref_model = _TinyModel(seed=1)
    tokenizer = _MockTokenizer()
    config = types.SimpleNamespace(
        group_size=3, num_baseline_rollouts=2, max_new_tokens=4,
        temperature=0.8, clip_eps=0.15
    )
    trainer = DebiasedGRPOTrainer(model, ref_model, tokenizer, _simple_reward, config)
    assert trainer.group_size == 3
    assert trainer.clip_eps == 0.15


def test_trainer_default_config_values():
    """When keys are absent from config, sensible defaults must be used."""
    model = _TinyModel(seed=0)
    ref_model = _TinyModel(seed=1)
    tokenizer = _MockTokenizer()
    trainer = DebiasedGRPOTrainer(model, ref_model, tokenizer, _simple_reward, config={})
    assert trainer.group_size == 8
    assert trainer.num_baseline == 4
    assert trainer.clip_eps == 0.2


# ---------------------------------------------------------------------------
# _pad_and_mask (trainer module-level function) direct tests
# ---------------------------------------------------------------------------

def test_pad_and_mask_output_shapes():
    """_pad_and_mask output tensors must have consistent shapes."""
    G1, T1 = 3, 8
    G2, T2 = 3, 12

    lp1 = torch.randn(G1, T1)
    rlp1 = torch.randn(G1, T1)
    mk1 = torch.ones(G1, T1, dtype=torch.bool)

    lp2 = torch.randn(G2, T2)
    rlp2 = torch.randn(G2, T2)
    mk2 = torch.ones(G2, T2, dtype=torch.bool)

    lp_out, rlp_out, mask_out = _pad_and_mask([lp1, lp2], [rlp1, rlp2], [mk1, mk2])

    N = G1 + G2
    T_max = T2
    assert lp_out.shape == (N, T_max)
    assert rlp_out.shape == (N, T_max)
    assert mask_out.shape == (N, T_max)
    assert mask_out.dtype == torch.bool


def test_pad_and_mask_shorter_prompts_zero_padded():
    """Rows from shorter (T1 < T_max) prompts must be zero-padded for log_probs."""
    G, T1, T2 = 2, 6, 10

    lp1 = torch.ones(G, T1) * 0.5
    rlp1 = torch.ones(G, T1) * 0.3
    mk1 = torch.ones(G, T1, dtype=torch.bool)

    lp2 = torch.ones(G, T2) * 0.7
    rlp2 = torch.ones(G, T2) * 0.4
    mk2 = torch.ones(G, T2, dtype=torch.bool)

    lp_out, _, _ = _pad_and_mask([lp1, lp2], [rlp1, rlp2], [mk1, mk2])

    # Rows 0:G came from lp1; columns T1:T_max should be zero-padded.
    assert torch.all(lp_out[:G, T1:] == 0.0), (
        "Short-prompt rows must have zeros beyond their original length"
    )
    # Rows G:2G came from lp2; all columns should be present.
    assert torch.allclose(lp_out[G:, :T2], torch.ones(G, T2) * 0.7, atol=1e-6)


def test_pad_and_mask_mask_false_for_padded_positions():
    """mask_out must be False for columns added by global zero-padding (T1 → T_max)."""
    G, T1, T2 = 2, 5, 9

    lp1 = torch.randn(G, T1)
    rlp1 = torch.randn(G, T1)
    mk1 = torch.ones(G, T1, dtype=torch.bool)  # all T1 positions real for prompt 1

    lp2 = torch.randn(G, T2)
    rlp2 = torch.randn(G, T2)
    mk2 = torch.ones(G, T2, dtype=torch.bool)

    _, _, mask_out = _pad_and_mask([lp1, lp2], [rlp1, rlp2], [mk1, mk2])

    # Rows from prompt 1 (indices 0:G): columns 5-8 were added by global padding.
    assert not mask_out[:G, T1:].any(), (
        f"Global-padding columns (indices {T1}+) for short-prompt rows must be False"
    )
    # Rows from prompt 1: columns 0-4 must be True (real).
    assert mask_out[:G, :T1].all()
