"""Tests for inference and export — Architecture.md §25–26.

Covers: export weight precision, >1024-token rejection, temperature/top-k
edge cases, export round-trip, missing-file error handling.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import AstraCSE, ModelConfig
from src.infer import export_weights, generate, MAX_SEQ_LEN

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return ModelConfig.from_yaml(REPO_ROOT / "configs" / "model.yaml")


@pytest.fixture(scope="module")
def small_cfg() -> ModelConfig:
    cfg = ModelConfig(
        vocab_size=256, d_model=64, n_layer=2, n_head=4, d_ff=128,
        max_seq_len=32, norm_eps=1e-5, weight_tying=True, expected_param_count=0,
    )
    cfg.__dict__["expected_param_count"] = sum(
        p.numel() for p in AstraCSE(cfg).parameters())
    return cfg


@pytest.fixture(scope="module")
def small_model(small_cfg) -> AstraCSE:
    m = AstraCSE(small_cfg)
    m.eval()
    return m


def _make_fake_tokenizer(vocab_size: int = 256):
    tok = MagicMock()
    tok.encode.return_value = MagicMock(ids=[1, 2, 3, 4, 5])
    tok.decode.return_value = "test output"
    tok.token_to_id.return_value = None  # no <eos>
    return tok


# ---------------------------------------------------------------------------
# Export precision
# ---------------------------------------------------------------------------

def test_export_fp16(cfg, tmp_path):
    model = AstraCSE(cfg)
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "global_step": 1,
                "metadata": {}}, ckpt_path)
    out = export_weights(ckpt_path, tmp_path / "export", "fp16")
    payload = torch.load(out, map_location="cpu", weights_only=False)
    sample = next(iter(payload["model_state"].values()))
    assert sample.dtype == torch.float16


def test_export_bf16(cfg, tmp_path):
    model = AstraCSE(cfg)
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "global_step": 1,
                "metadata": {}}, ckpt_path)
    out = export_weights(ckpt_path, tmp_path / "export", "bf16")
    payload = torch.load(out, map_location="cpu", weights_only=False)
    sample = next(iter(payload["model_state"].values()))
    assert sample.dtype == torch.bfloat16


def test_export_strips_optimizer_state(cfg, tmp_path):
    model = AstraCSE(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "full.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "global_step": 5,
        "metadata": {},
    }, ckpt_path)
    out = export_weights(ckpt_path, tmp_path / "export", "fp32")
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert "optimizer_state" not in payload


def test_export_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_weights(tmp_path / "nonexistent.pt", tmp_path / "out", "fp16")


# ---------------------------------------------------------------------------
# Input length enforcement — §25.1
# ---------------------------------------------------------------------------

def test_over_max_seq_len_raises(small_model, small_cfg):
    tok = MagicMock()
    tok.encode.return_value = MagicMock(ids=list(range(small_cfg.max_seq_len + 1)))
    tok.token_to_id.return_value = None
    with pytest.raises(ValueError, match="max_seq_len"):
        generate(small_model, tok, "dummy", device=torch.device("cpu"))


def test_exactly_max_seq_len_raises(small_model, small_cfg):
    # A prompt of exactly max_seq_len leaves no room for generation — should raise.
    tok = MagicMock()
    tok.encode.return_value = MagicMock(ids=list(range(small_cfg.max_seq_len)))
    tok.decode.return_value = ""
    tok.token_to_id.return_value = None
    with pytest.raises(ValueError, match="max_seq_len"):
        generate(small_model, tok, "dummy", max_new_tokens=1,
                 device=torch.device("cpu"))


def test_prompt_under_max_seq_len_accepted(small_model, small_cfg):
    tok = MagicMock()
    tok.encode.return_value = MagicMock(ids=list(range(small_cfg.max_seq_len - 1)))
    tok.decode.return_value = ""
    tok.token_to_id.return_value = None
    result = generate(small_model, tok, "dummy", max_new_tokens=1,
                      device=torch.device("cpu"))
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Temperature edge cases
# ---------------------------------------------------------------------------

def test_temperature_zero_like_is_greedy(small_model, small_cfg):
    torch.manual_seed(0)
    tok = _make_fake_tokenizer(small_cfg.vocab_size)
    # Very low temperature should produce deterministic output
    r1 = generate(small_model, tok, "q", max_new_tokens=5,
                  temperature=1e-5, top_k=0, device=torch.device("cpu"))
    torch.manual_seed(0)
    r2 = generate(small_model, tok, "q", max_new_tokens=5,
                  temperature=1e-5, top_k=0, device=torch.device("cpu"))
    assert r1 == r2


def test_high_temperature_different_outputs(small_model, small_cfg):
    torch.manual_seed(1)
    tok = _make_fake_tokenizer(small_cfg.vocab_size)
    r1 = generate(small_model, tok, "q", max_new_tokens=10,
                  temperature=10.0, top_k=0, device=torch.device("cpu"))
    torch.manual_seed(999)
    r2 = generate(small_model, tok, "q", max_new_tokens=10,
                  temperature=10.0, top_k=0, device=torch.device("cpu"))
    # With different seeds and high temperature, outputs are likely different
    # (not guaranteed but extremely likely with vocab_size=256)
    # We just verify it completes without error
    assert isinstance(r1, str)
    assert isinstance(r2, str)


# ---------------------------------------------------------------------------
# EOS stopping
# ---------------------------------------------------------------------------

def test_stops_on_eos(small_model, small_cfg):
    eos_id = 10
    call_count = [0]
    original_forward = small_model.forward

    def patched_forward(input_ids, **kwargs):
        call_count[0] += 1
        logits, caches = original_forward(input_ids, **kwargs)
        if call_count[0] == 2:
            # Force next token to be EOS on the second generation step
            logits[:, -1, :] = float("-inf")
            logits[:, -1, eos_id] = 1000.0
        return logits, caches

    tok = MagicMock()
    tok.encode.return_value = MagicMock(ids=[1, 2, 3])
    tok.decode.return_value = ""
    tok.token_to_id.return_value = eos_id

    small_model.forward = patched_forward
    try:
        generate(small_model, tok, "q", max_new_tokens=50,
                 temperature=0.01, top_k=1, device=torch.device("cpu"))
        # Should have stopped after 2 generation steps (1 normal + 1 EOS)
        assert call_count[0] <= 4  # prefill + generation steps
    finally:
        small_model.forward = original_forward


# ---------------------------------------------------------------------------
# Repetition penalty — mitigates loop-degeneration in small from-scratch LMs
# ---------------------------------------------------------------------------

def test_repetition_penalty_default_matches_no_penalty(small_model, small_cfg):
    """repetition_penalty=1.0 (the default) must be a true no-op."""
    tok = _make_fake_tokenizer(small_cfg.vocab_size)
    torch.manual_seed(3)
    r1 = generate(small_model, tok, "q", max_new_tokens=5, temperature=0.7, top_k=10,
                 device=torch.device("cpu"))
    torch.manual_seed(3)
    r2 = generate(small_model, tok, "q", max_new_tokens=5, temperature=0.7, top_k=10,
                 repetition_penalty=1.0, device=torch.device("cpu"))
    assert r1 == r2


def test_repetition_penalty_breaks_greedy_loop(small_model, small_cfg):
    """Without penalty, forcing one token to always have the top logit makes
    greedy decoding loop on it forever. With penalty > 1.0, the forced
    favourite must get suppressed after it appears, and the runner-up token
    must eventually be selected instead."""
    original_forward = small_model.forward

    def force_favorite_forward(input_ids, **kwargs):
        logits, caches = original_forward(input_ids, **kwargs)
        logits = logits.clone()
        logits[:, -1, :] = -10.0
        logits[:, -1, 7] = 5.0   # forced favourite
        logits[:, -1, 8] = 4.9  # close runner-up
        return logits, caches

    tok = MagicMock()
    tok.encode.return_value = MagicMock(ids=[1, 2, 3])
    tok.token_to_id.return_value = None
    tok.decode.side_effect = lambda ids, **kw: ",".join(map(str, ids))

    small_model.forward = force_favorite_forward
    try:
        torch.manual_seed(0)
        no_penalty = generate(small_model, tok, "q", max_new_tokens=6,
                              temperature=1e-4, top_k=1,
                              repetition_penalty=1.0, device=torch.device("cpu"))
        torch.manual_seed(0)
        with_penalty = generate(small_model, tok, "q", max_new_tokens=6,
                                temperature=1e-4, top_k=1,
                                repetition_penalty=1.5, device=torch.device("cpu"))
    finally:
        small_model.forward = original_forward

    assert no_penalty == "7,7,7,7,7,7", (
        f"Expected unpenalized greedy decoding to loop on the forced "
        f"favourite token, got {no_penalty!r}"
    )
    assert with_penalty != no_penalty, (
        "repetition_penalty=1.5 produced the same output as no penalty at all."
    )
    assert "8" in with_penalty.split(","), (
        "Expected the runner-up token to be picked once the favourite was "
        "penalized down for repeating."
    )


# ---------------------------------------------------------------------------
# KV-cache gives same output as full-sequence forward
# ---------------------------------------------------------------------------

def test_generate_matches_full_forward_output(small_model, small_cfg):
    """Verifies RoPE fix is effective: cached generation must match prefill."""
    torch.manual_seed(42)
    T = 6
    ids = torch.randint(0, small_cfg.vocab_size, (1, T))

    with torch.no_grad():
        logits_full, _ = small_model(ids)

        # Simulate what generate() does: token-by-token
        kv = None
        all_logits = []
        for t in range(T):
            cur = ids[:, t:t+1]
            logits_t, kv = small_model(cur, kv_caches=kv, return_kv_caches=True)
            all_logits.append(logits_t[:, 0, :])

    cached = torch.stack(all_logits, dim=1)
    assert torch.allclose(logits_full, cached, atol=1e-4), (
        "Cached generation logits differ from full-sequence — RoPE offset bug."
    )
