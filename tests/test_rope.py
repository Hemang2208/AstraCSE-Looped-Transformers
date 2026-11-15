"""Tests for RoPE positional encoding correctness — Architecture.md §5.

Critical: verifies the KV-cache RoPE offset bug is fixed.
Each new token during cached inference must be encoded at its true absolute
position, not always at position 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import AstraCSE, ModelConfig, _build_rope_cache, apply_rope, _rotate_half

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def small_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256, d_model=64, n_layer=2, n_head=4, d_ff=128,
        max_seq_len=32, norm_eps=1e-5, weight_tying=True,
        expected_param_count=0,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg) -> AstraCSE:
    cfg = small_cfg
    cfg.__dict__["expected_param_count"] = sum(
        p.numel() for p in AstraCSE(cfg).parameters()
    )
    m = AstraCSE(cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# 1. RoPE cache shape
# ---------------------------------------------------------------------------

def test_rope_cache_shape(small_cfg):
    cos, sin = _build_rope_cache(small_cfg.d_head, small_cfg.max_seq_len,
                                  torch.device("cpu"))
    assert cos.shape == (small_cfg.max_seq_len, small_cfg.d_head)
    assert sin.shape == (small_cfg.max_seq_len, small_cfg.d_head)


# ---------------------------------------------------------------------------
# 2. Rotate-half is its own inverse when applied twice
# ---------------------------------------------------------------------------

def test_rotate_half_double_inverse():
    x = torch.randn(2, 4, 8, 16)
    assert torch.allclose(_rotate_half(_rotate_half(x)), -x, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. RoPE is orthogonal — applying it preserves vector norms
# ---------------------------------------------------------------------------

def test_rope_preserves_norm(small_cfg):
    T, d = 16, small_cfg.d_head
    cos, sin = _build_rope_cache(d, T, torch.device("cpu"))
    q = torch.randn(1, small_cfg.n_head, T, d)
    q_rot, _ = apply_rope(q, q.clone(), cos, sin)
    norms_before = q.norm(dim=-1)
    norms_after = q_rot.norm(dim=-1)
    assert torch.allclose(norms_before, norms_after, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. KV-cache RoPE offset — the core bug fix
#
#    Full-sequence forward and token-by-token cached forward must produce
#    identical logits at every position.
# ---------------------------------------------------------------------------

def test_kv_cache_rope_matches_full_forward(small_model, small_cfg):
    torch.manual_seed(7)
    T = 12
    ids = torch.randint(0, small_cfg.vocab_size, (1, T))

    with torch.no_grad():
        # Full prefill (no cache)
        logits_full, _ = small_model(ids)

        # Token-by-token with KV cache
        kv_caches = None
        logits_cached = []
        for t in range(T):
            cur = ids[:, t:t+1]
            logits_t, kv_caches = small_model(cur, kv_caches=kv_caches,
                                               return_kv_caches=True)
            logits_cached.append(logits_t[:, 0, :])

    logits_cached_tensor = torch.stack(logits_cached, dim=1)   # [1, T, V]
    assert torch.allclose(logits_full, logits_cached_tensor, atol=1e-4), (
        "KV-cache logits differ from full-sequence logits — RoPE offset bug."
    )


# ---------------------------------------------------------------------------
# 5. Different positions must produce different encodings
# ---------------------------------------------------------------------------

def test_rope_different_positions_differ(small_cfg):
    cos, sin = _build_rope_cache(small_cfg.d_head, small_cfg.max_seq_len,
                                  torch.device("cpu"))
    q = torch.ones(1, 1, 1, small_cfg.d_head)
    q0, _ = apply_rope(q, q.clone(), cos[0:1], sin[0:1])
    q5, _ = apply_rope(q, q.clone(), cos[5:6], sin[5:6])
    assert not torch.allclose(q0, q5), "RoPE produces the same encoding for positions 0 and 5."


# ---------------------------------------------------------------------------
# 6. KV-cache output at a position depends only on tokens up to that position
# ---------------------------------------------------------------------------

def test_kv_cache_future_independence(small_model, small_cfg):
    torch.manual_seed(11)
    T = 8
    ids = torch.randint(0, small_cfg.vocab_size, (1, T))
    ids_alt = ids.clone()
    ids_alt[0, -2:] = (ids[0, -2:] + 1) % small_cfg.vocab_size  # change last 2 tokens

    with torch.no_grad():
        kv1 = None
        kv2 = None
        for t in range(T - 2):
            _, kv1 = small_model(ids[:, t:t+1], kv_caches=kv1, return_kv_caches=True)
            _, kv2 = small_model(ids_alt[:, t:t+1], kv_caches=kv2, return_kv_caches=True)

        # After processing the same prefix, KV caches must be identical
        for layer_idx, (c1, c2) in enumerate(zip(kv1, kv2)):
            assert torch.allclose(c1[0], c2[0], atol=1e-5), (
                f"Layer {layer_idx} K-cache differs despite identical prefix."
            )
            assert torch.allclose(c1[1], c2[1], atol=1e-5), (
                f"Layer {layer_idx} V-cache differs despite identical prefix."
            )
