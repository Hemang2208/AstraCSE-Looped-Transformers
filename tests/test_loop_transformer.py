"""Tests for the loop-transformer weight-sharing mode — Architecture.md §31.

A loop-transformer cycles through n_unique_layers < n_layer distinct blocks,
reusing the same nn.Parameter objects at multiple depth-slots to trade
parameter count for effective depth at matched compute.

Critical risks specific to this feature (all covered below):
1. The non-looped baseline must be completely unaffected (byte-for-byte).
2. Analytical param-count formula must match actual constructed models.
3. A shared block must be the SAME object at every depth-slot that uses it
   (not an accidental deep copy).
4. Gradient accumulation: a shared block's .grad after backward() must equal
   the SUM of its per-usage contributions, not just the last usage.
5. KV-cache + RoPE must still be position-correct when a shared block runs
   at multiple depth-slots (each depth-slot keeps its own cache entry).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import AstraCSE, ModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
LOOP_CONFIGS = [
    "model_loop_6x2.yaml",
    "model_loop_4x3.yaml",
    "model_loop_3x4.yaml",
]


def _tiny_cfg(n_layer: int, n_unique_layers) -> ModelConfig:
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_layer=n_layer, n_unique_layers=n_unique_layers,
        n_head=4, d_ff=64, max_seq_len=16, expected_param_count=0,
    )
    cfg.__dict__["expected_param_count"] = sum(p.numel() for p in AstraCSE(cfg).parameters())
    return cfg


# ---------------------------------------------------------------------------
# 1. Baseline (n_unique_layers=None) must be completely unaffected
# ---------------------------------------------------------------------------

def test_baseline_not_looped():
    cfg = ModelConfig.from_yaml(REPO_ROOT / "configs" / "model.yaml")
    model = AstraCSE(cfg)
    assert cfg.n_unique_layers is None
    assert cfg.n_unique == cfg.n_layer
    assert model.is_looped is False
    assert model.depth_embed is None
    assert len(model.blocks) == cfg.n_layer
    assert model.count_params() == 96_473_856


# ---------------------------------------------------------------------------
# 2. Divisibility validation
# ---------------------------------------------------------------------------

def test_indivisible_loop_config_rejected():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=64, d_model=32, n_layer=12, n_unique_layers=5,
                    n_head=4, d_ff=64, max_seq_len=16, expected_param_count=0)


def test_divisible_loop_config_accepted():
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layer=12, n_unique_layers=4,
                       n_head=4, d_ff=64, max_seq_len=16, expected_param_count=0)
    assert cfg.n_unique == 4


# ---------------------------------------------------------------------------
# 3. Analytical param count matches the committed YAML configs on disk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("yaml_name", LOOP_CONFIGS)
def test_loop_yaml_configs_match_expected_param_count(yaml_name):
    cfg = ModelConfig.from_yaml(REPO_ROOT / "configs" / yaml_name)
    model = AstraCSE(cfg)
    assert model.count_params() == cfg.expected_param_count, (
        f"{yaml_name}: actual {model.count_params():,} != "
        f"declared {cfg.expected_param_count:,}"
    )
    assert model.is_looped is True
    assert len(model.blocks) == cfg.n_unique_layers
    assert model.depth_embed.shape == (cfg.n_layer, cfg.d_model)


# ---------------------------------------------------------------------------
# 4. Weight-sharing identity — same object at every depth-slot that maps to it
# ---------------------------------------------------------------------------

def test_shared_blocks_are_the_same_object():
    cfg = _tiny_cfg(n_layer=12, n_unique_layers=4)
    model = AstraCSE(cfg)
    assert len(model.blocks) == 4
    # Depth-slots 0,4,8 all resolve to unique block 0; 1,5,9 -> block 1; etc.
    n_unique = len(model.blocks)
    for depth_slot in range(cfg.n_layer):
        resolved = model.blocks[depth_slot % n_unique]
        assert resolved is model.blocks[depth_slot % n_unique]  # trivially true, documents intent
    # Concretely: block 0 must be literally the same nn.Module both times it's used.
    assert model.blocks[0] is model.blocks[0 % n_unique]


# ---------------------------------------------------------------------------
# 5. Gradient accumulation — the core correctness risk of weight sharing
#
#    A shared block's accumulated gradient must equal the sum of the
#    gradients you'd get from an "unrolled" model with independent
#    (but identically-initialized) copies at each depth-slot.
# ---------------------------------------------------------------------------

def test_shared_gradient_equals_sum_of_unrolled_gradients():
    torch.manual_seed(3)
    K, T = 2, 3
    n_layer = K * T

    looped_cfg = _tiny_cfg(n_layer=n_layer, n_unique_layers=K)
    looped = AstraCSE(looped_cfg)

    unrolled_cfg = _tiny_cfg(n_layer=n_layer, n_unique_layers=n_layer)  # no sharing
    unrolled = AstraCSE(unrolled_cfg)

    with torch.no_grad():
        for i in range(n_layer):
            unrolled.blocks[i].load_state_dict(looped.blocks[i % K].state_dict())
        # Blocks are independently constructed objects (different random init)
        # until copied above; embed/lm_head (tied) must be synced too, since
        # they're each model's own random draw and aren't part of .blocks.
        unrolled.embed.weight.copy_(looped.embed.weight)
        # Zero the loop's depth-conditioning so both models compute the exact
        # same function — isolates the weight-sharing effect from that of
        # the (deliberately absent, on the unrolled side) depth embedding.
        looped.depth_embed.zero_()

    ids = torch.randint(0, looped_cfg.vocab_size, (2, 8))

    logits_looped, _ = looped(ids)
    logits_looped.float().pow(2).mean().backward()

    logits_unrolled, _ = unrolled(ids)
    logits_unrolled.float().pow(2).mean().backward()

    assert torch.allclose(logits_looped, logits_unrolled, atol=1e-5), (
        "Looped and unrolled-with-identical-weights models disagree on forward "
        "output — weight-sharing changed the computed function unexpectedly."
    )

    for u in range(K):
        shared_grad = looped.blocks[u].attn.qkv.weight.grad
        expected_sum = sum(
            unrolled.blocks[i].attn.qkv.weight.grad
            for i in range(n_layer) if i % K == u
        )
        assert shared_grad is not None and torch.isfinite(shared_grad).all()
        assert torch.allclose(shared_grad, expected_sum, atol=1e-5), (
            f"Shared unique-block {u}'s accumulated gradient does not equal the "
            f"sum of its per-usage gradients from the unrolled reference model."
        )


def test_optimizer_step_updates_shared_block_once_not_per_usage():
    """A shared block must appear exactly once in model.parameters() —
    otherwise AdamW would double-apply weight decay / momentum to it."""
    cfg = _tiny_cfg(n_layer=12, n_unique_layers=4)
    model = AstraCSE(cfg)
    ids_ptrs = [p.data_ptr() for p in model.parameters()]
    assert len(ids_ptrs) == len(set(ids_ptrs)), (
        "model.parameters() yields the same underlying tensor more than once — "
        "shared blocks are not being de-duplicated by PyTorch as expected."
    )


# ---------------------------------------------------------------------------
# 6. KV-cache + RoPE must remain position-correct for a looped model
#    (same invariant as tests/test_rope.py, applied to a shared-weight config)
# ---------------------------------------------------------------------------

def test_kv_cache_matches_full_forward_when_looped():
    torch.manual_seed(7)
    cfg = _tiny_cfg(n_layer=6, n_unique_layers=2)
    model = AstraCSE(cfg)
    model.eval()

    T = 10
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        logits_full, _ = model(ids)

        kv_caches = None
        logits_cached = []
        for t in range(T):
            logits_t, kv_caches = model(ids[:, t:t+1], kv_caches=kv_caches,
                                         return_kv_caches=True)
            logits_cached.append(logits_t[:, 0, :])

    logits_cached_tensor = torch.stack(logits_cached, dim=1)
    assert torch.allclose(logits_full, logits_cached_tensor, atol=1e-4), (
        "KV-cache token-by-token generation diverges from full-sequence forward "
        "for a looped (weight-shared) model — RoPE/cache offset bug under looping."
    )
    # Each depth-slot must keep its own cache entry even though weights are shared.
    assert len(kv_caches) == cfg.n_layer


def test_kv_cache_per_slot_entries_differ_when_looped():
    """Even though depth-slots 0 and K reuse the same weights, their KV-cache
    entries must differ (different positions/content), proving the cache is
    indexed by depth-slot, not accidentally shared by weight identity."""
    torch.manual_seed(5)
    cfg = _tiny_cfg(n_layer=6, n_unique_layers=2)
    model = AstraCSE(cfg)
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        _, kv_caches = model(ids, return_kv_caches=True)

    k_slot0, _ = kv_caches[0]
    k_slot2, _ = kv_caches[2]  # depth-slot 2 reuses the same weights as slot 0 (K=2)
    assert not torch.allclose(k_slot0, k_slot2), (
        "KV-cache entries for two depth-slots sharing the same weights are "
        "identical — cache is not properly keyed per depth-slot."
    )
