"""Phase 2 sanity tests — Architecture.md §22.1.

Tests:
1. Parameter count assertion: must equal 96,473,856 exactly.
2. Output shapes correct for forward pass.
3. Causal mask: token i cannot attend to token j > i.
4. Checkpoint round-trip: save → load → identical output.
5. Overfit-a-tiny-batch: loss approaches 0 on a single repeated batch (CPU).

Run with: pytest tests/test_model.py -v
"""

import copy
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

from src.model import AstraCSE, ModelConfig, load_model

CFG_PATH = REPO_ROOT / "configs" / "model.yaml"


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return ModelConfig.from_yaml(CFG_PATH)


@pytest.fixture(scope="module")
def model(cfg) -> AstraCSE:
    m = AstraCSE(cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# 1. Parameter count
# ---------------------------------------------------------------------------

def test_param_count(model):
    actual = model.count_params()
    assert actual == 96_473_856, \
        f"Parameter count {actual:,} != expected 96,473,856 (Architecture.md §3)"


# ---------------------------------------------------------------------------
# 2. Output shapes
# ---------------------------------------------------------------------------

def test_forward_shape(model, cfg):
    B, T = 2, 64
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape == (B, T, cfg.vocab_size), \
        f"Expected logits shape ({B}, {T}, {cfg.vocab_size}), got {logits.shape}"


def test_max_seq_len_accepted(model, cfg):
    ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape[1] == cfg.max_seq_len


def test_over_max_seq_len_rejected(model, cfg):
    ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len + 1))
    with pytest.raises(AssertionError):
        model(ids)


# ---------------------------------------------------------------------------
# 3. Causal mask — token i must not attend to future tokens
# ---------------------------------------------------------------------------

def test_causal_mask(cfg):
    model = AstraCSE(cfg)
    model.eval()
    T = 8
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        logits_full, _ = model(ids)

    # Mutate only the last token's embedding, rerun: only position T-1 should change
    ids_alt = ids.clone()
    ids_alt[0, -1] = (ids[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_alt, _ = model(ids_alt)

    # Positions 0..T-2 must be identical (causal masking)
    assert torch.allclose(logits_full[0, :T-1], logits_alt[0, :T-1], atol=1e-5), \
        "Causal mask broken: early positions changed when only a future token was modified."

    # Position T-1 must differ
    assert not torch.allclose(logits_full[0, T-1], logits_alt[0, T-1], atol=1e-5), \
        "Last-position logits unchanged after modifying last input token — unexpected."


# ---------------------------------------------------------------------------
# 4. Checkpoint round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(cfg, tmp_path):
    model = AstraCSE(cfg)
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits_before, _ = model(ids)

    ckpt_path = tmp_path / "test_checkpoint.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)

    model2 = AstraCSE(cfg)
    payload = torch.load(ckpt_path, map_location="cpu")
    model2.load_state_dict(payload["model_state"])
    model2.eval()

    with torch.no_grad():
        logits_after, _ = model2(ids)

    assert torch.allclose(logits_before, logits_after, atol=1e-6), \
        "Checkpoint round-trip produced different outputs."


# ---------------------------------------------------------------------------
# 5. Overfit-a-tiny-batch (Architecture.md §22.1)
# ---------------------------------------------------------------------------

def test_overfit_tiny_batch(cfg):
    torch.manual_seed(42)
    # Use a tiny config to make this CPU-fast
    tiny_cfg = ModelConfig(
        vocab_size=cfg.vocab_size,
        d_model=64,
        n_layer=2,
        n_head=4,
        d_ff=128,
        max_seq_len=cfg.max_seq_len,
        norm_eps=cfg.norm_eps,
        weight_tying=True,
        expected_param_count=0,  # skip assertion for tiny test model
    )
    # Patch expected count to avoid assertion
    tiny_cfg.__dict__["expected_param_count"] = sum(
        p.numel() for p in AstraCSE(tiny_cfg).parameters()
    )
    model = AstraCSE(tiny_cfg)
    model.train()

    ids = torch.randint(0, tiny_cfg.vocab_size, (4, 32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    initial_loss = None
    for step in range(300):
        optimizer.zero_grad()
        logits, _ = model(ids)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, tiny_cfg.vocab_size),
            ids[:, 1:].reshape(-1),
        )
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    # Loss should drop significantly from ln(vocab_size) ≈ 9.62
    assert final_loss < initial_loss * 0.1, \
        f"Tiny-batch overfit failed: initial={initial_loss:.3f}, final={final_loss:.3f} " \
        f"(expected <{initial_loss * 0.1:.3f})"


# ---------------------------------------------------------------------------
# 6. Weight tying: embed and lm_head share the same tensor
# ---------------------------------------------------------------------------

def test_weight_tying(model):
    assert model.embed.weight.data_ptr() == model.lm_head.weight.data_ptr(), \
        "embed.weight and lm_head.weight are not tied (Architecture.md §3.2)"


# ---------------------------------------------------------------------------
# 7. Initial loss near ln(vocab_size) = 9.615...
# ---------------------------------------------------------------------------

def test_initial_loss_near_entropy(cfg):
    torch.manual_seed(0)
    model = AstraCSE(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 64))
    with torch.no_grad():
        logits, _ = model(ids)
    loss = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        ids[:, 1:].reshape(-1),
    )
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 1.0, \
        f"Initial loss {loss.item():.3f} far from ln({cfg.vocab_size})={expected:.3f} " \
        f"— initialization may be wrong (Architecture.md §3.3)"
