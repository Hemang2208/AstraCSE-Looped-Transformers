"""Tests for training utilities — Architecture.md §16–21.

Covers: LR schedule, checkpoint save/load/corruption, NaN guard,
RNG state restore, GradScaler removed, config loading.
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import AstraCSE, ModelConfig
from src.train import (
    get_lr,
    save_checkpoint,
    load_checkpoint,
    _ckpt_hash,
    eval_loss,
)
from src.data import DataloaderState

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return ModelConfig.from_yaml(REPO_ROOT / "configs" / "model.yaml")


@pytest.fixture()
def tiny_model_and_opt(cfg):
    model = AstraCSE(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


# ---------------------------------------------------------------------------
# LR Schedule (§16.2)
# ---------------------------------------------------------------------------

def test_lr_warmup_starts_at_zero():
    lr = get_lr(step=0, warmup_steps=100, total_steps=1000,
                peak_lr=6e-4, min_lr=6e-5)
    assert lr == 0.0


def test_lr_at_warmup_end_equals_peak():
    peak = 6e-4
    lr = get_lr(step=100, warmup_steps=100, total_steps=1000,
                peak_lr=peak, min_lr=6e-5)
    assert abs(lr - peak) < 1e-10


def test_lr_at_total_steps_equals_min():
    min_lr = 6e-5
    lr = get_lr(step=1000, warmup_steps=100, total_steps=1000,
                peak_lr=6e-4, min_lr=min_lr)
    assert lr == min_lr


def test_lr_beyond_total_steps_clamped():
    min_lr = 6e-5
    lr = get_lr(step=9999, warmup_steps=100, total_steps=1000,
                peak_lr=6e-4, min_lr=min_lr)
    assert lr == min_lr


def test_lr_is_monotone_decreasing_after_warmup():
    lrs = [get_lr(s, 10, 100, 6e-4, 6e-5) for s in range(10, 101)]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i+1], f"LR not monotone at step {10+i}"


def test_lr_midpoint_close_to_half_range():
    peak, minimum = 6e-4, 6e-5
    mid_step = 550  # midpoint of [100, 1000]
    lr = get_lr(mid_step, 100, 1000, peak, minimum)
    # cosine midpoint ≈ (peak+min)/2
    expected = minimum + 0.5 * (peak - minimum)
    assert abs(lr - expected) < 1e-6


# ---------------------------------------------------------------------------
# Checkpoint save / load / integrity (§17)
# ---------------------------------------------------------------------------

def test_checkpoint_save_creates_files(cfg, tmp_path, tiny_model_and_opt):
    model, opt = tiny_model_and_opt
    dl_state = DataloaderState(shard_name="shard_0000.bin", offset=0)
    path = save_checkpoint(42, model, opt, {}, dl_state, "cfghash", "tokhash", tmp_path)
    assert path.exists()
    assert path.with_suffix(".sha256").exists()


def test_checkpoint_hash_file_matches(cfg, tmp_path, tiny_model_and_opt):
    model, opt = tiny_model_and_opt
    dl_state = DataloaderState()
    path = save_checkpoint(1, model, opt, {}, dl_state, "h1", "h2", tmp_path)
    stored = path.with_suffix(".sha256").read_text().strip()
    computed = _ckpt_hash(path)
    assert stored == computed


def test_checkpoint_load_restores_step(cfg, tmp_path, tiny_model_and_opt):
    model, opt = tiny_model_and_opt
    dl_state = DataloaderState()
    save_checkpoint(99, model, opt, {"best_val_loss": 1.5}, dl_state,
                    "ch", "th", tmp_path)
    ckpt_path = tmp_path / "step-0000099.pt"
    payload = load_checkpoint(ckpt_path, model, opt)
    assert payload["global_step"] == 99


def test_checkpoint_corruption_raises(cfg, tmp_path, tiny_model_and_opt):
    model, opt = tiny_model_and_opt
    dl_state = DataloaderState()
    path = save_checkpoint(5, model, opt, {}, dl_state, "c", "t", tmp_path)
    # Corrupt the file
    data = path.read_bytes()
    path.write_bytes(data[:-10] + b"\x00" * 10)
    with pytest.raises(RuntimeError, match="corruption"):
        load_checkpoint(path, model, opt)


def test_checkpoint_no_sha256_file_loads_without_check(cfg, tmp_path, tiny_model_and_opt):
    model, opt = tiny_model_and_opt
    dl_state = DataloaderState()
    path = save_checkpoint(3, model, opt, {}, dl_state, "c", "t", tmp_path)
    sha_path = path.with_suffix(".sha256")
    sha_path.unlink()
    # Should load without error (no hash file → skip integrity check)
    payload = load_checkpoint(path, model, opt)
    assert payload["global_step"] == 3


def test_checkpoint_rng_state_restored(cfg, tmp_path, tiny_model_and_opt):
    model, opt = tiny_model_and_opt
    torch.manual_seed(12345)
    dl_state = DataloaderState()
    path = save_checkpoint(7, model, opt, {}, dl_state, "c", "t", tmp_path)

    # Advance RNG so it diverges
    _ = torch.randn(100)
    load_checkpoint(path, model, opt)

    # After restore, next random draw should match the saved state
    r1 = torch.randn(5).tolist()
    # Re-restore and draw again
    load_checkpoint(path, model, opt)
    r2 = torch.randn(5).tolist()
    assert r1 == r2, "RNG state not correctly restored from checkpoint."


def test_checkpoint_dataloader_state_persisted(cfg, tmp_path, tiny_model_and_opt):
    model, opt = tiny_model_and_opt
    dl_state = DataloaderState(shard_name="shard_0003.bin", offset=8192)
    path = save_checkpoint(10, model, opt, {}, dl_state, "c", "t", tmp_path)
    payload = load_checkpoint(path, model, opt)
    restored = DataloaderState.from_dict(payload["dataloader_state"])
    assert restored.shard_name == "shard_0003.bin"
    assert restored.offset == 8192


# ---------------------------------------------------------------------------
# NaN/Inf guard (§21.1) — train_step returns None on non-finite loss
# ---------------------------------------------------------------------------

def test_train_step_returns_none_on_nan(cfg, tmp_path):
    from src.train import train_step
    device = torch.device("cpu")
    model = AstraCSE(cfg)
    model.train()

    # Force NaN by setting all weights to NaN
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))

    inp = torch.randint(0, cfg.vocab_size, (1, 16))
    tgt = torch.randint(0, cfg.vocab_size, (1, 16))
    result = train_step(model, inp, tgt, device, accum_steps=1)
    assert result is None, "train_step should return None when loss is NaN."


# ---------------------------------------------------------------------------
# No GradScaler in train_step signature (§15.2 — BF16 doesn't need it)
# ---------------------------------------------------------------------------

def test_train_step_has_no_grad_scaler_param():
    import inspect
    from src.train import train_step
    params = inspect.signature(train_step).parameters
    assert "scaler" not in params, (
        "train_step should not accept a GradScaler (BF16 doesn't need it)."
    )


# ---------------------------------------------------------------------------
# eval_loss runs without error on a trivial dataset
# ---------------------------------------------------------------------------

def test_eval_loss_basic(cfg, tmp_path):
    import numpy as np
    from src.data import SEQ_LEN, make_dataloader

    shard_dir = tmp_path / "val"
    shard_dir.mkdir()
    tokens = np.arange(0, (SEQ_LEN + 1) * 4, dtype=np.uint16) % cfg.vocab_size
    mm = np.memmap(str(shard_dir / "shard_0000.bin"), dtype=np.uint16,
                   mode="w+", shape=(len(tokens),))
    mm[:] = tokens
    mm.flush()

    device = torch.device("cpu")
    model = AstraCSE(cfg)
    model.eval()
    val_loader, _ = make_dataloader(shard_dir, batch_size=2, shuffle=False)
    loss = eval_loss(model, val_loader, device, max_batches=2)
    assert math.isfinite(loss) and loss > 0


# ---------------------------------------------------------------------------
# Config loading — missing expected_param_count falls back to default
# ---------------------------------------------------------------------------

def test_config_missing_expected_param_count_uses_default(tmp_path):
    import yaml
    cfg_data = {
        "model": {
            "vocab_size": 15000, "d_model": 768, "n_layer": 12,
            "n_head": 12, "d_ff": 2048, "max_seq_len": 1024,
            # no expected_param_count key
        }
    }
    p = tmp_path / "model.yaml"
    p.write_text(yaml.dump(cfg_data), encoding="utf-8")
    cfg = ModelConfig.from_yaml(p)
    assert cfg.expected_param_count == 96_473_856
