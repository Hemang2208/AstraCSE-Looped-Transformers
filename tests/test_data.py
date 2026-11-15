"""Tests for the dataloader — Architecture.md §12–13, §17.1.

Verifies:
1. Shard iteration yields correct (input, target) pairs.
2. Sequences are exactly SEQ_LEN tokens.
3. Resume by shard_name (not epoch-position) works correctly even with shuffle.
4. End-of-epoch state resets to DataloaderState().
5. Missing shard directory raises FileNotFoundError.
6. count_tokens returns the correct total.
"""
from __future__ import annotations

import sys
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DataloaderState, ShardedDataset, count_tokens, SEQ_LEN


def _make_shard(path: Path, n_tokens: int, start_val: int = 0) -> None:
    arr = np.arange(start_val, start_val + n_tokens, dtype=np.uint16)
    mm = np.memmap(str(path), dtype=np.uint16, mode="w+", shape=(n_tokens,))
    mm[:] = arr
    mm.flush()


@pytest.fixture()
def shard_dir(tmp_path) -> Path:
    d = tmp_path / "shards"
    d.mkdir()
    # 3 shards, each with exactly 3 full sequences + 1 token leftover
    for i in range(3):
        _make_shard(d / f"shard_{i:04d}.bin",
                    n_tokens=(SEQ_LEN + 1) * 3 + 1,
                    start_val=i * 10_000)
    return d


# ---------------------------------------------------------------------------
# 1. Basic iteration — shape and type
# ---------------------------------------------------------------------------

def test_yields_correct_shapes(shard_dir):
    ds = ShardedDataset(shard_dir, seq_len=SEQ_LEN, shuffle=False)
    for inp, tgt in ds:
        assert inp.shape == (SEQ_LEN,)
        assert tgt.shape == (SEQ_LEN,)
        assert inp.dtype == torch.int64
        break


# ---------------------------------------------------------------------------
# 2. input/target offset by 1
# ---------------------------------------------------------------------------

def test_input_target_offset(shard_dir):
    ds = ShardedDataset(shard_dir, seq_len=SEQ_LEN, shuffle=False)
    inp, tgt = next(iter(ds))
    # target[i] == input[i+1] for all but the wrap boundary
    assert torch.all(inp[1:] == tgt[:-1])


# ---------------------------------------------------------------------------
# 3. No padding — every sequence is exactly SEQ_LEN
# ---------------------------------------------------------------------------

def test_all_sequences_full_length(shard_dir):
    ds = ShardedDataset(shard_dir, seq_len=SEQ_LEN, shuffle=False)
    for inp, tgt in ds:
        assert inp.shape[0] == SEQ_LEN
        assert tgt.shape[0] == SEQ_LEN


# ---------------------------------------------------------------------------
# 4. Resume from middle of a shard — correct continuation
# ---------------------------------------------------------------------------

def test_resume_from_shard_state(shard_dir):
    ds_full = ShardedDataset(shard_dir, seq_len=SEQ_LEN, shuffle=False)
    all_batches = list(ds_full)

    # Collect the first 2 batches, save state
    ds = ShardedDataset(shard_dir, seq_len=SEQ_LEN, shuffle=False)
    it = iter(ds)
    _ = next(it)
    _ = next(it)
    saved_state = DataloaderState(shard_name=ds.state.shard_name,
                                   offset=ds.state.offset)

    # Resume from saved state
    ds_resumed = ShardedDataset(shard_dir, seq_len=SEQ_LEN, shuffle=False,
                                 resume_state=saved_state)
    resumed_batches = list(ds_resumed)

    # Full dataset minus first 2 batches must equal resumed
    expected = all_batches[2:]
    assert len(resumed_batches) == len(expected), (
        f"Resume gave {len(resumed_batches)} batches, expected {len(expected)}"
    )
    for (ri, rt), (ei, et) in zip(resumed_batches, expected):
        assert torch.equal(ri, ei), "Resumed input_ids don't match"
        assert torch.equal(rt, et), "Resumed target_ids don't match"


# ---------------------------------------------------------------------------
# 5. Resume is stable across shuffle — shard_name not epoch-position
# ---------------------------------------------------------------------------

def test_resume_stable_across_shuffle(shard_dir):
    """With shuffle=True, resume must land on the correct shard by filename."""
    ds = ShardedDataset(shard_dir, seq_len=SEQ_LEN, shuffle=True, seed=42)
    it = iter(ds)
    first_inp, _ = next(it)
    saved_state = DataloaderState(shard_name=ds.state.shard_name,
                                   offset=ds.state.offset)

    # Resume — different shuffle seed to prove we're using name not position
    ds_resumed = ShardedDataset(shard_dir, seq_len=SEQ_LEN, shuffle=True,
                                 resume_state=saved_state, seed=99)
    resumed = list(ds_resumed)

    # All sequences from a fresh full run after the save point must appear in
    # the resumed stream (order may differ due to shuffle, but all should be present).
    # Here we verify at least one batch matches the next batch from the original run.
    second_inp, _ = next(it)
    found = any(torch.equal(r[0], second_inp) for r in resumed)
    assert found, "Resume skipped a batch that should have appeared."


# ---------------------------------------------------------------------------
# 6. End-of-epoch state reset
# ---------------------------------------------------------------------------

def test_state_resets_after_epoch(shard_dir):
    ds = ShardedDataset(shard_dir, seq_len=SEQ_LEN, shuffle=False)
    list(ds)   # exhaust the epoch
    assert ds.state.shard_name == ""
    assert ds.state.offset == 0


# ---------------------------------------------------------------------------
# 7. Missing shard directory raises
# ---------------------------------------------------------------------------

def test_missing_shard_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ShardedDataset(tmp_path / "nonexistent", seq_len=SEQ_LEN)


# ---------------------------------------------------------------------------
# 8. count_tokens
# ---------------------------------------------------------------------------

def test_count_tokens(shard_dir):
    # Each shard has (SEQ_LEN+1)*3 + 1 tokens
    expected_per_shard = (SEQ_LEN + 1) * 3 + 1
    total = count_tokens(shard_dir)
    assert total == 3 * expected_per_shard


# ---------------------------------------------------------------------------
# 9. DataloaderState round-trip serialisation
# ---------------------------------------------------------------------------

def test_dataloader_state_roundtrip():
    state = DataloaderState(shard_name="shard_0007.bin", offset=4096)
    d = state.to_dict()
    restored = DataloaderState.from_dict(d)
    assert restored.shard_name == state.shard_name
    assert restored.offset == state.offset


def test_dataloader_state_backcompat():
    # Old checkpoints stored shard_index (int) not shard_name — must not crash
    old_format = {"shard_index": 3, "offset": 2048}
    state = DataloaderState.from_dict(old_format)
    assert state.shard_name == ""
    assert state.offset == 2048
