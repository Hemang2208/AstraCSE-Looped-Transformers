"""Tests for the eval contamination check — Architecture.md §11.4.

Verifies:
1. Identical n-gram windows hash to the same value; different windows differ.
2. A training shard that contains a benchmark question verbatim is flagged.
3. A shard of unrelated tokens is not flagged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from check_contamination import window_hashes, main  # noqa: E402

TOK_DIR = REPO_ROOT / "tokenizer"


def test_window_hashes_are_positional_and_deterministic():
    tokens = np.array([1, 2, 3, 1, 2, 3], dtype=np.uint64)
    h = window_hashes(tokens, 3)
    assert len(h) == 4
    assert h[0] == h[3]          # (1,2,3) appears at position 0 and 3
    assert h[0] != h[1]          # (1,2,3) != (2,3,1)


def _benchmark_file(tmp_path: Path) -> Path:
    q = {
        "id": "t_001",
        "subject": "computer_programming",
        "category": "mcq",
        "question": "Which data structure gives first-in first-out order when traversed by a queue of tasks",
        "choices": ["A) stack", "B) queue", "C) heap", "D) tree"],
        "answer": "B",
    }
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"questions": [q]}), encoding="utf-8")
    return path


def _write_shard(path: Path, tokens: np.ndarray) -> None:
    tokens.astype(np.uint16).tofile(path)


@pytest.mark.skipif(not (TOK_DIR / "vocab.json").exists(), reason="tokenizer not present")
def test_shard_containing_benchmark_text_is_flagged(tmp_path, monkeypatch):
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer(str(TOK_DIR / "vocab.json"), str(TOK_DIR / "merges.txt"))
    bench = _benchmark_file(tmp_path)
    q = json.loads(bench.read_text())["questions"][0]
    text = q["question"] + " " + " ".join(q["choices"])
    ids = np.array(tok.encode(text).ids)

    shards = tmp_path / "shards"
    shards.mkdir()
    _write_shard(shards / "shard_0000.bin", np.concatenate([ids, ids]))

    monkeypatch.setattr(sys, "argv", [
        "check_contamination.py",
        "--shards-dir", str(shards),
        "--tok-dir", str(TOK_DIR),
        "--benchmark", str(bench),
    ])
    assert main() == 1


@pytest.mark.skipif(not (TOK_DIR / "vocab.json").exists(), reason="tokenizer not present")
def test_unrelated_shard_passes(tmp_path, monkeypatch):
    bench = _benchmark_file(tmp_path)
    shards = tmp_path / "shards"
    shards.mkdir()
    rng = np.random.default_rng(0)
    _write_shard(shards / "shard_0000.bin", rng.integers(0, 15000, size=5000))

    monkeypatch.setattr(sys, "argv", [
        "check_contamination.py",
        "--shards-dir", str(shards),
        "--tok-dir", str(TOK_DIR),
        "--benchmark", str(bench),
    ])
    assert main() == 0
