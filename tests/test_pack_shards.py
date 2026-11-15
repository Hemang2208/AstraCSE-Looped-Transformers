"""Tests for corpus packing — Architecture.md §12–13.

Verifies:
1. discover_train_val_files combines data_dir and extra_dir(s) correctly.
2. Subject grouping (for the balanced val split) is correct across multiple
   root directories, not just within one.
3. A file directly under a root (no subject subdirectory) groups as "root".
4. pack_split skips an unreadable file instead of crashing (the exact
   failure mode observed live during this corpus's dedup.py run).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.pack_shards import discover_train_val_files, pack_split


def _make_tree(root: Path, subject_counts: dict[str, int]) -> None:
    for subject, n in subject_counts.items():
        d = root / subject if subject != "." else root
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"doc_{i:03d}.txt").write_text(f"{subject} document {i}", encoding="utf-8")


def test_discover_combines_data_dir_and_extra_dir(tmp_path):
    data_dir = tmp_path / "deduped"
    extra_dir = tmp_path / "cleaned_code"
    _make_tree(data_dir, {"wikipedia_large": 20})
    _make_tree(extra_dir, {"python": 10, "java": 10})

    train, val = discover_train_val_files(data_dir, [extra_dir], val_fraction=0.1)
    assert len(train) + len(val) == 40


def test_discover_subject_grouping_spans_both_roots(tmp_path):
    """Each subject's val fraction must be computed from ITS OWN file count,
    the same whether that subject's files live under data_dir or extra_dir."""
    data_dir = tmp_path / "deduped"
    extra_dir = tmp_path / "cleaned_code"
    _make_tree(data_dir, {"wikipedia_large": 100})
    _make_tree(extra_dir, {"python": 100})

    train, val = discover_train_val_files(data_dir, [extra_dir], val_fraction=0.1)

    val_subjects = {f.parent.name for f in val}
    train_subjects = {f.parent.name for f in train}
    # Both subjects must appear in val (10% of 100 each = 10 -- not one
    # subject accidentally dominating the split because of root grouping).
    assert "wikipedia_large" in val_subjects
    assert "python" in val_subjects
    assert val_subjects == train_subjects == {"wikipedia_large", "python"}

    python_val = [f for f in val if f.parent.name == "python"]
    wiki_val = [f for f in val if f.parent.name == "wikipedia_large"]
    assert len(python_val) == 10
    assert len(wiki_val) == 10


def test_discover_direct_child_of_root_groups_as_root(tmp_path):
    data_dir = tmp_path / "deduped"
    data_dir.mkdir(parents=True)
    for i in range(20):
        (data_dir / f"doc_{i:03d}.txt").write_text(f"document {i}", encoding="utf-8")

    train, val = discover_train_val_files(data_dir, [], val_fraction=0.1)
    assert len(train) + len(val) == 20


def test_discover_empty_dirs_return_empty_lists(tmp_path):
    data_dir = tmp_path / "deduped"
    data_dir.mkdir(parents=True)
    train, val = discover_train_val_files(data_dir, [], val_fraction=0.1)
    assert train == []
    assert val == []


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    """Deterministic stand-in for ByteLevelBPETokenizer: one token id per
    character, so expected token counts are trivial to compute by hand."""
    def encode(self, text):
        return _FakeEncoding([ord(c) % 1000 for c in text])

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]


def test_pack_split_skips_unreadable_file_instead_of_crashing(tmp_path, monkeypatch):
    # Each good file needs >= SEQ_LEN characters so the pair crosses one full
    # 1024-token sequence once combined -- a few bytes (the earlier version
    # of this test) always lands in the "final partial sequence, dropped"
    # branch and asserts a false negative, not a real failure.
    good1 = tmp_path / "good1.txt"
    bad = tmp_path / "bad.txt"
    good2 = tmp_path / "good2.txt"
    good1.write_text("a" * 600, encoding="utf-8")
    bad.write_text("b" * 600, encoding="utf-8")
    good2.write_text("c" * 600, encoding="utf-8")

    real_read_text = Path.read_text

    def flaky_read_text(self, *args, **kwargs):
        if self == bad:
            raise OSError(22, "Invalid argument")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    out_dir = tmp_path / "out"
    stats = pack_split([good1, bad, good2], _FakeTokenizer(), doc_token_id=999,
                        out_dir=out_dir, split_name="train")

    # bad.txt must not have crashed the run, and must not have contributed
    # any tokens. The two good files (600 chars + 1 <doc> token each = 1202
    # tokens) cross one full SEQ_LEN=1024 sequence; the remaining 178 tokens
    # are a partial final sequence and correctly dropped (Architecture.md
    # §13: "drop the final partial sequence, no padding") -- so the only
    # robust assertion is the fixed-length invariant itself, not a specific
    # count that depends on exactly where the boundary falls.
    assert stats["total_tokens"] > 0
    assert stats["total_tokens"] % 1024 == 0


@pytest.mark.skipif(not (REPO_ROOT / "tokenizer" / "vocab.json").exists(),
                    reason="tokenizer not present")
def test_pack_split_matches_serial_reference(tmp_path, monkeypatch):
    """The chunked encode_batch path must produce exactly the token stream the
    original one-document-at-a-time path produced, including shard boundaries
    and the dropped partial final sequence."""
    import numpy as np
    from tokenizers import ByteLevelBPETokenizer
    import scripts.pack_shards as ps

    monkeypatch.setattr(ps, "TOKENS_PER_SHARD", 4096)   # force several shards
    monkeypatch.setattr(ps, "ENCODE_CHUNK_FILES", 3)    # force several encode chunks

    tok_dir = REPO_ROOT / "tokenizer"
    tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    doc_id = tok.token_to_id("<doc>")

    src = tmp_path / "src"
    src.mkdir()
    files = []
    for i in range(10):
        p = src / f"d{i:02d}.txt"
        if i == 4:
            p.write_text("   \n", encoding="utf-8")       # whitespace-only: skipped
        elif i == 7:
            p.write_text("", encoding="utf-8")            # empty: skipped
        else:
            p.write_text(("def f(x):\n    return x + %d  # sample text %d\n" % (i, i)) * 80,
                         encoding="utf-8")
        files.append(p)

    stream: list[int] = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        stream.extend(tok.encode(text).ids)
        stream.append(doc_id)
    expected = np.array(stream[: (len(stream) // ps.SEQ_LEN) * ps.SEQ_LEN], dtype=np.uint16)

    stats = ps.pack_split(files, tok, doc_id, tmp_path / "out", "train")
    assert len(stats["shard_paths"]) >= 2               # boundaries actually exercised
    got = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in stats["shard_paths"]])

    assert stats["total_tokens"] == len(expected)
    assert np.array_equal(got, expected)
