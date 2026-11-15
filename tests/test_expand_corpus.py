"""Tests for corpus v2 construction (scripts/expand_corpus.py).

The filters decide what enters the training data, so each is tested directly:
CS-term matching (including "c++"/"c#" and word boundaries), the Wikipedia and
Stack Exchange site rules, benchmark 13-gram contamination, v1-val overlap, and
the packer (exact <doc>-separated stream, SEQ_LEN-aligned shards). The pack stage
is also run end to end on small fake extract outputs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import scripts.expand_corpus as ec  # noqa: E402
from scripts.check_contamination import window_hashes  # noqa: E402


@pytest.fixture(scope="module")
def rx():
    return ec.build_cs_regex(ec.CS_TERMS)


def test_terms_match_case_insensitively_with_punctuation_terms(rx):
    distinct, total = ec.cs_score("We used C++ and C# on Linux. LINUX again; the CPU stalled.", rx)
    assert distinct == 4 and total == 5          # c++, c#, linux (x2), cpu


def test_terms_respect_word_boundaries(rx):
    # "api" inside "rapid", "java" inside "javanese", "git" inside "digital" must not count
    assert ec.cs_score("A rapid javanese digital legitimate capital", rx) == (0, 0)


def test_score_reads_only_the_start(rx):
    text = "x " * 20_000 + "compiler kernel linux"
    assert ec.cs_score(text, rx, max_chars=1000) == (0, 0)


def test_wikipedia_urls_are_excluded():
    assert not ec.keep_web_url("https://en.wikipedia.org/wiki/Compiler")
    assert not ec.keep_web_url("http://wikipedia.org/x")
    assert ec.keep_web_url("https://notwikipedia.org/x")
    assert ec.keep_web_url("https://www.geeksforgeeks.org/quicksort/")


def test_stack_exchange_site_rules():
    assert ec.keep_se_site("cs.stackexchange.com", "1")
    assert not ec.keep_se_site("cs.meta.stackexchange.com", "1")
    assert not ec.keep_se_site("cooking.stackexchange.com", "1")
    kept = sum(ec.keep_se_site("stackoverflow.com", str(i)) for i in range(6000))
    assert 800 < kept < 1200                      # about 1 in STACKOVERFLOW_KEEP_MOD
    assert ec.keep_se_site("stackoverflow.com", "42") == ec.keep_se_site("stackoverflow.com", "42")


def _cs_text(n_terms: int, words: int = 200) -> str:
    terms = ["compiler", "kernel", "linux", "mutex", "deadlock", "semaphore", "tcp", "dns"]
    return " ".join(terms[:n_terms]) + " filler" * words


def test_accept_rules(rx):
    meta = {"url": "https://example.edu/os", "int_score": 3}
    assert ec.accept("fineweb", _cs_text(6), meta, rx) == (True, "")
    assert ec.accept("fineweb", _cs_text(5), meta, rx) == (False, "not_cs")
    assert ec.accept("fineweb", _cs_text(8, words=10), meta, rx) == (False, "short")
    assert ec.accept("fineweb", _cs_text(8), {**meta, "url": "https://en.wikipedia.org/wiki/X"},
                     rx) == (False, "wikipedia")
    assert ec.accept("fineweb", _cs_text(8), {**meta, "int_score": 2}, rx) == (False, "edu_score")
    se_meta = {"site": "unix.stackexchange.com", "id": "1"}
    assert ec.accept("stackexchange", "How do I grep? " * 40, se_meta, rx) == (True, "")
    assert ec.accept("stackexchange", "short question", se_meta, rx) == (False, "short")


def test_cosmopedia_rules(rx):
    textbook = {"format": "textbook", "audience": "college_students", "prompt": ""}
    bisac = {**textbook, "prompt": 'Write a unit related to "COMPUTERS / Information Theory".'}
    plain = "Shannon entropy measures information. " * 60          # no CS terms
    assert ec.accept("cosmopedia", plain, bisac, rx) == (True, "")
    assert ec.accept("cosmopedia", plain, textbook, rx) == (False, "not_cs")
    assert ec.accept("cosmopedia", _cs_text(6), textbook, rx) == (True, "")
    assert ec.accept("cosmopedia", _cs_text(8), {**bisac, "format": "story_children"},
                     rx) == (False, "format")
    assert ec.accept("cosmopedia", _cs_text(8), {**bisac, "audience": "young_children"},
                     rx) == (False, "audience")


def test_weak_terms_cannot_carry_a_document(rx):
    weak_only = "computer software hardware programming python java api sdk " * 30
    assert ec.cs_score(weak_only, rx)[0] == ec.WEAK_CAP


def test_trie_regex_matches_a_naive_alternation(rx):
    import random, re
    words = sorted({t.lower() for t in ec.CS_TERMS})
    naive = re.compile(r"(?<![a-z0-9])(?:" + "|".join(
        re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")(?![a-z0-9])")
    rng = random.Random(0)
    vocab = words + ["the", "data", "rapid", "javanese", "c+", "tcp/", "sql;", "x86-64", "b-trees"]
    for _ in range(200):
        text = " ".join(rng.choice(vocab) for _ in range(40)).lower()
        assert sorted(rx.findall(text)) == sorted(naive.findall(text))


def test_benchmark_ngram_detection():
    bench_doc = np.arange(100, 130, dtype=np.uint16)
    bench = np.unique(window_hashes(bench_doc, 13))
    clean = np.arange(1000, 1200, dtype=np.uint16)
    dirty = np.concatenate([clean[:50], bench_doc[5:18], clean[50:]])  # one exact 13-gram
    assert not ec.has_benchmark_ngram(clean, bench)
    assert ec.has_benchmark_ngram(dirty, bench)
    assert not ec.has_benchmark_ngram(dirty[:12], bench)                # shorter than 13
    assert not ec.has_benchmark_ngram(dirty, np.empty(0, np.uint64))


def test_val_overlap_fraction():
    val = np.arange(0, 1000, dtype=np.uint16)
    val_sorted = np.unique(window_hashes(val, 13))
    copy = val[100:300]
    unrelated = np.arange(5000, 5200, dtype=np.uint16)
    half = np.concatenate([val[100:200], unrelated[:100]])
    assert ec.val_overlap_fraction(copy, val_sorted) == 1.0
    assert ec.val_overlap_fraction(unrelated, val_sorted) == 0.0
    assert 0.35 < ec.val_overlap_fraction(half, val_sorted) < 0.55


def test_text_hash_ignores_case_and_whitespace():
    assert ec.text_hash("Hello   World\n") == ec.text_hash("hello world")
    assert ec.text_hash("hello world") != ec.text_hash("hello world!")


def test_normalize_text():
    assert ec.normalize_text("a\r\nb  \r\n\x00c\n\n\n\n\nd") == "a\nb\nc\n\n\nd"


def test_pack_documents_exact_stream_and_alignment(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "TOKENS_PER_SHARD", 4096)
    monkeypatch.setattr(ec, "SEQ_LEN", 1024)
    rng = np.random.default_rng(0)
    docs = [rng.integers(0, 15000, size=int(n), dtype=np.uint16) for n in rng.integers(50, 900, 40)]
    paths, total = ec.pack_documents(docs, 9999, tmp_path, start_index=1000)
    stream = np.concatenate([np.concatenate([d, [9999]]) for d in docs]).astype(np.uint16)
    expected = stream[:(len(stream) // 1024) * 1024]
    got = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in paths])
    assert np.array_equal(got, expected) and total == len(expected)
    assert Path(paths[0]).name == "shard_1000.bin"
    assert all(np.fromfile(p, dtype=np.uint16).size == 4096 for p in paths[:-1])
    assert np.fromfile(paths[-1], dtype=np.uint16).size % 1024 == 0


@pytest.mark.skipif(not (REPO_ROOT / "tokenizer" / "vocab.json").exists(),
                    reason="tokenizer not present")
def test_pack_end_to_end_dedups_caps_and_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "TOKENS_PER_SHARD", 2048)
    monkeypatch.setattr(ec, "SEQ_LEN", 1024)
    work = tmp_path / "work"

    def fake_source(name, docs, hashes):
        d = work / name
        d.mkdir(parents=True)
        np.savez(d / "part-0000.npz", tokens=np.concatenate(docs),
                 lengths=np.array([len(x) for x in docs], dtype=np.uint32),
                 hashes=np.array(hashes, dtype=np.uint64))
        (d / "part-0000.json").write_text(json.dumps(
            {"file": "f", "docs": len(docs), "kept": len(docs),
             "tokens": int(sum(len(x) for x in docs)), "rejected": {"not_cs": 3}}))

    a_docs = [np.full(300, i, dtype=np.uint16) for i in range(1, 11)]
    fake_source("a", a_docs, [10 + i for i in range(10)])
    b_docs = [np.full(300, 50 + i, dtype=np.uint16) for i in range(10)]
    fake_source("b", b_docs, [10] + [1000 + i for i in range(9)])   # first doc duplicates a's
    args = type("A", (), {"workdir": str(work), "out_dir": str(tmp_path / "out"),
                          "sources": ["a", "b"], "caps": ["b=1500"], "start_index": 1000,
                          "seed": 0, "tok_dir": str(REPO_ROOT / "tokenizer")})
    assert ec.pack(args) == 0
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert m["sources"]["b"]["duplicates_dropped"] == 1
    b_used = m["sources"]["b"]["train_tokens"] + m["sources"]["b"]["val_new_tokens"]
    assert 1500 <= b_used <= 1800                 # cap reached, at most one doc over
    assert m["train_tokens"] % 1024 == 0 and m["train_shards"] >= 2
    assert m["sources"]["a"]["rejected"] == {"not_cs": 3}


def test_pack_cap_samples_across_the_whole_source(tmp_path, monkeypatch):
    """With a cap, kept documents must come from all files, not just the first ones."""
    monkeypatch.setattr(ec, "TOKENS_PER_SHARD", 2048)
    work = tmp_path / "work" / "se"
    work.mkdir(parents=True)
    for f in range(10):  # ten files, each with its own token value (like one site per file)
        docs = [np.full(100, 10 + f, dtype=np.uint16) for _ in range(20)]
        np.savez(work / f"part-{f:04d}.npz", tokens=np.concatenate(docs),
                 lengths=np.full(20, 100, dtype=np.uint32),
                 hashes=np.arange(f * 100, f * 100 + 20, dtype=np.uint64) + 7)
        (work / f"part-{f:04d}.json").write_text(json.dumps(
            {"file": str(f), "docs": 20, "kept": 20, "tokens": 2000, "rejected": {}}))
    if not (REPO_ROOT / "tokenizer" / "vocab.json").exists():
        pytest.skip("tokenizer not present")
    args = type("A", (), {"workdir": str(tmp_path / "work"), "out_dir": str(tmp_path / "out"),
                          "sources": ["se"], "caps": ["se=5000"], "start_index": 1000,
                          "seed": 0, "tok_dir": str(REPO_ROOT / "tokenizer")})
    assert ec.pack(args) == 0
    stream = np.concatenate([np.fromfile(p, dtype=np.uint16)
                             for p in sorted((tmp_path / "out" / "train").glob("*.bin"))])
    files_seen = {int(v) - 10 for v in np.unique(stream) if 10 <= v < 20}
    assert len(files_seen) >= 8
