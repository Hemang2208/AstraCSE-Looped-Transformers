"""Phase 1 — eval-set contamination check.

Architecture.md §11.4: n-gram overlap check between the packed training
shards and the benchmark question sets.

Memory-bounded approach: instead of holding every training n-gram in a set
(infeasible for ~1B tokens), this builds the n-gram hash set of the benchmark
questions (a few tens of thousands of entries) and scans each shard once,
checking every window against it with vectorised numpy operations.

Each n-gram is hashed to 64 bits (polynomial rolling hash, wraparound). A
collision false-positive rate is ~(benchmark n-grams) / 2^64 per training
window — negligible at this scale.

Usage:
    python scripts/check_contamination.py \
        --shards-dir C:/astra_data/shards/train \
        --tok-dir tokenizer \
        --benchmark eval/benchmark.json eval/mmlu_cs_benchmark.json

Exits 0 if no benchmark item exceeds the overlap threshold, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from tokenizers import ByteLevelBPETokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

NGRAM_SIZE = 13          # 13-gram overlap is a common contamination signal
MAX_OVERLAP_FRACTION = 0.01   # Flag a benchmark item if >1% of its 13-grams hit training
HASH_BASE = np.uint64(1_000_003)


def load_tokenizer(tok_dir: Path) -> "ByteLevelBPETokenizer":
    return ByteLevelBPETokenizer(
        str(tok_dir / "vocab.json"),
        str(tok_dir / "merges.txt"),
    )


def window_hashes(tokens: np.ndarray, n: int) -> np.ndarray:
    """64-bit rolling hash of every length-n window of `tokens`."""
    arr = tokens.astype(np.uint64)
    n_windows = len(arr) - n + 1
    if n_windows <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(n_windows, dtype=np.uint64)
    for j in range(n):
        h = h * HASH_BASE + arr[j:j + n_windows]
    return h


def benchmark_item_hashes(benchmark_paths: list[Path], tok: "ByteLevelBPETokenizer",
                          n: int) -> list[dict]:
    """Return one record per benchmark item with its unique n-gram hashes."""
    items: list[dict] = []
    for bp in benchmark_paths:
        questions = json.loads(bp.read_text(encoding="utf-8")).get("questions", [])
        for q in questions:
            text = q.get("question", "") + " " + " ".join(q.get("choices", []))
            ids = np.array(tok.encode(text).ids, dtype=np.uint64)
            hashes = np.unique(window_hashes(ids, n))
            if len(hashes) == 0:
                continue
            items.append({
                "id": q.get("id", "?"),
                "subject": q.get("subject", "?"),
                "source": bp.name,
                "hashes": hashes,
            })
    return items


def scan_shards(shards_dir: Path, bench_hashes: np.ndarray, n: int) -> set[int]:
    """Return the set of benchmark n-gram hashes that occur in the shards."""
    shard_files = sorted(shards_dir.rglob("*.bin"))
    if not shard_files:
        raise FileNotFoundError(f"No .bin shards under {shards_dir}")

    found: set[int] = set()
    for i, sf in enumerate(shard_files):
        arr = np.memmap(str(sf), dtype=np.uint16, mode="r")
        hashes = window_hashes(np.asarray(arr), n)
        mask = np.isin(hashes, bench_hashes)
        if mask.any():
            found.update(int(h) for h in np.unique(hashes[mask]))
        if (i + 1) % 25 == 0 or i + 1 == len(shard_files):
            print(f"  scanned {i + 1}/{len(shard_files)} shards, "
                  f"{len(found):,} benchmark n-grams seen so far")
    return found


def main() -> int:
    if not TOKENIZERS_AVAILABLE:
        print("ERROR: tokenizers not installed", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description="Eval contamination check.")
    ap.add_argument("--shards-dir", type=Path, default=Path("data/shards/train"))
    ap.add_argument("--tok-dir", type=Path, default=Path("tokenizer"))
    ap.add_argument("--benchmark", type=Path, nargs="+",
                    default=[Path("eval/benchmark.json"), Path("eval/mmlu_cs_benchmark.json")])
    ap.add_argument("--ngram-size", type=int, default=NGRAM_SIZE)
    ap.add_argument("--threshold", type=float, default=MAX_OVERLAP_FRACTION)
    args = ap.parse_args()

    for p in [args.shards_dir, args.tok_dir, *args.benchmark]:
        if not p.exists():
            print(f"ERROR: {p} does not exist", file=sys.stderr)
            return 1

    tok = load_tokenizer(args.tok_dir)
    items = benchmark_item_hashes(args.benchmark, tok, args.ngram_size)
    if not items:
        print("ERROR: no benchmark items with enough tokens to check", file=sys.stderr)
        return 1

    all_bench = np.unique(np.concatenate([it["hashes"] for it in items]))
    print(f"{len(items)} benchmark items, {len(all_bench):,} unique {args.ngram_size}-grams to check")
    print(f"Scanning training shards in {args.shards_dir} ...")
    found = scan_shards(args.shards_dir, all_bench, args.ngram_size)
    found_arr = np.fromiter(found, dtype=np.uint64, count=len(found))

    contaminated = []
    for it in items:
        hit = np.isin(it["hashes"], found_arr).sum()
        overlap = hit / len(it["hashes"])
        if overlap > args.threshold:
            contaminated.append({"id": it["id"], "subject": it["subject"],
                                 "source": it["source"], "overlap": overlap})

    if contaminated:
        print(f"FAIL: {len(contaminated)} benchmark items contaminated (>{args.threshold:.0%} overlap):")
        for c in contaminated:
            print(f"  [{c['source']}] id={c['id']}  subject={c['subject']}  overlap={c['overlap']:.2%}")
        return 1

    print(f"PASS: 0 / {len(items)} benchmark items show >{args.threshold:.0%} "
          f"{args.ngram_size}-gram overlap with the training shards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
