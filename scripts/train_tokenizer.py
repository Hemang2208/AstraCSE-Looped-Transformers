"""Phase 1 — train byte-level BPE tokenizer on the cleaned corpus.

Architecture.md §10:
- Byte-level BPE, vocab_size=15000
- Special tokens: <bos>, <eos>, <pad>, <doc>
- Trained on the actual domain corpus (not a generic corpus)
- Preserves leading whitespace (code indentation)
- Freeze by writing tokenizer + SHA-256 hash to tokenizer/

Dependencies: tokenizers  (pip install tokenizers)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from tokenizers import ByteLevelBPETokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("ERROR: 'tokenizers' library not installed.", file=sys.stderr)
    print("Install with: pip install tokenizers", file=sys.stderr)

VOCAB_SIZE = 15_000
SPECIAL_TOKENS = ["<bos>", "<eos>", "<pad>", "<doc>"]
MIN_FREQUENCY = 2


def collect_text_files(root: Path, sample_per_dir: int = 0) -> list[Path]:
    paths = sorted(root.rglob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found under {root}")
    if sample_per_dir:
        by_dir: dict[Path, list[Path]] = {}
        for p in paths:
            by_dir.setdefault(p.parent, []).append(p)
        sampled: list[Path] = []
        for d in sorted(by_dir):
            ps = by_dir[d]
            if len(ps) <= sample_per_dir:
                sampled.extend(ps)
            else:
                step = len(ps) / sample_per_dir
                sampled.extend(ps[int(i * step)] for i in range(sample_per_dir))
        paths = sampled
    return paths


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    if not TOKENIZERS_AVAILABLE:
        return 1

    ap = argparse.ArgumentParser(description="Train byte-level BPE tokenizer.")
    ap.add_argument("--data-dir", type=Path, default=Path("data/deduped"),
                    help="Root of cleaned/deduped text files")
    ap.add_argument("--out-dir", type=Path, default=Path("tokenizer"),
                    help="Output directory for tokenizer artifacts")
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--sample-per-dir", type=int, default=0,
                    help="Train on at most N evenly-spaced files per directory "
                         "(0 = all). Keeps memory bounded on a multi-GB corpus.")
    args = ap.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: {args.data_dir} does not exist", file=sys.stderr)
        return 1

    files = collect_text_files(args.data_dir, args.sample_per_dir)
    print(f"Training on {len(files)} files from {args.data_dir}")

    tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
    tokenizer.train(
        files=[str(p) for p in files],
        vocab_size=args.vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(args.out_dir))
    print(f"Tokenizer saved to {args.out_dir}/")

    # Verify bytes-per-token per subject domain
    print("\nBytes-per-token per subject domain (sample):")
    subject_stats: dict[str, dict] = {}
    for f in files:
        subject = f.parent.name if f.parent != args.data_dir else "root"
        text_sample = f.read_text(encoding="utf-8", errors="replace")[:50_000]
        if not text_sample.strip():
            continue
        encoded = tokenizer.encode(text_sample)
        bpt = len(text_sample.encode("utf-8")) / max(len(encoded.ids), 1)
        if subject not in subject_stats:
            subject_stats[subject] = {"bytes": 0, "tokens": 0}
        subject_stats[subject]["bytes"] += len(text_sample.encode("utf-8"))
        subject_stats[subject]["tokens"] += len(encoded.ids)

    stats_rows = []
    for subj, s in sorted(subject_stats.items()):
        bpt = s["bytes"] / max(s["tokens"], 1)
        print(f"  {subj:<30s}  {bpt:.2f} bytes/token")
        stats_rows.append({"subject": subj, "bytes": s["bytes"],
                            "tokens": s["tokens"], "bytes_per_token": round(bpt, 3)})

    # Write stats log
    stats_path = args.out_dir / "bytes_per_token.json"
    stats_path.write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")
    print(f"\nBytes-per-token stats written to {stats_path}")

    # Hash the vocab + merges files and write a version file
    vocab_path = args.out_dir / "vocab.json"
    merges_path = args.out_dir / "merges.txt"
    if vocab_path.exists() and merges_path.exists():
        combined = hash_file(vocab_path) + hash_file(merges_path)
        tokenizer_hash = hashlib.sha256(combined.encode()).hexdigest()
        version = {
            "vocab_size": args.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "vocab_hash": hash_file(vocab_path),
            "merges_hash": hash_file(merges_path),
            "combined_hash": tokenizer_hash,
        }
        version_path = args.out_dir / "tokenizer_version.json"
        version_path.write_text(json.dumps(version, indent=2), encoding="utf-8")
        print(f"\nTokenizer frozen. Hash: {tokenizer_hash[:16]}...")
        print(f"Version file: {version_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
