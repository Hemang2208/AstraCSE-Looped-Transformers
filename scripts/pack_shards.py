"""Phase 1 — tokenize and pack corpus into fixed-length uint16 memmap shards.

Architecture.md §12–13:
- Insert <doc> boundary token between documents.
- Pack into sequences of exactly SEQ_LEN=1024 tokens.
- Drop the final partial sequence (no padding).
- Store as uint16 numpy memmap shards.
- Carve out a held-out validation split before packing.

Output layout:
  data/shards/train/shard_0000.bin  ...
  data/shards/val/shard_0000.bin

Dependencies: tokenizers numpy  (pip install tokenizers numpy)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    from tokenizers import ByteLevelBPETokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("ERROR: 'tokenizers' not installed. pip install tokenizers", file=sys.stderr)

SEQ_LEN = 1024
TOKENS_PER_SHARD = SEQ_LEN * 4096   # 4M tokens per shard (~8 MB at uint16)
VAL_FRACTION = 0.02                   # 2% of documents held out for validation
ENCODE_CHUNK_FILES = 512              # files read and encoded per encode_batch call
READ_WORKERS = 16                     # parallel file reads (I/O latency bound)


def load_tokenizer(tok_dir: Path) -> "ByteLevelBPETokenizer":
    vocab = str(tok_dir / "vocab.json")
    merges = str(tok_dir / "merges.txt")
    tok = ByteLevelBPETokenizer(vocab, merges)
    return tok


def verify_tokenizer_hash(tok_dir: Path) -> bool:
    version_path = tok_dir / "tokenizer_version.json"
    if not version_path.exists():
        print("WARNING: tokenizer_version.json missing — cannot verify hash.")
        return True
    version = json.loads(version_path.read_text())
    vocab_hash = hashlib.sha256((tok_dir / "vocab.json").read_bytes()).hexdigest()
    merges_hash = hashlib.sha256((tok_dir / "merges.txt").read_bytes()).hexdigest()
    if vocab_hash != version["vocab_hash"] or merges_hash != version["merges_hash"]:
        print("ERROR: tokenizer files do not match frozen version hash!", file=sys.stderr)
        return False
    print(f"Tokenizer hash verified: {version['combined_hash'][:16]}...")
    return True


def write_shard(tokens: list[int], out_path: Path) -> None:
    arr = np.array(tokens, dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(str(out_path), dtype=np.uint16, mode="w+", shape=(len(arr),))
    mm[:] = arr
    mm.flush()


def _read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        # At 1M+ files, a sporadic single-file glitch (AV scan lock,
        # filesystem race -- seen live during dedup.py's run on this
        # corpus) is survivable: skip it, don't crash hours into packing.
        print(f"  SKIP (unreadable: {e}): {path.name}")
        return None


def pack_split(
    files: list[Path],
    tok: "ByteLevelBPETokenizer",
    doc_token_id: int,
    out_dir: Path,
    split_name: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer = np.empty(0, dtype=np.uint16)
    shard_idx = 0
    total_tokens = 0
    shard_paths = []

    # Encode in chunks: tokenizers' encode_batch parallelises across all cores,
    # while encoding one document per call left the whole pass single-threaded
    # (measured ~11k tokens/s, ~20h for the full corpus). Output is identical:
    # documents are still concatenated in file order with <doc> after each.
    read_pool = ThreadPoolExecutor(max_workers=READ_WORKERS)
    for start in range(0, len(files), ENCODE_CHUNK_FILES):
        # Reads run in threads: per-file I/O latency on this machine (OneDrive /
        # Defender) dominated -- ~30 files/s serial vs ~600 files/s with 16
        # threads. map() preserves file order, so the token stream is unchanged.
        texts = [t for t in read_pool.map(_read_text_or_none,
                                          files[start:start + ENCODE_CHUNK_FILES])
                 if t is not None and t.strip()]
        if not texts:
            continue

        pieces: list[np.ndarray] = []
        for enc in tok.encode_batch(texts):
            pieces.append(np.asarray(enc.ids, dtype=np.uint16))
            pieces.append(np.array([doc_token_id], dtype=np.uint16))
        buffer = np.concatenate([buffer, *pieces])

        while len(buffer) >= TOKENS_PER_SHARD:
            chunk, buffer = buffer[:TOKENS_PER_SHARD], buffer[TOKENS_PER_SHARD:]
            # Full sequences only; TOKENS_PER_SHARD is a multiple of SEQ_LEN
            chunk = chunk[:(len(chunk) // SEQ_LEN) * SEQ_LEN]
            shard_path = out_dir / f"shard_{shard_idx:04d}.bin"
            write_shard(chunk, shard_path)
            shard_paths.append(str(shard_path))
            total_tokens += len(chunk)
            print(f"  [{split_name}] shard {shard_idx:04d}: {len(chunk):,} tokens")
            shard_idx += 1

    read_pool.shutdown()

    # Flush remaining buffer; drop any partial final sequence (Architecture.md §13)
    if len(buffer) >= SEQ_LEN:
        chunk = buffer[:(len(buffer) // SEQ_LEN) * SEQ_LEN]
        shard_path = out_dir / f"shard_{shard_idx:04d}.bin"
        write_shard(chunk, shard_path)
        shard_paths.append(str(shard_path))
        total_tokens += len(chunk)
        print(f"  [{split_name}] shard {shard_idx:04d}: {len(chunk):,} tokens (final flush)")
        shard_idx += 1

    return {"split": split_name, "shards": shard_idx, "total_tokens": total_tokens,
            "shard_paths": shard_paths}


def discover_train_val_files(
    data_dir: Path,
    extra_dirs: list[Path],
    val_fraction: float = VAL_FRACTION,
) -> tuple[list[Path], list[Path]]:
    """Find all .txt files under data_dir plus any extra_dirs (e.g.
    data/cleaned/the_stack, read verbatim when dedup.py's
    --skip-near-dup-for skipped copying it into data_dir), then split into
    train/val per subject (immediate parent dir name) to keep subjects
    balanced across the split regardless of which root they came from."""
    all_roots = [data_dir, *extra_dirs]
    all_files: list[Path] = []
    for root in all_roots:
        root_files = sorted(root.rglob("*.txt"))
        print(f"  {root}: {len(root_files):,} files")
        all_files.extend(root_files)

    subjects: dict[str, list[Path]] = {}
    for f in all_files:
        is_direct_child_of_a_root = any(f.parent == root for root in all_roots)
        subj = "root" if is_direct_child_of_a_root else f.parent.name
        subjects.setdefault(subj, []).append(f)

    train_files: list[Path] = []
    val_files: list[Path] = []
    for subj, files in sorted(subjects.items()):
        n_val = max(1, int(len(files) * val_fraction))
        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])

    return train_files, val_files


def main() -> int:
    if not TOKENIZERS_AVAILABLE:
        return 1

    ap = argparse.ArgumentParser(description="Pack deduped corpus into uint16 memmap shards.")
    ap.add_argument("--data-dir", type=Path, default=Path("data/deduped"))
    ap.add_argument("--extra-dir", type=Path, action="append", default=[],
                    help="Additional directory tree(s) to include verbatim, in "
                         "addition to --data-dir (e.g. data/cleaned/the_stack, "
                         "when dedup.py's --skip-near-dup-for was used to skip "
                         "copying that tree into --data-dir). Repeatable.")
    ap.add_argument("--tok-dir", type=Path, default=Path("tokenizer"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/shards"))
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    args = ap.parse_args()

    for d in [args.data_dir, args.tok_dir, *args.extra_dir]:
        if not d.exists():
            print(f"ERROR: {d} does not exist", file=sys.stderr)
            return 1

    if not verify_tokenizer_hash(args.tok_dir):
        return 1

    tok = load_tokenizer(args.tok_dir)

    # Resolve <doc> token id
    doc_token_id = tok.token_to_id("<doc>")
    if doc_token_id is None:
        print("ERROR: <doc> token not found in tokenizer vocab", file=sys.stderr)
        return 1

    train_files, val_files = discover_train_val_files(
        args.data_dir, args.extra_dir, args.val_fraction)
    if not train_files and not val_files:
        print(f"ERROR: no .txt files under {args.data_dir} or {args.extra_dir}",
              file=sys.stderr)
        return 1

    print(f"Files: {len(train_files)} train / {len(val_files)} val")

    train_stats = pack_split(train_files, tok, doc_token_id,
                             args.out_dir / "train", "train")
    val_stats = pack_split(val_files, tok, doc_token_id,
                           args.out_dir / "val", "val")

    manifest = {"train": train_stats, "val": val_stats,
                "seq_len": SEQ_LEN, "dtype": "uint16"}
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nPacking complete.")
    print(f"  Train: {train_stats['shards']} shards, {train_stats['total_tokens']:,} tokens")
    print(f"  Val:   {val_stats['shards']} shards,  {val_stats['total_tokens']:,} tokens")
    print(f"  Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
