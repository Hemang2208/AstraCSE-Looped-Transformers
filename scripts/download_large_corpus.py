"""Phase 0 (extended) — Download large-scale corpus from HuggingFace.

Target: ~1.3B tokens from 3 sources:
  1. Wikipedia English (CS-filtered)   ~500M tokens
  2. ArXiv CS papers (9 categories)    ~300M tokens
  3. The Stack (Python/C/Java/C++)     ~500M tokens

All sources are openly licensed and approved in data/SOURCES.md.

Key design choices:
  - Streaming mode: never loads full dataset into RAM.
  - Resume: saves a progress checkpoint every CHECKPOINT_EVERY docs
    so a killed run continues from where it left off.
  - Disk space guard: checks available space before writing each batch.
  - Token estimation: counts tokens on-the-fly; stops at TARGET_TOKENS.
  - Rate-limit safe: respects HuggingFace download limits.
  - All text written as plain UTF-8 .txt files into data/raw/<source>/
    so existing clean_text.py + dedup.py pipeline handles them.

Dependencies:
    pip install datasets huggingface_hub tqdm

Usage:
    python scripts/download_large_corpus.py
    python scripts/download_large_corpus.py --sources wikipedia arxiv
    python scripts/download_large_corpus.py --target-tokens 500_000_000
    python scripts/download_large_corpus.py --resume          # continue from checkpoint
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from tqdm import tqdm

RAW_DIR = Path("data/raw")
CHECKPOINT_FILE = Path("data/.download_checkpoint.json")

# Stop when this many tokens are estimated (rough: 1 token ≈ 4 bytes)
TARGET_TOKENS = 1_500_000_000        # 1.5B tokens target
BYTES_PER_TOKEN_ESTIMATE = 4.0

# Flush to disk every N documents
CHECKPOINT_EVERY = 5_000

# Minimum free disk space required at all times (bytes)
MIN_FREE_DISK_BYTES = 5 * 1024 ** 3  # 5 GB headroom

# Max file size per output file (split large sources into chunks)
MAX_FILE_BYTES = 50 * 1024 * 1024    # 50 MB per file


# ---------------------------------------------------------------------------
# Disk / progress helpers
# ---------------------------------------------------------------------------

def check_disk_space(path: Path = Path(".")) -> int:
    """Return free bytes on the disk containing `path`."""
    return shutil.disk_usage(path).free


def estimate_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // int(BYTES_PER_TOKEN_ESTIMATE))


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Text writer — splits into MAX_FILE_BYTES chunks
# ---------------------------------------------------------------------------

class ChunkWriter:
    """Writes text to a directory, splitting into fixed-size files."""

    def __init__(self, out_dir: Path, prefix: str = "chunk"):
        self.out_dir = out_dir
        self.prefix = prefix
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_idx = self._find_next_chunk()
        self._current_file = None
        self._current_bytes = 0

    def _find_next_chunk(self) -> int:
        existing = sorted(self.out_dir.glob(f"{self.prefix}_*.txt"))
        if not existing:
            return 0
        last = existing[-1]
        m = re.search(r"_(\d+)\.txt$", last.name)
        return (int(m.group(1)) + 1) if m else 0

    def _open_next(self) -> None:
        if self._current_file:
            self._current_file.close()
        path = self.out_dir / f"{self.prefix}_{self._chunk_idx:05d}.txt"
        self._current_file = open(path, "a", encoding="utf-8")
        self._chunk_idx += 1
        self._current_bytes = path.stat().st_size if path.exists() else 0

    def write(self, text: str) -> None:
        encoded = text.encode("utf-8")
        if self._current_file is None or self._current_bytes + len(encoded) > MAX_FILE_BYTES:
            self._open_next()
        self._current_file.write(text + "\n\n")
        self._current_bytes += len(encoded) + 2

    def close(self) -> None:
        if self._current_file:
            self._current_file.close()
            self._current_file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# 1. Wikipedia — CS-filtered
# ---------------------------------------------------------------------------

WIKIPEDIA_CS_CATEGORIES = {
    # Must appear in title or first 500 chars of text
    "algorithm", "data structure", "sorting", "graph", "tree", "hash",
    "operating system", "process", "thread", "memory", "scheduler",
    "file system", "network", "protocol", "tcp", "http", "dns", "router",
    "database", "sql", "query", "index", "transaction", "normalization",
    "compiler", "interpreter", "programming", "python", "java", "c++",
    "computer architecture", "cpu", "cache", "instruction set", "risc",
    "software engineering", "testing", "debugging", "version control",
    "cryptography", "security", "encryption", "vulnerability", "owasp",
    "automata", "turing", "complexity", "np-complete", "regular language",
    "machine learning", "neural network", "deep learning", "transformer",
    "computer science", "discrete mathematics", "logic", "boolean",
}


def _is_cs_article(title: str, text: str) -> bool:
    title_lower = title.lower()
    text_sample = text[:500].lower()
    return any(kw in title_lower or kw in text_sample
               for kw in WIKIPEDIA_CS_CATEGORIES)


def download_wikipedia(
    out_dir: Path,
    target_tokens: int,
    resume_from: int = 0,
) -> int:
    """Stream Wikipedia English, keep CS articles. Returns tokens written."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        return 0

    print(f"\n[wikipedia] Streaming... (resume from doc {resume_from:,})")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "wikipedia", "20220301.en",
        split="train", streaming=True, trust_remote_code=True,
    )

    tokens_written = 0
    docs_seen = 0
    docs_written = 0
    checkpoint_state = load_checkpoint()

    with ChunkWriter(out_dir, "wiki") as writer:
        for doc in tqdm(ds, desc="Wikipedia", unit="doc", mininterval=5):
            docs_seen += 1

            # Resume: skip already-processed docs
            if docs_seen <= resume_from:
                continue

            title = doc.get("title", "")
            text = doc.get("text", "")

            if not text or len(text) < 200:
                continue

            if not _is_cs_article(title, text):
                continue

            # Disk space guard
            if check_disk_space(out_dir) < MIN_FREE_DISK_BYTES:
                print(f"\n[wikipedia] WARNING: low disk space — stopping at {docs_written:,} docs")
                break

            writer.write(f"=== {title} ===\n\n{text}")
            tokens_written += estimate_tokens(text)
            docs_written += 1

            if docs_written % CHECKPOINT_EVERY == 0:
                checkpoint_state["wikipedia_docs_seen"] = docs_seen
                checkpoint_state["wikipedia_tokens"] = tokens_written
                save_checkpoint(checkpoint_state)
                print(f"  [wikipedia] {docs_written:,} docs | ~{tokens_written/1e6:.1f}M tokens")

            if tokens_written >= target_tokens:
                print(f"\n[wikipedia] Target reached: {tokens_written/1e6:.1f}M tokens")
                break

    checkpoint_state["wikipedia_docs_seen"] = docs_seen
    checkpoint_state["wikipedia_tokens"] = tokens_written
    save_checkpoint(checkpoint_state)
    print(f"[wikipedia] Done: {docs_written:,} CS articles | ~{tokens_written/1e6:.1f}M tokens")
    return tokens_written


# ---------------------------------------------------------------------------
# 2. ArXiv CS papers
# ---------------------------------------------------------------------------

ARXIV_CS_CATEGORIES = {
    "cs.DS",   # Data Structures and Algorithms
    "cs.OS",   # Operating Systems (uses cs.DC in ArXiv)
    "cs.DC",   # Distributed/Parallel Computing
    "cs.NI",   # Networking and Internet Architecture
    "cs.DB",   # Databases
    "cs.SE",   # Software Engineering
    "cs.CR",   # Cryptography and Security
    "cs.PL",   # Programming Languages
    "cs.AR",   # Hardware Architecture
    "cs.CC",   # Computational Complexity
    "cs.FL",   # Formal Languages and Automata
    "cs.LG",   # Machine Learning (general CS grounding)
}


def _is_cs_category(categories_str: str) -> bool:
    if not categories_str:
        return False
    cats = set(c.strip() for c in categories_str.split())
    return bool(cats & ARXIV_CS_CATEGORIES)


def download_arxiv(
    out_dir: Path,
    target_tokens: int,
    resume_from: int = 0,
) -> int:
    """Stream ArXiv CS papers (abstract + title). Returns tokens written."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        return 0

    print(f"\n[arxiv] Streaming CS papers... (resume from doc {resume_from:,})")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ArXiv dataset on HuggingFace — CC-BY license, abstracts only
    ds = load_dataset(
        "arxiv_dataset", split="train", streaming=True, trust_remote_code=True,
    )

    tokens_written = 0
    docs_seen = 0
    docs_written = 0
    checkpoint_state = load_checkpoint()

    with ChunkWriter(out_dir, "arxiv") as writer:
        for doc in tqdm(ds, desc="ArXiv", unit="doc", mininterval=5):
            docs_seen += 1

            if docs_seen <= resume_from:
                continue

            categories = doc.get("categories", "")
            if not _is_cs_category(categories):
                continue

            title = doc.get("title", "").strip()
            abstract = doc.get("abstract", "").strip()
            # Some versions have 'text' field with full paper
            body = doc.get("text", "").strip()

            text = f"{title}\n\n{abstract}"
            if body and len(body) > len(abstract):
                text += f"\n\n{body}"

            if len(text) < 100:
                continue

            if check_disk_space(out_dir) < MIN_FREE_DISK_BYTES:
                print(f"\n[arxiv] WARNING: low disk space — stopping")
                break

            writer.write(text)
            tokens_written += estimate_tokens(text)
            docs_written += 1

            if docs_written % CHECKPOINT_EVERY == 0:
                checkpoint_state["arxiv_docs_seen"] = docs_seen
                checkpoint_state["arxiv_tokens"] = tokens_written
                save_checkpoint(checkpoint_state)
                print(f"  [arxiv] {docs_written:,} papers | ~{tokens_written/1e6:.1f}M tokens")

            if tokens_written >= target_tokens:
                print(f"\n[arxiv] Target reached: {tokens_written/1e6:.1f}M tokens")
                break

    checkpoint_state["arxiv_docs_seen"] = docs_seen
    checkpoint_state["arxiv_tokens"] = tokens_written
    save_checkpoint(checkpoint_state)
    print(f"[arxiv] Done: {docs_written:,} papers | ~{tokens_written/1e6:.1f}M tokens")
    return tokens_written


# ---------------------------------------------------------------------------
# 3. The Stack — permissively-licensed code
# ---------------------------------------------------------------------------

STACK_LANGUAGES = {
    "python":     500_000,   # max files per language
    "c":          300_000,
    "cpp":        300_000,
    "java":       300_000,
    "javascript": 200_000,
    "go":         200_000,
    "rust":       150_000,
}

# Only these SPDX license identifiers are allowed (Architecture.md §11.3 cat 4)
ALLOWED_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
    "ISC", "Unlicense", "CC0-1.0", "0BSD",
}

MAX_FILE_CHARS = 50_000   # skip files longer than this (usually generated/minified)
MIN_FILE_CHARS = 100      # skip trivially small files


def _license_ok(license_str: str) -> bool:
    """Check SPDX license field — must be in allowed set."""
    if not license_str:
        return False
    # HuggingFace The Stack has a list of license strings
    parts = re.split(r"[,\s]+", license_str.upper())
    return any(p in {lic.upper() for lic in ALLOWED_LICENSES} for p in parts)


def download_stack(
    out_dir: Path,
    target_tokens: int,
    resume_state: dict,
) -> int:
    """Stream The Stack, filtered by language + license. Returns tokens written."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        return 0

    total_tokens = 0
    checkpoint_state = load_checkpoint()

    for lang, max_files in STACK_LANGUAGES.items():
        lang_dir = out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        resume_from = resume_state.get(f"stack_{lang}_docs_seen", 0)
        print(f"\n[stack/{lang}] Streaming... (resume from {resume_from:,})")

        try:
            ds = load_dataset(
                "bigcode/the-stack",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  [stack/{lang}] FAILED to load: {e}")
            continue

        tokens_written = 0
        docs_seen = 0
        docs_written = 0

        with ChunkWriter(lang_dir, f"stack_{lang}") as writer:
            for doc in tqdm(ds, desc=f"Stack/{lang}", unit="file", mininterval=5):
                docs_seen += 1

                if docs_seen <= resume_from:
                    continue

                license_str = doc.get("license", "") or ""
                if not _license_ok(license_str):
                    continue

                content = doc.get("content", "").strip()
                if not content:
                    continue

                if len(content) < MIN_FILE_CHARS or len(content) > MAX_FILE_CHARS:
                    continue

                # Skip auto-generated files (common heuristic)
                first_line = content.split("\n", 1)[0].lower()
                if any(kw in first_line for kw in
                       ("auto-generated", "do not edit", "generated by", "autogenerated")):
                    continue

                if check_disk_space(lang_dir) < MIN_FREE_DISK_BYTES:
                    print(f"\n[stack/{lang}] WARNING: low disk space — stopping")
                    break

                writer.write(content)
                tokens_written += estimate_tokens(content)
                docs_written += 1

                if docs_written % CHECKPOINT_EVERY == 0:
                    checkpoint_state[f"stack_{lang}_docs_seen"] = docs_seen
                    checkpoint_state[f"stack_{lang}_tokens"] = tokens_written
                    save_checkpoint(checkpoint_state)
                    print(f"  [stack/{lang}] {docs_written:,} files | "
                          f"~{tokens_written/1e6:.1f}M tokens")

                if docs_written >= max_files:
                    print(f"\n[stack/{lang}] Per-language cap reached ({max_files:,} files)")
                    break

                if tokens_written >= target_tokens:
                    print(f"\n[stack/{lang}] Token target reached")
                    break

        checkpoint_state[f"stack_{lang}_docs_seen"] = docs_seen
        checkpoint_state[f"stack_{lang}_tokens"] = tokens_written
        save_checkpoint(checkpoint_state)
        total_tokens += tokens_written
        print(f"[stack/{lang}] Done: {docs_written:,} files | ~{tokens_written/1e6:.1f}M tokens")

        if total_tokens >= target_tokens:
            print(f"\n[stack] Total target reached: {total_tokens/1e6:.1f}M tokens")
            break

    return total_tokens


# ---------------------------------------------------------------------------
# Disk space estimator — print before starting
# ---------------------------------------------------------------------------

def print_disk_estimate(target_tokens: int) -> None:
    estimated_gb = (target_tokens * BYTES_PER_TOKEN_ESTIMATE) / (1024 ** 3)
    free_gb = check_disk_space(RAW_DIR if RAW_DIR.exists() else Path(".")) / (1024 ** 3)
    print(f"\nDisk estimate:")
    print(f"  Target tokens:    {target_tokens/1e9:.1f}B")
    print(f"  Estimated size:   ~{estimated_gb:.1f} GB raw text")
    print(f"  Free disk:        {free_gb:.1f} GB")
    if free_gb < estimated_gb + 5:
        print(f"  WARNING: may not have enough disk space!")
        print(f"  Need at least {estimated_gb + 5:.0f} GB free.")
    else:
        print(f"  OK: enough space.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SOURCES = {
    "wikipedia": (
        RAW_DIR / "wikipedia_large",
        500_000_000,    # 500M token target for this source
    ),
    "arxiv": (
        RAW_DIR / "arxiv_cs",
        300_000_000,    # 300M token target
    ),
    "stack": (
        RAW_DIR / "the_stack",
        500_000_000,    # 500M token target
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Download large corpus for AstraCSE training.")
    ap.add_argument("--sources", nargs="*", choices=list(SOURCES),
                    default=list(SOURCES))
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS,
                    help=f"Total token target (default {TARGET_TOKENS:,})")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last checkpoint")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print estimates and exit")
    args = ap.parse_args()

    print_disk_estimate(args.target_tokens)

    if args.dry_run:
        print("Dry run — exiting.")
        return 0

    # Check datasets installed
    try:
        import datasets
        print(f"datasets version: {datasets.__version__}")
    except ImportError:
        print("ERROR: pip install datasets huggingface_hub", file=sys.stderr)
        return 1

    checkpoint = load_checkpoint() if args.resume else {}
    if args.resume and checkpoint:
        print(f"Resuming from checkpoint: {CHECKPOINT_FILE}")
        for k, v in checkpoint.items():
            print(f"  {k}: {v:,}")
        print()

    total_tokens = 0
    per_source_target = args.target_tokens // len(args.sources)

    for source in args.sources:
        out_dir, default_target = SOURCES[source]
        source_target = min(default_target, per_source_target)

        print(f"\n{'='*60}")
        print(f"Source: {source}  |  Target: ~{source_target/1e6:.0f}M tokens")
        print(f"Output: {out_dir}")
        print(f"{'='*60}")

        try:
            if source == "wikipedia":
                resume_from = checkpoint.get("wikipedia_docs_seen", 0)
                written = download_wikipedia(out_dir, source_target, resume_from)
            elif source == "arxiv":
                resume_from = checkpoint.get("arxiv_docs_seen", 0)
                written = download_arxiv(out_dir, source_target, resume_from)
            elif source == "stack":
                written = download_stack(out_dir, source_target, checkpoint)
            else:
                written = 0
        except KeyboardInterrupt:
            print(f"\n[{source}] Interrupted — checkpoint saved.")
            break
        except Exception as e:
            print(f"\n[{source}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        total_tokens += written
        print(f"\n[{source}] Contributed ~{written/1e6:.1f}M tokens")
        print(f"Running total: ~{total_tokens/1e6:.1f}M tokens")

    print(f"\n{'='*60}")
    print(f"Download complete.")
    print(f"Total estimated tokens: ~{total_tokens/1e6:.0f}M")
    print(f"\nNext steps:")
    print(f"  1. python scripts/clean_text.py --raw-dir data/raw --out-dir data/cleaned")
    print(f"  2. python scripts/dedup.py")
    print(f"  3. python scripts/train_tokenizer.py")
    print(f"  4. python scripts/pack_shards.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
