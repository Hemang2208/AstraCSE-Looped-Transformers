"""Phase 1 — text extraction and normalization.

Reads raw .txt / .md / .html files from data/raw/<subject>/
and writes cleaned plain-text to data/cleaned/<subject>/.

Rules (Architecture.md §12):
- Strip HTML markup when present.
- Collapse repeated blank lines to at most one.
- Preserve leading whitespace (indentation) in code blocks.
- Drop lines that are purely navigation chrome (very short lines that are
  all punctuation or single words — heuristic, good enough for this scale).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

_BLANK_RUN = re.compile(r"\n{3,}")
_NAV_LINE = re.compile(r"^\s{0,4}[^\w\s]{0,3}\s*\w{0,8}\s*[^\w\s]{0,3}\s*$")


def strip_html(text: str) -> str:
    if not BS4_AVAILABLE:
        # Fallback: crude tag removal
        text = re.sub(r"<[^>]+>", " ", text)
        return text
    soup = BeautifulSoup(text, "html.parser")
    # Remove script/style entirely
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def normalize(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    in_code_block = False
    for line in lines:
        # Detect fenced code blocks (markdown ```), toggle preservation mode
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            cleaned.append(line)
            continue
        if in_code_block:
            # Preserve line as-is (indentation matters)
            cleaned.append(line)
            continue
        # Drop nav-chrome lines (outside code only) then strip trailing whitespace
        if _NAV_LINE.match(line):
            continue
        cleaned.append(line.rstrip())
    text = "\n".join(cleaned)
    # Collapse 3+ blank lines to 2
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Document splitting — scripts/download_parts.py's ChunkWriter packs many
# documents into one 50MB .txt file for I/O efficiency. Everything downstream
# (exact-hash dedup below, dedup.py's near-dup pass, pack_shards.py's <doc>
# boundary insertion, and the val-split-by-file logic) assumes one file ==
# one document — a chunk file must be split back into individual documents
# BEFORE any of that, or all four silently operate at the wrong granularity.
# ---------------------------------------------------------------------------

_WIKI_TITLE_RE = re.compile(r"(?=^=== .+ ===$)", re.MULTILINE)


def _looks_like_title(block: str) -> bool:
    """Heuristic for ArXiv chunks: a title line is short, single-line, and
    has no internal blank line — an abstract paragraph never looks like this."""
    return bool(block) and "\n" not in block and len(block) < 300


def split_documents(rel_path: Path, raw: str) -> list[str]:
    """Recover individual documents from a possibly-multi-document raw chunk
    file. Falls back to "the whole file is one document" for sources that
    were never chunked this way (the original small-corpus sources)."""
    path_str = rel_path.as_posix()

    if "the_stack" in path_str:
        # Post-fix ChunkWriter prefixes every document with "\x00".
        parts = raw.split("\x00")
        return [p.strip() for p in parts if p.strip()]

    if "wikipedia_large" in path_str:
        parts = _WIKI_TITLE_RE.split(raw)
        return [p.strip() for p in parts if p.strip()]

    if path_str.startswith("arxiv_"):
        # No unique delimiter was written for ArXiv chunks. Recover per-paper
        # granularity via a title-detection heuristic on "\n\n"-separated
        # blocks: a short single-line block starts a new document, everything
        # else (including a rare internal blank line inside one abstract)
        # accumulates into the current one. Verified empirically against this
        # project's actual ArXiv output before relying on it.
        blocks = raw.split("\n\n")
        docs: list[str] = []
        buf: list[str] = []
        for block in blocks:
            stripped = block.strip()
            if buf and _looks_like_title(stripped):
                docs.append("\n\n".join(buf))
                buf = [block]
            else:
                buf.append(block)
        if buf:
            docs.append("\n\n".join(buf))
        return [d.strip() for d in docs if d.strip()]

    # Small-corpus sources: one file already is one document.
    return [raw]


# ---------------------------------------------------------------------------
# Dedup (exact hash — near-dup handled separately in dedup.py)
# ---------------------------------------------------------------------------

def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Clean raw text files for AstraCSE corpus.")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                    help="Root of raw source files (default: data/raw)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/cleaned"),
                    help="Output root for cleaned files (default: data/cleaned)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done without writing files")
    args = ap.parse_args()

    if not args.raw_dir.exists():
        print(f"ERROR: raw dir {args.raw_dir} does not exist", file=sys.stderr)
        return 1

    seen_hashes: set = set()
    written = skipped_dup = skipped_empty = 0
    files_seen = 0

    for src in sorted(args.raw_dir.rglob("*")):
        if not src.is_file():
            continue
        if src.suffix not in {".txt", ".md", ".html", ".htm", ".rst"}:
            continue

        files_seen += 1
        rel = src.relative_to(args.raw_dir)
        is_html = src.suffix in {".html", ".htm"}
        raw_full = src.read_text(encoding="utf-8", errors="replace")

        documents = split_documents(rel, raw_full)
        multi = len(documents) > 1
        if multi:
            print(f"  SPLIT: {rel}  ->  {len(documents):,} documents")

        for doc_idx, raw_doc in enumerate(documents):
            text = strip_html(raw_doc) if is_html else raw_doc
            cleaned = normalize(text)
            if not cleaned:
                skipped_empty += 1
                continue

            h = file_hash(cleaned)
            if h in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(h)

            out = (args.out_dir / rel.parent / f"{rel.stem}__{doc_idx:05d}.txt"
                   if multi else args.out_dir / rel.with_suffix(".txt"))

            if not args.dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(cleaned, encoding="utf-8")
            written += 1
            if written % 20_000 == 0:
                print(f"  ... {written:,} documents written so far "
                      f"({skipped_dup:,} exact-dup, {skipped_empty:,} empty)")

    print(f"\nDone. raw_files_scanned={files_seen}  written={written}  "
          f"skipped_dup={skipped_dup}  skipped_empty={skipped_empty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
