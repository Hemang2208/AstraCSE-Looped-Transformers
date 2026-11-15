"""Phase 1 — extract plain text from PDF files in data/raw/.

Writes extracted text to data/cleaned/<same_path>.txt
(alongside the HTML/txt files already cleaned by clean_text.py).

Dependencies: pymupdf  (pip install pymupdf)

Usage:
    python scripts/extract_pdfs.py
    python scripts/extract_pdfs.py --raw-dir data/raw --out-dir data/cleaned
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import fitz   # pymupdf
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False


def extract_pdf(src: Path) -> str:
    """Extract all text from a PDF, preserving paragraph breaks."""
    doc = fitz.open(str(src))
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text.strip())
    doc.close()
    return "\n\n".join(pages)


def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    if not FITZ_AVAILABLE:
        print("ERROR: pymupdf not installed. Run: pip install pymupdf", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description="Extract text from PDFs into data/cleaned/")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/cleaned"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.raw_dir.exists():
        print(f"ERROR: {args.raw_dir} does not exist", file=sys.stderr)
        return 1

    pdfs = sorted(args.raw_dir.rglob("*.pdf"))
    if not pdfs:
        print("No PDF files found.")
        return 0

    written = skipped = failed = 0
    seen_hashes: set = set()

    for src in pdfs:
        rel = src.relative_to(args.raw_dir)
        out = args.out_dir / rel.with_suffix(".txt")

        if out.exists() and out.stat().st_size > 0:
            print(f"  SKIP (exists): {rel}")
            skipped += 1
            continue

        try:
            text = extract_pdf(src)
        except Exception as e:
            print(f"  FAILED: {rel}: {e}")
            failed += 1
            continue

        if not text or len(text) < 100:
            print(f"  EMPTY: {rel}")
            skipped += 1
            continue

        h = file_hash(text)
        if h in seen_hashes:
            print(f"  DUP: {rel}")
            skipped += 1
            continue
        seen_hashes.add(h)

        if args.dry_run:
            print(f"  WOULD WRITE: {out}  ({len(text):,} chars)")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"  WROTE: {out.name}  ({len(text):,} chars)")
        written += 1

    print(f"\nDone. written={written}  skipped={skipped}  failed={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
