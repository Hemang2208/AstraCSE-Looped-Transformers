"""Download MMLU CS-relevant subjects and convert to AstraCSE's eval format.

Honest limitation, stated up front: MMLU (Hendrycks et al. 2021) has 57
subjects total, and none of them line up 1:1 with AstraCSE's 10-subject
taxonomy (Architecture.md Section 24). Only a handful are CS-relevant at
all, and even those are broader or narrower than AstraCSE's split — MMLU
has no dedicated category for operating systems, computer networks,
database systems, or software engineering specifically.

This script does NOT force those into AstraCSE's 10 buckets. Each question
keeps MMLU's own subject name, and the output records which AstraCSE
subjects have any MMLU coverage at all versus none, so nobody downstream
mistakes "we ran MMLU" for "we covered all 10 subjects."

Usage:
    python scripts/download_mmlu.py
    python scripts/download_mmlu.py --split test --out eval/mmlu_cs_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# MMLU subject -> rough AstraCSE-subject correspondence, for reporting only
# (not used to rewrite the question's own subject field). None is a stretch
# in both directions: MMLU's own subjects mix topics AstraCSE keeps separate,
# and AstraCSE's subjects are each broader than any single MMLU category.
MMLU_TO_ASTRA_HINT = {
    "computer_security":            "cybersecurity",
    "college_computer_science":     "algorithms / theory_of_computation (mixed)",
    "high_school_computer_science": "computer_programming",
    "machine_learning":             None,   # not one of AstraCSE's 10 subjects
    "electrical_engineering":       "computer_architecture (partial)",
    "abstract_algebra":             "theory_of_computation (partial)",
}

ASTRA_SUBJECTS = {
    "computer_programming", "data_structures", "algorithms", "operating_systems",
    "computer_networks", "database_systems", "computer_architecture",
    "theory_of_computation", "software_engineering", "cybersecurity",
}


def convert_split(examples, mmlu_subject: str) -> list[dict]:
    out = []
    for i, ex in enumerate(examples):
        choices = ex["choices"]
        if len(choices) != 4:
            continue
        lettered = [f"{chr(ord('A') + j)}) {c}" for j, c in enumerate(choices)]
        answer_letter = chr(ord("A") + ex["answer"])
        out.append({
            "id": f"mmlu_{mmlu_subject}_{i:04d}",
            "subject": mmlu_subject,          # kept as MMLU's own name — not remapped
            "category": "mcq",
            "question": ex["question"],
            "choices": lettered,
            "answer": answer_letter,
            "source": "mmlu",
        })
    return out


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description="Download MMLU CS-relevant subjects for AstraCSE eval.")
    ap.add_argument("--out", type=Path, default=Path("eval/mmlu_cs_benchmark.json"))
    ap.add_argument("--split", default="test", choices=["test", "validation", "dev"])
    args = ap.parse_args()

    all_questions: list[dict] = []
    per_subject_counts: dict[str, int] = {}

    for mmlu_subject in MMLU_TO_ASTRA_HINT:
        print(f"Downloading MMLU/{mmlu_subject} ({args.split} split)...")
        try:
            ds = load_dataset("cais/mmlu", mmlu_subject, split=args.split)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        converted = convert_split(ds, mmlu_subject)
        all_questions.extend(converted)
        per_subject_counts[mmlu_subject] = len(converted)
        print(f"  {len(converted)} questions")

    covered_hint = {v for v in MMLU_TO_ASTRA_HINT.values() if v}
    astra_touched = {s for s in ASTRA_SUBJECTS
                     if any(s in (h or "") for h in MMLU_TO_ASTRA_HINT.values())}
    astra_uncovered = sorted(ASTRA_SUBJECTS - astra_touched)

    out_data = {
        "version": "1.0",
        "description": (
            "MMLU (Hendrycks et al. 2021) CS-relevant subset. Questions keep MMLU's "
            "own subject names (not remapped to AstraCSE's 10-subject taxonomy) because "
            "the correspondence is inexact. See astra_subjects_with_no_mmlu_coverage — "
            "this is NOT a full 10-subject benchmark on its own; use alongside "
            "eval/benchmark.json, not as a replacement for it."
        ),
        "mmlu_subject_to_astra_hint": MMLU_TO_ASTRA_HINT,
        "astra_subjects_with_no_mmlu_coverage": astra_uncovered,
        "per_mmlu_subject_counts": per_subject_counts,
        "categories": ["mcq"],
        "questions": all_questions,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_questions)} questions to {args.out}")
    print(f"AstraCSE subjects with ZERO MMLU coverage: {astra_uncovered}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
