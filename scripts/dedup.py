"""Phase 1 — near-duplicate removal using MinHash/LSH.

Run after clean_text.py. Reads data/cleaned/, removes near-duplicate
documents (Jaccard similarity > threshold), writes survivors to data/deduped/.

Architecture.md §11.4: exact-hash dedup is done in clean_text.py; this
script handles the near-duplicate pass.

Processes one IMMEDIATE PARENT DIRECTORY at a time (e.g. wikipedia_large,
the_stack/python, the_stack/java — clean_text.py's split_documents() already
gives each source this per-directory granularity) with its own fresh
MinHashLSH, instead of one LSH for the whole ~1.1M-document corpus at once.
This bounds peak memory to whatever the single largest source directory
needs rather than the full corpus — deliberately conservative given this
runs on a personal laptop with other applications competing for RAM, not a
dedicated box. Groups still larger than MAX_GROUP_BATCH_SIZE (e.g.
arxiv_algo_theory at ~314k files — too large for even one group's LSH to
reliably fit in the free RAM actually observed on this machine) are further
split into independent batches, each with its own fresh LSH. Trade-off,
stated plainly: a near-duplicate pair that happens to land in two different
batches of the same oversized group will not be caught — only within-batch
near-dups are. Exact duplicates are unaffected (clean_text.py's hash-based
pass already catches those corpus-wide, batching or not).

Dependencies: datasketch psutil  (pip install datasketch psutil)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    print("WARNING: datasketch not installed — near-dup pass will be skipped.",
          file=sys.stderr)
    print("Install with: pip install datasketch", file=sys.stderr)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not installed — no memory safety check will run.",
          file=sys.stderr)
    print("Install with: pip install psutil", file=sys.stderr)


_SHINGLE_RE = re.compile(r"\w+")
SHINGLE_SIZE = 5
NUM_PERM = 128
DEFAULT_THRESHOLD = 0.8
MIN_FREE_RAM_GB = 1.5   # abort a group rather than risk the OS killing us
MAX_GROUP_BATCH_SIZE = 15_000   # cap files per LSH instance -- bounds peak
                                 # memory for the few groups much larger than
                                 # this (e.g. arxiv_algo_theory at 314k files)
                                 # independent of how much RAM happens to be
                                 # free when the script runs.


def shingles(text: str, k: int = SHINGLE_SIZE) -> set:
    tokens = _SHINGLE_RE.findall(text.lower())
    return {" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}


def make_minhash(text: str) -> "MinHash":
    m = MinHash(num_perm=NUM_PERM)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m


def _free_ram_gb() -> float:
    if not PSUTIL_AVAILABLE:
        return float("inf")  # can't check — don't block on it
    return psutil.virtual_memory().available / 1024**3


def _dedup_one_group(files: list[Path], src_dir: Path, out_dir: Path,
                     threshold: float, dry_run: bool) -> tuple[int, int, bool]:
    """Dedup one directory's worth of files with its own fresh LSH.
    Returns (kept, removed, low_memory_abort)."""
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    kept = removed = 0

    for idx, f in enumerate(files):
        if idx % 5000 == 0:
            free_gb = _free_ram_gb()
            if free_gb < MIN_FREE_RAM_GB:
                print(f"\n  LOW MEMORY ({free_gb:.1f} GB free < {MIN_FREE_RAM_GB} GB "
                      f"threshold) — stopping this group early, kept so far preserved.")
                return kept, removed, True

        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            # At ~1.1M files, a sporadic single-file glitch (AV scan lock,
            # filesystem race) is survivable — skip it, don't crash the group.
            print(f"\n  SKIP (unreadable: {e}): {f.name}")
            continue
        if not text.strip():
            continue
        mh = make_minhash(text)
        key = str(idx)

        neighbors = lsh.query(mh)
        if neighbors:
            removed += 1
            continue

        lsh.insert(key, mh)
        out = out_dir / f.relative_to(src_dir)
        if not dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        kept += 1

    return kept, removed, False


def _load_batch_state(state_path: Path, batch_size: int) -> set[int]:
    if not state_path.exists():
        return set()
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    if data.get("batch_size") != batch_size:
        # MAX_GROUP_BATCH_SIZE changed since this state was written -- batch
        # index N now covers a different slice of files than it used to.
        # Trusting it would resume at the wrong file offset. Start over.
        print(f"    (batch size changed since last run — ignoring stale "
              f"batch state, redoing this group's batches from scratch)")
        return set()
    return set(data.get("completed_batches", []))


def _save_batch_state(state_path: Path, completed: set[int], batch_size: int) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"batch_size": batch_size, "completed_batches": sorted(completed)}),
        encoding="utf-8")


def _dedup_group_batched(files: list[Path], src_dir: Path, out_dir: Path,
                         out_group_dir: Path, threshold: float,
                         dry_run: bool) -> tuple[int, int, bool]:
    """Groups at or under MAX_GROUP_BATCH_SIZE run through _dedup_one_group
    unchanged. Larger ones split into independent batches, each with its own
    fresh LSH -- bounds peak memory to one batch regardless of total group
    size, and each batch's completion is tracked separately, so a low-memory
    abort partway through a huge group only costs that one batch's progress,
    not the whole group's."""
    if len(files) <= MAX_GROUP_BATCH_SIZE:
        return _dedup_one_group(files, src_dir, out_dir, threshold, dry_run)

    n_batches = math.ceil(len(files) / MAX_GROUP_BATCH_SIZE)
    state_path = out_group_dir / ".dedup_batch_state.json"
    completed = _load_batch_state(state_path, MAX_GROUP_BATCH_SIZE)

    total_kept = total_removed = 0
    for b in range(n_batches):
        if b in completed:
            continue
        batch_files = files[b * MAX_GROUP_BATCH_SIZE:(b + 1) * MAX_GROUP_BATCH_SIZE]
        print(f"    batch {b + 1}/{n_batches} ({len(batch_files):,} files)...")
        kept, removed, low_mem = _dedup_one_group(batch_files, src_dir, out_dir,
                                                   threshold, dry_run)
        total_kept += kept
        total_removed += removed
        if low_mem:
            return total_kept, total_removed, True
        completed.add(b)
        if not dry_run:
            _save_batch_state(state_path, completed, MAX_GROUP_BATCH_SIZE)

    if not dry_run:
        state_path.unlink(missing_ok=True)   # all batches done -- state no longer needed
    return total_kept, total_removed, False


def _copy_group(files: list[Path], src_dir: Path, out_dir: Path, dry_run: bool) -> int:
    """Copy a group through unchanged -- used for sources where exact-hash
    dedup (already done in clean_text.py) is the dominant redundancy and the
    near-dup MinHash pass isn't worth its cost (see --skip-near-dup-for)."""
    n = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"\n  SKIP (unreadable: {e}): {f.name}")
            continue
        if not text.strip():
            continue
        out = out_dir / f.relative_to(src_dir)
        if not dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        n += 1
    return n


def dedup_files(src_dir: Path, out_dir: Path,
                threshold: float = DEFAULT_THRESHOLD,
                dry_run: bool = False,
                skip_near_dup_for: tuple[str, ...] = ()) -> None:
    all_files = sorted(src_dir.rglob("*.txt"))
    if not all_files:
        print(f"No .txt files found under {src_dir}")
        return

    if not DATASKETCH_AVAILABLE:
        print("datasketch unavailable — copying all files without near-dup pass.")
        for f in all_files:
            rel = f.relative_to(src_dir)
            out = out_dir / rel
            if not dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        return

    # Group by immediate parent dir (wikipedia_large, the_stack/python, ...) so
    # each group gets its own LSH — bounds peak memory to one group, not 1.1M
    # documents' worth of signatures held simultaneously (see module docstring).
    groups: dict[Path, list[Path]] = {}
    for f in all_files:
        groups.setdefault(f.parent, []).append(f)

    print(f"{len(all_files):,} files across {len(groups)} groups "
          f"(largest: {max(len(v) for v in groups.values()):,} files)")
    if PSUTIL_AVAILABLE:
        print(f"Free RAM at start: {_free_ram_gb():.1f} GB "
              f"(will pause a group if it drops below {MIN_FREE_RAM_GB} GB)")

    sorted_groups = sorted(groups.items())
    total_kept = total_removed = 0
    for i, (group_dir, files) in enumerate(sorted_groups):
        rel_group = group_dir.relative_to(src_dir)
        out_group_dir = out_dir / rel_group
        marker = out_group_dir / ".dedup_complete"

        if not dry_run and marker.exists():
            print(f"  {rel_group}: already fully done — skipping (delete "
                  f"{out_group_dir} to force redo)")
            continue

        batch_state_path = out_group_dir / ".dedup_batch_state.json"
        is_batched = len(files) > MAX_GROUP_BATCH_SIZE
        has_resumable_batch_state = is_batched and batch_state_path.exists()

        if (not dry_run and out_group_dir.exists() and any(out_group_dir.iterdir())
                and not has_resumable_batch_state):
            # Output exists but no completion marker, and no in-progress batch
            # state to resume from -- this is a PARTIAL result from a run
            # that stopped early (e.g. low memory) on a non-batched group.
            # Its LSH state died with that process, so simply appending more
            # files now would use a fresh, empty LSH that doesn't know about
            # the already-kept ones -- a near-duplicate among the remaining
            # files could slip through unflagged. Redo the whole group.
            print(f"  {rel_group}: found incomplete output (no completion "
                  f"marker) — deleting and redoing the whole group from scratch")
            shutil.rmtree(out_group_dir)

        if any(str(rel_group).startswith(p) for p in skip_near_dup_for):
            copied = _copy_group(files, src_dir, out_dir, dry_run)
            total_kept += copied
            print(f"  {rel_group}: copied={copied:,} (near-dup pass skipped — "
                  f"exact-dedup from clean_text.py already applied)")
            if not dry_run:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
            continue

        kept, removed, low_mem = _dedup_group_batched(
            files, src_dir, out_dir, out_group_dir, threshold, dry_run)
        total_kept += kept
        total_removed += removed
        print(f"  {rel_group}: kept={kept:,} removed={removed:,}"
              f"{' [STOPPED EARLY - LOW MEMORY]' if low_mem else ''}")
        if not low_mem and not dry_run:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        if low_mem:
            # Don't keep looping into more groups while memory is critical —
            # the next group is likely to hit the same wall immediately.
            pending = [g.relative_to(src_dir) for g, _ in sorted_groups[i + 1:]]
            print(f"\nStopping here — {len(pending)} group(s) not yet attempted: "
                  f"{', '.join(str(g) for g in pending[:5])}"
                  f"{', ...' if len(pending) > 5 else ''}")
            print("Close other applications to free RAM, then re-run with the same "
                  "command — completed groups are unaffected (each group is "
                  "independent; re-running just redoes that one group's dedup pass).")
            break

    print(f"\nDedup complete. kept={total_kept:,}  removed_as_near_dup={total_removed:,}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Near-duplicate removal for AstraCSE corpus.")
    ap.add_argument("--cleaned-dir", type=Path, default=Path("data/cleaned"))
    ap.add_argument("--out-dir",     type=Path, default=Path("data/deduped"))
    ap.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--dry-run",     action="store_true")
    ap.add_argument("--skip-near-dup-for", nargs="*", default=["the_stack"],
                    help="Group-path prefixes to copy through without the MinHash "
                         "near-dup pass (default: the_stack — code's dominant "
                         "redundancy is exact duplication, already handled by "
                         "clean_text.py; pass --skip-near-dup-for with no values "
                         "to disable skipping and near-dup everything).")
    args = ap.parse_args()

    if not args.cleaned_dir.exists():
        print(f"ERROR: {args.cleaned_dir} does not exist", file=sys.stderr)
        return 1

    dedup_files(args.cleaned_dir, args.out_dir,
                threshold=args.threshold, dry_run=args.dry_run,
                skip_near_dup_for=tuple(args.skip_near_dup_for))
    return 0


if __name__ == "__main__":
    sys.exit(main())
