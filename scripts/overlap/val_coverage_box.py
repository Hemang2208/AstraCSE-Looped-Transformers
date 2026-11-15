"""Validation-side coverage on the GPU box (run at nice 19 and idle I/O).

Every v1 validation token is marked if it lies inside a verbatim copy of any
13-token window found in the v1 training shards. Two numbers per val shard:

  any:  tokens covered by at least one matched 13-token window
  long: tokens covered by runs of at least 100 consecutive matched tokens

Long runs are copied passages. Short runs are boilerplate. Read-only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ubuntu/astra_overlap/scripts")
import screen_box as sb  # noqa: E402

WINDOW = sb.WINDOW
LONG_WINDOWS = 100 - (WINDOW - 1)


def runs(mask):
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(padded)
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return starts, ends - starts


def token_cover(win_mask, n_tokens):
    n = win_mask.shape[0]
    c = np.concatenate([[0], np.cumsum(win_mask, dtype=np.int64)])
    t = np.arange(n_tokens)
    return (c[np.minimum(t, n - 1) + 1] - c[np.maximum(t - (WINDOW - 1), 0)]) > 0


def long_run_mask(win_mask, min_windows):
    starts, lens = runs(win_mask)
    keep = lens >= min_windows
    diff = np.zeros(win_mask.shape[0] + 1, dtype=np.int64)
    np.add.at(diff, starts[keep], 1)
    np.add.at(diff, starts[keep] + lens[keep], -1)
    return np.cumsum(diff[:-1]) > 0


def coverage(val_paths, train_paths, log=print):
    val_tokens = [sb.load(p) for p in val_paths]
    val_hash = [sb.window_hashes(a) for a in val_tokens]
    offsets = np.cumsum([0] + [h.shape[0] for h in val_hash])
    h_all = np.concatenate(val_hash)
    order = np.argsort(h_all, kind="stable")
    h_sorted = h_all[order]
    n = h_sorted.shape[0]
    first = np.concatenate([[True], h_sorted[1:] != h_sorted[:-1]])
    group_of = np.cumsum(first) - 1
    group_end = np.append(np.flatnonzero(first)[1:], n)

    diff = np.zeros(n + 1, dtype=np.int64)
    for i, p in enumerate(train_paths, 1):
        h = sb.window_hashes(sb.load(p))
        lo = np.minimum(np.searchsorted(h_sorted, h), n - 1)
        hit = h_sorted[lo] == h
        if hit.any():
            lo_h = lo[hit]
            np.add.at(diff, lo_h, 1)
            np.add.at(diff, group_end[group_of[lo_h]], -1)
        if i % 25 == 0:
            log(f"train shards scanned: {i}/{len(train_paths)}")
    covered = np.zeros(n, dtype=bool)
    covered[order] = np.cumsum(diff[:-1]) > 0

    shards = []
    for k, a in enumerate(val_tokens):
        cov = covered[offsets[k]:offsets[k + 1]]
        n_tok = int(a.shape[0])
        any_tok = token_cover(cov, n_tok)
        long_tok = token_cover(long_run_mask(cov, LONG_WINDOWS), n_tok)
        shards.append({"shard": val_paths[k].name, "tokens": n_tok,
                       "any_tokens": int(any_tok.sum()), "long_tokens": int(long_tok.sum())})
    return shards


def main():
    repo = Path("/home/ubuntu/AstraCSE")
    work = Path("/home/ubuntu/astra_overlap")
    val = sorted((repo / "data" / "shards" / "val").glob("*.bin"))
    train = sorted((work / "v1" / "train").glob("*.bin"))
    assert len(val) == 6 and len(train) == 275, (len(val), len(train))
    shards = coverage(val, train, log=lambda m: print(m, flush=True))
    total = sum(s["tokens"] for s in shards)
    summary = {
        "val_tokens": total,
        "any_fraction": sum(s["any_tokens"] for s in shards) / total,
        "long_fraction": sum(s["long_tokens"] for s in shards) / total,
        "train_shards": len(train),
        "shards": shards,
    }
    out = work / "out" / "val_coverage.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"val tokens {total:,}; any-match fraction {summary['any_fraction']:.4f}; "
          f"long-run (>=100 tokens) fraction {summary['long_fraction']:.4f}", flush=True)
    for s in shards:
        print(f"  {s['shard']}: tokens {s['tokens']:,}, any {s['any_tokens']/s['tokens']:.4f}, "
              f"long {s['long_tokens']/s['tokens']:.4f}", flush=True)
    print("val coverage written", flush=True)


if __name__ == "__main__":
    main()
