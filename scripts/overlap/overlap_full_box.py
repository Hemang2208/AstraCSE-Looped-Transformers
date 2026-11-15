"""Full-pass overlap check on the GPU box (run at nice 19 and idle I/O, read-only).

One pass over every v1 training shard does two checks:

  1. MMLU-CS: the repo's own rule (scripts/check_contamination.py). An item is
     flagged when more than 1% of its unique 13-grams occur in the training shards.
  2. Validation: marks every validation token that lies inside a verbatim copy of
     a training window. Long copies (runs of at least 100 tokens) give the per-token
     mask used for the clean-token re-score. Clean = not inside a long copy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

LONG_TOKENS = 100


def runs(mask):
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(padded)
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return starts, ends - starts


def token_cover(win_mask, n_tokens, n):
    m = win_mask.shape[0]
    c = np.concatenate([[0], np.cumsum(win_mask, dtype=np.int64)])
    t = np.arange(n_tokens)
    return (c[np.minimum(t, m - 1) + 1] - c[np.maximum(t - (n - 1), 0)]) > 0


def long_run_mask(win_mask, min_windows):
    starts, lens = runs(win_mask)
    keep = lens >= min_windows
    diff = np.zeros(win_mask.shape[0] + 1, dtype=np.int64)
    np.add.at(diff, starts[keep], 1)
    np.add.at(diff, starts[keep] + lens[keep], -1)
    return np.cumsum(diff[:-1]) > 0


def mmlu_items(cc, tok, mmlu_path, n):
    items = []
    for q in json.loads(Path(mmlu_path).read_text(encoding="utf-8")).get("questions", []):
        text = q.get("question", "") + " " + " ".join(q.get("choices", []))
        ids = np.array(tok.encode(text).ids, dtype=np.uint64)
        hashes = np.unique(cc.window_hashes(ids, n))
        if hashes.shape[0] == 0:
            continue
        items.append({"id": q.get("id", "?"), "subject": q.get("subject", "?"),
                      "question": q.get("question", ""), "hashes": hashes})
    return items


def run(cc, val_dir, train_dir, mmlu_path, tok_dir, out_dir, log):
    n = cc.NGRAM_SIZE
    tok = cc.load_tokenizer(Path(tok_dir))
    items = mmlu_items(cc, tok, mmlu_path, n)
    all_mmlu = np.unique(np.concatenate([it["hashes"] for it in items]))
    log(f"MMLU-CS items: {len(items)}, unique {n}-grams: {all_mmlu.shape[0]:,}")

    val_paths = sorted(Path(val_dir).glob("*.bin"))
    val_tokens = [np.fromfile(str(p), dtype=np.uint16) for p in val_paths]
    val_h = [cc.window_hashes(a, n) for a in val_tokens]
    offsets = np.cumsum([0] + [h.shape[0] for h in val_h])
    h_all = np.concatenate(val_h)
    order = np.argsort(h_all, kind="stable")
    h_sorted = h_all[order]
    total_windows = h_sorted.shape[0]
    first = np.concatenate([[True], h_sorted[1:] != h_sorted[:-1]])
    group_of = np.cumsum(first) - 1
    group_end = np.append(np.flatnonzero(first)[1:], total_windows)

    diff = np.zeros(total_windows + 1, dtype=np.int64)
    found_mmlu = set()
    train_paths = sorted(Path(train_dir).glob("*.bin"))
    for i, p in enumerate(train_paths, 1):
        h = cc.window_hashes(np.fromfile(str(p), dtype=np.uint16), n)
        lo = np.minimum(np.searchsorted(h_sorted, h), total_windows - 1)
        hit = h_sorted[lo] == h
        if hit.any():
            lo_h = lo[hit]
            np.add.at(diff, lo_h, 1)
            np.add.at(diff, group_end[group_of[lo_h]], -1)
        m = np.isin(h, all_mmlu)
        if m.any():
            found_mmlu.update(int(x) for x in np.unique(h[m]))
        if i % 25 == 0 or i == len(train_paths):
            log(f"train shards scanned: {i}/{len(train_paths)}; MMLU {n}-grams seen: {len(found_mmlu):,}")

    covered = np.zeros(total_windows, dtype=bool)
    covered[order] = np.cumsum(diff[:-1]) > 0

    found_arr = np.array(sorted(found_mmlu), dtype=np.uint64)
    rows = []
    for it in items:
        seen = int(np.isin(it["hashes"], found_arr).sum()) if found_arr.size else 0
        total = int(it["hashes"].shape[0])
        rows.append({"id": it["id"], "subject": it["subject"], "ngrams": total,
                     "found": seen, "fraction": seen / total, "question": it["question"][:160]})
    rows.sort(key=lambda r: -r["fraction"])
    by_subject = {}
    for r in rows:
        s = by_subject.setdefault(r["subject"], {"items": 0, "any_hit": 0, "flagged": 0})
        s["items"] += 1
        s["any_hit"] += int(r["found"] > 0)
        s["flagged"] += int(r["fraction"] > cc.MAX_OVERLAP_FRACTION)
    mmlu_summary = {
        "rule": f"flag if more than {cc.MAX_OVERLAP_FRACTION:.0%} of unique {n}-grams occur in training",
        "items": len(rows),
        "flagged": sum(1 for r in rows if r["fraction"] > cc.MAX_OVERLAP_FRACTION),
        "any_hit": sum(1 for r in rows if r["found"] > 0),
        "by_subject": by_subject,
        "top_items": rows[:25],
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_rows = []
    total_tok = any_tok = long_tok = 0
    for k, (p, a) in enumerate(zip(val_paths, val_tokens)):
        cov = covered[offsets[k]:offsets[k + 1]]
        n_tok = int(a.shape[0])
        any_mask = token_cover(cov, n_tok, n)
        long_mask = token_cover(long_run_mask(cov, LONG_TOKENS - (n - 1)), n_tok, n)
        np.save(out_dir / f"val_long_mask_{k}.npy", long_mask)
        total_tok += n_tok
        any_tok += int(any_mask.sum())
        long_tok += int(long_mask.sum())
        val_rows.append({"index": k, "shard": p.name, "tokens": n_tok,
                         "any_fraction": float(any_mask.mean()), "long_fraction": float(long_mask.mean())})
    val_summary = {"val_tokens": total_tok, "any_fraction": any_tok / total_tok,
                   "long_fraction": long_tok / total_tok, "clean_fraction": 1 - long_tok / total_tok,
                   "shards": val_rows}
    (out_dir / "val_long_manifest.json").write_text(
        json.dumps([r["shard"] for r in val_rows], indent=2), encoding="utf-8")
    result = {"mmlu_cs": mmlu_summary, "validation": val_summary, "train_shards": len(train_paths)}
    (out_dir / "overlap_full.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"MMLU-CS: {mmlu_summary['flagged']}/{mmlu_summary['items']} flagged, "
        f"{mmlu_summary['any_hit']} with any hit")
    log(f"validation: any {val_summary['any_fraction']:.4f}, long {val_summary['long_fraction']:.4f}, "
        f"clean {val_summary['clean_fraction']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts-dir", default="/home/ubuntu/AstraCSE/scripts")
    ap.add_argument("--tok-dir", default="/home/ubuntu/AstraCSE/tokenizer")
    ap.add_argument("--val-dir", default="/home/ubuntu/AstraCSE/data/shards/val")
    ap.add_argument("--train-dir", default="/home/ubuntu/astra_overlap/v1/train")
    ap.add_argument("--mmlu", default="/home/ubuntu/AstraCSE/eval/mmlu_cs_benchmark.json")
    ap.add_argument("--out-dir", default="/home/ubuntu/astra_overlap/out")
    args = ap.parse_args()
    sys.path.insert(0, args.scripts_dir)
    import check_contamination as cc  # the repo's own functions and thresholds
    run(cc, args.val_dir, args.train_dir, args.mmlu, args.tok_dir, args.out_dir,
        lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
