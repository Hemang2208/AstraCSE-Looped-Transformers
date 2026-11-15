"""On-box overlap screen. Runs at nice 19 and idle I/O, no GPU, read-only on the
training data.

Question: do 13-token windows from the v1 validation shards (the shards the eval
scores) appear in new v2 shards more often than in v1 training shards?

  queries: 2000 random 13-token windows from data/shards/val
  control: 75 random v1 training shards (synced from S3 data/shards/train)
  sample:  75 new v2 shards from data/shards_v2/train. Candidates are v2 objects
           whose S3 ETag matches no v1 train object. A candidate is rejected if
           its bytes equal any v1 training shard (md5 check).

Matching is exact on 13 token ids. The 64-bit rolling hash only filters; every
candidate is compared byte for byte.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from tokenizers import ByteLevelBPETokenizer

WINDOW = 13
MULT = np.uint64(0x9E3779B97F4A7C15)


def load(path):
    return np.fromfile(str(path), dtype=np.uint16)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def window_hashes(arr):
    n = arr.shape[0] - WINDOW + 1
    h = np.zeros(n, dtype=np.uint64)
    for j in range(WINDOW):
        h = h * MULT + arr[j:j + n].astype(np.uint64)
    return h


def scan(paths, windows):
    qh = np.sort(np.array([window_hashes(w)[0] for w in windows], dtype=np.uint64))
    table = {w.tobytes(): k for k, w in enumerate(windows)}
    counts = np.zeros(len(windows), dtype=np.int64)
    hits = []
    for p in paths:
        arr = load(p)
        h = window_hashes(arr)
        idx = np.minimum(np.searchsorted(qh, h), len(qh) - 1)
        for pos in np.nonzero(qh[idx] == h)[0]:
            k = table.get(arr[pos:pos + WINDOW].tobytes())
            if k is not None:
                counts[k] += 1
                hits.append((k, Path(p).name, int(pos)))
    return counts, hits


def self_test(paths, doc_id, rng):
    found = {}
    for p in paths:
        a = load(p)
        for i in rng.integers(0, a.shape[0] - WINDOW, size=20):
            w = a[i:i + WINDOW].copy()
            if doc_id not in w:
                found.setdefault(w.tobytes(), w)
    windows = list(found.values())
    counts, _ = scan(paths, windows)
    return float((counts > 0).mean()), len(windows)


def build_queries(val_paths, doc_id, n, rng):
    arrays = [load(p) for p in val_paths]
    lengths = np.array([a.shape[0] - WINDOW + 1 for a in arrays])
    probs = lengths / lengths.sum()
    windows, origin, seen = [], [], set()
    while len(windows) < n:
        s = int(rng.choice(len(arrays), p=probs))
        i = int(rng.integers(lengths[s]))
        w = arrays[s][i:i + WINDOW].copy()
        if doc_id in w or w.tobytes() in seen:
            continue
        seen.add(w.tobytes())
        windows.append(w)
        origin.append(f"{val_paths[s].name}:{i}")
    return windows, origin


def read_etags(path):
    out = {}
    for line in open(path):
        key, etag = line.split()
        out[key] = etag.strip('"')
    return out


def pick_sample(candidates, v2_dir, v1_md5s, n, rng):
    order = list(candidates)
    rng.shuffle(order)
    accepted, rejected = [], []
    for key in order:
        path = v2_dir / key.rsplit("/", 1)[-1]
        if md5(path) in v1_md5s:
            rejected.append(path.name)
            continue
        accepted.append(path)
        if len(accepted) == n:
            break
    return accepted, rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--n-queries", type=int, default=2000)
    args = ap.parse_args()

    tok_dir = args.repo / "tokenizer"
    tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    doc_id = tok.token_to_id("<doc>")
    assert doc_id is not None, "<doc> token not in tokenizer"

    v1_etags = read_etags(args.work / "meta" / "v1_train_etags.txt")
    v2_etags = read_etags(args.work / "meta" / "v2_train_etags.txt")
    v1_values = set(v1_etags.values())
    candidates = sorted(k for k, e in v2_etags.items() if e not in v1_values)
    print(f"v1 train objects {len(v1_etags)}, v2 train objects {len(v2_etags)}, "
          f"ETag candidates {len(candidates)}", flush=True)

    control_all = sorted((args.work / "v1" / "train").glob("*.bin"))
    assert len(control_all) == 275, len(control_all)
    v1_md5s = {md5(p) for p in control_all}
    print(f"md5 of {len(v1_md5s)} v1 train shards done", flush=True)

    v2_dir = args.repo / "data" / "shards_v2" / "train"
    sample_paths, rejected = pick_sample(candidates, v2_dir, v1_md5s, 75, random.Random(0))
    print(f"sample: {len(sample_paths)} v2 shards; rejected as identical to v1: {rejected}", flush=True)
    control = sorted(random.Random(0).sample(control_all, 75))

    val_paths = sorted((args.repo / "data" / "shards" / "val").glob("*.bin"))
    assert len(val_paths) == 6, len(val_paths)
    rng = np.random.default_rng(0)
    self_rate, self_n = self_test(control[:3], doc_id, rng)
    print(f"self-test: {self_rate:.3f} of {self_n} windows found in their own shards", flush=True)
    queries, origin = build_queries(val_paths, doc_id, args.n_queries, rng)
    ctrl_counts, ctrl_hits = scan(control, queries)
    print("control scan done", flush=True)
    samp_counts, samp_hits = scan(sample_paths, queries)
    print("sample scan done", flush=True)

    def tokens(paths):
        return sum(os.path.getsize(p) // 2 for p in paths)

    def show(hit):
        k, name, pos = hit
        return {"val_window": origin[k], "matched_in": name, "pos": pos,
                "text": tok.decode([int(t) for t in queries[k]])}

    report = {
        "queries": len(queries),
        "self_test": {"windows": self_n, "hit_rate": self_rate},
        "rejected_as_v1_copies": rejected,
        "control": {"shards": len(control), "tokens": tokens(control),
                    "queries_with_match": int((ctrl_counts > 0).sum()),
                    "total_matches": int(ctrl_counts.sum())},
        "sample": {"shards": len(sample_paths), "tokens": tokens(sample_paths),
                   "shard_names": [p.name for p in sample_paths],
                   "queries_with_match": int((samp_counts > 0).sum()),
                   "total_matches": int(samp_counts.sum())},
        "control_examples": [show(h) for h in ctrl_hits[:8]],
        "sample_examples": [show(h) for h in samp_hits[:12]],
    }
    out = args.work / "out" / "report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for label in ("control", "sample"):
        r = report[label]
        print(f"{label}: {r['queries_with_match']}/{len(queries)} val windows matched "
              f"({100 * r['queries_with_match'] / len(queries):.2f}%), "
              f"{r['total_matches']} total matches, {r['tokens']:,} tokens scanned", flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
