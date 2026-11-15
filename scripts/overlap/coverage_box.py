"""Full-coverage overlap check on the GPU box (run at nice 19 and idle I/O).

Every 13-token window of each pool shard is looked up in the set of all windows
of the v1 validation shards. A run of consecutive matched windows is a verbatim
token run shared with val; its token length is windows + 12. Long runs are copied
passages. Short runs are boilerplate.

  control: the 75 v1 training shards used for the control in screen_box.py
  sample:  the 75 new v2 shards accepted in screen_box.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from tokenizers import ByteLevelBPETokenizer

sys.path.insert(0, "/home/ubuntu/astra_overlap/scripts")
import screen_box as sb  # noqa: E402

WINDOW = sb.WINDOW


def runs(mask):
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(padded)
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return starts, ends - starts


def scan_pool(paths, val_sorted, tok):
    stats = {"shards": len(paths), "tokens": 0, "windows": 0, "matched_windows": 0,
             "runs_ge_30": 0, "runs_ge_100": 0, "runs_ge_200": 0, "max_run_tokens": 0}
    best = []
    for p in paths:
        arr = sb.load(p)
        h = sb.window_hashes(arr)
        idx = np.minimum(np.searchsorted(val_sorted, h), len(val_sorted) - 1)
        mask = val_sorted[idx] == h
        stats["tokens"] += int(arr.shape[0])
        stats["windows"] += int(mask.shape[0])
        stats["matched_windows"] += int(mask.sum())
        starts, lens = runs(mask)
        tok_lens = lens + (WINDOW - 1)
        stats["runs_ge_30"] += int((tok_lens >= 30).sum())
        stats["runs_ge_100"] += int((tok_lens >= 100).sum())
        stats["runs_ge_200"] += int((tok_lens >= 200).sum())
        if tok_lens.size:
            stats["max_run_tokens"] = max(stats["max_run_tokens"], int(tok_lens.max()))
        for s, t in zip(starts, tok_lens):
            if t >= 30:
                best.append((int(t), p, int(s)))
    best.sort(key=lambda x: -x[0])
    examples = []
    for t, p, s in best[:8]:
        examples.append({"shard": p.name, "start": s, "tokens": t,
                         "text": tok.decode([int(x) for x in sb.load(p)[s:s + t]])[:300]})
    return {"stats": stats, "top_runs": examples}


def main():
    repo = Path("/home/ubuntu/AstraCSE")
    work = Path("/home/ubuntu/astra_overlap")
    tok = ByteLevelBPETokenizer(str(repo / "tokenizer" / "vocab.json"), str(repo / "tokenizer" / "merges.txt"))
    report = json.loads((work / "out" / "report.json").read_text(encoding="utf-8"))
    control_all = sorted((work / "v1" / "train").glob("*.bin"))
    control = sorted(random.Random(0).sample(control_all, 75))
    sample = [repo / "data" / "shards_v2" / "train" / n for n in report["sample"]["shard_names"]]
    val = sorted((repo / "data" / "shards" / "val").glob("*.bin"))
    val_sorted = np.sort(np.concatenate([sb.window_hashes(sb.load(p)) for p in val]))
    print(f"val windows {val_sorted.shape[0]:,} from {len(val)} shards", flush=True)
    out = {}
    for label, paths in (("control", control), ("sample", sample)):
        out[label] = scan_pool(paths, val_sorted, tok)
        print(label, out[label]["stats"], flush=True)
    (work / "out" / "coverage.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("coverage report written", flush=True)


if __name__ == "__main__":
    main()
