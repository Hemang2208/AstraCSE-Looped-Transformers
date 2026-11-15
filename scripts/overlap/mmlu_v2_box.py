"""MMLU-CS overlap scan over the v2 training shards that the v2 runs use (GPU box).

Uses the repo's own rule and functions (scripts/check_contamination.py): an item is
flagged when more than 1% of its unique 13-grams occur in the shards. Read-only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ubuntu/AstraCSE/scripts")
sys.path.insert(0, "/home/ubuntu/astra_overlap/scripts")
import check_contamination as cc  # noqa: E402
import overlap_full_box as ob  # noqa: E402

REPO_TOK = Path("/home/ubuntu/AstraCSE/tokenizer")
MMLU = Path("/home/ubuntu/AstraCSE/eval/mmlu_cs_benchmark.json")
SHARDS = Path("/home/ubuntu/AstraCSE/data/shards_v2/train")
OUT = Path("/home/ubuntu/astra_overlap/out/mmlu_v2.json")

n = cc.NGRAM_SIZE
tok = cc.load_tokenizer(REPO_TOK)
items = ob.mmlu_items(cc, tok, MMLU, n)
all_mmlu = np.unique(np.concatenate([it["hashes"] for it in items]))
print(f"MMLU-CS items: {len(items)}, unique {n}-grams: {all_mmlu.shape[0]:,}", flush=True)
shards = sorted(SHARDS.rglob("*.bin"))
print(f"v2 train shards: {len(shards)}", flush=True)

found = cc.scan_shards(SHARDS, all_mmlu, n)
found_arr = np.array(sorted(found), dtype=np.uint64)
rows = []
for it in items:
    seen = int(np.isin(it["hashes"], found_arr).sum()) if found_arr.size else 0
    total = int(it["hashes"].shape[0])
    rows.append({"id": it["id"], "subject": it["subject"], "ngrams": total, "found": seen,
                 "fraction": seen / total})
flagged = [r for r in rows if r["fraction"] > cc.MAX_OVERLAP_FRACTION]
result = {"rule": f"flag if more than {cc.MAX_OVERLAP_FRACTION:.0%} of unique {n}-grams occur in the shards",
          "shards": len(shards), "items": len(rows), "flagged": len(flagged),
          "any_hit": sum(1 for r in rows if r["found"] > 0),
          "unique_ngrams_seen": len(found), "top_items": sorted(rows, key=lambda r: -r["fraction"])[:10]}
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"MMLU-CS in v2 train: {result['flagged']}/{result['items']} flagged, "
      f"{result['any_hit']} with any hit, {len(found):,} benchmark 13-grams seen", flush=True)
