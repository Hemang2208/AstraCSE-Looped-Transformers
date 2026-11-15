"""Prefix and per-shard check of the two v2 runs on the GPU box (one short pause).

Why: loop6x2_v2's in-training prefix val (first 50 unstrided batches) reads about 1.6, but its
official strided value is 2.08. baseline_v2 reads about 2.9 on the prefix and 2.01 strided. This
script reproduces both protocols from the final checkpoints, and breaks the strided value down by
validation shard and by position decile, so the disagreement can be located.

  prefix:   src/train.py eval_loss protocol: first 50 batches of 8 sequences, cross-entropy inside
            autocast, mean of batch means.
  strided:  eval/run_eval.py protocol: every 8th batch of 4 sequences. The full value must match the
            reported number. Per-shard and per-decile token means are added on the same batches.

Read-only on the training data. Accepts the same flags as rescore_clean.py so pause_rescore.sh can
run it unchanged.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SEQ = 1024
MODELS = [
    ("loop6x2_v2", "configs/model_loop_6x2.yaml"),
    ("baseline_v2", "configs/model.yaml"),
]


def sequence_starts(lengths):
    for k, n in enumerate(lengths):
        pos = 0
        while pos + SEQ + 1 <= n:
            yield k, pos
            pos += SEQ


def prefix_loss(model, shards_dir, device, micro_bs=8, batches=50):
    # Same loader and loop as src/train.py eval_loss with max_batches=50.
    from src.data import make_dataloader
    loader, _ = make_dataloader(shards_dir, micro_bs, seq_len=SEQ, shuffle=False)
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i, (inp, tgt) in enumerate(loader):
            if i >= batches:
                break
            inp, tgt = inp.to(device), tgt.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(inp)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
            total += loss.item()
            count += 1
    return total / max(count, 1)


def strided_breakdown(model, val_tokens, device):
    # Same batches as eval/run_eval.py compute_val_loss (every 8th batch of 4 sequences).
    lengths = [t.shape[0] for t in val_tokens]
    offsets = np.cumsum([0] + lengths)
    total_tokens = int(offsets[-1])
    seqs = list(sequence_starts(lengths))
    vocab = model.cfg.vocab_size
    batch_means = []
    shard_sum = np.zeros(len(val_tokens))
    shard_n = np.zeros(len(val_tokens))
    dec_sum = np.zeros(10)
    dec_n = np.zeros(10)
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(seqs), 4):
            if (b0 // 4) % 8:
                continue
            chunk = seqs[b0:b0 + 4]
            inp = np.stack([val_tokens[k][p:p + SEQ] for k, p in chunk]).astype(np.int64)
            tgt = np.stack([val_tokens[k][p + 1:p + 1 + SEQ] for k, p in chunk]).astype(np.int64)
            inp_t = torch.from_numpy(inp).to(device)
            tgt_t = torch.from_numpy(tgt).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(inp_t)
            # Official mean, computed outside autocast as in compute_val_loss.
            batch_means.append(F.cross_entropy(logits.reshape(-1, vocab), tgt_t.reshape(-1)).item())
            tok = F.cross_entropy(logits.reshape(-1, vocab), tgt_t.reshape(-1),
                                  reduction="none").float().reshape(tgt_t.shape).cpu().numpy()
            for row, (k, p) in enumerate(chunk):
                shard_sum[k] += tok[row].sum()
                shard_n[k] += SEQ
                d = min(int((offsets[k] + p) * 10 / total_tokens), 9)
                dec_sum[d] += tok[row].sum()
                dec_n[d] += SEQ
    return {
        "strided_full": float(np.mean(batch_means)),
        "per_shard": [float(x) for x in shard_sum / np.maximum(shard_n, 1)],
        "per_decile": [float(x) for x in dec_sum / np.maximum(dec_n, 1)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="/home/ubuntu/AstraCSE")
    ap.add_argument("--val-dir", default="/home/ubuntu/AstraCSE/data/shards/val")
    ap.add_argument("--masks-dir", default="/home/ubuntu/astra_overlap/out")
    ap.add_argument("--work", default="/home/ubuntu/astra_overlap/ckpt")
    ap.add_argument("--out", default="/home/ubuntu/astra_overlap/out/subject_check.json")
    args = ap.parse_args()

    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, "/home/ubuntu/astra_overlap/scripts")
    os.chdir(repo)
    from eval.run_eval import load_model  # the repo's own loader
    import rescore_clean as rc  # bucket, checkpoint layout and reported values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_paths = sorted(Path(args.val_dir).glob("*.bin"))
    val_tokens = [np.fromfile(str(p), dtype=np.uint16) for p in val_paths]
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    results = []
    for name, config in MODELS:
        key = rc.s3_final_key(name)
        ckpt = work / f"{name}.pt"
        subprocess.run(["aws", "s3", "cp", f"s3://{rc.BUCKET}/{rc.ckpt_prefix(name)}{key}", str(ckpt),
                        "--region", rc.REGION, "--only-show-errors"], check=True)
        model, precision = load_model(ckpt, repo / config)
        model = model.to(device)
        prefix = prefix_loss(model, Path(args.val_dir), device)
        br = strided_breakdown(model, val_tokens, device)
        results.append({
            "name": name, "checkpoint": key, "precision": precision,
            "prefix_50x8": prefix,
            "strided_full": br["strided_full"],
            "reported": rc.reported_val_loss(name),
            "per_shard": br["per_shard"],
            "per_decile": br["per_decile"],
        })
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"{name}: prefix {prefix:.4f}, strided {br['strided_full']:.4f} "
              f"(reported {results[-1]['reported']:.4f})", flush=True)
        del model
        ckpt.unlink()
    print("subject check written", flush=True)


if __name__ == "__main__":
    main()
