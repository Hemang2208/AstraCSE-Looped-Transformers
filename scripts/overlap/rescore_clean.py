"""Re-score completed checkpoints on the clean validation tokens (GPU box).

Mirrors eval/run_eval.py::compute_val_loss: the same shards (sorted order), the
same sequence layout as src/data.py (1,024 targets per sequence, no overlap),
batches of 4 sequences, and every 8th batch. Each target token is labelled
copied (inside a verbatim copy of 100 or more tokens, from overlap_full_box.py)
or clean. The full-set value must reproduce the reported val_loss before the
clean value is trusted.
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
BATCH = 4
EVERY = 8
BUCKET = "your-bucket-name"
REGION = "us-east-1"

RUNS = [
    ("baseline", "configs/model.yaml"),
    ("dense_6l", "configs/model_dense_6l.yaml"),
    ("dense_4l", "configs/model_dense_4l.yaml"),
    ("dense_3l", "configs/model_dense_3l.yaml"),
    ("loop_6x2", "configs/model_loop_6x2.yaml"),
    ("loop_6x2_lr3e-4", "configs/model_loop_6x2_lr3e-4.yaml"),
    ("loop_4x3", "configs/model_loop_4x3.yaml"),
    ("loop_3x4", "configs/model_loop_3x4.yaml"),
    ("baseline_v2", "configs/model.yaml"),
    ("loop6x2_v2", "configs/model_loop_6x2.yaml"),
    ("d1024_v2", "configs/model_d1024.yaml"),
    ("d1024_ctx2048", "configs/model_d1024_ctx2048.yaml"),
    ("dense_6l_v2", "configs/model_dense_6l.yaml"),
    ("loop_6x2_wu171", "configs/model_loop_6x2_wu171.yaml"),
    ("baseline_s1", "configs/model.yaml"),
    ("loop_6x2_s1", "configs/model_loop_6x2.yaml"),
    ("dense_6l_s1", "configs/model_dense_6l.yaml"),
    ("d1024_loop6x2_v2", "configs/model_d1024_loop6x2.yaml"),
    ("baseline_v2_s1", "configs/model.yaml"),
]


def sequence_starts(lengths):
    for k, n in enumerate(lengths):
        pos = 0
        while pos + SEQ + 1 <= n:
            yield k, pos
            pos += SEQ


def rescore(model, val_tokens, copied_masks, device):
    """val_tokens: uint16 arrays in sorted shard order. copied_masks: bool arrays of the same lengths."""
    seqs = list(sequence_starts([t.shape[0] for t in val_tokens]))
    vocab = model.cfg.vocab_size
    full_sum = full_n = 0.0
    clean_sum = copied_sum = 0.0
    clean_n = copied_n = 0
    batches = 0
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(seqs), BATCH):
            if (b0 // BATCH) % EVERY:
                continue
            chunk = seqs[b0:b0 + BATCH]
            inp = np.stack([val_tokens[k][p:p + SEQ] for k, p in chunk]).astype(np.int64)
            tgt = np.stack([val_tokens[k][p + 1:p + 1 + SEQ] for k, p in chunk]).astype(np.int64)
            copied = np.stack([copied_masks[k][p + 1:p + 1 + SEQ] for k, p in chunk])
            inp_t = torch.from_numpy(inp).to(device)
            tgt_t = torch.from_numpy(tgt).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(inp_t)
            # Same call as eval/run_eval.py compute_val_loss, so the full-set value reproduces exactly.
            full_sum += F.cross_entropy(logits.reshape(-1, vocab), tgt_t.reshape(-1)).item()
            full_n += 1
            tok_loss = F.cross_entropy(logits.reshape(-1, vocab), tgt_t.reshape(-1),
                                       reduction="none").float().reshape(tgt_t.shape)
            copied_t = torch.from_numpy(copied).to(device)
            clean_t = ~copied_t
            clean_sum += tok_loss[clean_t].sum().item()
            clean_n += int(clean_t.sum())
            copied_sum += tok_loss[copied_t].sum().item()
            copied_n += int(copied_t.sum())
            batches += 1
    return {
        "full_val_loss": full_sum / max(full_n, 1),
        "clean_loss": clean_sum / clean_n if clean_n else None,
        "copied_loss": copied_sum / copied_n if copied_n else None,
        "clean_tokens": clean_n,
        "copied_tokens": copied_n,
        "batches": batches,
    }


def ckpt_prefix(name):
    # The queue keeps baseline in the unprefixed layout (checkpoints/); every other run is checkpoints/<name>/.
    return "checkpoints/" if name == "baseline" else f"checkpoints/{name}/"


def s3_final_key(name):
    out = subprocess.run(["aws", "s3", "ls", f"s3://{BUCKET}/{ckpt_prefix(name)}", "--region", REGION],
                         capture_output=True, text=True, check=True).stdout
    keys = [line.split()[-1] for line in out.splitlines() if "-final.pt" in line]
    if not keys:
        raise RuntimeError(f"no final checkpoint found for {name}")
    return keys[-1]


def reported_val_loss(name):
    out = subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/results/{name}/mmlu_cs.json", "-",
                          "--region", REGION], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["val_loss"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="/home/ubuntu/AstraCSE")
    ap.add_argument("--val-dir", default="/home/ubuntu/AstraCSE/data/shards/val")
    ap.add_argument("--masks-dir", default="/home/ubuntu/astra_overlap/out")
    ap.add_argument("--work", default="/home/ubuntu/astra_overlap/ckpt")
    ap.add_argument("--out", default="/home/ubuntu/astra_overlap/out/rescore.json")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from eval.run_eval import load_model  # the repo's own loader

    if args.threads:
        torch.set_num_threads(args.threads)
    use_cuda = torch.cuda.is_available() and args.device in ("auto", "cuda")
    device = torch.device("cuda" if use_cuda else "cpu")

    val_paths = sorted(Path(args.val_dir).glob("*.bin"))
    masks_dir = Path(args.masks_dir)
    manifest = json.loads((masks_dir / "val_long_manifest.json").read_text(encoding="utf-8"))
    assert manifest == [p.name for p in val_paths], "mask manifest does not match the validation shards"
    val_tokens = [np.fromfile(str(p), dtype=np.uint16) for p in val_paths]
    copied_masks = [np.load(masks_dir / f"val_long_mask_{k}.npy") for k in range(len(val_paths))]
    for t, m in zip(val_tokens, copied_masks):
        assert t.shape[0] == m.shape[0], "mask length does not match shard length"

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    wanted = set(args.runs) if args.runs else None
    results = []
    print(f"device: {device}; validation tokens: {sum(t.shape[0] for t in val_tokens):,}", flush=True)
    for name, config in RUNS:
        if wanted and name not in wanted:
            continue
        key = s3_final_key(name)
        ckpt = work / f"{name}.pt"
        subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{ckpt_prefix(name)}{key}", str(ckpt),
                        "--region", REGION, "--only-show-errors"], check=True)
        model, precision = load_model(ckpt, repo / config)
        model = model.to(device)
        r = rescore(model, val_tokens, copied_masks, device)
        reported = reported_val_loss(name)
        r.update({"name": name, "config": config, "checkpoint": key, "precision": precision,
                  "reported_val_loss": reported,
                  "reproduces_reported": abs(r["full_val_loss"] - reported) < 1e-3})
        results.append(r)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"{name}: full {r['full_val_loss']:.4f} (reported {reported:.4f}, "
              f"reproduces={r['reproduces_reported']}), clean {r['clean_loss']:.4f}, "
              f"copied {r['copied_loss']:.4f}", flush=True)
        del model
        ckpt.unlink()
    print("re-score written", flush=True)


if __name__ == "__main__":
    main()
