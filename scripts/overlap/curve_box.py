"""Validation loss over training for the two v2 runs (GPU box, one short pause).

Scores periodic checkpoints of baseline_v2 and loop6x2_v2 with the same protocol as the official evaluation
(rescore_clean.rescore: every 8th batch of 4 x 1,024 tokens, fresh data, full and clean tokens). The final
checkpoint of one run is scored again and must reproduce its reported value, which checks the pipeline.

Accepts the flags pause_rescore.sh passes (--repo --val-dir --masks-dir --work --out) plus --points.
Read-only on the training data. Each checkpoint is deleted after it is scored.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def parse_points(items):
    out = []
    for item in items:
        run, step = item.split(":")
        out.append((run, int(step)))
    return out


def evaluate_points(points, check_final, fetch, load, reported, rescore, val_tokens, masks, device, log, write):
    """fetch(run, step or None) -> local path; load(path, run) -> model; reported(run) -> float."""
    results = []
    jobs = [(run, step) for run, step in points] + [(check_final, None)] if check_final else list(points)
    for run, step in jobs:
        path = fetch(run, step)
        model = load(path, run).to(device)
        r = rescore(model, val_tokens, masks, device)
        row = {"name": run, "step": step if step is not None else "final",
               "full_val_loss": r["full_val_loss"], "clean_loss": r["clean_loss"], "batches": r["batches"]}
        if step is None:
            row["reported_val_loss"] = reported(run)
            row["reproduces_reported"] = abs(r["full_val_loss"] - row["reported_val_loss"]) < 1e-3
        results.append(row)
        write(results)
        log(f"{run} step {row['step']}: full {r['full_val_loss']:.4f}, clean {r['clean_loss']:.4f}"
            + (f" (reported {row['reported_val_loss']:.4f}, reproduces={row['reproduces_reported']})" if step is None else ""))
        del model
        os.remove(path)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="/home/ubuntu/AstraCSE")
    ap.add_argument("--val-dir", default="/home/ubuntu/AstraCSE/data/shards/val")
    ap.add_argument("--masks-dir", default="/home/ubuntu/astra_overlap/out")
    ap.add_argument("--work", default="/home/ubuntu/astra_overlap/ckpt")
    ap.add_argument("--out", default="/home/ubuntu/astra_overlap/out/curve.json")
    ap.add_argument("--points", nargs="+", required=True, help="run:step, e.g. baseline_v2:2400")
    ap.add_argument("--check-final", default="loop6x2_v2")
    args = ap.parse_args()

    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, "/home/ubuntu/astra_overlap/scripts")
    os.chdir(repo)
    from eval.run_eval import load_model
    import rescore_clean as rc

    configs = dict(rc.RUNS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_paths = sorted(Path(args.val_dir).glob("*.bin"))
    manifest = json.loads((Path(args.masks_dir) / "val_long_manifest.json").read_text(encoding="utf-8"))
    assert manifest == [p.name for p in val_paths], "mask manifest does not match the validation shards"
    val_tokens = [np.fromfile(str(p), dtype=np.uint16) for p in val_paths]
    masks = [np.load(Path(args.masks_dir) / f"val_long_mask_{k}.npy") for k in range(len(val_paths))]
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    def fetch(run, step):
        key = rc.s3_final_key(run) if step is None else f"step-{step:07d}.pt"
        local = work / f"{run}-{key}"
        subprocess.run(["aws", "s3", "cp", f"s3://{rc.BUCKET}/{rc.ckpt_prefix(run)}{key}", str(local),
                        "--region", rc.REGION, "--only-show-errors"], check=True)
        return local

    def load(path, run):
        model, _ = load_model(path, repo / configs[run])
        return model

    def write(results):
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"device: {device}; points: {len(args.points)} + final check of {args.check_final}", flush=True)
    evaluate_points(parse_points(args.points), args.check_final, fetch, load, rc.reported_val_loss, rc.rescore,
                    val_tokens, masks, device, lambda m: print(m, flush=True), write)
    print("curve written", flush=True)


if __name__ == "__main__":
    main()
