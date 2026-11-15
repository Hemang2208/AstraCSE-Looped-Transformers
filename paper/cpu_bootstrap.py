"""CPU-only work for the review revision (no GPU, no cloud).

1. Statistics at initialisation for Dense-6 and Loop-6x2 (seed 0, first official validation batch):
   residual-stream std and variance after every layer application, output-projection std, gradient norms.
2. Validation losses per 1,024-token sequence on the official subset (every 8th batch of the v1 validation
   split, as in eval/run_eval.py). The full official subset is used for Dense-6 (v2) to check the pipeline against
   the reported number; the other models use every k-th batch of that subset, with k chosen so the run fits the
   time budget.
3. Paired bootstrap over batches (each batch = four sequences) for the matched and compute-matched comparisons.

Weights are the verified fp32 files kept in the local backup. Outputs go to paper/data/ (JSON).
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
BACKUP_W = Path(os.environ["ASTRA_BACKUP"]) / "weights"
VAL_DIR = Path(os.environ["ASTRA_BACKUP"]) / "data" / "shards_v1" / "val"
RESULTS = REPO / "paper" / "data" / "results"
OUT = REPO / "paper" / "data"
WORK = Path(os.environ.get("ASTRA_WORK", Path.home() / "astra_cpu_work"))
WORK.mkdir(parents=True, exist_ok=True)
BUDGET_S = 100 * 60
N_OFFICIAL = 738

sys.path.insert(0, str(REPO))
from src.model import AstraCSE, ModelConfig  # noqa: E402
from src.data import make_dataloader  # noqa: E402
from eval.run_eval import load_model  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


MODELS = {
    "dense_6l_v2": "model_dense_6l.yaml",
    "loop6x2_v2": "model_loop_6x2.yaml",
    "dense_6l_s1": "model_dense_6l.yaml",
    "loop_6x2_s1": "model_loop_6x2.yaml",
    "baseline_s1": "model.yaml",
    "baseline_v2": "model.yaml",
}
PAIRS = {
    "v1_seed1_matched": ("loop_6x2_s1", "dense_6l_s1"),
    "v2_matched": ("loop6x2_v2", "dense_6l_v2"),
    "v1_seed1_compute": ("loop_6x2_s1", "baseline_s1"),
    "v2_compute": ("loop6x2_v2", "baseline_v2"),
}


def first_batch():
    loader, _ = make_dataloader(VAL_DIR, batch_size=4, shuffle=False)
    for inp, tgt in loader:
        return inp, tgt


# ---------------------------------------------------------------- 1. statistics at initialisation
def init_stats(cfg_name, inp, tgt):
    cfg = ModelConfig.from_yaml(REPO / "configs" / cfg_name)
    torch.manual_seed(0)
    model = AstraCSE(cfg)
    model.train()
    n_unique = len(model.blocks)
    stats = {"n_applications": cfg.n_layer, "n_unique_blocks": n_unique,
             "target_out_proj_std": 0.02 / (2 * cfg.n_layer) ** 0.5}
    with torch.no_grad():
        x = model.embed(inp)
        stats["embedding_std"] = float(x.std())
        res_std, res_var = [], []
        for i in range(cfg.n_layer):
            block = model.blocks[i % n_unique]
            if model.depth_embed is not None:
                x = x + model.depth_embed[i]
            x, _ = block(x, None)
            res_std.append(float(x.std()))
            res_var.append(float(x.var()))
    stats["residual_std_by_application"] = res_std
    stats["residual_var_by_application"] = res_var
    stats["out_proj_std_block0"] = float(model.blocks[0].attn.out_proj.weight.std())
    stats["mlp_down_std_block0"] = float(model.blocks[0].mlp.down.weight.std())
    model.zero_grad()
    logits, _ = model(inp)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), tgt.view(-1))
    loss.backward()
    stats["initial_loss"] = float(loss)
    block_norms = []
    for blk in model.blocks:
        sq = sum(float(p.grad.pow(2).sum()) for p in blk.parameters() if p.grad is not None)
        block_norms.append(sq ** 0.5)
    stats["grad_norm_per_unique_block"] = block_norms
    if model.depth_embed is not None:
        stats["grad_norm_depth_embed"] = float(model.depth_embed.grad.norm())
    total = sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None)
    stats["grad_norm_total"] = total ** 0.5
    return stats


# ---------------------------------------------------------------- 2. per-batch validation losses
def batch_losses(model, inp, tgt, use_bf16):
    with torch.inference_mode():
        if use_bf16:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits, _ = model(inp)
        else:
            logits, _ = model(inp)
        tok = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none")
        return tok.view(inp.shape[0], inp.shape[1]).mean(dim=1).double().numpy()


def official_val_loss(run):
    path = RESULTS / f"{run}.mmlu_cs.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["val_loss"]


def main():
    started = time.time()
    summary = {"started": time.strftime("%Y-%m-%d %H:%M"), "cpu_threads": torch.get_num_threads()}

    inp, tgt = first_batch()
    log(f"first official batch shape {tuple(inp.shape)}")

    # ---- part 1: initialisation statistics (fast)
    init = {"dense_6l": init_stats("model_dense_6l.yaml", inp, tgt),
            "loop_6x2": init_stats("model_loop_6x2.yaml", inp, tgt)}
    (OUT / "init_stats.json").write_text(json.dumps(init, indent=1), encoding="utf-8")
    log("init statistics written: " + ", ".join(f"{k} grad_norm={v['grad_norm_total']:.4f}" for k, v in init.items()))

    # ---- part 2: load models and time one batch each
    models = {}
    for run, cfg_name in MODELS.items():
        model, precision = load_model(BACKUP_W / f"{run}.weights.pt", REPO / "configs" / cfg_name)
        models[run] = model
        log(f"loaded {run} ({precision})")

    t_fp32 = {}
    t_bf16 = {}
    t0 = time.time(); batch_losses(models["loop_6x2_s1"], inp, tgt, False); t_fp32["loop_6x2_s1"] = time.time() - t0
    t0 = time.time(); batch_losses(models["loop_6x2_s1"], inp, tgt, True); t_bf16["loop_6x2_s1"] = time.time() - t0
    use_bf16 = t_bf16["loop_6x2_s1"] < t_fp32["loop_6x2_s1"]
    log(f"12-layer batch: fp32 {t_fp32['loop_6x2_s1']:.2f}s, bf16 autocast {t_bf16['loop_6x2_s1']:.2f}s -> use bf16={use_bf16}")
    per_batch_s = {}
    for run in MODELS:
        t0 = time.time(); batch_losses(models[run], inp, tgt, use_bf16); per_batch_s[run] = time.time() - t0
    summary["dtype_on_cpu"] = "bf16 autocast" if use_bf16 else "fp32"
    summary["seconds_per_batch"] = per_batch_s

    # ---- choose the subsampling factor k so the run fits the budget
    def total_seconds(set_runs, k):
        tot = 0.0
        for run in set_runs:
            kk = 1 if run == "dense_6l_v2" else k
            tot += (N_OFFICIAL / kk) * per_batch_s[run]
        return tot

    candidates = []
    big = list(MODELS)
    small = ["dense_6l_v2", "loop6x2_v2", "dense_6l_s1", "loop_6x2_s1"]
    for label, runs in [("all six", big), ("four (no compute-matched)", small)]:
        for k in [2, 4, 8, 16]:
            if total_seconds(runs, k) <= BUDGET_S:
                candidates.append((label, runs, k))
                break
        if candidates:
            break
    if not candidates:
        raise SystemExit("no subsampling factor fits the time budget")
    label, active, k_loop = candidates[0]
    k_of = {run: (1 if run == "dense_6l_v2" else k_loop) for run in active}
    summary["model_set"] = label
    summary["k_subsampling"] = k_of
    summary["estimated_hours"] = round(total_seconds(active, k_loop) / 3600, 2)
    log(f"model set: {label}; subsampling k={k_loop} (dense_6l_v2 uses all batches); "
        f"estimated {summary['estimated_hours']} h")

    # ---- part 3: one pass over the validation loader
    per_batch = {run: {} for run in active}
    loader, _ = make_dataloader(VAL_DIR, batch_size=4, shuffle=False)
    done = 0
    for i, (x_in, y_tgt) in enumerate(loader):
        if i % 8:
            continue
        j = i // 8
        for run in active:
            if j % k_of[run]:
                continue
            per_batch[run][j] = float(batch_losses(models[run], x_in, y_tgt, use_bf16).mean())
        done += 1
        if done % 40 == 0:
            log(f"official batches processed: {done} / {N_OFFICIAL}")
            (WORK / "checkpoint_per_batch.json").write_text(json.dumps(per_batch), encoding="utf-8")
    assert len(per_batch["dense_6l_v2"]) == N_OFFICIAL, ("official subset size", len(per_batch["dense_6l_v2"]))
    log("pass complete")
    (WORK / "checkpoint_per_batch.json").write_text(json.dumps(per_batch), encoding="utf-8")

    # ---- sanity check: full official subset for dense_6l_v2 against the reported number
    full = np.array([per_batch["dense_6l_v2"][j] for j in sorted(per_batch["dense_6l_v2"])])
    summary["sanity_dense_6l_v2"] = {
        "batches": int(full.size),
        "cpu_mean": float(full.mean()),
        "reported_val_loss": official_val_loss("dense_6l_v2"),
    }
    summary["sanity_dense_6l_v2"]["difference"] = (summary["sanity_dense_6l_v2"]["cpu_mean"]
                                                   - summary["sanity_dense_6l_v2"]["reported_val_loss"])
    log(f"sanity dense_6l_v2: cpu {full.mean():.4f} vs reported {summary['sanity_dense_6l_v2']['reported_val_loss']:.4f}")

    # ---- part 4: subset means and batch-level paired bootstrap
    rng = np.random.default_rng(0)
    model_stats = {}
    for run in active:
        vals = list(per_batch[run].values())
        model_stats[run] = {"subset_mean": float(np.mean(vals)), "batches": len(vals),
                            "reported_full_val_loss": official_val_loss(run)}
    summary["models"] = model_stats

    pair_results = {}
    for name, (loop_run, dense_run) in PAIRS.items():
        if loop_run not in active or dense_run not in active:
            pair_results[name] = {"status": "not run (model not in the time budget)"}
            continue
        js = sorted(set(per_batch[loop_run]) & set(per_batch[dense_run]))
        d = np.array([per_batch[loop_run][j] - per_batch[dense_run][j] for j in js])
        boot = np.empty(10000)
        for b in range(10000):
            boot[b] = d[rng.integers(0, d.size, d.size)].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pair_results[name] = {
            "loop": loop_run, "dense": dense_run,
            "mean_difference": float(d.mean()), "ci95": [float(lo), float(hi)],
            "bootstrap_se": float(boot.std()), "batches": int(d.size),
            "subsampled": bool(k_loop > 1 and dense_run != "dense_6l_v2" or k_loop > 1),
            "unit": "batch of four 1,024-token sequences",
        }
        log(f"{name}: {loop_run} - {dense_run} = {d.mean():+.4f} nats, 95% CI [{lo:+.4f}, {hi:+.4f}], n={d.size}")
    summary["pairs"] = pair_results
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M")
    summary["hours"] = round((time.time() - started) / 3600, 2)
    (OUT / "cpu_bootstrap.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log("outputs written: paper/data/init_stats.json, paper/data/cpu_bootstrap.json")


if __name__ == "__main__":
    try:
        main()
        (WORK / "DONE.txt").write_text("ok\n", encoding="utf-8")
    except Exception:
        (WORK / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
