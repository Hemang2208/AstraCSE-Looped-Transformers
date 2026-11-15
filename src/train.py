"""AstraCSE-100M training loop — Architecture.md §16–21.

Features:
  - AdamW + cosine LR schedule with warmup (§16.1–16.2)
  - BF16 mixed precision via torch.autocast (§15)
  - NaN/Inf guard with consecutive-failure escalation (§21.1)
  - OOM auto-recovery with micro-batch halving (§21.3)
  - Checkpoint save/load with full state incl. RNG restoration (§17)
  - SIGTERM emergency checkpoint handler (§18)
  - Early stopping on validation loss plateau (§21.5)
  - S3 upload with checksum verification (§17.2)

Usage:
    python src/train.py --config configs/model.yaml \
        --train-shards data/shards/train \
        --val-shards data/shards/val \
        --checkpoint-dir checkpoints/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DataloaderState, make_dataloader
from src.model import AstraCSE, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("train")
# Explicit, so INFO lines reach train.log even when the root logger was configured
# elsewhere first (basicConfig is then a no-op and the root stays at WARNING).
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# LR Schedule (§16.2)
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, total_steps: int,
           peak_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint helpers (§17)
# ---------------------------------------------------------------------------

def _ckpt_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(
    step: int,
    model: AstraCSE,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict,
    dl_state: DataloaderState,
    config_hash: str,
    tok_hash: str,
    ckpt_dir: Path,
    tag: str = "",
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    label = f"step-{step:07d}" + (f"-{tag}" if tag else "")
    ckpt_path = ckpt_dir / f"{label}.pt"

    payload = {
        "global_step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler_state,
        "dataloader_state": dl_state.to_dict(),
        "rng_state": {
            "torch": torch.get_rng_state().tolist(),
            "cuda": torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None,
        },
        "metadata": {
            "config_hash": config_hash,
            "tokenizer_hash": tok_hash,
            "timestamp": time.time(),
        },
    }
    torch.save(payload, ckpt_path)
    digest = _ckpt_hash(ckpt_path)
    ckpt_path.with_suffix(".sha256").write_text(digest, encoding="utf-8")
    log.info("Checkpoint saved: %s  sha256=%s", ckpt_path.name, digest[:16])
    return ckpt_path


def load_checkpoint(ckpt_path: Path, model: AstraCSE,
                    optimizer: torch.optim.Optimizer) -> dict:
    sha_path = ckpt_path.with_suffix(".sha256")
    if sha_path.exists():
        actual = _ckpt_hash(ckpt_path)
        stored = sha_path.read_text().strip()
        if actual != stored:
            raise RuntimeError(
                f"Checkpoint corruption: {ckpt_path.name} "
                f"hash {actual[:16]} != expected {stored[:16]}"
            )
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])

    # Restore RNG state so stochastic ops reproduce after resume (§17.1)
    rng = payload.get("rng_state", {})
    if rng.get("torch"):
        torch.set_rng_state(torch.tensor(rng["torch"], dtype=torch.uint8))
    if rng.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state(torch.tensor(rng["cuda"], dtype=torch.uint8))

    log.info("Resumed from step %d  (%s)", payload["global_step"], ckpt_path.name)
    return payload


PERIODIC_CKPT_RE = re.compile(r"step-(\d{7})\.pt")
BEST_CKPT_RE = re.compile(r"step-(\d{7})-best\.pt")


def prune_local_checkpoints(ckpt_dir: Path, keep: int, pattern: re.Pattern) -> list[Path]:
    """Delete all but the newest `keep` local checkpoints whose name fully matches
    `pattern`, with their .sha256 files. Call only after the newest one is in S3:
    S3 keeps every checkpoint, this only bounds local disk use on long runs."""
    found = sorted((int(m.group(1)), p) for p in ckpt_dir.glob("*.pt")
                   if (m := pattern.fullmatch(p.name)))
    removed = []
    for _, p in found[:max(len(found) - keep, 0)]:
        p.unlink(missing_ok=True)
        p.with_suffix(".sha256").unlink(missing_ok=True)
        removed.append(p)
    if removed:
        log.info("Pruned local checkpoints: %s", ", ".join(p.name for p in removed))
    return removed


FINAL_CKPT_RE = re.compile(r"step-(\d{7})-final\.pt")


def newest_final_key(s3, bucket: str, prefix: str) -> str:
    """Key of the highest-step step-NNNNNNN-final.pt directly under prefix."""
    best = None
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(prefix):]
            m = FINAL_CKPT_RE.fullmatch(name)
            if m and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), obj["Key"])
    if best is None:
        raise FileNotFoundError(f"no final checkpoint under s3://{bucket}/{prefix}")
    return best[1]


def init_weights(source: str, model: AstraCSE) -> Path:
    """Load only the model weights of a checkpoint: a local path, an s3:// URI, or an
    s3:// prefix ending in "/" (its newest final checkpoint, so a follow-up phase can
    be queued before the run it starts from has finished). The .sha256 next to a
    local checkpoint is checked when present."""
    path = Path(source)
    if source.startswith("s3://"):
        import boto3
        s3 = boto3.client("s3")
        bucket, key = source.replace("s3://", "").split("/", 1)
        if key.endswith("/"):
            key = newest_final_key(s3, bucket, key)
            source = f"s3://{bucket}/{key}"
        # Outside ckpt_dir: the queue uploads every .pt in a job's directory to S3,
        # and this copy of another run's checkpoint does not belong there.
        init_dir = Path(tempfile.gettempdir()) / "astra-init"
        init_dir.mkdir(parents=True, exist_ok=True)
        path = init_dir / key.replace("/", "__")
        if not path.exists():
            s3.download_file(bucket, key, str(path))
    sha_path = path.with_suffix(".sha256")
    if sha_path.exists() and _ckpt_hash(path) != sha_path.read_text().strip():
        raise RuntimeError(f"Checkpoint corruption: {path.name}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    log.info("Initialised weights from %s (step %s)", source, payload.get("global_step"))
    return path


def _try_s3_upload(local_path: Path, s3_uri: str) -> bool:
    try:
        import boto3
    except ImportError:
        log.warning("boto3 not installed — skipping S3 upload.")
        return False
    try:
        bucket, key = s3_uri.replace("s3://", "").split("/", 1)
        boto3.client("s3").upload_file(str(local_path), bucket, key)
        log.info("Uploaded %s → s3://%s/%s", local_path.name, bucket, key)
        return True
    except Exception as exc:
        log.warning("S3 upload failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model: AstraCSE,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    device: torch.device,
    accum_steps: int,
) -> Optional[float]:
    """Forward + backward for one micro-batch; returns unscaled loss or None on NaN/Inf."""
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits, _ = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
        ) / accum_steps

    if not loss.isfinite():
        return None

    loss.backward()
    return loss.item() * accum_steps


@torch.no_grad()
def eval_loss(model: AstraCSE, val_loader, device: torch.device,
              max_batches: int = 50) -> float:
    model.eval()
    total = 0.0
    count = 0
    for i, (inp, tgt) in enumerate(val_loader):
        if i >= max_batches:
            break
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(inp)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), tgt.view(-1))
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args) -> None:
    # The run's own log next to its checkpoints, so the queue can archive the loss
    # and validation curves to S3 with them (appended across resumes).
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(Path(args.checkpoint_dir) / "train.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    try:
        _train(args)
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def _train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    # Recorded so every run can be repeated; random unless --seed is given.
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    torch.manual_seed(seed)
    log.info("seed=%d", seed)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)

    model_cfg = ModelConfig.from_yaml(cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    tok_version = Path(args.tok_dir) / "tokenizer_version.json"
    tok_hash = (json.loads(tok_version.read_text(encoding="utf-8"))["combined_hash"]
                if tok_version.exists() else "unknown")

    model = AstraCSE(model_cfg).to(device)
    model.verify_param_count()
    log.info("Model parameters: %d", model.count_params())

    opt_cfg = full_cfg.get("optimizer", {})
    decay_params   = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": opt_cfg.get("weight_decay", 0.1)},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=full_cfg["lr_schedule"]["peak_lr"],
        betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
        eps=opt_cfg.get("eps", 1e-8),
    )

    # Resume
    resume_state: Optional[DataloaderState] = None
    start_step = 0
    best_val_loss = float("inf")
    val_plateau_count = 0

    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            log.error("Resume checkpoint not found: %s", ckpt_path)
            sys.exit(1)
        payload = load_checkpoint(ckpt_path, model, optimizer)
        start_step = payload["global_step"]
        resume_state = DataloaderState.from_dict(payload["dataloader_state"])
        sched = payload.get("scheduler_state", {})
        best_val_loss = sched.get("best_val_loss", float("inf"))
        val_plateau_count = sched.get("val_plateau_count", 0)
    elif args.init_from:
        # A new training phase (e.g. context extension) from a finished run: weights
        # only, with a fresh optimizer, LR schedule and data order. --resume, when
        # the bootstrap finds a checkpoint of this phase, takes priority.
        init_weights(args.init_from, model)

    micro_bs = args.micro_batch_size
    seq_len = model_cfg.max_seq_len
    target_effective = full_cfg["training"].get("target_effective_batch_tokens", 500_000)
    grad_accum = max(1, target_effective // (micro_bs * seq_len))
    log.info("micro_bs=%d  grad_accum=%d  effective_batch_tokens=%d",
             micro_bs, grad_accum, micro_bs * seq_len * grad_accum)

    ckpt_dir = Path(args.checkpoint_dir)
    # Sequences of the model's context length (grad_accum above assumes it). The
    # loader's default is 1024, the length of every config before context extension.
    train_loader, train_ds = make_dataloader(
        Path(args.train_shards), micro_bs, seq_len=seq_len, shuffle=True,
        resume_state=resume_state)
    val_loader, _ = make_dataloader(
        Path(args.val_shards), micro_bs, seq_len=seq_len, shuffle=False)

    lr_cfg = full_cfg["lr_schedule"]
    peak_lr = lr_cfg["peak_lr"]
    min_lr = peak_lr * lr_cfg.get("min_lr_ratio", 0.1)
    total_steps = args.total_steps
    warmup_steps = max(1, int(total_steps * lr_cfg.get("warmup_ratio", 0.02)))
    grad_clip = opt_cfg.get("grad_clip_norm", 1.0)

    _sigterm_received = [False]

    def _sigterm_handler(signum, frame):  # noqa: ARG001
        log.warning("SIGTERM received — will checkpoint after current step.")
        _sigterm_received[0] = True

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # torch.compile wraps the model for the forward/backward passes only; checkpoints,
    # resume and --init-from keep using the plain module, so their state-dict keys
    # do not change. Off by default.
    train_model = torch.compile(model) if args.compile else model
    if args.compile:
        log.info("torch.compile enabled")

    model.train()
    step = start_step
    nan_counter = 0
    loss_history: list[float] = []
    spike_counter = 0
    last_ckpt_step = step

    log.info("Training from step %d", step)
    optimizer.zero_grad()
    accum_count = 0

    for input_ids, target_ids in train_loader:
        if step >= total_steps:
            break

        lr = get_lr(step, warmup_steps, total_steps, peak_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        loss_val = None
        try:
            loss_val = train_step(train_model, input_ids, target_ids, device, grad_accum)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            micro_bs = max(1, micro_bs // 2)
            grad_accum = max(1, target_effective // (micro_bs * seq_len))
            log.warning("OOM — micro_bs→%d  grad_accum→%d (§21.3)", micro_bs, grad_accum)
            optimizer.zero_grad()
            accum_count = 0
            continue

        # NaN/Inf guard (§21.1)
        if loss_val is None:
            nan_counter += 1
            log.warning("NaN/Inf step=%d consecutive=%d", step, nan_counter)
            optimizer.zero_grad()
            accum_count = 0
            if nan_counter >= 3:
                log.error("3 consecutive NaN/Inf — halting and checkpointing.")
                save_checkpoint(step, model, optimizer,
                                {"best_val_loss": best_val_loss,
                                 "val_plateau_count": val_plateau_count},
                                train_ds.state, config_hash, tok_hash,
                                ckpt_dir, tag="nan-halt")
                return
            continue

        nan_counter = 0
        accum_count += 1

        if accum_count < grad_accum:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        accum_count = 0
        step += 1

        loss_history.append(loss_val)

        # Loss spike detection (§21.2)
        if len(loss_history) >= 100:
            moving_avg = sum(loss_history[-100:]) / 100
            if loss_val > 3 * moving_avg:
                spike_counter += 1
                log.warning("Spike step=%d loss=%.4f avg=%.4f count=%d",
                            step, loss_val, moving_avg, spike_counter)
                if spike_counter > 5:
                    log.warning("5+ spikes — consider lowering LR on resume.")

        if step % args.log_every == 0:
            log.info("step=%d  loss=%.4f  lr=%.2e", step, loss_val, lr)

        # Validation + early stopping (§21.5)
        if step % args.eval_every == 0:
            val_loss = eval_loss(train_model, val_loader, device)
            log.info("step=%d  val_loss=%.4f", step, val_loss)
            sched_state = {"best_val_loss": best_val_loss,
                           "val_plateau_count": val_plateau_count}
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_plateau_count = 0
                p_best = save_checkpoint(step, model, optimizer, sched_state,
                                         train_ds.state, config_hash, tok_hash,
                                         ckpt_dir, tag="best")
                # With --keep-local, upload each best checkpoint under its own name
                # (the end-of-run sync then skips it) and keep only the newest locally.
                if args.s3_prefix and args.keep_local > 0:
                    if _try_s3_upload(p_best, f"{args.s3_prefix}/{p_best.name}"):
                        prune_local_checkpoints(ckpt_dir, 1, BEST_CKPT_RE)
            else:
                val_plateau_count += 1
                if args.early_stop_patience > 0 and val_plateau_count >= args.early_stop_patience:
                    log.warning("Early stopping: val loss plateaued (§21.5).")
                    break

        # Periodic checkpoint (§17.1)
        if step % args.ckpt_every == 0 and step != last_ckpt_step:
            p = save_checkpoint(step, model, optimizer,
                                {"best_val_loss": best_val_loss,
                                 "val_plateau_count": val_plateau_count},
                                train_ds.state, config_hash, tok_hash, ckpt_dir)
            last_ckpt_step = step
            if args.s3_prefix:
                uploaded = _try_s3_upload(p, f"{args.s3_prefix}/step-{step:07d}.pt")
                if uploaded and args.keep_local > 0:
                    prune_local_checkpoints(ckpt_dir, args.keep_local, PERIODIC_CKPT_RE)

        # SIGTERM emergency checkpoint (§18)
        if _sigterm_received[0]:
            p = save_checkpoint(step, model, optimizer,
                                {"best_val_loss": best_val_loss,
                                 "val_plateau_count": val_plateau_count},
                                train_ds.state, config_hash, tok_hash,
                                ckpt_dir, tag="sigterm")
            if args.s3_prefix:
                _try_s3_upload(p, f"{args.s3_prefix}/sigterm-step-{step:07d}.pt")
            log.info("Emergency checkpoint saved. Exiting.")
            sys.exit(0)

    log.info("Training complete. step=%d  best_val_loss=%.4f", step, best_val_loss)
    save_checkpoint(step, model, optimizer,
                    {"best_val_loss": best_val_loss, "val_plateau_count": val_plateau_count},
                    train_ds.state, config_hash, tok_hash, ckpt_dir, tag="final")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train AstraCSE-100M")
    ap.add_argument("--config",            default="configs/model.yaml")
    ap.add_argument("--train-shards",      default="data/shards/train")
    ap.add_argument("--val-shards",        default="data/shards/val")
    ap.add_argument("--tok-dir",           default="tokenizer")
    ap.add_argument("--checkpoint-dir",    default="checkpoints")
    ap.add_argument("--resume",            default=None)
    ap.add_argument("--micro-batch-size",  type=int, default=8)
    ap.add_argument("--total-steps",       type=int, default=100_000)
    ap.add_argument("--log-every",         type=int, default=10)
    ap.add_argument("--eval-every",        type=int, default=1000)
    ap.add_argument("--ckpt-every",        type=int, default=1000)
    ap.add_argument("--s3-prefix",         default=None)
    ap.add_argument("--keep-local",        type=int, default=0,
                    help="Keep only the newest N periodic checkpoints (and the newest best) "
                         "on local disk once uploaded to S3. 0 keeps everything.")
    ap.add_argument("--early-stop-patience", type=int, default=2,
                    help="Stop after this many evals without val improvement. 0 disables.")
    ap.add_argument("--compile", action="store_true",
                    help="Run the forward/backward passes through torch.compile.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed for weight init (random and logged when not given).")
    ap.add_argument("--init-from", default=None,
                    help="Start a new phase from this checkpoint's weights (path or s3:// URI); "
                         "ignored when --resume is given.")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
