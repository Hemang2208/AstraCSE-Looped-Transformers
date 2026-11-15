"""Measure the training-step speed-up of torch.compile on this GPU.

For each config: times forward+backward micro-steps (bf16 autocast, the training
batch shape) in eager mode and compiled, checks the two give the same loss on the
same batch, and reports peak memory. Uses random tokens, so it needs no data.

    python scripts/bench_compile.py configs/model.yaml configs/model_d1024.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.model import AstraCSE, ModelConfig  # noqa: E402


def micro_step(model, inp, tgt) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(inp)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
    loss.backward()
    return loss.detach()


def time_steps(model, inp, tgt, warmup: int, steps: int) -> tuple[float, float]:
    for _ in range(warmup):
        micro_step(model, inp, tgt)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        loss = micro_step(model, inp, tgt)
    torch.cuda.synchronize()
    return (time.time() - t0) / steps, float(loss)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--micro-batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    for cfg_path in args.configs:
        cfg = ModelConfig.from_yaml(Path(cfg_path))
        torch.manual_seed(0)
        model = AstraCSE(cfg).cuda()
        g = torch.Generator(device="cuda").manual_seed(0)
        inp = torch.randint(0, cfg.vocab_size, (args.micro_batch_size, cfg.max_seq_len),
                            device="cuda", generator=g)
        tgt = torch.randint(0, cfg.vocab_size, inp.shape, device="cuda", generator=g)

        torch.cuda.reset_peak_memory_stats()
        eager_s, eager_loss = time_steps(model, inp, tgt, warmup=3, steps=args.steps)
        eager_mem = torch.cuda.max_memory_allocated() / 2**30

        compiled = torch.compile(model)
        t0 = time.time()
        micro_step(compiled, inp, tgt)          # first call compiles
        torch.cuda.synchronize()
        compile_s = time.time() - t0
        torch.cuda.reset_peak_memory_stats()
        comp_s, comp_loss = time_steps(compiled, inp, tgt, warmup=2, steps=args.steps)
        comp_mem = torch.cuda.max_memory_allocated() / 2**30

        tokens = args.micro_batch_size * cfg.max_seq_len
        print(f"{Path(cfg_path).name}: {model.count_params()/1e6:.1f}M params | "
              f"eager {eager_s*1000:.0f} ms ({tokens/eager_s:,.0f} tok/s, {eager_mem:.1f} GiB) | "
              f"compiled {comp_s*1000:.0f} ms ({tokens/comp_s:,.0f} tok/s, {comp_mem:.1f} GiB) | "
              f"speed-up {eager_s/comp_s:.2f}x | compile {compile_s:.0f}s | "
              f"loss eager {eager_loss:.4f} vs compiled {comp_loss:.4f}", flush=True)
        del model, compiled
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
