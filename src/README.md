# src/

Model, training and inference code. The design is in `Architecture.md` (§2–9 model, §12–13 data, §16–21 training, §25–26 inference, §31 loop variant).

Files:
- `model.py`: decoder-only transformer (RoPE, RMSNorm, SwiGLU, causal attention, tied embeddings). Loop mode: set `n_unique_layers` below `n_layer` to reuse K unique blocks, with a learned depth-conditioning embedding.
- `data.py`: memmap shard dataloader with filename-based resume state.
- `train.py`: training loop with AdamW, cosine LR, BF16 autocast, optional `torch.compile` (`--compile`), NaN and OOM guards, SIGTERM checkpoints, and resume from the newest checkpoint. Other flags: `--init-from` (weights only, for context extension), `--seed`, `--keep-local`, `--early-stop-patience`. Writes `train.log` next to the checkpoints.
- `infer.py`: export to FP16, BF16 or FP32, and KV-cache generation with a repetition penalty.

Tests live in `tests/` (125 tests). Run `pytest tests/ -q` from the repo root.

Experiment documentation: `docs/METHODS.md` (how the runs were done), `docs/RESULTS.md` (numbers), `docs/FINDINGS.md` (interpretation).
