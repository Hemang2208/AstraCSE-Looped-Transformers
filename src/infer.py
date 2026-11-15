"""AstraCSE-100M inference and export — Architecture.md §25–26.

Usage:
    python src/infer.py export --checkpoint checkpoints/best.pt \
        --out-dir export/ --precision fp16

    python src/infer.py generate \
        --model-path export/model_fp16.pt \
        --tok-dir tokenizer/ \
        --prompt "What is a binary search tree?" \
        --max-new-tokens 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import AstraCSE, ModelConfig

MAX_SEQ_LEN = 1024   # §5.2 — hard limit, never silently exceed

_PRECISIONS = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


# ---------------------------------------------------------------------------
# Export (§26.1)
# ---------------------------------------------------------------------------

def export_weights(checkpoint_path: Path, out_dir: Path, precision: str) -> Path:
    """Strip optimizer state and save model-only weights at target precision."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = payload["model_state"]
    dtype = _PRECISIONS[precision]
    converted = {k: v.to(dtype) for k, v in model_state.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"model_{precision}.pt"
    torch.save({
        "model_state": converted,
        "precision": precision,
        "global_step": payload.get("global_step"),
        "metadata": payload.get("metadata", {}),
    }, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Exported {precision} weights → {out_path}  ({size_mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# KV-cache generation (§25)
# ---------------------------------------------------------------------------

def generate(
    model: AstraCSE,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Autoregressive generation with per-layer KV cache.

    The RoPE fix: we pass only the NEW token(s) each step, but keep
    `input_ids` growing so that `past_len = kv_cache[0].shape[2]` inside
    CausalSelfAttention correctly encodes each new token at its true
    absolute position.

    `repetition_penalty` (default 1.0 = disabled) mitigates the text-
    degeneration failure mode common in small from-scratch LMs (getting
    stuck re-emitting the same token/short loop, e.g. inside generated
    code). Applied to every already-generated token's logit, standard
    formula: score/penalty if score > 0 else score*penalty — penalty > 1.0
    discourages repeats, matching the same technique HF `generate()` uses.
    """
    model.eval()
    model_max = model.cfg.max_seq_len
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    if input_ids.shape[1] >= model_max:
        raise ValueError(
            f"Prompt length {input_ids.shape[1]} is >= max_seq_len "
            f"{model_max}. At least one token of generation headroom is required (§25.1)."
        )

    kv_caches: Optional[list] = None
    generated_ids: list[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Prefill: full prompt on first call; single new token on subsequent calls.
            cur_input = input_ids if kv_caches is None else input_ids[:, -1:]
            logits, kv_caches = model(cur_input, kv_caches=kv_caches,
                                      return_kv_caches=True)
            next_logits = logits[:, -1, :].clone()

            if repetition_penalty != 1.0:
                seen = torch.unique(input_ids[0])
                seen_scores = next_logits[0, seen]
                next_logits[0, seen] = torch.where(
                    seen_scores < 0, seen_scores * repetition_penalty,
                    seen_scores / repetition_penalty,
                )

            next_logits = next_logits / max(temperature, 1e-6)

            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(
                    next_logits < topk_vals[:, -1].unsqueeze(-1), float("-inf"))

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_token.item())

            # Grow input_ids — `past_len` inside CausalSelfAttention reads
            # kv_cache[0].shape[2] which equals len(input_ids) after appending.
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if input_ids.shape[1] > model_max:
                break

            eos_id = tokenizer.token_to_id("<eos>")
            if eos_id is not None and next_token.item() == eos_id:
                break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_export(args) -> None:
    export_weights(Path(args.checkpoint), Path(args.out_dir), args.precision)


def cmd_generate(args) -> None:
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError:
        print("ERROR: pip install tokenizers", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_dir = Path(args.tok_dir)
    tokenizer = ByteLevelBPETokenizer(
        str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    precision = payload.get("precision", "fp32")
    dtype = _PRECISIONS.get(precision, torch.float32)

    cfg = ModelConfig.from_yaml(cfg_path)
    model = AstraCSE(cfg)
    model.load_state_dict(payload["model_state"])
    model = model.to(dtype).to(device)

    print(generate(model, tokenizer, args.prompt,
                   max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature,
                   top_k=args.top_k,
                   repetition_penalty=args.repetition_penalty,
                   device=device))


def main() -> None:
    ap = argparse.ArgumentParser(description="AstraCSE-100M inference and export")
    sub = ap.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export")
    exp.add_argument("--checkpoint", required=True)
    exp.add_argument("--out-dir",    default="export")
    exp.add_argument("--precision",  choices=list(_PRECISIONS), default="fp16")

    gen = sub.add_parser("generate")
    gen.add_argument("--model-path",      required=True)
    gen.add_argument("--tok-dir",         default="tokenizer")
    gen.add_argument("--config",          default="configs/model.yaml")
    gen.add_argument("--prompt",          required=True)
    gen.add_argument("--max-new-tokens",  type=int,   default=200)
    gen.add_argument("--temperature",     type=float, default=0.8)
    gen.add_argument("--top-k",           type=int,   default=50)
    gen.add_argument("--repetition-penalty", type=float, default=1.15,
                     help="1.0 disables; >1.0 discourages repeating tokens "
                          "(mitigates loop degeneration in code generation)")

    {"export": cmd_export, "generate": cmd_generate}[ap.parse_args().cmd](ap.parse_args())


if __name__ == "__main__":
    main()
