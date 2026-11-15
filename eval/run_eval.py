"""Phase 5 — Evaluation runner (Architecture.md §23–24).

Runs the 10-subject benchmark against a trained/exported model checkpoint.
Reports: validation loss, perplexity, MCQ accuracy, per-subject breakdown.

Usage:
    python eval/run_eval.py \
        --model-path export/model_fp16.pt \
        --tok-dir tokenizer/ \
        --config configs/model.yaml \
        --benchmark eval/benchmark.json \
        --val-shards data/shards/val
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import AstraCSE, ModelConfig


def load_model(model_path: Path, config_path: Path) -> tuple[AstraCSE, str]:
    cfg = ModelConfig.from_yaml(config_path)
    payload = torch.load(model_path, map_location="cpu")
    precision = payload.get("precision", "fp32")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    model = AstraCSE(cfg)
    model.load_state_dict(payload["model_state"])
    model = model.to(dtype)
    model.eval()
    return model, precision


@torch.no_grad()
def score_mcq(model: AstraCSE, tokenizer, questions: list[dict],
              device: torch.device) -> dict:
    """Score MCQ questions by choosing the answer with lowest perplexity."""
    model.eval()
    correct = 0
    total = 0
    per_subject: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for q in questions:
        if q.get("category") != "mcq":
            continue
        question_text = q["question"]
        choices = q.get("choices", [])
        gold = q.get("answer", "").strip().upper()
        if not choices or not gold:
            continue

        perplexities = []
        for choice in choices:
            text = question_text + "\n" + choice
            ids = tokenizer.encode(text).ids[:model.cfg.max_seq_len]
            if len(ids) < 2:
                perplexities.append(float("inf"))
                continue
            inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            tgt = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(inp)
            loss = F.cross_entropy(logits.view(-1, model.cfg.vocab_size), tgt.view(-1))
            perplexities.append(loss.item())

        pred_idx = perplexities.index(min(perplexities))
        # Map index to letter: 0→A, 1→B, etc.
        pred_letter = chr(ord("A") + pred_idx)
        is_correct = (pred_letter == gold)
        if is_correct:
            correct += 1
            per_subject[q["subject"]]["correct"] += 1
        total += 1
        per_subject[q["subject"]]["total"] += 1

    return {
        "mcq_accuracy": correct / max(total, 1),
        "mcq_correct": correct,
        "mcq_total": total,
        "per_subject": {
            subj: {
                "accuracy": d["correct"] / max(d["total"], 1),
                "correct": d["correct"],
                "total": d["total"],
            }
            for subj, d in per_subject.items()
        },
    }


@torch.no_grad()
def compute_val_loss(model: AstraCSE, shards_dir: Path,
                     device: torch.device) -> dict:
    from src.data import make_dataloader

    loader, _ = make_dataloader(shards_dir, batch_size=4, shuffle=False)
    total_loss = 0.0
    count = 0
    # Every 8th batch of the whole validation set (~2.5M tokens, ~1/8 of the full
    # pass). val shards are packed subject by subject, so a capped prefix only
    # measures the first subject(s); striding covers all of them at a fraction of
    # the full pass's time.
    for i, (inp, tgt) in enumerate(loader):
        if i % 8:
            continue
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(inp)
        loss = F.cross_entropy(logits.view(-1, model.cfg.vocab_size), tgt.view(-1))
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    return {"val_loss": avg_loss, "perplexity": math.exp(avg_loss)}


def main() -> None:
    ap = argparse.ArgumentParser(description="AstraCSE-100M evaluation")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tok-dir", default="tokenizer")
    ap.add_argument("--config", default="configs/model.yaml")
    ap.add_argument("--benchmark", default="eval/benchmark.json")
    ap.add_argument("--val-shards", default="data/shards/val")
    ap.add_argument("--out", default=None, help="Write results JSON to this path")
    args = ap.parse_args()

    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError:
        print("ERROR: pip install tokenizers", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, precision = load_model(Path(args.model_path), Path(args.config))
    model = model.to(device)
    print(f"Model loaded ({precision}). Parameters: {model.count_params():,}")

    tok_dir = Path(args.tok_dir)
    tokenizer = ByteLevelBPETokenizer(
        str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))

    benchmark = json.loads(Path(args.benchmark).read_text())
    questions = benchmark.get("questions", [])

    results: dict = {}

    # Language modeling metrics
    if Path(args.val_shards).exists():
        print("Computing validation loss and perplexity...")
        lm_results = compute_val_loss(model, Path(args.val_shards), device)
        results.update(lm_results)
        print(f"  val_loss:   {lm_results['val_loss']:.4f}")
        print(f"  perplexity: {lm_results['perplexity']:.4f}")
        # Architecture.md §23.1 targets
        target_val_loss = (1.6, 2.1)
        if target_val_loss[0] <= lm_results["val_loss"] <= target_val_loss[1]:
            print(f"  -> Within target range [{target_val_loss[0]}, {target_val_loss[1]}] ✓")
        else:
            print(f"  -> Outside target range [{target_val_loss[0]}, {target_val_loss[1]}]")

    # MCQ evaluation
    print(f"\nScoring {len(questions)} benchmark questions...")
    mcq_results = score_mcq(model, tokenizer, questions, device)
    results.update(mcq_results)
    acc = mcq_results["mcq_accuracy"]
    print(f"  MCQ accuracy: {acc:.1%}  ({mcq_results['mcq_correct']}/{mcq_results['mcq_total']})")
    # Architecture.md §24.1 target: 60–70%
    if 0.60 <= acc <= 0.70:
        print(f"  -> Within target range [60%, 70%] ✓")
    else:
        print(f"  -> Outside target range [60%, 70%]")

    print("\nPer-subject breakdown:")
    for subj, s in sorted(mcq_results["per_subject"].items()):
        print(f"  {subj:<30s}  {s['accuracy']:.1%}  ({s['correct']}/{s['total']})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
