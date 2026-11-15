"""Conditional-likelihood MCQ evaluation on the 650-question MMLU-CS set (GPU box, one short pause).

Scoring rules, fixed before any result was seen (2026-11-04):
  cf_norm (primary)  Cloze: context "<doc>Question: {q}\\nAnswer:", continuation " {choice text}".
                     Score = sum of continuation token log-probs / characters in the continuation.
  cf_sum             Same context and continuation; score = sum of continuation token log-probs.
  mcf                Letter format: context lists the four choices as "A. ...", ..., then "Answer:";
                     continuation " A" / " B" / " C" / " D"; score = log-prob of the continuation.
The highest score wins; ties go to the first choice. The "A) " prefixes stored in the benchmark choices are
removed for the cloze rules. The context starts with <doc>, the token that precedes every document in training.

Also records, per rule, the distribution of predicted letters and every per-question prediction, so positional
bias and paired comparisons between models can be checked. The answer key is not uniform (C 187, A 169, B 150,
D 144), so always answering C would score 28.8%.

Accepts the flags pause_rescore.sh passes (--repo --val-dir --masks-dir --work --out; val and masks unused).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

PREFIX = re.compile(r"^\s*[A-D][\)\.:]\s*")
LETTERS = "ABCD"
RULES = ("cf_norm", "cf_sum", "mcf")


def strip_prefix(choice):
    return PREFIX.sub("", choice, count=1).strip()


def build_items(questions, encode, doc_id, max_len):
    """One item per (question, rule-family, choice): token ids, continuation start, continuation characters."""
    items, truncated = [], 0
    for qi, q in enumerate(questions):
        texts = [strip_prefix(c) for c in q["choices"]]
        cf_ctx = [doc_id] + encode(f"Question: {q['question']}\nAnswer:")
        listing = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(texts))
        mcf_ctx = [doc_id] + encode(f"Question: {q['question']}\n{listing}\nAnswer:")
        for family, ctx, conts in (("cf", cf_ctx, [" " + t for t in texts]),
                                   ("mcf", mcf_ctx, [" " + LETTERS[i] for i in range(len(texts))])):
            for ci, cont in enumerate(conts):
                cont_ids = encode(cont)
                if len(cont_ids) >= max_len:
                    raise ValueError(f"question {qi}: continuation of {len(cont_ids)} tokens does not fit in {max_len}")
                ids = ctx + cont_ids
                if len(ids) > max_len:
                    ids = ids[len(ids) - max_len:]
                    truncated += 1
                items.append({"q": qi, "family": family, "c": ci, "ids": ids,
                              "start": len(ids) - len(cont_ids), "chars": len(cont)})
    return items, truncated


@torch.no_grad()
def score_items(model, items, device, batch_size=32):
    """Sum of continuation log-probs for every item, batched with right padding."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out = [0.0] * len(items)
    model.eval()
    for b0 in range(0, len(order), batch_size):
        idx = order[b0:b0 + batch_size]
        width = max(len(items[i]["ids"]) for i in idx)
        batch = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):
            ids = items[i]["ids"]
            batch[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(batch)
        logp = F.log_softmax(logits.float(), dim=-1)
        for r, i in enumerate(idx):
            it = items[i]
            pos = torch.arange(it["start"] - 1, len(it["ids"]) - 1, device=device)
            tgt = batch[r, it["start"]:len(it["ids"])]
            out[i] = logp[r, pos, tgt].sum().item()
    return out


def decide(questions, items, scores):
    by = defaultdict(dict)
    for it, s in zip(items, scores):
        by[(it["q"], it["family"])][it["c"]] = (s, it["chars"])
    res = {}
    for rule in RULES:
        family = "mcf" if rule == "mcf" else "cf"
        preds, per_subject = [], defaultdict(lambda: [0, 0])
        for qi, q in enumerate(questions):
            cand = by[(qi, family)]
            vals = [cand[c][0] / cand[c][1] if rule == "cf_norm" else cand[c][0] for c in range(len(cand))]
            p = max(range(len(vals)), key=lambda c: (vals[c], -c))
            preds.append(LETTERS[p])
            ok = LETTERS[p] == q["answer"].strip().upper()
            per_subject[q["subject"]][0] += ok
            per_subject[q["subject"]][1] += 1
        correct = sum(ps[0] for ps in per_subject.values())
        res[rule] = {"correct": correct, "total": len(questions), "accuracy": correct / len(questions),
                     "per_subject": {s: {"correct": c, "total": t} for s, (c, t) in per_subject.items()},
                     "pred_letters": dict(Counter(preds)), "predictions": "".join(preds)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="/home/ubuntu/AstraCSE")
    ap.add_argument("--val-dir", default=None)
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--work", default="/home/ubuntu/astra_overlap/ckpt")
    ap.add_argument("--out", default="/home/ubuntu/astra_overlap/out/mcq_cond.json")
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, "/home/ubuntu/astra_overlap/scripts")
    os.chdir(repo)
    from eval.run_eval import load_model
    from tokenizers import ByteLevelBPETokenizer
    import rescore_clean as rc

    tok = ByteLevelBPETokenizer(str(repo / "tokenizer" / "vocab.json"), str(repo / "tokenizer" / "merges.txt"))
    doc_id = tok.token_to_id("<doc>")
    assert doc_id is not None
    bench = json.loads((repo / "eval" / "mmlu_cs_benchmark.json").read_text(encoding="utf-8"))
    questions = [q for q in bench["questions"] if q.get("category") == "mcq"]
    assert len(questions) == 650 and all(len(q["choices"]) == 4 for q in questions)
    configs = dict(rc.RUNS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    gold = "".join(q["answer"].strip().upper() for q in questions)
    results = {"rules_fixed_before_results": True, "answer_key_letters": dict(Counter(gold)), "gold": gold, "runs": {}}
    print(f"device: {device}; questions: {len(questions)}; runs: {args.runs}", flush=True)
    for run in args.runs:
        key = rc.s3_final_key(run)
        local = work / f"{run}-{key}"
        subprocess.run(["aws", "s3", "cp", f"s3://{rc.BUCKET}/{rc.ckpt_prefix(run)}{key}", str(local),
                        "--region", rc.REGION, "--only-show-errors"], check=True)
        model, _ = load_model(local, repo / configs[run])
        model = model.to(device)
        items, truncated = build_items(questions, lambda s: tok.encode(s).ids, doc_id, model.cfg.max_seq_len)
        res = decide(questions, items, score_items(model, items, device))
        res["checkpoint"], res["truncated_items"] = key, truncated
        results["runs"][run] = res
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"{run}: " + ", ".join(f"{r} {res[r]['correct']}/650 ({100 * res[r]['accuracy']:.1f}%)" for r in RULES)
              + f" | cf_norm letters {res['cf_norm']['pred_letters']} | truncated {truncated}", flush=True)
        del model
        os.remove(local)
    print("mcq conditional written", flush=True)


if __name__ == "__main__":
    main()
