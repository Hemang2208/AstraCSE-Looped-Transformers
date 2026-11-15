"""Build the paper's LaTeX tables from the repo's configs and result files.

Every number in the tables comes from:
  configs/*.yaml                           shapes and parameter counts
  paper/data/results/<run>.mmlu_cs.json    official eval (eval/run_eval.py), copied from S3 results/<run>/
  paper/data/rescore_*.json                clean-token re-score (scripts/overlap/rescore_clean.py)
  paper/data/subject_check.json            per-shard breakdown (scripts/overlap/subject_check_box.py)

Run from the repo root:  python paper/make_tables.py
Writes paper/tables/*.tex and prints the derived numbers used in the text.
"""
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "data"
OUT = ROOT / "paper" / "tables"

# run name -> (config file, label used in the paper)
RUNS = {
    "baseline": ("model.yaml", "Dense-12"),
    "dense_6l": ("model_dense_6l.yaml", "Dense-6"),
    "dense_4l": ("model_dense_4l.yaml", "Dense-4"),
    "dense_3l": ("model_dense_3l.yaml", "Dense-3"),
    "loop_6x2": ("model_loop_6x2.yaml", "Loop-6$\\times$2"),
    "loop_6x2_lr3e-4": ("model_loop_6x2_lr3e-4.yaml", "Loop-6$\\times$2 (LR 3e-4)"),
    "loop_4x3": ("model_loop_4x3.yaml", "Loop-4$\\times$3"),
    "loop_3x4": ("model_loop_3x4.yaml", "Loop-3$\\times$4"),
    "baseline_v2": ("model.yaml", "Dense-12"),
    "loop6x2_v2": ("model_loop_6x2.yaml", "Loop-6$\\times$2"),
    "d1024_v2": ("model_d1024.yaml", "Dense-12, $d{=}1024$"),
    "d1024_ctx2048": ("model_d1024_ctx2048.yaml", "\\quad + 2,048 context"),
    "dense_6l_v2": ("model_dense_6l.yaml", "Dense-6"),
    "loop_6x2_wu171": ("model_loop_6x2_wu171.yaml", "Loop-6$\\times$2 (warmup 171)"),
    "d1024_loop6x2_v2": ("model_d1024_loop6x2.yaml", "Loop-6$\\times$2, $d{=}1024$"),
    "baseline_s1": ("model.yaml", "Dense-12, seed 1"),
    "dense_6l_s1": ("model_dense_6l.yaml", "Dense-6, seed 1"),
    "loop_6x2_s1": ("model_loop_6x2.yaml", "Loop-6$\\times$2, seed 1"),
    "baseline_v2_s1": ("model.yaml", "Dense-12, seed 1 (v2)"),
}
MCQ_FILES = ["mcq_cond.json", "mcq_rescore_d1024.json", "mcq_rescore_wu171.json", "mcq_rescore_seeds.json",
             "mcq_rescore_d1024loop.json", "mcq_rescore_baselinev2s1.json"]
RESCORE_FILES = ["rescore_v1_and_baseline_v2.json", "rescore_loop6x2_v2.json", "rescore_d1024.json", "rescore_wu171.json",
                 "rescore_seeds.json", "rescore_d1024loop.json", "rescore_baselinev2s1.json"]


def mcq_runs():
    out = {}
    for f in MCQ_FILES:
        d = json.loads((DATA / f).read_text(encoding="utf-8"))
        assert d["rules_fixed_before_results"]
        out.update(d["runs"])
    return out
V1_ORDER = ["baseline", "dense_6l", "dense_4l", "dense_3l", "loop_6x2", "loop_6x2_lr3e-4", "loop_6x2_wu171",
            "loop_4x3", "loop_3x4"]
PAIRS = [(6, "loop_6x2", "dense_6l"), (4, "loop_4x3", "dense_4l"), (3, "loop_3x4", "dense_3l")]
ABLATION_PAIR = ("6, warmup 171", "loop_6x2_wu171", "dense_6l")
V1_TOKENS = 1_151_073_280
V2_TOKENS = 4_296_117_248


def config(run):
    raw = yaml.safe_load((ROOT / "configs" / RUNS[run][0]).read_text(encoding="utf-8"))["model"]
    k = raw.get("n_unique_layers") or raw["n_layer"]
    return {"d": raw["d_model"], "L": raw["n_layer"], "K": k, "T": raw["n_layer"] // k,
            "params": raw["expected_param_count"]}


def official(run):
    return json.loads((DATA / "results" / f"{run}.mmlu_cs.json").read_text(encoding="utf-8"))


def cf_norm_acc(run):
    """Primary MCQ accuracy (%): conditional-likelihood cloze, per character (scripts/overlap/mcq_cond_box.py)."""
    return 100 * mcq_runs()[run]["cf_norm"]["accuracy"]


def clean_scores():
    out = {}
    for f in RESCORE_FILES:
        for r in json.loads((DATA / f).read_text(encoding="utf-8")):
            assert r["reproduces_reported"], r["name"]
            out[r["name"]] = r
    return out


def m(x):
    return f"{x / 1e6:.1f}M"


MEMORY_MODELS = [
    ("Dense-12", "model.yaml"),
    ("Dense-6", "model_dense_6l.yaml"),
    ("Dense-4", "model_dense_4l.yaml"),
    ("Dense-3", "model_dense_3l.yaml"),
    ("Loop-6$\\times$2", "model_loop_6x2.yaml"),
    ("Loop-4$\\times$3", "model_loop_4x3.yaml"),
    ("Loop-3$\\times$4", "model_loop_3x4.yaml"),
    ("Dense-12, $d{=}1024$", "model_d1024.yaml"),
    ("Loop-6$\\times$2, $d{=}1024$", "model_d1024_loop6x2.yaml"),
]


def memory_lines():
    """Stored parameters, weight memory and forward compute per token, computed from the configs."""
    import yaml
    rows = []
    for label, cfg_name in MEMORY_MODELS:
        mcfg = yaml.safe_load((ROOT / "configs" / cfg_name).read_text(encoding="utf-8"))["model"]
        d, vocab, layers = mcfg["d_model"], mcfg["vocab_size"], mcfg["n_layer"]
        unique = mcfg.get("n_unique_layers", layers)
        total = mcfg["expected_param_count"]
        # outside the blocks: tied embedding (vocab x d), final norm (d), depth embedding (layers x d, looped only)
        fixed = vocab * d + d + (layers * d if unique < layers else 0)
        block = (total - fixed) / unique            # weights of one unique block
        flops = 2 * (layers * block + vocab * d)    # forward matmuls per token, 2 FLOPs per weight
        rows.append((label, total, layers, total * 2 / 1e6, flops / 1e9))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Stored parameters, weight memory and forward compute per token. Weight memory is the parameter "
        "count stored in bfloat16 (2 bytes each). Forward GFLOPs count two operations per weight applied to a token, "
        "including the tied output projection; attention score products, which are the same for every 12-layer model of "
        "a given width, are excluded. Sharing reduces storage but not compute.}",
        "\\label{tab:memory}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & Unique params & Layers per token & Weights (MB) & Fwd GFLOPs \\\\",
        "\\midrule",
    ]
    for label, total, layers, mb, gflops in rows:
        lines.append(f"{label} & {total / 1e6:.1f}M & {layers} & {mb:.1f} & {gflops:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return lines


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body, encoding="utf-8")


def main():
    clean = clean_scores()
    derived = {}

    # Table: model family
    rows = []
    for run in ["baseline", "dense_6l", "dense_4l", "dense_3l", "loop_6x2", "loop_4x3", "loop_3x4"]:
        c = config(run)
        shape = f"{c['K']}" if c["T"] == 1 else f"{c['K']} $\\times$ {c['T']}"
        apps = c["K"] * c["T"]
        rows.append(f"{RUNS[run][1]} & {shape} & {apps} & {c['params']:,} \\\\")
    for cfg, label in [("model_d1024.yaml", "Dense-12, $d{=}1024$"), ("model_d1024_loop6x2.yaml", "Loop-6$\\times$2, $d{=}1024$")]:
        raw = yaml.safe_load((ROOT / "configs" / cfg).read_text(encoding="utf-8"))["model"]
        k = raw.get("n_unique_layers") or raw["n_layer"]
        t = raw["n_layer"] // k
        shape = f"{k}" if t == 1 else f"{k} $\\times$ {t}"
        rows.append(f"{label} & {shape} & {k * t} & {raw['expected_param_count']:,} \\\\")
    write("models.tex", "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Model family. Unique blocks $\\times$ applications; ``Apps'' is layer applications per token. "
        "All models: $d{=}768$ unless noted, 12 heads, $d_{ff}{=}2048$, vocabulary 15{,}000, context 1{,}024.}",
        "\\label{tab:models}", "\\begin{tabular}{lcrr}", "\\toprule",
        "Model & Blocks & Apps & Parameters \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # Table: v1 results
    rows = []
    for run in V1_ORDER:
        c, o, cl = config(run), official(run), clean[run]
        acc = cf_norm_acc(run)
        rows.append(f"{RUNS[run][1]} & {m(c['params'])} & {c['K'] * c['T']} & {o['val_loss']:.3f} & "
                    f"{cl['clean_loss']:.3f} & {o['perplexity']:.1f} & {acc:.1f} \\\\")
    write("memory.tex", "\n".join(memory_lines()))
    write("v1_results.tex", "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Corpus v1 (1.15B tokens, one epoch, 2{,}302 steps). Validation loss in nats on the full strided subset "
        "and on the clean tokens (Section~\\ref{sec:overlap}). Perplexity is on the full subset. MCQ is accuracy on the "
        "650 MMLU-CS questions with the primary CF-norm rule (Table~\\ref{tab:mcq}; chance 25\\%, 95\\% interval about "
        "$\\pm$3.3 points). One seed per row.}",
        "\\label{tab:v1}", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Model & Params & Apps & Val & Clean & PPL & MCQ \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # Table: matched-parameter gaps
    rows = []
    for k, loop, dense in PAIRS:
        lf, df = official(loop)["val_loss"], official(dense)["val_loss"]
        lc, dc = clean[loop]["clean_loss"], clean[dense]["clean_loss"]
        derived[f"gap_K{k}_full"], derived[f"gap_K{k}_clean"] = lf - df, lc - dc
        rows.append(f"{k} & {lf:.3f} & {df:.3f} & {lf - df:.2f} & {lc - dc:.2f} \\\\")
    label, loop, dense = ABLATION_PAIR
    lf, df = official(loop)["val_loss"], official(dense)["val_loss"]
    lc, dc = clean[loop]["clean_loss"], clean[dense]["clean_loss"]
    derived["gap_wu171_full"], derived["gap_wu171_clean"] = lf - df, lc - dc
    derived["wu171_recovered_clean"] = clean["loop_6x2"]["clean_loss"] - lc
    derived["wu171_recovered_fraction"] = derived["wu171_recovered_clean"] / derived["gap_K6_clean"]
    rows.append("\\midrule")
    rows.append(f"{label} & {lf:.3f} & {df:.3f} & {lf - df:.2f} & {lc - dc:.2f} \\\\")
    write("v1_gaps.tex", "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Matched-parameter comparison on corpus v1. Each loop model is paired with the dense model that has the "
        "same $K$ unique blocks and no sharing (unique parameters equal up to the 9{,}216-parameter depth embedding). "
        "Gap = loop $-$ dense validation loss (nats), on the full subset and on clean tokens. The last row is the warmup "
        "ablation: Loop-6$\\times$2 with 171 instead of 46 warmup steps, everything else unchanged.}",
        "\\label{tab:gaps}", "\\begin{tabular}{crrrr}", "\\toprule",
        "$K$ & Loop & Dense & Gap (full) & Gap (clean) \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # Table: v1 -> v2
    rows = []
    for v1, v2 in [("baseline", "baseline_v2"), ("loop_6x2", "loop6x2_v2")]:
        c = config(v2)
        a, b = clean[v1]["clean_loss"], clean[v2]["clean_loss"]
        o2 = official(v2)
        derived[f"{v2}_change_clean"] = b - a
        acc = cf_norm_acc(v2)
        rows.append(f"{RUNS[v2][1]} & {m(c['params'])} & {a:.3f} & {o2['val_loss']:.3f} & {b:.3f} & ${b - a:+.2f}$ & {acc:.1f} \\\\")
    for run in ["d1024_v2", "d1024_ctx2048", "d1024_loop6x2_v2"]:
        c, o2, b = config(run), official(run), clean[run]["clean_loss"]
        rows.append(f"{RUNS[run][1]} & {m(c['params'])} & -- & {o2['val_loss']:.3f} & {b:.3f} & -- & {cf_norm_acc(run):.1f} \\\\")
    derived["d1024_minus_baseline_v2_clean"] = clean["d1024_v2"]["clean_loss"] - clean["baseline_v2"]["clean_loss"]
    derived["d1024_minus_baseline_v2_full"] = official("d1024_v2")["val_loss"] - official("baseline_v2")["val_loss"]
    derived["ctx2048_minus_d1024_clean"] = clean["d1024_ctx2048"]["clean_loss"] - clean["d1024_v2"]["clean_loss"]
    a, b = clean["dense_6l"]["clean_loss"], clean["dense_6l_v2"]["clean_loss"]
    rows.insert(2, f"Dense-6 & {m(config('dense_6l_v2')['params'])} & {a:.3f} & {official('dense_6l_v2')['val_loss']:.3f} & "
                   f"{b:.3f} & ${b - a:+.2f}$ & {cf_norm_acc('dense_6l_v2'):.1f} \\\\")
    derived["dense_6l_v2_change_clean"] = b - a
    derived["v2_matched_gap_clean"] = clean["loop6x2_v2"]["clean_loss"] - clean["dense_6l_v2"]["clean_loss"]
    derived["v2_matched_gap_full"] = official("loop6x2_v2")["val_loss"] - official("dense_6l_v2")["val_loss"]
    write("v2_results.tex", "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Second training regime (corpus v2: 4.30B tokens, one epoch, 8{,}595 steps, 171 warmup steps; the v1 "
        "tokens are a subset of v2). Validation loss in nats on the same v1 validation set. $\\Delta$ = v2 clean $-$ v1 "
        "clean. The $d{=}1024$ rows have no v1 run. ``+ 2,048 context'' continues the $d{=}1024$ model for 400 steps at "
        "2{,}048 tokens and is evaluated at 1{,}024 tokens. MCQ is CF-norm accuracy (\\%, Table~\\ref{tab:mcq}). "
        "One seed per row.}",
        "\\label{tab:v2}", "\\setlength{\\tabcolsep}{2.8pt}", "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Model & Params & v1 clean & v2 val & v2 clean & $\\Delta$ & MCQ \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # Table: per-shard gap on v2
    sc = {r["name"]: r for r in json.loads((DATA / "subject_check.json").read_text(encoding="utf-8"))}
    lb, ll = sc["baseline_v2"]["per_shard"], sc["loop6x2_v2"]["per_shard"]
    rows = [f"{i} & {b:.3f} & {l:.3f} & ${l - b:+.3f}$ \\\\" for i, (b, l) in enumerate(zip(lb, ll))]
    derived["v2_shard_gap_min"] = min(l - b for b, l in zip(lb, ll))
    derived["v2_shard_gap_max"] = max(l - b for b, l in zip(lb, ll))
    dl, db = sc["loop6x2_v2"]["per_decile"], sc["baseline_v2"]["per_decile"]
    derived["v2_decile_gap_min"] = min(l - b for b, l in zip(db, dl))
    derived["v2_decile_gap_max"] = max(l - b for b, l in zip(db, dl))
    write("v2_shards.tex", "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Corpus v2: mean token loss (nats) per validation shard, on the strided batches of the official "
        "evaluation. The gap is nearly constant across shards.}",
        "\\label{tab:shards}", "\\begin{tabular}{crrr}", "\\toprule",
        "Shard & Dense-12 & Loop-6$\\times$2 & Gap \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # Table: validation loss over training on v2 (periodic checkpoints)
    curve = json.loads((DATA / "curve_v2.json").read_text(encoding="utf-8"))
    fin = [r for r in curve if r["step"] == "final"]
    assert fin and all(r["reproduces_reported"] for r in fin), "curve pipeline did not reproduce the final value"
    cb = {r["step"]: r["clean_loss"] for r in curve if r["name"] == "baseline_v2" and r["step"] != "final"}
    cl = {r["step"]: r["clean_loss"] for r in curve if r["name"] == "loop6x2_v2" and r["step"] != "final"}
    cb[8595], cl[8595] = clean["baseline_v2"]["clean_loss"], clean["loop6x2_v2"]["clean_loss"]
    rows = [f"{s:,} & {cb[s]:.3f} & {cl[s]:.3f} & ${cl[s] - cb[s]:+.3f}$ \\\\" for s in sorted(cb)]
    for s in sorted(cb):
        derived[f"curve_gap_{s}"] = cl[s] - cb[s]
    v1_gap = clean["loop_6x2"]["clean_loss"] - clean["baseline"]["clean_loss"]
    rows.append("\\midrule")
    rows.append(f"v1 end (2,302) & {clean['baseline']['clean_loss']:.3f} & {clean['loop_6x2']['clean_loss']:.3f} & ${v1_gap:+.3f}$ \\\\")
    write("v2_curve.tex", "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Clean validation loss (nats) of the two v2 runs at periodic checkpoints, scored with the official "
        "protocol. Step 8{,}595 is the final checkpoint. The last row is the end of the v1 runs, for comparison at a "
        "similar step count.}",
        "\\label{tab:curve}", "\\begin{tabular}{rrrr}", "\\toprule",
        "Step & Dense-12 & Loop-6$\\times$2 & Gap \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # Table: MCQ under four scoring rules (old rule from the official eval; the others from mcq_cond.json)
    mq = json.loads((DATA / "mcq_cond.json").read_text(encoding="utf-8"))
    assert mq["rules_fixed_before_results"]
    n = len(mq["gold"])
    key = mq["answer_key_letters"]
    allmq = mcq_runs()
    assert all(len(r[rule]["predictions"]) == len(mq["gold"]) for r in allmq.values() for rule in ("cf_norm", "cf_sum", "mcf"))
    rows = []
    for run in V1_ORDER + ["baseline_s1", "dense_6l_s1", "loop_6x2_s1", "baseline_v2", "baseline_v2_s1", "loop6x2_v2",
                           "dense_6l_v2", "d1024_v2", "d1024_ctx2048", "d1024_loop6x2_v2"]:
        r = allmq[run]
        old = official(run)["mcq_correct"]
        a_share = r["mcf"]["pred_letters"].get("A", 0) / n
        label = RUNS[run][1] + (" (v2)" if run.endswith("_v2") else "")
        rows.append(f"{label} & {100 * old / n:.1f} & {100 * r['cf_norm']['accuracy']:.1f} & "
                    f"{100 * r['cf_sum']['accuracy']:.1f} & {100 * r['mcf']['accuracy']:.1f} & {100 * a_share:.0f} \\\\")
    accs = [allmq[r]["cf_norm"]["accuracy"] for r in allmq]
    derived["mcq_cf_norm_min"], derived["mcq_cf_norm_max"] = 100 * min(accs), 100 * max(accs)
    derived["mcq_cf_norm_max_z"] = max((a - 0.25) / math.sqrt(0.25 * 0.75 / n) for a in accs)
    shares = [allmq[r]["mcf"]["pred_letters"].get("A", 0) / n for r in allmq]
    derived["mcq_mcf_A_share_min"], derived["mcq_mcf_A_share_max"] = 100 * min(shares), 100 * max(shares)
    derived["mcq_always_C"], derived["mcq_always_A"] = 100 * key["C"] / n, 100 * key["A"] / n
    write("mcq.tex", "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Accuracy (\\%) on the 650 MMLU-CS questions under four scoring rules. ``Old'' is the mean loss of "
        "question plus choice; CF-norm (primary, fixed before results) and CF-sum are the cloze log-likelihood of the "
        "choice given the question, per character and in total; MCF is the log-likelihood of the answer letter. "
        f"``A\\%'' is how often MCF picks A. Chance is 25\\%; always answering C scores {100 * key['C'] / n:.1f}\\% and "
        f"always A {100 * key['A'] / n:.1f}\\%. The 95\\% interval is about $\\pm$3.3 points.}}",
        "\\label{tab:mcq}", "\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Model & Old & CF-norm & CF-sum & MCF & A\\% \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # Table: second seed (v1 and v2) and the gaps across seeds
    rows = []
    for a, b, label in [("baseline", "baseline_s1", "Dense-12"), ("dense_6l", "dense_6l_s1", "Dense-6"),
                        ("loop_6x2", "loop_6x2_s1", "Loop-6$\\times$2")]:
        fa, fb = official(a)["val_loss"], official(b)["val_loss"]
        ca, cb = clean[a]["clean_loss"], clean[b]["clean_loss"]
        derived[f"seed_delta_{a}"] = fb - fa
        rows.append(f"{label} & {fa:.3f} & {fb:.3f} & ${fb - fa:+.3f}$ \\\\")
    rows.append("\\midrule")
    m0 =(official("loop_6x2")["val_loss"] - official("dense_6l")["val_loss"], clean["loop_6x2"]["clean_loss"] - clean["dense_6l"]["clean_loss"])
    m1 = (official("loop_6x2_s1")["val_loss"] - official("dense_6l_s1")["val_loss"], clean["loop_6x2_s1"]["clean_loss"] - clean["dense_6l_s1"]["clean_loss"])
    c0 = (official("loop_6x2")["val_loss"] - official("baseline")["val_loss"], clean["loop_6x2"]["clean_loss"] - clean["baseline"]["clean_loss"])
    c1 = (official("loop_6x2_s1")["val_loss"] - official("baseline_s1")["val_loss"], clean["loop_6x2_s1"]["clean_loss"] - clean["baseline_s1"]["clean_loss"])
    derived["seed_matched_gap_full_s0"], derived["seed_matched_gap_full_s1"] = m0[0], m1[0]
    derived["seed_compute_gap_full_s0"], derived["seed_compute_gap_full_s1"] = c0[0], c1[0]
    rows.append(f"Gap, matched (vs Dense-6) & {m0[0]:.3f} & {m1[0]:.3f} & ${m1[0] - m0[0]:+.3f}$ \\\\")
    rows.append(f"Gap, compute (vs Dense-12) & {c0[0]:.3f} & {c1[0]:.3f} & ${c1[0] - c0[0]:+.3f}$ \\\\")
    write("v1_seeds.tex", "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Second seed on corpus v1. Validation loss (nats, full set) for seed 0 and seed 1, the change between "
        "seeds. The gap rows are loop minus the comparison model, for each seed.}",
        "\\label{tab:seeds}", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{lrrr}", "\\toprule",
        "Model / gap & Seed 0 & Seed 1 & Change \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # Derived numbers for the text
    cv = json.loads((DATA / "val_coverage.json").read_text(encoding="utf-8"))
    derived["width_loop_minus_dense_full"] = official("d1024_loop6x2_v2")["val_loss"] - official("d1024_v2")["val_loss"]
    derived["width_loop_minus_dense_clean"] = clean["d1024_loop6x2_v2"]["clean_loss"] - clean["d1024_v2"]["clean_loss"]
    derived["width_loop_minus_baseline_v2_full"] = official("d1024_loop6x2_v2")["val_loss"] - official("baseline_v2")["val_loss"]
    derived["baseline_v2_seed_delta"] = official("baseline_v2_s1")["val_loss"] - official("baseline_v2")["val_loss"]
    derived["width_loop_cf_norm"] = cf_norm_acc("d1024_loop6x2_v2")
    derived["val_any_fraction"] = cv["any_fraction"]
    derived["val_long_fraction"] = cv["long_fraction"]
    derived["v1_compute_matched_gap_full"] = official("loop_6x2")["val_loss"] - official("baseline")["val_loss"]
    derived["v1_compute_matched_gap_clean"] = clean["loop_6x2"]["clean_loss"] - clean["baseline"]["clean_loss"]
    derived["v2_compute_matched_gap_full"] = official("loop6x2_v2")["val_loss"] - official("baseline_v2")["val_loss"]
    derived["v2_compute_matched_gap_clean"] = clean["loop6x2_v2"]["clean_loss"] - clean["baseline_v2"]["clean_loss"]
    derived["clean_shift_min"] = min(clean[r]["clean_loss"] - official(r)["val_loss"] for r in clean)
    derived["clean_shift_max"] = max(clean[r]["clean_loss"] - official(r)["val_loss"] for r in clean)
    derived["lr_check_change"] = official("loop_6x2_lr3e-4")["val_loss"] - official("loop_6x2")["val_loss"]
    for run, tokens in [("baseline", V1_TOKENS), ("dense_6l", V1_TOKENS), ("baseline_v2", V2_TOKENS), ("loop6x2_v2", V2_TOKENS)]:
        derived[f"tokens_per_param_{run}"] = tokens / config(run)["params"]
    accs = [100 * official(r)["mcq_correct"] / 650 for r in RUNS]
    derived["mcq_min"], derived["mcq_max"] = min(accs), max(accs)
    derived["mcq_95ci_points_at_27pct"] = 196 * math.sqrt(0.27 * 0.73 / 650)
    for k, v in derived.items():
        print(f"{k:34} {v:.4f}")
    print(f"tables written to {OUT}")


if __name__ == "__main__":
    main()
