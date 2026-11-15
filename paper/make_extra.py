"""Figures and tables for the review revision, generated from stored results (no GPU).

Inputs: paper/data/rescore_*.json (clean-token losses), paper/data/results/*.mmlu_cs.json (official full-set losses),
configs/ (parameter counts), paper/data/init_stats.json and paper/data/cpu_bootstrap.json (CPU analysis).
Outputs: paper/figures/fig_pareto.pdf, paper/tables/init_stats.tex, paper/tables/paired.tex.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "data"
RESULTS = DATA / "results"
TABLES = ROOT / "paper" / "tables"
FIGS = ROOT / "paper" / "figures"

BLUE, ORANGE = "#0072B2", "#E69F00"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


CLEAN = {}
for f in ["rescore_v1_and_baseline_v2.json", "rescore_loop6x2_v2.json", "rescore_d1024.json", "rescore_wu171.json",
          "rescore_seeds.json", "rescore_d1024loop.json", "rescore_baselinev2s1.json"]:
    for r in load(DATA / f):
        CLEAN[r["name"]] = r["clean_loss"]

OFFICIAL = {p.name.replace(".mmlu_cs.json", ""): load(p)["val_loss"] for p in RESULTS.glob("*.mmlu_cs.json")}


def params(cfg_name):
    return yaml.safe_load((ROOT / "configs" / cfg_name).read_text(encoding="utf-8"))["model"]["expected_param_count"]


# (label, marker, corpus, run keys, config)
GROUPS = [
    ("Dense-12", "o", "v1", ["baseline", "baseline_s1"], "model.yaml"),
    ("Dense-6", "o", "v1", ["dense_6l", "dense_6l_s1"], "model_dense_6l.yaml"),
    ("Dense-4", "o", "v1", ["dense_4l"], "model_dense_4l.yaml"),
    ("Dense-3", "o", "v1", ["dense_3l"], "model_dense_3l.yaml"),
    ("Loop-6$\\times$2", "s", "v1", ["loop_6x2", "loop_6x2_s1"], "model_loop_6x2.yaml"),
    ("Loop-4$\\times$3", "s", "v1", ["loop_4x3"], "model_loop_4x3.yaml"),
    ("Loop-3$\\times$4", "s", "v1", ["loop_3x4"], "model_loop_3x4.yaml"),
    ("Dense-12 (v2)", "o", "v2", ["baseline_v2", "baseline_v2_s1"], "model.yaml"),
    ("Dense-6 (v2)", "o", "v2", ["dense_6l_v2"], "model_dense_6l.yaml"),
    ("Loop-6$\\times$2 (v2)", "s", "v2", ["loop6x2_v2"], "model_loop_6x2.yaml"),
    ("Dense-12, $d{=}1024$ (v2)", "^", "v2", ["d1024_v2"], "model_d1024.yaml"),
    ("Loop-6$\\times$2, $d{=}1024$ (v2)", "^", "v2", ["d1024_loop6x2_v2"], "model_d1024_loop6x2.yaml"),
]


def fig_pareto():
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    frontier = {"v1": [], "v2": []}
    for label, marker, corpus, runs, cfg in GROUPS:
        mem = params(cfg) * 2 / 1e6
        vals = [CLEAN[r] for r in runs if r in CLEAN]
        if not vals:
            continue
        color = BLUE if corpus == "v1" else ORANGE
        mean = sum(vals) / len(vals)
        if len(vals) == 2:
            ax.errorbar(mem, mean, yerr=[[mean - min(vals)], [max(vals) - mean]], fmt=marker, color=color,
                        markeredgecolor="black", markeredgewidth=0.3, markersize=4, capsize=2, linewidth=0.7, zorder=3)
        else:
            ax.scatter(mem, mean, marker=marker, color=color, edgecolor="black", linewidth=0.3, s=16, zorder=3)
        frontier[corpus].append((mem, mean))
    for corpus, color in [("v1", BLUE), ("v2", ORANGE)]:
        pts = sorted(frontier[corpus])
        best, keep = float("inf"), []
        for mem, loss in pts:
            if loss < best:
                best = loss
                keep.append((mem, loss))
        if len(keep) > 1:
            ax.step([p[0] for p in keep], [p[1] for p in keep], where="post", color=color, linestyle="--", linewidth=0.6)
    ax.set_xlabel("Weights in bfloat16 (MB)")
    ax.set_ylabel("Clean val. loss (nats)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor="black", markersize=4),
                       plt.Line2D([], [], marker="o", ls="", color=ORANGE, markeredgecolor="black", markersize=4)],
              labels=["corpus v1", "corpus v2"], frameon=False, loc="upper right")
    fig.savefig(FIGS / "fig_pareto.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", FIGS / "fig_pareto.pdf")


def fmt(x, digits=3):
    if x is None:
        return "--"
    s = f"{x:.{digits}f}"
    return s.replace("-", "$-$")


def init_table():
    init = load(DATA / "init_stats.json")
    d, l = init["dense_6l"], init["loop_6x2"]
    rows = [
        ("Layer applications per token", str(d["n_applications"]), str(l["n_applications"])),
        ("Unique blocks", str(d["n_unique_blocks"]), str(l["n_unique_blocks"])),
        ("Output-projection std, target $0.02/\\sqrt{2L}$", fmt(d["target_out_proj_std"], 4), fmt(l["target_out_proj_std"], 4)),
        ("Output-projection std, measured (block 0)", fmt(d["out_proj_std_block0"], 4), fmt(l["out_proj_std_block0"], 4)),
        ("Residual std after application 1", fmt(d["residual_std_by_application"][0], 3), fmt(l["residual_std_by_application"][0], 3)),
        ("Residual std after last application", fmt(d["residual_std_by_application"][-1], 3), fmt(l["residual_std_by_application"][-1], 3)),
        ("Residual variance after last application", fmt(d["residual_var_by_application"][-1], 3), fmt(l["residual_var_by_application"][-1], 3)),
        ("Initial loss (nats)", fmt(d["initial_loss"], 3), fmt(l["initial_loss"], 3)),
        ("Gradient norm, total", fmt(d["grad_norm_total"], 3), fmt(l["grad_norm_total"], 3)),
        ("Gradient norm, mean per unique block", fmt(sum(d["grad_norm_per_unique_block"]) / len(d["grad_norm_per_unique_block"]), 3),
         fmt(sum(l["grad_norm_per_unique_block"]) / len(l["grad_norm_per_unique_block"]), 3)),
    ]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Statistics at initialisation (seed 0, first validation batch of four 1,024-token sequences). "
             "Dense-6 and Loop-6$\\times$2 differ in the number of layer applications, so their output projections have "
             "different target scales (Section~\\ref{sec:models}). Residual statistics are taken after the stated layer "
             "application; the gradient of a shared block is the sum over its applications.}",
             "\\label{tab:init}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lrr}", "\\toprule", "Quantity & Dense-6 & Loop-6$\\times$2 \\\\", "\\midrule"]
    lines += [f"{q} & {a} & {b} \\\\" for q, a, b in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (TABLES / "init_stats.tex").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", TABLES / "init_stats.tex")


def paired_table():
    boot = load(DATA / "cpu_bootstrap.json")
    pairs = boot.get("pairs", {})
    k_note = boot.get("k_subsampling", {})

    def ci(name):
        p = pairs.get(name, {})
        if "mean_difference" not in p:
            return "not run"
        lo, hi = p["ci95"]
        return f"{p['mean_difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]".replace("-", "$-$")

    def row(label, seed0, seed1, mean, extra):
        return f"{label} & {seed0} & {seed1} & {mean} & {extra} \\\\"

    o = OFFICIAL
    d12 = [o["baseline"], o["baseline_s1"]]
    d6 = [o["dense_6l"], o["dense_6l_s1"]]
    l6 = [o["loop_6x2"], o["loop_6x2_s1"]]
    d12v2 = [o["baseline_v2"], o["baseline_v2_s1"]]
    # state the subsampling actually used (k and batch count come from cpu_bootstrap.json)
    k_sub = k_note.get("loop6x2_v2", 1)
    ord_k = {2: "2nd", 4: "4th", 8: "8th", 16: "16th"}.get(k_sub, f"{k_sub}th")
    n_sub = pairs.get("v2_matched", {}).get("batches", "?")
    sub_desc = f"every {ord_k} official batch, {n_sub} of the 738 batches"
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Two seeds, first regime (full validation set, official values) and second regime (Dense-12, "
             "two seeds). Gaps are loop minus the comparison model, per seed. The last column gives the paired "
             "difference on the CPU subsample with a 95\\% interval from resampling batches of four sequences "
             f"({sub_desc}; Section~\\ref{{sec:setup}}). Intervals reflect sampling of batches, not seed "
             "variation. Batch-level bootstrap intervals ignore correlation between batches from the same document or subject, so they likely underestimate uncertainty.}",
             "\\label{tab:paired}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lrrrl}", "\\toprule",
             "Model / gap & Seed 0 & Seed 1 & Mean & Subsample diff. [95\\% CI] \\\\", "\\midrule"]
    lines.append(row("Dense-12, v1", fmt(d12[0]), fmt(d12[1]), fmt(sum(d12) / 2), "--"))
    lines.append(row("Dense-6, v1", fmt(d6[0]), fmt(d6[1]), fmt(sum(d6) / 2), "--"))
    lines.append(row("Loop-6$\\times$2, v1", fmt(l6[0]), fmt(l6[1]), fmt(sum(l6) / 2), "--"))
    lines.append("\\midrule")
    gm = [l6[i] - d6[i] for i in range(2)]
    gc = [l6[i] - d12[i] for i in range(2)]
    lines.append(row("Matched gap (loop $-$ Dense-6)", fmt(gm[0]), fmt(gm[1]), fmt(sum(gm) / 2), ci("v1_seed1_matched")))
    lines.append(row("Compute gap (loop $-$ Dense-12)", fmt(gc[0]), fmt(gc[1]), fmt(sum(gc) / 2), ci("v1_seed1_compute")))
    lines.append("\\midrule")
    lines.append(row("Dense-12, v2", fmt(d12v2[0]), fmt(d12v2[1]), fmt(sum(d12v2) / 2), "--"))
    lines.append(row("Matched gap, v2 (loop $-$ Dense-6)", fmt(o["loop6x2_v2"] - o["dense_6l_v2"]), "--", "--", ci("v2_matched")))
    gv2 = [o["loop6x2_v2"] - d12v2[0], o["loop6x2_v2"] - d12v2[1]]
    lines.append(row("Compute gap, v2 (loop $-$ Dense-12)", fmt(gv2[0]), fmt(gv2[1]), fmt(sum(gv2) / 2), ci("v2_compute")))
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    (TABLES / "paired.tex").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", TABLES / "paired.tex", "| subsampling:", k_note)


if __name__ == "__main__":
    fig_pareto()
    init_table()
    paired_table()
    print("done")
