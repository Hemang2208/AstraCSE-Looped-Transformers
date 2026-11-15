"""Draw the paper's figures as vector PDFs in paper/figures/ from the result files in paper/data/.

Every number comes from paper/data (official results, clean re-scores, conditional MCQ, v2 curve) and the
configs. Run from the repo root:  python paper/make_figures.py
The architecture diagram is TikZ, in paper/figures/architecture.tex (no script needed).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "data"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE, ORANGE, GREY, GREEN = "#0072B2", "#E69F00", "#7F7F7F", "#009E73"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


OFFICIAL = {p.name.split(".mmlu_cs")[0]: load(p)["val_loss"]
            for p in (DATA / "results").glob("*.mmlu_cs.json")}
CLEAN = {}
for f in ["rescore_v1_and_baseline_v2.json", "rescore_loop6x2_v2.json", "rescore_d1024.json", "rescore_wu171.json",
          "rescore_seeds.json", "rescore_d1024loop.json", "rescore_baselinev2s1.json"]:
    for r in load(DATA / f):
        CLEAN[r["name"]] = r["clean_loss"]
MCQ = {}
for f in ["mcq_cond.json", "mcq_rescore_d1024.json", "mcq_rescore_wu171.json", "mcq_rescore_seeds.json",
          "mcq_rescore_d1024loop.json", "mcq_rescore_baselinev2s1.json"]:
    for k, v in load(DATA / f)["runs"].items():
        MCQ[k] = v["cf_norm"]["accuracy"]
CONF = {"baseline": "model.yaml", "baseline_s1": "model.yaml", "baseline_v2": "model.yaml", "baseline_v2_s1": "model.yaml",
        "dense_3l": "model_dense_3l.yaml", "dense_4l": "model_dense_4l.yaml", "dense_6l": "model_dense_6l.yaml",
        "dense_6l_s1": "model_dense_6l.yaml", "dense_6l_v2": "model_dense_6l.yaml",
        "loop_3x4": "model_loop_3x4.yaml", "loop_4x3": "model_loop_4x3.yaml", "loop_6x2": "model_loop_6x2.yaml",
        "loop_6x2_s1": "model_loop_6x2.yaml", "loop_6x2_lr3e-4": "model_loop_6x2_lr3e-4.yaml",
        "loop_6x2_wu171": "model_loop_6x2_wu171.yaml", "loop6x2_v2": "model_loop_6x2.yaml",
        "d1024_v2": "model_d1024.yaml", "d1024_ctx2048": "model_d1024_ctx2048.yaml",
        "d1024_loop6x2_v2": "model_d1024_loop6x2.yaml"}


def params(run):
    return yaml.safe_load((ROOT / "configs" / CONF[run]).read_text(encoding="utf-8"))["model"]["expected_param_count"]


def gap(a, b, clean=False):
    if clean:
        return CLEAN[a] - CLEAN[b]
    return OFFICIAL[a] - OFFICIAL[b]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# Figure: matched and compute-matched gaps (loop minus comparison), full set
def fig_gaps():
    rows = [
        ("v1, K=6, seed 0", gap("loop_6x2", "dense_6l"), BLUE, False),
        ("v1, K=6, seed 1", gap("loop_6x2_s1", "dense_6l_s1"), BLUE, False),
        ("v1, K=4", gap("loop_4x3", "dense_4l"), BLUE, False),
        ("v1, K=3", gap("loop_3x4", "dense_3l"), BLUE, False),
        ("v1, K=6, warmup 171", gap("loop_6x2_wu171", "dense_6l"), BLUE, False),
        ("v1, vs Dense-12, seed 0", gap("loop_6x2", "baseline"), BLUE, True),
        ("v1, vs Dense-12, seed 1", gap("loop_6x2_s1", "baseline_s1"), BLUE, True),
        ("v2, K=6 matched (Dense-6)", gap("loop6x2_v2", "dense_6l_v2"), ORANGE, False),
        ("v2, vs Dense-12", gap("loop6x2_v2", "baseline_v2"), ORANGE, True),
        ("Width 1,024, vs dense", gap("d1024_loop6x2_v2", "d1024_v2"), GREEN, False),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    ys = list(range(len(rows)))[::-1]
    for y, (label, value, color, compute) in zip(ys, rows):
        ax.barh(y, value, color=color, hatch="//" if compute else None, edgecolor="black", linewidth=0.4, height=0.6)
        ax.text(value + (0.03 if value >= 0 else -0.03), y, f"{value:+.2f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Loop minus comparison (nats); negative = loop better")
    ax.set_xlim(-0.4, 2.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREEN)],
              labels=["corpus v1", "corpus v2", "width 1,024"], loc="lower right", frameon=False)
    save(fig, "fig_gaps")


# Figure: v2 training curve (clean loss) and the gap over training
def fig_curve():
    curve = load(DATA / "curve_v2.json")
    def series(name):
        pts = [(8595 if r["step"] == "final" else r["step"], r["clean_loss"]) for r in curve if r["name"] == name]
        return sorted(pts)
    b, l = series("baseline_v2"), series("loop6x2_v2")
    fig, (top, bot) = plt.subplots(2, 1, figsize=(3.45, 3.3), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    top.plot([s for s, _ in b], [v for _, v in b], "-o", color=BLUE, markersize=2.5, linewidth=0.9, label="Dense-12 (96.5M)")
    top.plot([s for s, _ in l], [v for _, v in l], "-s", color=ORANGE, markersize=2.5, linewidth=0.9, label="Loop-6$\\times$2 (54.0M)")
    top.set_ylabel("Clean val. loss (nats)")
    top.set_xscale("log")
    top.legend(frameon=False, loc="upper right")
    top.spines[["top", "right"]].set_visible(False)
    gaps = [(s, lv - bv) for (s, bv), (_, lv) in zip(b, l)]
    bot.plot([s for s, _ in gaps], [g for _, g in gaps], "-o", color="black", markersize=2.5, linewidth=0.9)
    bot.axhline(0, color="black", linewidth=0.5)
    bot.set_ylabel("Loop minus\nDense-12")
    bot.set_xlabel("Training step (log scale)")
    bot.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_curve_v2")


# Figure: validation loss against parameters, all runs with a clean value
def fig_scatter():
    runs = [r for r in CLEAN if r in CONF]
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    marker = lambda r: "^" if r.startswith("d1024") else ("s" if "loop" in r else "o")
    for r in sorted(runs):
        color = ORANGE if r.endswith("_v2") or r.startswith("d1024") or r.endswith("_v2_s1") else BLUE
        ax.scatter(params(r) / 1e6, CLEAN[r], marker=marker(r), color=color, s=14, edgecolor="black", linewidth=0.3, zorder=3)
    labels = {"dense_6l": "Dense-6", "loop_6x2": "Loop-6$\\times$2", "baseline_v2": "Dense-12 (v2)",
              "loop6x2_v2": "Loop-6$\\times$2 (v2)", "d1024_v2": "Dense-12, d=1024 (v2)", "d1024_loop6x2_v2": "Loop-6$\\times$2, d=1024 (v2)"}
    offs = {"dense_6l": (4, -6), "loop_6x2": (4, 4), "baseline_v2": (4, 2), "loop6x2_v2": (4, -7),
            "d1024_v2": (-30, 7), "d1024_loop6x2_v2": (-6, -11)}
    for r, text in labels.items():
        if r.startswith("d1024"):
            continue
        dx, dy = offs[r]
        ax.annotate(text, (params(r) / 1e6, CLEAN[r]), xytext=(dx, dy), textcoords="offset points", fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("Clean val. loss (nats)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor="black", markersize=4),
                       plt.Line2D([], [], marker="o", ls="", color=ORANGE, markeredgecolor="black", markersize=4)],
              labels=["corpus v1 (1.15B tokens)", "corpus v2 (4.30B tokens)"], frameon=False, loc="upper right")
    save(fig, "fig_scatter")


# Figure: conditional-likelihood MCQ accuracy (CF-norm) for all runs, with chance and 95% interval
FIG5_LABELS = {
    "baseline": "Dense-12", "baseline_s1": "Dense-12, seed 1", "baseline_v2": "Dense-12 (v2)",
    "baseline_v2_s1": "Dense-12, seed 1 (v2)", "dense_3l": "Dense-3", "dense_4l": "Dense-4", "dense_6l": "Dense-6",
    "dense_6l_s1": "Dense-6, seed 1", "dense_6l_v2": "Dense-6 (v2)", "loop_3x4": "Loop-3$\\times$4",
    "loop_4x3": "Loop-4$\\times$3", "loop_6x2": "Loop-6$\\times$2", "loop_6x2_s1": "Loop-6$\\times$2, seed 1",
    "loop_6x2_lr3e-4": "Loop-6$\\times$2 (LR 3e-4)", "loop_6x2_wu171": "Loop-6$\\times$2 (warmup 171)",
    "loop6x2_v2": "Loop-6$\\times$2 (v2)", "d1024_v2": "Dense-12, $d{=}1024$ (v2)",
    "d1024_ctx2048": "Dense-12, $d{=}1024$ + 2,048 context", "d1024_loop6x2_v2": "Loop-6$\\times$2, $d{=}1024$ (v2)",
}


def fig_mcq():
    order = sorted(MCQ.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(3.45, 3.4))
    ax.axvspan(25 - 3.3, 25 + 3.3, color=GREY, alpha=0.18, linewidth=0)
    ax.axvline(25, color="black", linewidth=0.7, linestyle="--")
    for i, (run, acc) in enumerate(order):
        color = ORANGE if (run.endswith("_v2") or run.startswith("d1024") or run.endswith("v2_s1")) else BLUE
        ax.scatter(100 * acc, i, color=color, s=14, edgecolor="black", linewidth=0.3, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FIG5_LABELS.get(r, r) for r, _ in order])
    ax.set_xlabel("MMLU-CS accuracy, CF-norm (%)")
    ax.set_xlim(20, 31)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=5.5)
    save(fig, "fig_mcq")


if __name__ == "__main__":
    fig_gaps()
    fig_curve()
    fig_scatter()
    fig_mcq()
