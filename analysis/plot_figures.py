"""Generate all paper figures.

Figure 1: Grouped bar chart of aggregate API vs Interface accuracy.
Figure 3: Heatmap of per-cell API-Interface accuracy gaps.

Usage:
    cd paper_reproduction
    python analysis/plot_figures.py
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RELEASE = Path(__file__).resolve().parent.parent
DATA_DIR = RELEASE / "analysis" / "artifacts" / "data"
FIG_DIR = RELEASE / "analysis" / "artifacts" / "figures"

# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Aggregate accuracy bars
# ═══════════════════════════════════════════════════════════════════════════

BAR_SYSTEMS = [
    ("claude-haiku",      "Haiku 4.5",       "#E69F00", "#F4D5A0"),
    ("claude-sonnet",     "Sonnet 4.6",      "#C46D08", "#EAB785"),
    ("claude-opus",       "Opus 4.6",        "#8C4708", "#D29070"),
    ("gemini-fast",       "Gemini 3 Fast",   "#56B4E9", "#B4DEF3"),
    ("gemini-thinking",   "Gemini 3 Think",  "#0072B2", "#7FB4D0"),
    ("chatgpt-instant",   "GPT 5.3 Inst.",   "#009E73", "#7AC7AC"),
    ("chatgpt-thinking",  "GPT 5.4 Think",   "#00604D", "#5C9783"),
]


def _pick_font():
    from matplotlib import font_manager
    for name in ("Inter", "Helvetica Neue", "Helvetica", "Arial",
                 "IBM Plex Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


def plot_aggregate_bars():
    rows = {r["model"]: r for r in csv.DictReader(open(DATA_DIR / "figure1_aggregate_bars.csv"))}

    labels, api_means, ifc_means, api_cols, ifc_cols = [], [], [], [], []
    for model_key, label, api_c, ifc_c in BAR_SYSTEMS:
        r = rows.get(model_key)
        if not r:
            continue
        labels.append(label)
        api_means.append(float(r["api_pct"]))
        ifc_means.append(float(r["ifc_pct"]))
        api_cols.append(api_c)
        ifc_cols.append(ifc_c)

    plt.rcParams.update({
        "font.family":       _pick_font(),
        "font.size":         11.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "axes.linewidth":    0.9,
    })

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    fig.patch.set_facecolor("white")

    bars_api = ax.bar(x - w / 2, api_means, w,
                      color=api_cols, edgecolor="none", linewidth=0)
    bars_ifc = ax.bar(x + w / 2, ifc_means, w,
                      color=ifc_cols, edgecolor="none", linewidth=0)

    for bars, vals in [(bars_api, api_means), (bars_ifc, ifc_means)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=10, color="#1f1f1f", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5, color="#1f1f1f")
    ax.tick_params(axis="y", labelsize=11.5)
    ax.set_ylabel("Mean accuracy across 9 benchmarks", fontsize=11.5,
                  labelpad=10, color="#555")
    ax.set_ylim(60, 100)
    ax.set_yticks(range(60, 101, 5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.tick_params(axis="x", length=0, pad=8, colors="#1f1f1f")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="y", linestyle="-", color="#ececec", linewidth=1.0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_color("#d0d0d0")
    ax.spines["bottom"].set_linewidth(1.0)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#444", edgecolor="white", label="API"),
        Patch(facecolor="#aaa", edgecolor="white", label="Interface"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False,
              fontsize=11, handlelength=1.4, handleheight=1.0)

    plt.tight_layout()
    out = FIG_DIR / "aggregate_accuracy_bars.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Capability deltas heatmap
# ═══════════════════════════════════════════════════════════════════════════

HEATMAP_SYSTEMS = [
    ("GPT 5.3 Instant",      "chatgpt-instant"),
    ("GPT 5.4 Think",        "chatgpt-thinking"),
    ("Claude Haiku 4.5",     "claude-haiku"),
    ("Claude Opus 4.6",      "claude-opus"),
    ("Claude Sonnet 4.6",    "claude-sonnet"),
    ("Gemini 3 Flash Think", "gemini-thinking"),
    ("Gemini 3 Flash Fast",  "gemini-fast"),
]
HEATMAP_BENCHES = [
    ("ARC",      "metabench-arc"),
    ("GSM8K",    "metabench-gsm8k"),
    ("HS",       "metabench-hellaswag"),
    ("MMLU",     "metabench-mmlu"),
    ("TQA",      "metabench-truthfulQA"),
    ("WG",       "metabench-winogrande"),
    ("BBQ",      "bbq"),
    ("AA-Omni.", "aa-omniscience"),
    ("AITA",     "elephant-flip"),
]

CMAP = LinearSegmentedColormap.from_list(
    "soft_diverge",
    [(0.00, "#2c6e8f"), (0.30, "#9bc3d0"), (0.50, "#f7f4ee"),
     (0.70, "#e8a78a"), (1.00, "#a83232")],
    N=256,
)


def bh_adjust(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [None] * m
    cum = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        cum = min(cum, pvals[i] * m / rank)
        q[i] = min(cum, 1.0)
    return q


def plot_heatmap():
    acc_csv = DATA_DIR / "accuracy_per_run.csv"
    lme_csv = DATA_DIR / "per_cell_lme.csv"
    if not lme_csv.exists():
        raise SystemExit(f"Missing {lme_csv}. Run compute_paper_stats.py first.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.bottom": False, "axes.spines.left": False,
    })

    acc_rows = list(csv.DictReader(open(acc_csv)))
    lme_rows = {(r["model"], r["benchmark"]): r for r in csv.DictReader(open(lme_csv))}

    deltas = np.zeros((len(HEATMAP_SYSTEMS), len(HEATMAP_BENCHES)))
    pvals = []
    flat = []

    for si, (_, mkey) in enumerate(HEATMAP_SYSTEMS):
        for bi, (_, bkey) in enumerate(HEATMAP_BENCHES):
            api_vals = [float(r["accuracy"]) * 100 for r in acc_rows
                        if r["model"] == mkey and r["benchmark"] == bkey and r["surface"] == "api"]
            ifc_vals = [float(r["accuracy"]) * 100 for r in acc_rows
                        if r["model"] == mkey and r["benchmark"] == bkey and r["surface"] == "interface"]
            d = (np.mean(api_vals) - np.mean(ifc_vals) if api_vals and ifc_vals else 0)
            deltas[si, bi] = d
            lme = lme_rows.get((mkey, bkey))
            p = float(lme["p"]) if lme and lme["p"] != "nan" else 1.0
            pvals.append(p)
            flat.append((si, bi))

    q = bh_adjust(pvals)
    vmax = float(np.ceil(np.max(np.abs(deltas)) / 5) * 5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(17.0, 7.5))
    im = ax.imshow(deltas, cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(HEATMAP_BENCHES) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(HEATMAP_SYSTEMS) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(HEATMAP_BENCHES)))
    ax.set_xticklabels([b[0] for b in HEATMAP_BENCHES], fontsize=16)
    ax.set_yticks(range(len(HEATMAP_SYSTEMS)))
    ax.set_yticklabels([s[0] for s in HEATMAP_SYSTEMS], fontsize=16)
    ax.tick_params(axis="both", which="major", length=0, pad=6)

    for k, (si, bi) in enumerate(flat):
        d = deltas[si, bi]
        sig = q[k] < 0.05
        weight = "bold" if sig else "normal"
        intensity = abs(d) / vmax
        color = "white" if intensity > 0.55 else "#2a2a2a"
        num_text = ax.text(bi, si, f"{d:+.1f}", ha="center", va="center",
                           fontsize=15, fontweight=weight, color=color)
        if sig:
            ax.annotate("*", xy=(1.0, 1.0), xycoords=num_text,
                        xytext=(1, 0), textcoords="offset points",
                        ha="left", va="top",
                        fontsize=14, fontweight="bold", color=color)

    tick_step = 10 if vmax >= 20 else 5
    tick_max = int(np.ceil(vmax / tick_step) * tick_step)
    ticks = np.arange(-tick_max, tick_max + 1, tick_step)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=ticks, aspect=28)
    cbar.set_label("API $-$ Interface (pp)", fontsize=17, labelpad=10)
    cbar.ax.set_yticklabels([f"{int(t):+d}" if t else "0" for t in ticks], fontsize=15)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, pad=5)

    plt.tight_layout()
    out = FIG_DIR / "capability_deltas_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_aggregate_bars()
    plot_heatmap()
