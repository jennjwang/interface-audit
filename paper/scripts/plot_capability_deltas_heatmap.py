"""Heatmap of API--Interface accuracy gaps (7 systems x 9 benchmarks).

Reads paper/tables/appendix_full_capability.csv, computes Wald p-values per cell,
applies Benjamini--Hochberg FDR across all 63 cells, and writes
paper/figures/capability_deltas_heatmap.{png,pdf}.

Cells are coloured by signed Δ (API − Iface) on a diverging scale; significance
markers (* q<0.05, ** q<0.01) are appended to the annotation.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
CSV_PATH = REPO / "paper" / "tables" / "appendix_full_capability.csv"
OUT_DIR = REPO / "paper" / "figures"
OUT_DIR.mkdir(exist_ok=True)

SYSTEMS = [
    ("GPT 5.3 Instant",      "GPT 5.3 Instant"),
    ("GPT 5.4 Think",        "GPT 5.4 Thinking"),
    ("Claude Haiku 4.5",     "Claude Haiku"),
    ("Claude Opus 4.6",      "Claude Opus"),
    ("Claude Sonnet 4.6",    "Claude Sonnet"),
    ("Gemini 3 Flash Think", "Gemini 3 Thinking"),
    ("Gemini 3 Flash Fast",  "Gemini 3 Fast"),
]
BENCHES = [
    ("ARC",      "ARC"),
    ("GSM8K",    "GSM8K"),
    ("HS",       "HellaSwag"),
    ("MMLU",     "MMLU"),
    ("TQA",      "TruthfulQA"),
    ("WG",       "WinoGrande"),
    ("BBQ",      "BBQ"),
    ("AA-Omni.", "AA-Omniscience"),
    ("AITA",     "Elephant Flip"),
]


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


# Custom soft diverging palette (muted teal -> off-white -> warm coral).
# Easier on the eyes than RdBu_r for tables that are mostly near-zero with
# a few extreme positive cells.
CMAP = LinearSegmentedColormap.from_list(
    "soft_diverge",
    [
        (0.00, "#2c6e8f"),  # deep teal
        (0.30, "#9bc3d0"),  # pale teal
        (0.50, "#f7f4ee"),  # ivory
        (0.70, "#e8a78a"),  # peach
        (1.00, "#a83232"),  # deep coral
    ],
    N=256,
)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": False,
        "axes.spines.left": False,
    })

    rows = {(r["model"], r["benchmark"]): r for r in csv.DictReader(CSV_PATH.open())}

    deltas = np.zeros((len(SYSTEMS), len(BENCHES)))
    pvals = []
    flat = []
    for si, (_, mkey) in enumerate(SYSTEMS):
        for bi, (_, bkey) in enumerate(BENCHES):
            r = rows[(mkey, bkey)]
            d = float(r["diff_pp"])
            se = float(r["se_pp"])
            deltas[si, bi] = d
            p = 1.0 if se == 0 else 2 * (1 - stats.norm.cdf(abs(d / se)))
            pvals.append(p)
            flat.append((si, bi))
    q = bh_adjust(pvals)

    # Symmetric colour range, clipped a touch above the max to keep mid-cells visible
    vmax = float(np.ceil(np.max(np.abs(deltas)) / 5) * 5)  # round up to nearest 5
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(17.0, 7.5))
    im = ax.imshow(deltas, cmap=CMAP, norm=norm, aspect="auto",
                   interpolation="nearest")

    # White grid between cells
    ax.set_xticks(np.arange(len(BENCHES) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(SYSTEMS) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(len(BENCHES)))
    ax.set_xticklabels([b[0] for b in BENCHES], fontsize=16)
    ax.set_yticks(range(len(SYSTEMS)))
    ax.set_yticklabels([s[0] for s in SYSTEMS], fontsize=16)
    ax.tick_params(axis="both", which="major", length=0, pad=6)

    # Annotate cells. Significance asterisks render as proper superscripts
    # (slightly larger than the number, anchored to its top-right).
    for k, (si, bi) in enumerate(flat):
        d = deltas[si, bi]
        qv = q[k]
        if qv < 0.01:
            sig = "**"
        elif qv < 0.05:
            sig = "*"
        else:
            sig = ""
        weight = "bold" if sig else "normal"
        intensity = abs(d) / vmax
        color = "white" if intensity > 0.55 else "#2a2a2a"
        num_text = ax.text(bi, si, f"{d:+.1f}", ha="center", va="center",
                           fontsize=15, fontweight=weight, color=color)
        if sig:
            ax.annotate(
                sig, xy=(1.0, 1.0), xycoords=num_text,
                xytext=(1, 0), textcoords="offset points",
                ha="left", va="top",
                fontsize=14, fontweight="bold", color=color,
            )

    # Colorbar with integer ticks
    tick_step = 10 if vmax >= 20 else 5
    tick_max = int(np.ceil(vmax / tick_step) * tick_step)
    ticks = np.arange(-tick_max, tick_max + 1, tick_step)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=ticks,
                        aspect=28)
    cbar.set_label("API $-$ Interface (pp)", fontsize=17, labelpad=10)
    cbar.ax.set_yticklabels([f"{int(t):+d}" if t else "0" for t in ticks],
                            fontsize=15)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, pad=5)

    plt.tight_layout()
    out = OUT_DIR / "capability_deltas_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
