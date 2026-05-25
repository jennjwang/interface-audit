"""Bar charts of API-minus-Interface accuracy gap per system.

Produces two figures:
  - paper/figures/aggregate_gap_bars.png         (original: all 7 systems,
        provider-family colours, small fonts, "Error bars:" footer)
  - paper/figures/aggregate_gap_bars_three.png   (condensed: Opus / Gemini
        Think / GPT Think only, large fonts, light-blue/yellow/red)

For each system, plots a single bar showing the unweighted mean Δ (API − Iface)
across the 9 capability benchmarks, with SE error bars propagated from the
per-cell paired item SE: SE_diff = sqrt(sum_i SE_i^2) / n.

Also writes paper/tables/aggregate_accuracy.csv with the underlying numbers
(per-system api_mean, ifc_mean, gap, and SEs) for the full 7-system set.
"""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
CSV_PATH = REPO / "paper" / "tables" / "appendix_full_capability.csv"
OUT_DIR = REPO / "paper" / "figures"
OUT_CSV = REPO / "paper" / "tables" / "aggregate_accuracy.csv"
OUT_DIR.mkdir(exist_ok=True)

# Original: all 7 systems with provider-family colours.
SYSTEMS_FULL = [
    ("Haiku 4.5",       "Claude Haiku",      "#e89060"),
    ("Sonnet 4.6",      "Claude Sonnet",     "#c96430"),
    ("Opus 4.6",        "Claude Opus",       "#a04020"),
    ("Gemini 3 Fast",   "Gemini 3 Fast",     "#8a6ab8"),
    ("Gemini 3 Think",  "Gemini 3 Thinking", "#5d3d8a"),
    ("GPT 5.3 Inst.",   "GPT 5.3 Instant",   "#4f8cc0"),
    ("GPT 5.4 Think",   "GPT 5.4 Thinking",  "#2c5079"),
]

# Condensed: one flagship thinking model per provider, with light-blue / yellow
# / red palette (per user request).
SYSTEMS_THREE = [
    ("Opus",        "Claude Opus",       "#7fb3d5"),
    ("Gemini Think","Gemini 3 Thinking", "#f5d76e"),
    ("GPT Think",   "GPT 5.4 Thinking",  "#e57373"),
]


def compute_rows(systems, by_model):
    """Aggregate per-system mean gap and SE across benchmarks."""
    out = []
    for label, key, color in systems:
        rs = by_model[key]
        api  = np.array([float(r["api_pct"])       for r in rs])
        ifc  = np.array([float(r["interface_pct"]) for r in rs])
        diff = np.array([float(r["diff_pp"])       for r in rs])
        se   = np.array([float(r["se_pp"])         for r in rs])
        n = len(rs)
        api_m = float(api.mean())
        ifc_m = float(ifc.mean())
        gap   = float(diff.mean())
        gap_se = math.sqrt((se ** 2).sum()) / n
        side_se = math.sqrt(((se / np.sqrt(2)) ** 2).sum()) / n
        out.append({
            "label": label, "key": key, "color": color,
            "api_mean": api_m, "api_se": side_se,
            "ifc_mean": ifc_m, "ifc_se": side_se,
            "gap": gap, "gap_se": gap_se, "n_benchmarks": n,
        })
    return out


def plot_chart(rows, out_path, *, fontsize_axis, fontsize_tick,
               fontsize_value, bar_width, figsize, footer_text=None):
    """Render a single gap-bars chart with the given style params."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size":   fontsize_tick,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
    })

    labels  = [r["label"] for r in rows]
    gaps    = [r["gap"]   for r in rows]
    gap_ses = [r["gap_se"] for r in rows]
    colors  = [r["color"] for r in rows]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    bars = ax.bar(
        x, gaps, bar_width, yerr=gap_ses, capsize=6,
        color=colors, edgecolor="white", linewidth=1.4,
        error_kw={"ecolor": "#333", "elinewidth": 1.3, "capthick": 1.3},
    )

    for b, v, s in zip(bars, gaps, gap_ses):
        ax.text(
            b.get_x() + b.get_width() / 2, v + s + 0.2,
            f"{v:+.1f}", ha="center", va="bottom",
            fontsize=fontsize_value, color="#2a2a2a",
        )

    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize_axis)
    ax.tick_params(axis="y", labelsize=fontsize_tick)
    ax.set_ylabel("API $-$ Interface (pp)", fontsize=fontsize_axis, labelpad=12)
    ax.set_ylim(0, max(gaps) + max(gap_ses) + 1.5)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="y", linestyle=":", color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)

    if footer_text:
        fig.text(0.5, -0.02, footer_text,
                 ha="center", fontsize=max(fontsize_tick - 2, 9), color="#666")

    plt.tight_layout()
    fig.savefig(out_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    rows = list(csv.DictReader(CSV_PATH.open()))
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    full = compute_rows(SYSTEMS_FULL, by_model)
    three = compute_rows(SYSTEMS_THREE, by_model)

    # Write per-system aggregate CSV (full 7-system set).
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "model", "api_mean", "api_se", "interface_mean", "interface_se",
            "gap", "gap_se", "n_benchmarks",
        ])
        w.writeheader()
        for r in full:
            w.writerow({
                "model": r["key"],
                "api_mean":       round(r["api_mean"], 2),
                "api_se":         round(r["api_se"], 2),
                "interface_mean": round(r["ifc_mean"], 2),
                "interface_se":   round(r["ifc_se"], 2),
                "gap":            round(r["gap"], 2),
                "gap_se":         round(r["gap_se"], 2),
                "n_benchmarks":   r["n_benchmarks"],
            })
    print(f"Wrote {OUT_CSV.relative_to(REPO)}")

    # Original 7-bar chart.
    out_full = OUT_DIR / "aggregate_gap_bars.png"
    plot_chart(
        full, out_full,
        fontsize_axis=16, fontsize_tick=15, fontsize_value=14,
        bar_width=0.6, figsize=(14.0, 5.8),
        footer_text=("Error bars: SE of the unweighted mean gap across 9 "
                     "benchmarks, propagated from per-cell paired item SE."),
    )
    print(f"Wrote {out_full.relative_to(REPO)}")

    # Condensed 3-bar chart with large fonts and pastel-vivid palette.
    out_three = OUT_DIR / "aggregate_gap_bars_three.png"
    plot_chart(
        three, out_three,
        fontsize_axis=30, fontsize_tick=26, fontsize_value=22,
        bar_width=0.78, figsize=(10.5, 6.4),
        footer_text=None,
    )
    print(f"Wrote {out_three.relative_to(REPO)}")


if __name__ == "__main__":
    main()
