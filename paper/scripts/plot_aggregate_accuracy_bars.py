"""Grouped bar chart: aggregate API vs Interface accuracy per system.

For each system, the bar height is the unweighted mean accuracy across the 9
capability benchmarks (each benchmark contributes equally regardless of N).
Error bars are SE of that mean, propagated from per-cell paired item SE:
  SE_agg = sqrt(sum_i (1/n)^2 * SE_i^2) = sqrt(mean(SE_i^2)) / sqrt(n)

Models are coloured by provider family:
  Anthropic (Haiku/Sonnet/Opus)   -> warm coral/orange
  Google    (Gemini Fast/Think)   -> purple
  OpenAI    (GPT 5.3/5.4)         -> steel blue
Darker shade = API, lighter shade = Interface.

Reads paper/tables/appendix_full_capability.csv. Writes
paper/figures/aggregate_accuracy_bars.{png,pdf}.
"""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
CSV_PATH = REPO / "paper" / "tables" / "appendix_full_capability.csv"
OUT_DIR = REPO / "paper" / "figures"
OUT_DIR.mkdir(exist_ok=True)

# (display, CSV model key, API colour, Interface colour)
# Colourblind-friendly palette based on Okabe-Ito hues.
#   Anthropic (Claude) -> orange  / vermillion gradient (Haiku light -> Opus dark)
#   Google    (Gemini) -> sky-blue / blue gradient      (Fast light -> Think dark)
#   OpenAI    (GPT)    -> bluish-green gradient         (5.3 light -> 5.4 dark)
# Within each model, the API bar is the saturated hue and the Interface is a
# softer tint of the same hue.
SYSTEMS = [
    ("Haiku 4.5",       "Claude Haiku",      "#E69F00", "#F4D5A0"),  # orange
    ("Sonnet 4.6",      "Claude Sonnet",     "#C46D08", "#EAB785"),  # darker orange
    ("Opus 4.6",        "Claude Opus",       "#8C4708", "#D29070"),  # vermillion / brown
    ("Gemini 3 Fast",   "Gemini 3 Fast",     "#56B4E9", "#B4DEF3"),  # sky blue
    ("Gemini 3 Think",  "Gemini 3 Thinking", "#0072B2", "#7FB4D0"),  # blue
    ("GPT 5.3 Inst.",   "GPT 5.3 Instant",   "#009E73", "#7AC7AC"),  # bluish green
    ("GPT 5.4 Think",   "GPT 5.4 Thinking",  "#00604D", "#5C9783"),  # deep green
]


def weighted_mean_sd(values_and_weights):
    n_tot = sum(w for _, w in values_and_weights)
    mu = sum(v * w for v, w in values_and_weights) / n_tot
    var = sum(w * (v - mu) ** 2 for v, w in values_and_weights) / n_tot
    return mu, math.sqrt(var)


def compute_rows(systems, by_model):
    """Compute per-system aggregate API/Interface means and SEs."""
    out = []
    for label, key, api_c, ifc_c in systems:
        rs = by_model[key]
        api_pcts = np.array([float(r["api_pct"]) for r in rs])
        ifc_pcts = np.array([float(r["interface_pct"]) for r in rs])
        ses      = np.array([float(r["se_pp"]) for r in rs])
        n = len(rs)
        api_m = float(api_pcts.mean())
        ifc_m = float(ifc_pcts.mean())
        side_se = math.sqrt(((ses / np.sqrt(2)) ** 2).sum()) / n
        out.append({
            "label": label, "api_c": api_c, "ifc_c": ifc_c,
            "api_m": api_m, "ifc_m": ifc_m, "se": side_se,
        })
    return out


def _pick_font():
    """Use a refined sans if available, else fall back to DejaVu Sans."""
    from matplotlib import font_manager
    for name in ("Inter", "Helvetica Neue", "Helvetica", "Arial",
                 "IBM Plex Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


def plot_chart(rows, out_path, *, fontsize_axis, fontsize_tick, fontsize_value,
               fontsize_legend, bar_width, figsize, y_range, y_step,
               legend_loc="upper left", value_pad=0.8, dpi=220,
               ylabel="Mean accuracy across 9 benchmarks",
               hide_y_ticks=False, show_gap_line=False,
               gap_color="#c0392b", gap_fontsize=None,
               show_error_bars=True, show_legend=True,
               per_bar_labels=False, title=None):
    plt.rcParams.update({
        "font.family":       _pick_font(),
        "font.size":         fontsize_tick,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "axes.linewidth":    0.9,
    })

    labels    = [r["label"] for r in rows]
    api_means = [r["api_m"] for r in rows]
    ifc_means = [r["ifc_m"] for r in rows]
    ses       = [r["se"]    for r in rows]
    api_cols  = [r["api_c"] for r in rows]
    ifc_cols  = [r["ifc_c"] for r in rows]

    x = np.arange(len(labels))
    w = bar_width

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    err_kw = {"yerr": ses, "capsize": 5,
              "error_kw": {"ecolor": "#444", "elinewidth": 1.2, "capthick": 1.2}}
    if not show_error_bars:
        err_kw = {}
    bars_api = ax.bar(
        x - w/2, api_means, w,
        color=api_cols, edgecolor="none", linewidth=0, **err_kw,
    )
    bars_ifc = ax.bar(
        x + w/2, ifc_means, w,
        color=ifc_cols, edgecolor="none", linewidth=0, **err_kw,
    )

    # Bar value labels above each bar.
    label_pad_ses = ses if show_error_bars else [0.0] * len(ses)
    for bars, vals in [(bars_api, api_means), (bars_ifc, ifc_means)]:
        for b, v, s in zip(bars, vals, label_pad_ses):
            ax.text(b.get_x() + b.get_width()/2, v + s + value_pad,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=fontsize_value, color="#1f1f1f",
                    fontweight="bold")

    if show_gap_line:
        gfs = gap_fontsize if gap_fontsize is not None else fontsize_value
        y_lo, y_hi = y_range
        bracket_lift = (y_hi - y_lo) * 0.16   # horizontal segment above API top
        stem_pad     = (y_hi - y_lo) * 0.035  # gap between bar top and stem start
        stem_inset   = 0.10                   # inward distance from bar inner edge
        for i, (api_v, ifc_v) in enumerate(zip(api_means, ifc_means)):
            x_api = i - w/2
            x_ifc = i + w/2
            gap = api_v - ifc_v
            bracket_top = api_v + bracket_lift
            left_x  = x_api + w/2 - stem_inset
            right_x = x_ifc - w/2 + stem_inset
            ax.plot(
                [left_x, left_x, right_x, right_x],
                [api_v + stem_pad, bracket_top, bracket_top, ifc_v + stem_pad],
                color=gap_color, linewidth=1.6, zorder=6,
                solid_capstyle="round", solid_joinstyle="round",
            )
            ax.text((left_x + right_x) / 2, bracket_top + stem_pad * 0.6,
                    f"+{gap:.1f} pp", ha="center", va="bottom",
                    color=gap_color, fontsize=gfs, fontweight="bold",
                    zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize_axis, color="#1f1f1f")
    ax.tick_params(axis="y", labelsize=fontsize_tick)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize_axis, labelpad=10,
                      color="#555")
    y_lo, y_hi = y_range
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(range(y_lo, y_hi + 1, y_step))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    if hide_y_ticks:
        ax.set_yticklabels([])
    ax.tick_params(axis="x", length=0, pad=8, colors="#1f1f1f")
    ax.tick_params(axis="y", length=0, pad=4)
    # Subtle horizontal gridlines.
    ax.grid(axis="y", linestyle="-", color="#ececec", linewidth=1.0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    # Slim baseline under the bars.
    ax.spines["bottom"].set_color("#d0d0d0")
    ax.spines["bottom"].set_linewidth(1.0)

    if show_legend:
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor="#444", edgecolor="white", label="API"),
            Patch(facecolor="#aaa", edgecolor="white", label="Interface"),
        ]
        ax.legend(handles=legend_handles, loc=legend_loc, frameon=False,
                  fontsize=fontsize_legend, handlelength=1.4, handleheight=1.0)

    if per_bar_labels:
        # Pick label colors from the bar palette: API label takes the
        # saturated API color, Interface label is muted gray.
        trans = ax.get_xaxis_transform()
        small_fs = int(fontsize_legend * 0.65)
        for xi, api_c in zip(x, api_cols):
            ax.text(xi - w/2, -0.04, "API", ha="center", va="top",
                    transform=trans, fontsize=small_fs, color=api_c,
                    fontweight="bold")
            ax.text(xi + w/2, -0.04, "INTERFACE", ha="center", va="top",
                    transform=trans, fontsize=small_fs, color="#8a8a8a",
                    fontweight="bold")
        ax.tick_params(axis="x", pad=int(small_fs * 2.4))

    if title:
        ax.set_title(title, fontsize=fontsize_axis + 2, color="#1f1f1f",
                     pad=18, loc="left", fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    rows = list(csv.DictReader(CSV_PATH.open()))
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    full = compute_rows(SYSTEMS, by_model)
    gpt  = compute_rows([s for s in SYSTEMS if s[0].startswith("GPT")], by_model)
    # Expand "Inst." → "Instant" for the zoomed GPT-only chart.
    for r in gpt:
        r["label"] = r["label"].replace("Inst.", "Instant")

    out_full = OUT_DIR / "aggregate_accuracy_bars.png"
    plot_chart(
        full, out_full,
        fontsize_axis=11.5, fontsize_tick=11.5, fontsize_value=10,
        fontsize_legend=11, bar_width=0.36, figsize=(10.8, 5.4),
        y_range=(60, 100), y_step=5,
    )
    print(f"Wrote {out_full.relative_to(REPO)}")

    # GPT-only zoomed chart with large fonts; tighter y-range (75-95) so the
    # 4-pp API/Interface gaps are visually obvious.
    out_gpt = OUT_DIR / "aggregate_accuracy_bars_gpt.png"
    plot_chart(
        gpt, out_gpt,
        fontsize_axis=32, fontsize_tick=28, fontsize_value=30,
        fontsize_legend=30, bar_width=0.42, figsize=(13.0, 9.0),
        y_range=(75, 100), y_step=5, value_pad=0.6,
        ylabel=None, hide_y_ticks=True,
        show_gap_line=True, gap_color="#d35f4a", gap_fontsize=30,
        show_error_bars=False, show_legend=False, per_bar_labels=True,
        title="Mean accuracy across 9 capability benchmarks", dpi=150,
    )
    print(f"Wrote {out_gpt.relative_to(REPO)}")


if __name__ == "__main__":
    main()
