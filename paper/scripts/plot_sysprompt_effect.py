"""How does the interface system prompt move accuracy relative to the
baseline API, compared to the actual Interface change?

Scatter plot: for each (system, benchmark) cell,
  x = Interface accuracy - baseline API accuracy   (the surface gap to explain)
  y = SP accuracy - baseline API accuracy          (what the system prompt actually does)

The y = x diagonal is "the system prompt perfectly reproduces the interface
shift". The y = 0 line is "the system prompt does nothing". Points along y = x
mean SP explains the surface gap; points between y = 0 and y = x mean SP
partially explains; points above y = x mean SP overshoots; points opposite-sign
from x mean SP moves the wrong way.

Reads paper/tables/appendix_system_prompt_ablation.csv. Writes
paper/figures/sysprompt_effect.png.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
CSV_PATH = REPO / "paper" / "tables" / "appendix_system_prompt_ablation.csv"
OUT_DIR = REPO / "paper" / "figures"
OUT_DIR.mkdir(exist_ok=True)

# (CSV model key, display label, dot colour)
MODELS = [
    ("gpt-5.3",            "GPT 5.3 Inst.",  "#009E73"),   # bluish green
    ("gpt-5.4",            "GPT 5.4 Think",  "#00604D"),   # deep green
    ("claude-opus-4.6",    "Claude Opus",    "#8C4708"),   # vermillion/brown
    ("claude-sonnet-4.6",  "Claude Sonnet",  "#C46D08"),   # orange
    ("gemini-3-flash",     "Gemini 3 Fast",  "#56B4E9"),   # sky blue
]


def main():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.9,
    })

    by_model = defaultdict(list)
    for r in csv.DictReader(CSV_PATH.open()):
        gap_to_close = float(r["iface_mean"]) - float(r["api_mean"])
        sp_change    = float(r["sp_mean"])    - float(r["api_mean"])
        by_model[r["model"]].append((r["benchmark"], gap_to_close, sp_change))

    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    fig.patch.set_facecolor("white")

    # Symmetric frame; clip outliers visually with a slightly tighter window
    lo, hi = -22, 13

    # Shading: wrong-direction quadrants (x<0, y>0) and (x>0, y<0)
    ax.fill_between([lo, 0], 0, hi, color="#fff0e0", alpha=0.55, zorder=0)
    ax.fill_between([0, hi], lo, 0, color="#fff0e0", alpha=0.55, zorder=0)

    # Reference lines
    ax.axhline(0, color="#888", linewidth=0.7, zorder=1)
    ax.axvline(0, color="#888", linewidth=0.7, zorder=1)
    ax.plot([lo, hi], [lo, hi], "--", color="#9aa9bd", linewidth=1.0, zorder=2,
            label="$y = x$ (SP fully matches Interface)")

    # Annotated outliers we want to call out
    callouts = {
        ("gemini-3-flash", "Elephant Flip"): ("right", "top",  -0.4, -0.5),
        ("gpt-5.4",        "WinoGrande"):    ("left",  "top",   0.6, -0.5),
        ("gpt-5.4",        "HellaSwag"):     ("right", "top",  -0.6, -0.5),
    }

    legend_handles = []
    for csv_key, label, color in MODELS:
        xs, ys = [], []
        for b, x, y in by_model.get(csv_key, []):
            # Clip wildly-OOB outliers into the visible range but mark them
            y_plot = max(lo + 0.5, min(hi - 0.5, y))
            xs.append(x); ys.append(y_plot)
            # Callouts
            if (csv_key, b) in callouts:
                ha, va, dx, dy = callouts[(csv_key, b)]
                ax.annotate(f"{b}\n({csv_key}: SP {y:+.1f}, Iface {x:+.1f})",
                            xy=(x, y_plot), xytext=(x + dx, y_plot + dy),
                            ha=ha, va=va, fontsize=8.5, color="#444",
                            arrowprops=dict(arrowstyle="-", color="#bbb",
                                            connectionstyle="arc3,rad=0.0",
                                            linewidth=0.7))
        ax.scatter(xs, ys, s=80, color=color, marker="o",
                   edgecolor="white", linewidth=1.2, zorder=3)
        legend_handles.append(
            plt.Line2D([], [], marker="o", linestyle="",
                       markerfacecolor=color, markeredgecolor="white",
                       markersize=10, label=label)
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Interface $-$ API (pp): actual surface gap",
                  fontsize=11.5, labelpad=8)
    ax.set_ylabel("SP $-$ API (pp): what the system prompt does",
                  fontsize=11.5, labelpad=8)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.grid(linestyle=":", color="#dddddd", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="box")

    # Quadrant captions, placed clear of the data
    ax.text(lo + 0.6, hi - 0.6, "Wrong direction\n(SP helps where Iface hurts)",
            ha="left", va="top", fontsize=9, color="#8a5a1c", fontstyle="italic")
    ax.text(hi - 0.6, lo + 0.6, "Wrong direction\n(SP hurts where Iface helps)",
            ha="right", va="bottom", fontsize=9, color="#8a5a1c", fontstyle="italic")

    leg_y = ax.legend(handles=[plt.Line2D([], [], linestyle="--",
                                          color="#9aa9bd",
                                          label="$y = x$ (SP fully matches Interface)")],
                      loc="upper right", frameon=False, fontsize=9.5)
    ax.add_artist(leg_y)
    ax.legend(handles=legend_handles, title="System",
              loc="lower left", frameon=False, fontsize=9.5,
              title_fontsize=10, handletextpad=0.3, borderaxespad=0.5)

    plt.tight_layout()
    out = OUT_DIR / "sysprompt_effect.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out.relative_to(REPO)}")

    # Print a short text summary
    same_dir = wrong_dir = overshoot = partial = total = 0
    for cells in by_model.values():
        for b, x, y in cells:
            total += 1
            if (x > 0 and y > 0) or (x < 0 and y < 0):
                same_dir += 1
                if abs(y) > abs(x):
                    overshoot += 1
                else:
                    partial += 1
            elif x != 0 and y != 0:
                wrong_dir += 1
    print(f"\nSP direction summary across {total} cells:")
    print(f"  same direction as Iface : {same_dir} ({100*same_dir/total:.0f}%)"
          f"  [partial: {partial}, overshoot: {overshoot}]")
    print(f"  wrong direction         : {wrong_dir} ({100*wrong_dir/total:.0f}%)")


if __name__ == "__main__":
    main()
