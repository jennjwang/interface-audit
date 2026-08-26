"""AA-Omniscience correct rate for Claude models: System Card vs API vs Interface.

Bar heights for API / Interface are pulled from paper/tables/appendix_full_capability.csv.
System-card values come from Anthropic's published AA-Omniscience figure
(the user's reference image). Sonnet 4.6 was not in that figure, so we use
the Sonnet 4.5 system-card value (33.6%) as the closest published reference
and annotate the version mismatch.

Per-side SE error bars are computed from experiments/plots/per_query.csv
(std across the 5 kept runs / sqrt(5)).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
PAPER_CSV = REPO / "paper" / "tables" / "appendix_full_capability.csv"
PER_Q = REPO / "experiments" / "plots" / "per_query.csv"
OUT_PNG = THIS.parent / "aa_omni_styled_bars.png"
OUT_PDF = THIS.parent / "aa_omni_styled_bars.pdf"

# (paper-label, display, provider, model_slug)
MODELS = [
    ("Claude Haiku",  "Haiku 4.5",  "claude", "haiku"),
    ("Claude Sonnet", "Sonnet 4.6", "claude", "sonnet"),
    ("Claude Opus",   "Opus 4.6",   "claude", "opus"),
]

# Anthropic system-card AA-Omniscience values.
# Sonnet 4.6 not in the figure → use Sonnet 4.5 (33.6%) as the closest
# published reference; annotated in the x-axis label.
CARD = {
    "Claude Haiku":  22.3,
    "Claude Sonnet": 33.6,    # Sonnet 4.5 system-card value
    "Claude Opus":   41.3,
}
# SE estimated by eye from the Anthropic system-card figure's error bar
# half-widths (assuming the figure plots ±SE).
CARD_SE = {
    "Claude Haiku":  2.7,
    "Claude Sonnet": 3.3,
    "Claude Opus":   3.0,
}
CARD_VERSION_NOTE = {
    "Claude Sonnet": "(Sonnet 4.5)",
}

COLORS = {
    "Claude Haiku":  ["#fde08a", "#e6b85c", "#a87b1d"],   # yellows
    "Claude Sonnet": ["#a3d3a6", "#3a9a4f", "#1d6132"],   # greens
    "Claude Opus":   ["#f3a48d", "#d65a3e", "#8c2d18"],   # salmons
}
SOURCE_ORDER = ["System Card", "API", "Interface"]
SHORT = {"System Card": "System Card", "API": "API", "Interface": "Iface"}


def per_side_se():
    """Compute SE per (model, side) from per_query.csv across the 5 kept runs."""
    runs = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    mkey = {(p, s): lbl for lbl, _, p, s in MODELS}
    with PER_Q.open() as f:
        for r in csv.DictReader(f):
            if r["benchmark"] != "aa-omniscience":
                continue
            key = mkey.get((r["provider"], r["model"]))
            if key is None:
                continue
            run = int(r["run"])
            for side, ext_col, corr_col in (
                ("api", "api_extracted", "api_correct"),
                ("ifc", "ifc_extracted", "ifc_correct"),
            ):
                if int(r[ext_col]):
                    runs[(key, side)][run][0] += 1
                    runs[(key, side)][run][1] += int(r[corr_col])

    out = {}
    for key, by_run in runs.items():
        accs = [c / e for e, c in by_run.values() if e > 0]
        if len(accs) < 2:
            continue
        out[key] = float(np.std(accs, ddof=1) / np.sqrt(len(accs))) * 100
    return out


AUDIT_DIR = THIS.parent / "score_audit_opus46"
MIN_COVERAGE = 170    # only count runs that scored ≥170 of 200 items


def opus46_from_audit():
    """Mean and SE for Opus 4.6 AA-Omniscience from the score_audit_opus46 CSVs.

    Filters out partial-coverage runs (<170 items) and reports per-run mean /
    SE across the remaining (essentially complete) runs.
    """
    from collections import defaultdict
    out = {}
    for side, fname in (("api", "api__aa_omniscience.csv"),
                         ("ifc", "interface__aa_omniscience.csv")):
        by_run = defaultdict(lambda: [0, 0])
        with (AUDIT_DIR / fname).open() as fh:
            for r in csv.DictReader(fh):
                c = r["correct"].strip().lower() == "true"
                by_run[r["run_id"]][0] += 1
                by_run[r["run_id"]][1] += c
        accs = [c / n for n, c in by_run.values() if n >= MIN_COVERAGE]
        m = float(np.mean(accs)) * 100
        s = float(np.std(accs, ddof=1) / np.sqrt(len(accs))) * 100
        out[side] = (m, s, len(accs))
    return out


def main():
    paper = {}
    with PAPER_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["benchmark"] != "AA-Omniscience":
                continue
            paper[r["model"]] = (float(r["api_pct"]), float(r["interface_pct"]))

    se = per_side_se()

    # Override Opus 4.6 with the audit-rescored numbers.
    opus = opus46_from_audit()
    paper["Claude Opus"] = (opus["api"][0], opus["ifc"][0])
    se[("Claude Opus", "api")] = opus["api"][1]
    se[("Claude Opus", "ifc")] = opus["ifc"][1]
    print(f"Opus 4.6 (audit, {opus['api'][2]} kept runs): "
          f"API={opus['api'][0]:.2f}% ± {opus['api'][1]:.2f}, "
          f"IFC={opus['ifc'][0]:.2f}% ± {opus['ifc'][1]:.2f}")

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=150)

    BAR_W = 0.78
    GAP = 0.6
    x_positions = []
    cur = 0.0
    model_centers = []
    bar_specs = []   # (x, h, err, color, src, model_label)

    for mlabel, _, _, _ in MODELS:
        start = cur
        api, ifc = paper.get(mlabel, (None, None))
        for src_idx, src in enumerate(SOURCE_ORDER):
            color = COLORS[mlabel][src_idx]
            if src == "System Card":
                h, err = CARD[mlabel], CARD_SE.get(mlabel)
            elif src == "API":
                h, err = api, se.get((mlabel, "api"))
            else:
                h, err = ifc, se.get((mlabel, "ifc"))
            bar_specs.append((cur, h, err, color, src, mlabel))
            x_positions.append(cur)
            cur += 1.0
        model_centers.append((start + cur - 1.0) / 2)
        cur += GAP

    for x, h, err, color, src, mlabel in bar_specs:
        if h is None or (isinstance(h, float) and h != h):
            ax.bar(x, 1.5, BAR_W, color="white", edgecolor="#aaa",
                   linewidth=0.8, hatch="///")
            ax.text(x, 2.5, "n/a", ha="center", va="bottom",
                    fontsize=8.5, color="#999")
            continue
        ax.bar(x, h, BAR_W, color=color, edgecolor="white", linewidth=0.6)
        if err is not None:
            ax.errorbar(x, h, yerr=err, color="#333",
                        linewidth=1.0, capsize=3, capthick=1.0)
        ax.text(x, h + (err or 0) + 0.8, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9, color="#222")

    ax.set_xticks(x_positions)
    # Add the version note under the Card tick label for Sonnet (so the
    # 4.5/4.6 mismatch is obvious without crowding the model label).
    xt_labels = []
    for (mlabel, *_), in zip(MODELS):
        for src in SOURCE_ORDER:
            lab = SHORT[src]
            if src == "System Card" and mlabel in CARD_VERSION_NOTE:
                lab = f"{lab}\n{CARD_VERSION_NOTE[mlabel]}"
            xt_labels.append(lab)
    ax.set_xticklabels(xt_labels, fontsize=9, color="#555")

    for (mlabel, disp, _, _), cx in zip(MODELS, model_centers):
        ax.text(cx, -10.0, disp, ha="center", va="top",
                fontsize=11, color="#222", fontweight="bold",
                transform=ax.transData)

    ax.set_title("AA-Omniscience: Correct Rate",
                 fontsize=13, pad=10, fontweight="bold")
    ax.set_ylim(0, 75)
    ax.set_yticks(range(0, 71, 10))
    ax.set_yticklabels([f"{y}%" for y in range(0, 71, 10)])
    ax.set_ylabel("Correct rate")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color="#ccc", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#666")
    ax.spines["bottom"].set_color("#666")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
