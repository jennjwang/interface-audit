"""Two-panel grouped bars: BBQ accuracy by source (System Card / API / Interface)
for the four models that have published system-card BBQ scores. Left panel:
ambiguous; right panel: disambiguated.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve()
SRC = THIS.parent / "bbq_ambig_split.csv"
OUT_PNG = THIS.parent / "bbq_three_way_bars.png"
OUT_PDF = THIS.parent / "bbq_three_way_bars.pdf"

# System-card values (ambig %, disambig %).
CARD = {
    "GPT 5.4 Thinking": (93.0, 88.0),   # gpt-5-thinking, no web search
    "Claude Haiku":     (98.0, 71.2),   # Claude Haiku 4.5
    "Claude Sonnet":    (97.5, 88.1),   # Claude Sonnet 4.6
    "Claude Opus":      (99.7, 90.9),   # Claude Opus 4.6
}

# Pretty x-axis labels (system-card model identifier in parens).
LABEL = {
    "GPT 5.4 Thinking": "GPT 5.4 Thinking\n(gpt-5-thinking)",
    "Claude Haiku":     "Claude Haiku\n(4.5)",
    "Claude Sonnet":    "Claude Sonnet\n(4.6)",
    "Claude Opus":      "Claude Opus\n(4.6)",
}
ORDER = ["GPT 5.4 Thinking", "Claude Haiku", "Claude Sonnet", "Claude Opus"]

# Load measured API/IFC accuracy split by ambig/disambig.
measured = {}
with SRC.open() as fh:
    for r in csv.DictReader(fh):
        if r["model"] in CARD and r["type"] in ("ambig", "disambig"):
            measured.setdefault(r["model"], {})[r["type"]] = {
                "api": float(r["api_pct"]),
                "ifc": float(r["ifc_pct"]),
            }

fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4), dpi=150, sharey=True)

C_CARD = "#5e3c99"
C_API  = "#2c7fb8"
C_IFC  = "#d95f0e"
SOURCE_COLORS = [C_CARD, C_API, C_IFC]
SOURCE_LABELS = ["System Card", "API", "Interface"]

x = np.arange(len(ORDER))
w = 0.26

for panel_idx, kind in enumerate(("ambig", "disambig")):
    ax = axes[panel_idx]
    card_vals = [CARD[m][0 if kind == "ambig" else 1] for m in ORDER]
    api_vals  = [measured[m][kind]["api"] for m in ORDER]
    ifc_vals  = [measured[m][kind]["ifc"] for m in ORDER]
    series = [card_vals, api_vals, ifc_vals]

    bars = []
    for j, (vals, color, label) in enumerate(zip(series, SOURCE_COLORS, SOURCE_LABELS)):
        offset = (j - 1) * w   # center the middle (API) bar at the model tick
        b = ax.bar(x + offset, vals, w, color=color, label=label,
                   edgecolor="white", linewidth=0.8)
        bars.append(b)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.4,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in ORDER], fontsize=9)
    ax.set_title(f"{'Ambiguous' if kind == 'ambig' else 'Disambiguated'} questions",
                 fontsize=12, pad=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color="#ccc", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#666")
    ax.spines["bottom"].set_color("#666")
    if panel_idx == 0:
        ax.set_ylabel("BBQ accuracy (%)")
        ax.legend(loc="lower left", frameon=False, fontsize=10)

axes[0].set_ylim(60, 105)
fig.suptitle("BBQ accuracy: System Card vs API vs Interface",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT_PNG, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
