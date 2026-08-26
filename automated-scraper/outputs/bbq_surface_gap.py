"""BBQ: API↔Interface surface gap with system-card anchor.

Per model and question type, draws a segment from API accuracy to Interface
accuracy; the system-card value is a single tick on the same axis. The
segment length is the surface gap the vendor card cannot see.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve()
SRC = THIS.parent / "bbq_ambig_split.csv"
OUT_PNG = THIS.parent / "bbq_surface_gap.png"
OUT_PDF = THIS.parent / "bbq_surface_gap.pdf"

CARD = {
    "GPT 5.4 Thinking": (93.0, 88.0),   # gpt-5-thinking, no web search
    "Claude Haiku":     (98.0, 71.2),
    "Claude Sonnet":    (97.5, 88.1),
    "Claude Opus":      (99.7, 90.9),
}
LABEL = {
    "GPT 5.4 Thinking": "GPT 5.4 Thinking",
    "Claude Haiku":     "Claude Haiku 4.5",
    "Claude Sonnet":    "Claude Sonnet 4.6",
    "Claude Opus":      "Claude Opus 4.6",
}
ORDER = ["GPT 5.4 Thinking", "Claude Haiku", "Claude Sonnet", "Claude Opus"]

measured = {}
with SRC.open() as fh:
    for r in csv.DictReader(fh):
        if r["model"] in CARD and r["type"] in ("ambig", "disambig"):
            measured.setdefault(r["model"], {})[r["type"]] = {
                "api": float(r["api_pct"]),
                "ifc": float(r["ifc_pct"]),
            }

C_API   = "#2c7fb8"
C_IFC   = "#d95f0e"
C_SEG   = "#888"
C_CARD  = "#5e3c99"

fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.0), dpi=150, sharex=True)

for panel_idx, kind in enumerate(("ambig", "disambig")):
    ax = axes[panel_idx]
    y_positions = np.arange(len(ORDER))[::-1]   # top-to-bottom in ORDER

    for y, model in zip(y_positions, ORDER):
        api = measured[model][kind]["api"]
        ifc = measured[model][kind]["ifc"]
        card = CARD[model][0 if kind == "ambig" else 1]
        gap = abs(api - ifc)

        # surface gap segment
        ax.plot([min(api, ifc), max(api, ifc)], [y, y],
                color=C_SEG, linewidth=4.0, solid_capstyle="round",
                zorder=2, alpha=0.55)
        # endpoint markers
        ax.scatter([api], [y], s=110, color=C_API, zorder=4,
                   edgecolor="white", linewidth=1.0, label="API" if y == y_positions[0] else None)
        ax.scatter([ifc], [y], s=110, color=C_IFC, zorder=4,
                   edgecolor="white", linewidth=1.0, label="Interface" if y == y_positions[0] else None)
        # system-card tick
        ax.scatter([card], [y], marker="X", s=160, color=C_CARD, zorder=5,
                   edgecolor="white", linewidth=1.2,
                   label="System Card" if y == y_positions[0] else None)

        # gap label to the right of the segment
        right_x = max(api, ifc, card)
        ax.text(right_x + 0.8, y, f"|API−IFC| = {gap:.1f}",
                va="center", ha="left", fontsize=9, color="#444")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([LABEL[m] for m in ORDER], fontsize=10)
    ax.set_title(f"{'Ambiguous' if kind == 'ambig' else 'Disambiguated'} questions",
                 fontsize=12, loc="left", pad=6)
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, color="#ccc", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#666")
    ax.spines["bottom"].set_color("#666")

axes[-1].set_xlabel("BBQ accuracy (%)")
axes[0].set_xlim(65, 108)

# Shared legend at top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10)

fig.suptitle("Surface gap the system card cannot see",
             fontsize=13, y=1.08, x=0.05, ha="left", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 1.0))
fig.savefig(OUT_PNG, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
