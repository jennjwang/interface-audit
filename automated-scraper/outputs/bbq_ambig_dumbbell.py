"""Dumbbell plot: BBQ ambig vs disambig accuracy per (model, side),
with the gpt-5-thinking system-card reference at the top.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

THIS = Path(__file__).resolve()
SRC = THIS.parent / "bbq_ambig_split.csv"
OUT_PNG = THIS.parent / "bbq_ambig_dumbbell.png"
OUT_PDF = THIS.parent / "bbq_ambig_dumbbell.pdf"

MODELS_ORDER = [
    "GPT 5.3 Instant",
    "GPT 5.4 Thinking",
    "Claude Haiku",
    "Claude Sonnet",
    "Claude Opus",
    "Gemini 3 Fast",
    "Gemini 3 Thinking",
]

# {(model, type): {"api_pct": float, "ifc_pct": float}}
data = {}
with SRC.open() as f:
    for r in csv.DictReader(f):
        if r["type"] == "all":
            continue
        data[(r["model"], r["type"])] = {
            "api": float(r["api_pct"]),
            "ifc": float(r["ifc_pct"]),
        }

# Build rows top-to-bottom: system card first, then each model (IFC above API
# so that a model's two rows sit together).
rows = []  # list of (label, ambig_pct, disambig_pct, kind)
rows.append(("gpt-5-thinking (system card)", 93.0, 88.0, "card"))
for m in MODELS_ORDER:
    a = data[(m, "ambig")]
    d = data[(m, "disambig")]
    rows.append((f"{m} — Interface", a["ifc"], d["ifc"], "ifc"))
    rows.append((f"{m} — API",       a["api"], d["api"], "api"))

# Plot
n = len(rows)
fig, ax = plt.subplots(figsize=(9.2, 0.45 * n + 1.2), dpi=150)

C_AMBIG    = "#2c7fb8"   # blue
C_DISAMBIG = "#d95f0e"   # orange
C_LINE     = "#999999"
C_CARD     = "#5e3c99"   # purple highlight for system card

for i, (label, amb, dis, kind) in enumerate(rows):
    y = n - 1 - i  # top row at top
    # connector
    ax.plot([min(amb, dis), max(amb, dis)], [y, y],
            color=C_LINE, linewidth=2, zorder=1, alpha=0.6)
    # endpoints
    ax.scatter([amb], [y], s=70, color=C_AMBIG, zorder=3,
               edgecolor="white", linewidth=0.8)
    ax.scatter([dis], [y], s=70, color=C_DISAMBIG, zorder=3,
               edgecolor="white", linewidth=0.8)
    # gap text on the right
    gap = amb - dis
    ax.text(101.0, y, f"Δ {gap:+.1f}",
            va="center", ha="left", fontsize=8,
            color="#444" if kind != "card" else C_CARD,
            fontweight="bold" if kind == "card" else "normal")

# y axis labels
ax.set_yticks(range(n))
labels_top_down = [r[0] for r in rows]
ax.set_yticklabels(list(reversed(labels_top_down)), fontsize=9)
# Highlight the card row label
for tick, (label, *_rest, kind) in zip(reversed(ax.get_yticklabels()), rows):
    if kind == "card":
        tick.set_color(C_CARD)
        tick.set_fontweight("bold")

# separator below the system-card row
ax.axhline(n - 1.5, color="#bbb", linewidth=0.8, linestyle="--")

# x axis
ax.set_xlim(65, 105)
ax.set_xlabel("BBQ accuracy (%)")
ax.set_title("BBQ accuracy: ambiguous vs disambiguated\n"
             "(5-run mean; 99 ambig / 101 disambig items)",
             fontsize=11)

# legend
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=C_AMBIG, markersize=9, label="Ambiguous"),
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=C_DISAMBIG, markersize=9, label="Disambiguated"),
]
ax.legend(handles=legend_handles, loc="lower left",
          frameon=False, fontsize=9)

# light grid
ax.grid(axis="x", linestyle=":", linewidth=0.7, color="#ccc", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#666")
ax.spines["bottom"].set_color("#666")

fig.tight_layout()
fig.savefig(OUT_PNG, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
