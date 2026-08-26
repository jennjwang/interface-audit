"""BBQ ambig vs disambig accuracy in the AA-Omniscience bar style.

Two side-by-side panels (ambig | disambig). Within each panel: per-model color
families with light→dark shades for System Card → API → Interface. Error bars
are per-run SE for API/Interface; the System Card has no SE (omitted).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
SUBSET = REPO / "benchmark_creation" / "results" / "bbq-subset-200.csv"
PER_Q = REPO / "experiments" / "plots" / "per_query.csv"
OUT_PNG = THIS.parent / "bbq_styled_bars.png"
OUT_PDF = THIS.parent / "bbq_styled_bars.pdf"

MODELS = [
    ("GPT 5.4 Thinking", "chatgpt", "gpt-5-4-thinking", "GPT 5.4 Thinking"),
    ("Claude Haiku",     "claude",  "haiku",            "Haiku 4.5"),
    ("Claude Sonnet",    "claude",  "sonnet",           "Sonnet 4.6"),
    ("Claude Opus",      "claude",  "opus",             "Opus 4.6"),
]

# Per-model note shown under the System Card tick label when the card was
# published under a different model name/version than what we measured.
CARD_VERSION_NOTE = {
    "GPT 5.4 Thinking": "(gpt-5-thinking)",
}

# OpenAI / Anthropic published system-card BBQ numbers.
CARD = {
    "GPT 5.4 Thinking": {"ambig": 93.0, "disambig": 88.0},
    "Claude Haiku":     {"ambig": 98.0, "disambig": 71.2},
    "Claude Sonnet":    {"ambig": 97.5, "disambig": 88.1},
    "Claude Opus":      {"ambig": 99.7, "disambig": 90.9},
}

# Color families per model (light → dark for System Card → API → Interface).
COLORS = {
    "GPT 5.4 Thinking": ["#a6cee3", "#1f78b4", "#08519c"],   # blues
    "Claude Haiku":     ["#fde08a", "#e6b85c", "#a87b1d"],   # yellows
    "Claude Sonnet":    ["#a3d3a6", "#3a9a4f", "#1d6132"],   # greens
    "Claude Opus":      ["#f3a48d", "#d65a3e", "#8c2d18"],   # salmons
}
SOURCE_ORDER = ["System Card", "API", "Interface"]
SOURCE_LABEL_SHORT = {"System Card": "System Card", "API": "API", "Interface": "Iface"}

UNK = {"can't be determined","can't answer",'cannot answer','cannot be determined',
       'not answerable','not enough info','not enough information','not known',
       'undetermined','unknown'}


def classify(query, gold):
    import re
    m = re.findall(r'^([ABC])\.\s*(.+?)\s*$', query, re.MULTILINE)
    opts = dict(m)
    unk = next((k for k,v in opts.items() if v.strip().lower() in UNK), None)
    if unk is None:
        return None
    return 'ambig' if gold == unk else 'disambig'


AUDIT_DIR = REPO / "automated-scraper" / "outputs" / "score_audit_opus46"
MIN_BBQ_EXT = 180   # ≥0.90 of 200 items scored


def opus46_bbq_from_audit(label):
    """Return {(kind, side): (mean_pct, se_pct)} and 'n_runs' for Opus 4.6 BBQ
    from the audit, restricted to runs with ≥0.90 extraction on both sides.
    """
    def load(path):
        items = defaultdict(dict); run_n = defaultdict(int)
        with open(path) as fh:
            for r in csv.DictReader(fh):
                c = (r.get("correct") or "").strip().lower()
                if c not in ("true", "false"):
                    continue
                run_n[r["run_id"]] += 1
                items[r["qid"]][r["run_id"]] = (c == "true")
        full = {rid for rid, n in run_n.items() if n >= MIN_BBQ_EXT}
        return items, full

    api_items, api_full = load(AUDIT_DIR / "api__bbq.csv")
    ifc_items, ifc_full = load(AUDIT_DIR / "interface__bbq.csv")
    paired = sorted(api_full & ifc_full)

    def pooled_acc(items, kind):
        n_ext = c_corr = 0
        for qid_str, runs in items.items():
            try:
                qid = int(qid_str)
            except ValueError:
                continue
            if label.get(qid) != kind:
                continue
            for rid in paired:
                if rid in runs:
                    n_ext += 1
                    c_corr += int(runs[rid])
        return n_ext, c_corr

    out = {"n_runs": len(paired)}
    for kind in ("ambig", "disambig"):
        for side, items in (("api", api_items), ("ifc", ifc_items)):
            n_ext, c_corr = pooled_acc(items, kind)
            if n_ext == 0:
                out[(kind, side)] = (float("nan"), 0.0)
                continue
            p = c_corr / n_ext
            se = (p * (1 - p) / n_ext) ** 0.5
            out[(kind, side)] = (p * 100, se * 100)
    return out


def main():
    # 1. ambig label per qid
    label = {}
    with SUBSET.open() as f:
        for r in csv.DictReader(f):
            c = classify(r['query'], r['answer'])
            if c is not None:
                label[int(r['id'])] = c

    # 2. per-run accuracy for each (model, side, kind). Side ∈ {api, ifc}.
    #    per_run[(label, kind, side)] = list of run-level accuracy values (one per run)
    per_run = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (k, run) -> [ext, cor]
    mkey = {(p, m): lbl for lbl, p, m, _ in MODELS}
    with PER_Q.open() as f:
        for r in csv.DictReader(f):
            if r['benchmark'] != 'bbq':
                continue
            key = mkey.get((r['provider'], r['model']))
            if key is None:
                continue
            try:
                qid = int(r['qid'])
            except ValueError:
                continue
            kind = label.get(qid)
            if kind is None:
                continue
            run = int(r['run'])
            for side, ext_col, corr_col in (
                ("api", "api_extracted", "api_correct"),
                ("ifc", "ifc_extracted", "ifc_correct"),
            ):
                ext = int(r[ext_col])
                cor = int(r[corr_col])
                if ext:
                    per_run[(key, kind, side)][run][0] += 1
                    per_run[(key, kind, side)][run][1] += cor

    # Convert per-run buckets → pooled accuracy and binomial SE.
    # SE = sqrt(p*(1-p)/n) where n = total item·run extracted attempts.
    means_se = {}
    for key, runs in per_run.items():
        n_ext = sum(e for e, _ in runs.values())
        n_cor = sum(c for _, c in runs.values())
        if n_ext < 2:
            continue
        p = n_cor / n_ext
        se = (p * (1 - p) / n_ext) ** 0.5
        means_se[key] = (p * 100, se * 100)

    # Override Opus 4.6 with audit-rescored BBQ numbers.
    opus_audit = opus46_bbq_from_audit(label)
    for kind in ("ambig", "disambig"):
        for side in ("api", "ifc"):
            means_se[("Claude Opus", kind, side)] = opus_audit[(kind, side)]
    print(f"Opus 4.6 BBQ (audit, {opus_audit['n_runs']} paired runs):")
    for kind in ("ambig", "disambig"):
        for side in ("api", "ifc"):
            m, s = opus_audit[(kind, side)]
            print(f"  {kind:8s} {side}: {m:.2f}% ± {s:.2f}")

    # 3. Plot setup. Two panels stacked vertically.
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 10.2), dpi=150)

    # Within each panel: bars laid out as [M1.Card, M1.API, M1.IFC, gap,
    # M2.Card, M2.API, M2.IFC, gap, ...]. Gap = visual spacer between model
    # groups (achieved by leaving x-positions empty).
    BAR_W = 0.78
    GAP = 0.6   # spacer between model groups

    for panel_idx, kind in enumerate(("ambig", "disambig")):
        ax = axes[panel_idx]
        x_positions = []
        bar_specs = []   # (x, height, yerr, color, model_label)
        cur = 0.0
        model_centers = []
        for (mlabel, _prov, _slug, _disp), _ in zip(MODELS, MODELS):
            start = cur
            for src_idx, src in enumerate(SOURCE_ORDER):
                if src == "System Card":
                    h = CARD[mlabel][kind]
                    err = None
                else:
                    side = "api" if src == "API" else "ifc"
                    m, se = means_se.get((mlabel, kind, side), (float("nan"), 0.0))
                    h = m
                    err = se
                color = COLORS[mlabel][src_idx]
                bar_specs.append((cur, h, err, color))
                x_positions.append(cur)
                cur += 1.0
            # center of this model group
            model_centers.append((start + cur - 1.0) / 2)
            cur += GAP   # spacer

        # Draw bars
        for (x, h, err, color), src in zip(bar_specs, SOURCE_ORDER * len(MODELS)):
            ax.bar(x, h, BAR_W, color=color, edgecolor="white", linewidth=0.6)
            if err is not None:
                ax.errorbar(x, h, yerr=err, color="#333", linewidth=1.0,
                            capsize=3, capthick=1.0)
            ax.text(x, h + 0.6, f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, color="#222")

        # Source labels under each bar (System Card / API / Iface),
        # with an optional version note appended below the System Card tick
        # when the card was published under a different model name.
        ax.set_xticks(x_positions)
        xt_labels = []
        for mlabel, *_ in MODELS:
            for src in SOURCE_ORDER:
                lab = SOURCE_LABEL_SHORT[src]
                if src == "System Card" and mlabel in CARD_VERSION_NOTE:
                    lab = f"{lab}\n{CARD_VERSION_NOTE[mlabel]}"
                xt_labels.append(lab)
        ax.set_xticklabels(xt_labels, fontsize=8.5, color="#555")

        # Model labels as a second row below (manually) — pushed further
        # down to leave a gap above the two-line System Card tick label.
        for (mlabel, _prov, _slug, disp), cx in zip(MODELS, model_centers):
            ax.text(cx, -16.0, disp, ha="center", va="top",
                    fontsize=10.5, color="#222", fontweight="bold",
                    transform=ax.transData)

        ax.set_title(
            f"{'Ambiguous' if kind == 'ambig' else 'Disambiguated'} questions",
            fontsize=12.5, pad=8,
        )
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 10))
        ax.set_yticklabels([f"{y}%" for y in range(0, 101, 10)])
        ax.grid(axis="y", linestyle=":", linewidth=0.7, color="#ccc", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#666")
        ax.spines["bottom"].set_color("#666")

        ax.set_ylabel("Correct rate")

    fig.suptitle("BBQ: Correct Rate",
                 fontsize=14, y=1.00, fontweight="bold")

    # Legend below the figure (grey swatches showing the light→dark mapping
    # to System Card / API / Interface used within each model family).
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#cccccc", edgecolor="white", label="System Card"),
        Patch(facecolor="#888888", edgecolor="white", label="API"),
        Patch(facecolor="#3a3a3a", edgecolor="white", label="Interface"),
    ]
    fig.legend(
        handles=legend_handles, ncol=3, loc="lower center",
        bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=10.5,
        handlelength=1.6, handleheight=1.0, columnspacing=1.6,
        title="Within each model: light → dark shade",
        title_fontsize=9,
    )

    fig.tight_layout()
    # leave a little extra room under each panel for the bold model labels and
    # the legend below.
    fig.subplots_adjust(bottom=0.13, hspace=0.55)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
