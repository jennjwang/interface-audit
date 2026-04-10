"""Heatmap of per-question answer patterns across runs for selected questions.

Usage:
    python plot_question_examples.py --data-dir metabench-mmlu/data-combined \
        --answer-key metabench-mmlu/queries/metabench_mmlu_5shot.csv \
        --questions 76 59

    python plot_question_examples.py --data-dir metabench-truthfulQA/data \
        --answer-key metabench-truthfulQA/queries/metabench_truthfulQA.csv

If --questions is omitted, auto-selects questions with highest answer variance.
"""
from __future__ import annotations

import argparse
import csv
import string
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import get_config

_BASE_COLORS = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899",
                "#06B6D4", "#84CC16", "#F97316", "#6366F1", "#14B8A6", "#E11D48"]
LETTER_COLORS = {}
for i, c in enumerate(string.ascii_uppercase):
    LETTER_COLORS[c] = _BASE_COLORS[i % len(_BASE_COLORS)]
LETTER_COLORS["-"] = "#D1D5DB"

ANSWER_CATEGORIES = list(string.ascii_uppercase)
CORRECT_CATEGORIES = ["True", "False"]


def load_all_answers(cfg, answer_key_path: Path | None):
    timestamps = cfg.get_timestamps()

    display_labels = []
    for label in cfg.labels:
        parts = label.split(": ", 1)
        display = f"{parts[0]}:\n{parts[1]}" if len(parts) == 2 else label
        display_labels.append(display)

    data = defaultdict(lambda: defaultdict(list))

    for label, (source, filename) in cfg.models.items():
        parts = label.split(": ", 1)
        display = f"{parts[0]}:\n{parts[1]}" if len(parts) == 2 else label
        for ts in timestamps:
            path = cfg.csv_path(source, filename, ts)
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        qid = int(row.get("id", ""))
                    except (ValueError, TypeError):
                        continue
                    ans = (row.get("answer") or "").strip().upper()
                    data[qid][display].append(ans if ans else "-")

    answer_key = {}
    if answer_key_path and answer_key_path.exists():
        with open(answer_key_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    answer_key[int(row["id"])] = row["answer"].strip().upper()
                except Exception:
                    pass

    return data, answer_key, timestamps, display_labels


def _find_high_variance_questions(data, answer_key, labels, n=2):
    """Auto-select questions with highest answer variance across conditions."""
    scores = []
    for qid in sorted(data.keys()):
        all_answers = []
        for label in labels:
            all_answers.extend(data[qid].get(label, []))
        unique = set(a for a in all_answers if a != "-")
        if len(unique) > 1:
            scores.append((qid, len(unique), len(all_answers)))
    scores.sort(key=lambda x: (-x[1], -x[2]))
    return [(qid, f"High variance ({n_unique} distinct answers)") for qid, n_unique, _ in scores[:1]]


def plot_question(ax, qid, gold, labels, data, n_runs):
    n_conds = len(labels)

    for ci, label in enumerate(labels):
        answers = data[qid].get(label, [])
        row_answers: list[str] = []

        # Exactly n_runs slots per condition; treat missing as "-"
        for ri in range(n_runs):
            if ri < len(answers):
                ans = (answers[ri] or "-") or "-"
            else:
                ans = "-"
            row_answers.append(ans)

            # Draw the cell
            color = LETTER_COLORS.get(ans, "#D1D5DB")
            is_correct = ans == gold

            rect = plt.Rectangle((ri - 0.42, ci - 0.38), 0.84, 0.76,
                                 facecolor=color, edgecolor="white", linewidth=0.8,
                                 alpha=0.85, zorder=2)
            ax.add_patch(rect)
            ax.text(ri, ci, ans, ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=3)

            if not is_correct and ans != "-":
                ax.plot(ri, ci + 0.28, marker="v", color="#1F2937", markersize=4, zorder=4)

        # Per-condition observed agreement (P_o) for this question, matching
        # the per-question component of Fleiss' kappa.
        # P_o = (1 / (n*(n-1))) * sum(n_j^2) - 1/(n-1)
        # where n_j is the count of raters in category j.
        answered_only = [a for a in row_answers if a != "-"]
        n = len(answered_only)
        if n >= 2:
            # FK(answer): agreement on answer identity (A/B/C/D/...)
            from collections import Counter
            ans_counts = Counter(answered_only)
            po_answer = (sum(c * c for c in ans_counts.values()) - n) / (n * (n - 1))

            # FK(correct): agreement on correctness (binary)
            if gold != "?":
                n_correct = sum(1 for a in answered_only if a == gold)
                n_wrong = n - n_correct
                po_correct = (n_correct**2 + n_wrong**2 - n) / (n * (n - 1))
            else:
                po_correct = None

            x_stats = n_runs + 0.1
            ax.text(
                x_stats, ci - 0.10,
                f"P\u2092(answer)={po_answer:.2f}",
                ha="left", va="center",
                fontsize=8.5, color="#4B5563",
            )
            if po_correct is not None:
                ax.text(
                    x_stats, ci + 0.18,
                    f"P\u2092(correct)={po_correct:.2f}",
                    ha="left", va="center",
                    fontsize=8.5, color="#111827",
                )

    ax.set_xlim(-0.5, n_runs + 3.0)
    ax.set_ylim(n_conds - 0.5, -0.5)
    ax.set_xticks(range(n_runs))
    ax.set_xticklabels([f"R{i+1}" for i in range(n_runs)], fontsize=7)
    ax.set_yticks(range(n_conds))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)

    n_api = sum(1 for l in labels if l.startswith("API"))
    if 0 < n_api < n_conds:
        ax.axhline(n_api - 0.5, color="#94A3B8", linewidth=1.5, linestyle="--", zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def main():
    parser = argparse.ArgumentParser(parents=[])
    parser.add_argument("--answer-key", default=None,
                        help="Path to answer key CSV (id, answer)")
    parser.add_argument("--questions", nargs="*", type=int, default=None,
                        help="Question IDs to plot (default: auto-select high-variance)")
    cfg = get_config()
    args, _ = parser.parse_known_args()

    ak_path = Path(args.answer_key) if args.answer_key else None
    if ak_path and not ak_path.is_absolute():
        ak_path = (Path.cwd() / ak_path).resolve()

    # If no answer-key was provided, try to infer a sensible default for known
    # datasets so gold answers are populated in the plot.
    if ak_path is None:
        dataset_name = cfg.base_dir.name
        inferred = cfg.base_dir / "queries" / "queries.csv"
        if inferred.exists():
            ak_path = inferred

    data, answer_key, timestamps, labels = load_all_answers(cfg, ak_path)
    n_runs = len(timestamps)
    dataset_name = cfg.base_dir.name

    if args.questions:
        questions = [(qid, "") for qid in args.questions]
    else:
        questions = _find_high_variance_questions(data, answer_key, labels, n=2)

    if not questions:
        print("No questions to plot.")
        return

    fig, axes = plt.subplots(len(questions), 1, figsize=(13, 3.2 + 3.0 * len(questions)),
                             gridspec_kw={"hspace": 0.45})
    if len(questions) == 1:
        axes = [axes]

    for ax, (qid, desc) in zip(axes, questions):
        gold = answer_key.get(qid, "?")
        
        plot_question(ax, qid, gold, labels, data, n_runs)
        if desc:
            title = f"Q{qid}: {desc}   (correct = {gold})"
        else:
            title = f"Q{qid}   (correct = {gold})"
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8, loc="left")

    seen_letters = set()
    for qid, _ in questions:
        for label in labels:
            seen_letters.update(data[qid].get(label, []))
    seen_letters.discard("-")
    legend_patches = [mpatches.Patch(facecolor=LETTER_COLORS[l], label=l)
                      for l in sorted(seen_letters) if l in LETTER_COLORS]
    legend_patches.append(plt.Line2D([0], [0], marker="v", color="#1F2937", linestyle="None",
                                     markersize=5, label="incorrect"))
    fig.legend(handles=legend_patches, loc="lower center", ncol=min(len(legend_patches), 8),
               fontsize=10, frameon=True, framealpha=0.9, edgecolor="#E5E7EB",
               bbox_to_anchor=(0.5, -0.01))

    # Explain per-condition statistics used in the right margin.
    fig.text(
        0.5, -0.06,
        "P\u2092(answer) = observed pairwise agreement on answer identity;  "
        "P\u2092(correct) = observed pairwise agreement on correctness.",
        ha="center", va="top", fontsize=9, color="#374151",
    )

    fig.suptitle(f"{dataset_name} — Answer Patterns Across {n_runs} Repeated Sessions",
                 fontsize=14, fontweight="bold", y=0.98)

    out = cfg.base_dir / "plots" / "question_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
