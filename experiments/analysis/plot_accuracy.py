"""Bar chart of accuracy by model/condition.

Usage:
    python plot_accuracy.py --data-dir metabench-mmlu/data-combined
    python plot_accuracy.py --data-dir metabench-truthfulQA/data --exclude chat-latest

Leaderboard mode (overall AVG across benchmarks):
    python plot_accuracy.py --leaderboard
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import get_config


def compute_accuracy(csv_path: Path) -> float | None:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scorable = [r for r in rows if (r.get("gold_answer") or "").strip()]
    if not scorable:
        return None
    correct = sum(1 for r in scorable if str(r.get("correct")) == "True")
    return correct / len(scorable)


def _mean_se(values: list[float]) -> tuple[float | None, float | None, int]:
    n = len(values)
    if n == 0:
        return None, None, 0
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0, 1
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return mu, math.sqrt(var / n), n


def _short_label(label: str) -> str:
    return label.split(": ", 1)[1] if ": " in label else label


def plot_leaderboard_avg_accuracy(
    coverage_csv: Path,
    out_dir: Path,
    complete_only: bool,
) -> None:
    with coverage_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_run: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if complete_only and str(r.get("complete_run", "")).strip() != "True":
            continue
        try:
            acc = float(r["accuracy"])
            run_index = int(r["run_index"])
        except (TypeError, ValueError, KeyError):
            continue
        cond = (r.get("condition") or "").strip()
        if not cond:
            continue
        by_run[cond][run_index].append(acc)

    # Per condition: mean across benchmarks per run, then mean/SE across runs.
    conditions = sorted(by_run.keys())
    per_cond_run_means: dict[str, list[float]] = {}
    for cond in conditions:
        run_means: list[float] = []
        for run_idx in sorted(by_run[cond]):
            vals = by_run[cond][run_idx]
            if vals:
                run_means.append(sum(vals) / len(vals))
        per_cond_run_means[cond] = run_means

    stats = {c: _mean_se(per_cond_run_means[c]) for c in conditions}
    conditions.sort(key=lambda c: (-(stats[c][0] or -1.0), c))

    means = [stats[c][0] or 0.0 for c in conditions]
    ses = [stats[c][1] or 0.0 for c in conditions]
    counts = [stats[c][2] for c in conditions]

    API_COLOR = "#4A90D9"
    IFACE_COLOR = "#E07B53"
    colors = [API_COLOR if c.startswith("API") else IFACE_COLOR for c in conditions]

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(conditions)), 5.5))

    x = np.arange(len(conditions))
    bars = ax.bar(
        x,
        means,
        yerr=ses,
        capsize=5,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        error_kw=dict(lw=1.5, capthick=1.2),
        zorder=3,
    )

    for bar, m, se, n in zip(bars, means, ses, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + se + 0.008,
            f"{m:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.03,
            f"n={n}",
            ha="center",
            va="top",
            fontsize=8,
            color="white",
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(c) for c in conditions], fontsize=11)
    ax.set_ylabel("AVG Accuracy", fontsize=12)
    title = "OpenLLM Leaderboard — AVG Accuracy by Condition"
    if complete_only:
        title += " (complete runs only)"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    y_min = max(0, min(means) - 0.15) if means else 0
    ax.set_ylim(y_min, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    groups = []
    if any(c.startswith("API") for c in conditions):
        groups.append(Patch(facecolor=API_COLOR, edgecolor="white", label="API"))
    if any(c.startswith("Interface") for c in conditions):
        groups.append(Patch(facecolor=IFACE_COLOR, edgecolor="white", label="Interface"))
    if groups:
        ax.legend(handles=groups, loc="lower right", fontsize=10, framealpha=0.9)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "avg_accuracy_by_condition.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to {out}")
    for c in conditions:
        m, se, n = stats[c]
        m_s = f"{m:.1%}" if m is not None else "-"
        se_s = f"{se:.1%}" if se is not None else "-"
        print(f"  {c:30s}  {m_s} ± {se_s}  (n={n})")


def main():
    script_dir = Path(__file__).resolve().parent
    default_coverage = script_dir.parent / "openllm_leaderboard" / "plots" / "coverage_by_run.csv"
    default_out_dir = script_dir.parent / "openllm_leaderboard" / "plots"

    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--leaderboard",
        action="store_true",
        help="Plot overall leaderboard AVG accuracy using openllm_leaderboard/coverage_by_run.csv",
    )
    p.add_argument(
        "--coverage-csv",
        default=None,
        help="Path to coverage_by_run.csv (implies --leaderboard)",
    )
    p.add_argument(
        "--out-dir",
        default=str(default_out_dir),
        help="Output directory for leaderboard plot",
    )
    p.add_argument(
        "--complete-only",
        action="store_true",
        help="Only include rows where complete_run=True (leaderboard mode)",
    )
    args, unknown = p.parse_known_args()

    if args.leaderboard or args.coverage_csv:
        coverage = Path(args.coverage_csv) if args.coverage_csv else default_coverage
        plot_leaderboard_avg_accuracy(coverage, Path(args.out_dir), complete_only=args.complete_only)
        return

    cfg = get_config(unknown)
    timestamps = cfg.get_timestamps()
    dataset_name = cfg.base_dir.name

    data: dict[str, list[float]] = {l: [] for l in cfg.labels}
    for label, (source, filename) in cfg.models.items():
        for ts in timestamps:
            path = cfg.csv_path(source, filename, ts)
            if path.exists():
                acc = compute_accuracy(path)
                if acc is not None:
                    data[label].append(acc)

    labels = cfg.labels
    means = [np.mean(data[l]) if data[l] else 0 for l in labels]
    stds = [np.std(data[l], ddof=1) if len(data[l]) > 1 else 0 for l in labels]
    counts = [len(data[l]) for l in labels]
    API_COLOR = "#4A90D9"
    IFACE_COLOR = "#E07B53"
    colors = [API_COLOR if l.startswith("API") else IFACE_COLOR for l in labels]

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(labels)), 5.5))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="white",
                  linewidth=1.2, error_kw=dict(lw=1.5, capthick=1.2), zorder=3)

    for bar, m, s, n in zip(bars, means, stds, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.008,
                f"{m:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.03,
                f"n={n}", ha="center", va="top", fontsize=8, color="white", alpha=0.9)

    short_labels = [cfg.short_label(l) for l in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{dataset_name} Accuracy by Model / Condition",
                 fontsize=14, fontweight="bold", pad=12)
    y_min = max(0, min(means) - 0.15) if means else 0
    ax.set_ylim(y_min, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    groups = []
    if cfg.api_labels:
        groups.append(Patch(facecolor=API_COLOR, edgecolor="white", label="API"))
    if cfg.iface_labels:
        groups.append(Patch(facecolor=IFACE_COLOR, edgecolor="white", label="Interface"))
    if groups:
        ax.legend(handles=groups, loc="lower right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out = cfg.base_dir / "plots" / "accuracy_by_model.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved to {out}")

    for l, m, s, n in zip(labels, means, stds, counts):
        print(f"  {l:30s}  {m:.1%} ± {s:.1%}  (n={n})")


if __name__ == "__main__":
    main()
