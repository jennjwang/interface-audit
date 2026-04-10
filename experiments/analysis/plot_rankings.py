"""Bump chart + frequency chart of ranking stability (Kendall's W).

Usage:
    python plot_rankings.py --data-dir metabench-mmlu/data-combined
    python plot_rankings.py --data-dir metabench-truthfulQA/data
"""
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import rankdata

from config import get_config


def compute_accuracy(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scorable = [r for r in rows if (r.get("gold_answer") or "").strip()]
    if not scorable:
        return None
    return sum(1 for r in scorable if str(r.get("correct")) == "True") / len(scorable)


def load_data(cfg):
    timestamps = cfg.get_timestamps()
    labels = cfg.labels
    acc = np.zeros((len(timestamps), len(labels)))

    for j, (label, (source, filename)) in enumerate(cfg.models.items()):
        for i, ts in enumerate(timestamps):
            path = cfg.csv_path(source, filename, ts)
            acc[i, j] = compute_accuracy(path) if path.exists() else np.nan

    return timestamps, labels, acc


def kendalls_w(rank_matrix):
    m, k = rank_matrix.shape
    if k < 2 or m < 2:
        return float("nan"), float("nan")
    R_j = rank_matrix.sum(axis=0)
    R_bar = R_j.mean()
    S = np.sum((R_j - R_bar) ** 2)
    W = (12 * S) / (m ** 2 * (k ** 3 - k))
    from scipy.stats import chi2 as chi2_dist
    chi2 = m * (k - 1) * W
    p_val = 1 - chi2_dist.cdf(chi2, df=k - 1)
    return W, p_val


def make_bump_chart(ax, labels, all_idx, ranks, acc_matrix, colors, n_runs, run_labels, title_prefix):
    for k, j in enumerate(all_idx):
        label = labels[j]
        short = label.split(": ")[1] if ": " in label else label
        color = colors[label]
        ax.plot(range(n_runs), ranks[:, k], marker="o", color=color,
                linewidth=2.2, markersize=7, label=short, zorder=3)
        for i in range(n_runs):
            ax.annotate(f"{acc_matrix[i, j]:.0%}", (i, ranks[i, k]),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=7, ha="center", color=color, fontweight="bold")
    if n_runs == 0:
        return

    n_items = len(all_idx)
    ax.set_yticks(range(1, n_items + 1))
    ax.set_yticklabels([f"#{i}" for i in range(1, n_items + 1)])
    ax.invert_yaxis()
    ax.set_xticks(range(n_runs))
    ax.set_xticklabels(run_labels, fontsize=8, rotation=45, ha="right")
    ax.set_title(title_prefix, fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(n_items + 0.6, 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Rank")


def make_freq_chart(ax, labels, all_idx, ranks, n_runs, highlight_color, faded_color):
    if n_runs == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    orderings = Counter()
    for i in range(n_runs):
        order = tuple(labels[all_idx[k]] for k in np.argsort(ranks[i]))
        orderings[order] += 1

    bars_data = orderings.most_common()
    bar_labels = [" > ".join(o.split(": ")[1] if ": " in o else o for o in order) for order, _ in bars_data]
    bar_counts = [c for _, c in bars_data]
    bar_colors = [highlight_color if c == max(bar_counts) else faded_color for c in bar_counts]
    bars = ax.barh(range(len(bar_labels)), bar_counts, color=bar_colors, edgecolor="white")
    ax.set_yticks(range(len(bar_labels)))
    ax.set_yticklabels(bar_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("# Runs", fontsize=10)
    ax.set_title("Ordering Frequency", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(bar_counts) + 1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, c in zip(bars, bar_counts):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{c}/{n_runs}", va="center", fontsize=10, fontweight="bold")


def main():
    cfg = get_config()
    timestamps, labels, acc = load_data(cfg)
    n_runs = len(timestamps)
    run_labels = [f"Run {i+1}" for i in range(n_runs)]
    dataset_name = cfg.base_dir.name

    has_api = len(cfg.api_labels) >= 2
    has_iface = len(cfg.iface_labels) >= 2
    n_groups = int(has_api) + int(has_iface)

    if n_groups == 0:
        print("Need at least 2 conditions in one group (API or Interface) to plot rankings.")
        return

    fig = plt.figure(figsize=(7 * n_groups, 8))
    gs = gridspec.GridSpec(2, n_groups, height_ratios=[3, 1.2], hspace=0.35, wspace=0.3)

    def rank_group(group_idx):
        valid_runs = [i for i in range(n_runs)
                      if not np.any(np.isnan(acc[i, group_idx]))]
        ranks = np.zeros((len(valid_runs), len(group_idx)))
        for ri, i in enumerate(valid_runs):
            ranks[ri] = rankdata(-acc[i, group_idx], method="average")
        return valid_runs, ranks

    col = 0
    if has_api:
        api_idx = [labels.index(l) for l in cfg.api_labels]
        api_valid, api_ranks = rank_group(api_idx)
        n_api_runs = len(api_valid)
        api_run_labels = [f"Run {i+1}" for i in api_valid]
        w_api, p_api = kendalls_w(api_ranks)
        p_str = f"p={p_api:.4f}" if p_api >= 0.0001 else "p<0.0001"
        w_str = f"{w_api:.2f}" if not np.isnan(w_api) else "N/A"

        ax_bump = fig.add_subplot(gs[0, col])
        acc_valid = acc[api_valid]
        make_bump_chart(ax_bump, labels, api_idx, api_ranks, acc_valid, cfg.colors,
                        n_api_runs, api_run_labels,
                        f"API Rankings\nKendall's W = {w_str}  ({p_str})")

        ax_freq = fig.add_subplot(gs[1, col])
        make_freq_chart(ax_freq, labels, api_idx, api_ranks, n_api_runs,
                        "#4A90D9", "#A0C4E8")
        col += 1

    if has_iface:
        iface_idx = [labels.index(l) for l in cfg.iface_labels]
        iface_valid, iface_ranks = rank_group(iface_idx)
        n_iface_runs = len(iface_valid)
        iface_run_labels = [f"Run {i+1}" for i in iface_valid]
        w_iface, p_iface = kendalls_w(iface_ranks)
        p_str = f"p={p_iface:.4f}" if p_iface >= 0.0001 else "p<0.0001"
        w_str = f"{w_iface:.2f}" if not np.isnan(w_iface) else "N/A"

        ax_bump = fig.add_subplot(gs[0, col])
        acc_valid = acc[iface_valid]
        make_bump_chart(ax_bump, labels, iface_idx, iface_ranks, acc_valid, cfg.colors,
                        n_iface_runs, iface_run_labels,
                        f"Interface Rankings\nKendall's W = {w_str}  ({p_str})")

        ax_freq = fig.add_subplot(gs[1, col])
        make_freq_chart(ax_freq, labels, iface_idx, iface_ranks, n_iface_runs,
                        "#E07B53", "#F0C4B0")

    fig.suptitle(f"{dataset_name} — Ranking Stability Across {n_runs} Repeated Sessions",
                 fontsize=14, y=1.01, fontweight="bold")

    out = cfg.base_dir / "plots" / "ranking_stability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
