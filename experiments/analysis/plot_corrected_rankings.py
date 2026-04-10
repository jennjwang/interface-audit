"""Statistically corrected model rankings via pairwise bootstrap tests
with Holm-Bonferroni correction.

For each pair of models, a paired bootstrap test on per-question correctness
determines whether their accuracy difference is significant.  Models that are
NOT significantly different are grouped into equivalence clusters and share the
same corrected rank.

Outputs:
  * corrected_rankings.png  — heatmap of pairwise p-values + corrected ranking table
  * corrected_rankings.csv  — per-model summary (mean accuracy, corrected rank, cluster)

Usage:
    python plot_corrected_rankings.py --data-dir metabench-mmlu/data-claude
    python plot_corrected_rankings.py --data-dir metabench-arc/data --only high low
    python plot_corrected_rankings.py --data-dir metabench-mmlu/data-claude --alpha 0.01
"""
from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import rankdata

from config import get_config


# ── data loading ──────────────────────────────────────────────────────────

def load_question_correctness(cfg) -> dict[str, dict[int, list[bool]]]:
    """Return {label: {qid: [correct_bool, ...]}} across all sessions."""
    timestamps = cfg.get_timestamps()
    result: dict[str, dict[int, list[bool]]] = {}

    for label, (source, filename) in cfg.models.items():
        qid_runs: dict[int, list[bool]] = defaultdict(list)
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
                    if not (row.get("gold_answer") or "").strip():
                        continue
                    qid_runs[qid].append(str(row.get("correct")) == "True")
        result[label] = dict(qid_runs)

    return result


def build_paired_arrays(
    data: dict[str, dict[int, list[bool]]],
    labels: list[str],
) -> tuple[list[int], np.ndarray]:
    """Build (n_questions, n_models) matrix of majority-vote correctness.

    For each model-question pair, use the majority vote across sessions to get a
    single binary indicator.  Only keep questions present in ALL models.
    """
    per_model_qids = [set(data[l].keys()) for l in labels]
    non_empty = [s for s in per_model_qids if s]
    if len(non_empty) < 2:
        return [], np.zeros((0, len(labels)))
    all_qids = set.intersection(*non_empty)
    qids = sorted(all_qids)

    n_q = len(qids)
    n_m = len(labels)
    mat = np.zeros((n_q, n_m), dtype=float)

    for j, label in enumerate(labels):
        for i, qid in enumerate(qids):
            runs = data[label].get(qid, [False])
            mat[i, j] = np.mean(runs)

    return qids, mat


# ── bootstrap pairwise test ──────────────────────────────────────────────

def paired_bootstrap_p(
    col_a: np.ndarray,
    col_b: np.ndarray,
    n_bootstrap: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Two-sided paired bootstrap test on the mean difference.

    Returns (p_value, observed_diff, bootstrap_se).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(col_a)
    diff = col_a - col_b
    obs_diff = diff.mean()

    boot_diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[b] = diff[idx].mean()

    centered = boot_diffs - obs_diff
    p_val = np.mean(np.abs(centered) >= np.abs(obs_diff))
    se = boot_diffs.std()
    return float(p_val), float(obs_diff), float(se)


# ── multiple comparison correction ───────────────────────────────────────

def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return list of booleans: True = reject H0 (significantly different)."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * m
    for rank_k, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (m - rank_k)
        if p <= threshold:
            rejected[orig_idx] = True
        else:
            break
    return rejected


# ── equivalence clustering ───────────────────────────────────────────────

def build_equivalence_clusters(
    labels: list[str],
    sig_matrix: np.ndarray,
    mean_acc: np.ndarray,
) -> list[tuple[int, list[str]]]:
    """Group models by transitivity of non-significance.

    Two models are in the same cluster if there exists a chain of
    non-significant pairwise comparisons connecting them.  Clusters are
    ranked by descending mean accuracy of their best member.
    """
    n = len(labels)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if not sig_matrix[i, j]:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    ranked = sorted(
        clusters.values(),
        key=lambda grp: max(mean_acc[i] for i in grp),
        reverse=True,
    )

    result = []
    for rank, grp in enumerate(ranked, start=1):
        members = sorted(grp, key=lambda i: -mean_acc[i])
        result.append((rank, [labels[i] for i in members]))

    return result


# ── visualisation ────────────────────────────────────────────────────────

def plot_results(
    labels: list[str],
    mean_acc: np.ndarray,
    p_matrix: np.ndarray,
    sig_matrix: np.ndarray,
    diff_matrix: np.ndarray,
    clusters: list[tuple[int, list[str]]],
    alpha: float,
    n_questions: int,
    n_sessions: int,
    cfg,
    group_name: str,
):
    n = len(labels)
    short = [cfg.short_label(l) for l in labels]
    order = sorted(range(n), key=lambda i: -mean_acc[i])

    fig = plt.figure(figsize=(max(8, 1.5 * n + 3), max(6, 1.2 * n + 3)))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.4)

    # ── left: p-value heatmap ──
    ax_hm = fig.add_subplot(gs[0, 0])
    display_p = p_matrix[np.ix_(order, order)]
    display_sig = sig_matrix[np.ix_(order, order)]
    display_diff = diff_matrix[np.ix_(order, order)]
    ordered_short = [short[i] for i in order]

    im = ax_hm.imshow(display_p, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax_hm.text(j, i, "—", ha="center", va="center",
                           fontsize=9, color="#666")
                continue
            p = display_p[i, j]
            d = display_diff[i, j]
            star = "*" if display_sig[i, j] else ""
            txt = f"{d:+.1%}\np={p:.3f}{star}"
            color = "white" if p < 0.15 else "black"
            ax_hm.text(j, i, txt, ha="center", va="center",
                       fontsize=7, color=color, fontweight="bold" if star else "normal")

    ax_hm.set_xticks(range(n))
    ax_hm.set_xticklabels(ordered_short, fontsize=9, rotation=45, ha="right")
    ax_hm.set_yticks(range(n))
    ax_hm.set_yticklabels(ordered_short, fontsize=9)
    ax_hm.set_title(f"Pairwise Accuracy Differences\n(paired bootstrap, α={alpha})",
                     fontsize=11, fontweight="bold", pad=10)

    cbar = fig.colorbar(im, ax=ax_hm, shrink=0.7, label="p-value")
    cbar.ax.tick_params(labelsize=8)

    # ── right: corrected ranking table ──
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")

    table_data = []
    cell_colors = []
    cluster_palette = plt.cm.Set3(np.linspace(0, 1, max(len(clusters), 1)))

    for rank, members in clusters:
        for k, label in enumerate(members):
            idx = labels.index(label)
            sl = cfg.short_label(label)
            prefix = "API" if label.startswith("API") else "Iface"
            table_data.append([
                str(rank),
                f"{sl}  ({prefix})",
                f"{mean_acc[idx]:.1%}",
            ])
            c = cluster_palette[(rank - 1) % len(cluster_palette)]
            cell_colors.append([(c[0], c[1], c[2], 0.35)] * 3)

    if table_data:
        tbl = ax_tbl.table(
            cellText=table_data,
            colLabels=["Rank", "Model", "Mean Acc"],
            cellColours=cell_colors,
            colColours=["#ddd"] * 3,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.5)

    ax_tbl.set_title("Corrected Rankings\n(Holm-Bonferroni clusters)",
                      fontsize=11, fontweight="bold", pad=10)

    dataset_name = cfg.base_dir.name
    fig.suptitle(
        f"{dataset_name} — {group_name} Statistically Corrected Rankings\n"
        f"({n_questions} questions × {n_sessions} sessions, {n*(n-1)//2} pairwise tests)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    return fig


# ── main ─────────────────────────────────────────────────────────────────

def run_analysis(
    cfg,
    labels: list[str],
    data: dict[str, dict[int, list[bool]]],
    alpha: float,
    n_bootstrap: int,
    group_name: str,
) -> tuple[plt.Figure, list[dict]] | None:
    labels = [l for l in labels if data.get(l)]
    if len(labels) < 2:
        return None

    qids, mat = build_paired_arrays(data, labels)
    n_q, n_m = mat.shape
    if n_q == 0:
        print(f"  Skipping {group_name}: no overlapping questions across all models.")
        return None
    n_sessions = len(cfg.get_timestamps())
    mean_acc = mat.mean(axis=0)

    rng = np.random.default_rng(42)
    p_matrix = np.ones((n_m, n_m))
    diff_matrix = np.zeros((n_m, n_m))
    p_list = []
    pair_list = []

    for i, j in combinations(range(n_m), 2):
        p_val, obs_diff, _ = paired_bootstrap_p(mat[:, i], mat[:, j], n_bootstrap, rng)
        p_matrix[i, j] = p_matrix[j, i] = p_val
        diff_matrix[i, j] = obs_diff
        diff_matrix[j, i] = -obs_diff
        p_list.append(p_val)
        pair_list.append((i, j))

    rejected = holm_bonferroni(p_list, alpha)
    sig_matrix = np.zeros((n_m, n_m), dtype=bool)
    for (i, j), rej in zip(pair_list, rejected):
        sig_matrix[i, j] = sig_matrix[j, i] = rej

    clusters = build_equivalence_clusters(labels, sig_matrix, mean_acc)

    fig = plot_results(
        labels, mean_acc, p_matrix, sig_matrix, diff_matrix,
        clusters, alpha, n_q, n_sessions, cfg, group_name,
    )

    label_to_rank = {}
    for rank, members in clusters:
        for m in members:
            label_to_rank[m] = rank

    rows = []
    for idx, label in enumerate(labels):
        rows.append({
            "condition": label,
            "mean_accuracy": round(float(mean_acc[idx]), 6),
            "raw_rank": int(rankdata(-mean_acc, method="min")[idx]),
            "corrected_rank": label_to_rank[label],
            "cluster_members": "; ".join(
                m for r, ms in clusters for m in ms if r == label_to_rank[label]
            ),
            "n_questions": n_q,
        })

    return fig, rows


def main():
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level (default 0.05)")
    p.add_argument("--n-bootstrap", type=int, default=10_000,
                   help="Number of bootstrap samples (default 10000)")
    args, _ = p.parse_known_args()

    cfg = get_config()
    data = load_question_correctness(cfg)
    dataset_name = cfg.base_dir.name
    out_dir = cfg.base_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for group_name, group_labels in [("API", cfg.api_labels), ("Interface", cfg.iface_labels)]:
        if len(group_labels) < 2:
            continue

        result = run_analysis(cfg, group_labels, data, args.alpha, args.n_bootstrap, group_name)
        if result is None:
            continue

        fig, rows = result
        tag = group_name.lower()
        fig_path = out_dir / f"corrected_rankings_{tag}.png"
        fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path}")
        all_rows.extend(rows)

    combined_labels = cfg.labels
    if len(combined_labels) >= 2:
        result = run_analysis(cfg, combined_labels, data, args.alpha, args.n_bootstrap, "All Models")
        if result is not None:
            fig, rows = result
            fig_path = out_dir / "corrected_rankings_all.png"
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fig_path}")

            all_rows = rows

    if all_rows:
        csv_path = out_dir / "corrected_rankings.csv"
        fields = ["condition", "mean_accuracy", "raw_rank", "corrected_rank",
                   "cluster_members", "n_questions"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(all_rows, key=lambda r: r["corrected_rank"]))
        print(f"Saved {csv_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  {dataset_name} — Statistically Corrected Rankings (α={args.alpha})")
    print(f"{'='*70}")
    for row in sorted(all_rows, key=lambda r: (r["corrected_rank"], -r["mean_accuracy"])):
        print(
            f"  Rank {row['corrected_rank']:>2d}  "
            f"{row['condition']:<35s}  "
            f"{row['mean_accuracy']:.1%}  "
            f"(raw #{row['raw_rank']})"
        )


if __name__ == "__main__":
    main()
