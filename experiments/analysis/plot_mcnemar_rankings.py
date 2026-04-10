"""Statistically corrected model rankings via McNemar's test with best practices.

Best-practice pipeline (MCQ leaderboards):
  1. Cochran's Q test  — global gate: "is there ANY difference among models?"
     Skip pairwise tests if Q is not significant.
  2. Pairwise McNemar's tests  — for every pair of models.
     Uses exact binomial test when discordant pairs < 25 (Chi-square approx breaks down).
  3. Benjamini-Hochberg (FDR) correction  — applied to all pairwise p-values.

Outputs:
  * mcnemar_rankings.png  — heatmap of pairwise p-values + corrected ranking table
  * mcnemar_rankings.csv  — per-model summary (mean accuracy, corrected rank, cluster)

Usage:
    python plot_mcnemar_rankings.py --data-dir metabench-mmlu/data-claude
    python plot_mcnemar_rankings.py --data-dir metabench-arc/data --only high low
    python plot_mcnemar_rankings.py --data-dir metabench-mmlu/data-claude --alpha 0.01
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
from scipy.stats import chi2 as chi2_dist, binom
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
    single binary indicator. Only keep questions present in ALL models.
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
            mat[i, j] = float(np.mean(runs) >= 0.5)  # majority vote → binary

    return qids, mat


# ── Cochran's Q test ──────────────────────────────────────────────────────

def cochrans_q_test(mat: np.ndarray) -> tuple[float, float]:
    """Run Cochran's Q test on a (n_questions, n_models) binary matrix.

    Returns (Q_statistic, p_value).
    H0: all models have the same proportion of correct answers.
    """
    n, k = mat.shape
    row_totals = mat.sum(axis=1)        # L_i: correct answers per question
    col_totals = mat.sum(axis=0)        # G_j: correct answers per model
    total = mat.sum()

    # Q = (k-1) * [k * sum(G_j^2) - T^2] / [k * T - sum(L_i^2)]
    numerator = (k - 1) * (k * np.sum(col_totals ** 2) - total ** 2)
    denominator = k * total - np.sum(row_totals ** 2)

    if denominator == 0:
        return 0.0, 1.0

    Q = numerator / denominator
    p_val = float(1 - chi2_dist.cdf(Q, df=k - 1))
    return float(Q), p_val


# ── McNemar's test (standard + exact) ────────────────────────────────────

def mcnemar_test(
    col_a: np.ndarray,
    col_b: np.ndarray,
) -> tuple[float, int]:
    """Two-sided McNemar's test between two binary vectors.

    Applies exact binomial test when the number of discordant pairs < 25
    (Chi-square approximation is unreliable with small cell counts).

    Returns (p_value, n_discordant).
    """
    a = col_a.astype(bool)
    b = col_b.astype(bool)

    # b01: A wrong, B correct  |  b10: A correct, B wrong
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n_discordant = b01 + b10

    if n_discordant == 0:
        return 1.0, 0

    if n_discordant < 25:
        # Exact binomial test: under H0, b10 ~ Binomial(n_discordant, 0.5)
        # Two-sided p-value
        p_val = float(2 * min(
            binom.cdf(b10, n_discordant, 0.5),
            1 - binom.cdf(b10 - 1, n_discordant, 0.5),
        ))
        p_val = min(p_val, 1.0)
    else:
        # Standard McNemar Chi-square (with continuity correction)
        chi2 = (abs(b01 - b10) - 1) ** 2 / n_discordant
        p_val = float(1 - chi2_dist.cdf(chi2, df=1))

    return p_val, n_discordant


# ── Benjamini-Hochberg FDR correction ────────────────────────────────────

def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return list of booleans: True = reject H0 after BH-FDR correction."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * m
    # Find the largest k such that p_(k) <= k/m * alpha
    last_reject = -1
    for rank_k, (orig_idx, p) in enumerate(indexed, start=1):
        if p <= (rank_k / m) * alpha:
            last_reject = rank_k - 1  # 0-indexed position in sorted list
    for pos in range(last_reject + 1):
        orig_idx = indexed[pos][0]
        rejected[orig_idx] = True
    return rejected


# ── equivalence clustering ────────────────────────────────────────────────

def build_equivalence_clusters(
    labels: list[str],
    sig_matrix: np.ndarray,
    mean_acc: np.ndarray,
) -> list[tuple[int, list[str]]]:
    """Group models by transitivity of non-significance (union-find)."""
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


# ── visualisation ─────────────────────────────────────────────────────────

def plot_results(
    labels: list[str],
    mean_acc: np.ndarray,
    p_matrix: np.ndarray,
    sig_matrix: np.ndarray,
    disc_matrix: np.ndarray,
    clusters: list[tuple[int, list[str]]],
    alpha: float,
    n_questions: int,
    n_sessions: int,
    cochrans_q: float,
    cochrans_p: float,
    cfg,
    group_name: str,
) -> plt.Figure:
    n = len(labels)
    short = [cfg.short_label(l) for l in labels]
    order = sorted(range(n), key=lambda i: -mean_acc[i])

    fig = plt.figure(figsize=(max(8, 1.5 * n + 3), max(6, 1.2 * n + 3)))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.4)

    # ── left: p-value heatmap ──
    ax_hm = fig.add_subplot(gs[0, 0])
    display_p = p_matrix[np.ix_(order, order)]
    display_sig = sig_matrix[np.ix_(order, order)]
    display_disc = disc_matrix[np.ix_(order, order)]
    ordered_short = [short[i] for i in order]

    im = ax_hm.imshow(display_p, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax_hm.text(j, i, "—", ha="center", va="center",
                           fontsize=9, color="#666")
                continue
            p = display_p[i, j]
            d = display_disc[i, j]
            star = "*" if display_sig[i, j] else ""
            method = "exact" if d < 25 else "χ²"
            txt = f"p={p:.3f}{star}\n(n={int(d)}, {method})"
            color = "white" if p < 0.15 else "black"
            ax_hm.text(j, i, txt, ha="center", va="center",
                       fontsize=7, color=color, fontweight="bold" if star else "normal")

    ax_hm.set_xticks(range(n))
    ax_hm.set_xticklabels(ordered_short, fontsize=9, rotation=45, ha="right")
    ax_hm.set_yticks(range(n))
    ax_hm.set_yticklabels(ordered_short, fontsize=9)

    q_str = f"Q={cochrans_q:.2f}, p={cochrans_p:.4f}" if cochrans_p >= 0.0001 else f"Q={cochrans_q:.2f}, p<0.0001"
    ax_hm.set_title(
        f"Pairwise McNemar Tests\n(BH-FDR correction, α={alpha})\nCochran's Q: {q_str}",
        fontsize=11, fontweight="bold", pad=10,
    )

    cbar = fig.colorbar(im, ax=ax_hm, shrink=0.7, label="p-value")
    cbar.ax.tick_params(labelsize=8)

    # ── right: corrected ranking table ──
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")

    table_data = []
    cell_colors = []
    cluster_palette = plt.cm.Set3(np.linspace(0, 1, max(len(clusters), 1)))

    for rank, members in clusters:
        for label in members:
            idx = labels.index(label)
            sl = cfg.short_label(label)
            prefix = "API" if label.startswith("API") else "Iface"
            table_data.append([str(rank), f"{sl}  ({prefix})", f"{mean_acc[idx]:.1%}"])
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

    ax_tbl.set_title(
        "Corrected Rankings\n(BH-FDR clusters)",
        fontsize=11, fontweight="bold", pad=10,
    )

    dataset_name = cfg.base_dir.name
    n_pairs = n * (n - 1) // 2
    fig.suptitle(
        f"{dataset_name} — {group_name} McNemar Rankings\n"
        f"({n_questions} questions × {n_sessions} sessions, {n_pairs} pairwise tests)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    return fig


# ── main analysis ─────────────────────────────────────────────────────────

def run_analysis(
    cfg,
    labels: list[str],
    data: dict[str, dict[int, list[bool]]],
    alpha: float,
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

    # ── Step 1: Cochran's Q test ──
    Q_stat, Q_p = cochrans_q_test(mat)
    q_str = f"p={Q_p:.4f}" if Q_p >= 0.0001 else "p<0.0001"
    print(f"\n[{group_name}] Cochran's Q = {Q_stat:.3f}, {q_str}")

    if Q_p >= alpha:
        print(f"  Cochran's Q not significant (p={Q_p:.4f} >= α={alpha}). "
              "No pairwise tests performed.")

    # ── Step 2: Pairwise McNemar's tests ──
    p_matrix = np.ones((n_m, n_m))
    disc_matrix = np.zeros((n_m, n_m))
    p_list: list[float] = []
    pair_list: list[tuple[int, int]] = []

    for i, j in combinations(range(n_m), 2):
        p_val, n_disc = mcnemar_test(mat[:, i], mat[:, j])
        p_matrix[i, j] = p_matrix[j, i] = p_val
        disc_matrix[i, j] = disc_matrix[j, i] = n_disc
        p_list.append(p_val)
        pair_list.append((i, j))
        method = "exact" if n_disc < 25 else "chi2"
        print(f"  {cfg.short_label(labels[i])} vs {cfg.short_label(labels[j])}: "
              f"p={p_val:.4f}, discordant={n_disc} ({method})")

    # ── Step 3: Benjamini-Hochberg FDR correction ──
    rejected = benjamini_hochberg(p_list, alpha)
    sig_matrix = np.zeros((n_m, n_m), dtype=bool)
    for (i, j), rej in zip(pair_list, rejected):
        sig_matrix[i, j] = sig_matrix[j, i] = rej

    clusters = build_equivalence_clusters(labels, sig_matrix, mean_acc)

    fig = plot_results(
        labels, mean_acc, p_matrix, sig_matrix, disc_matrix,
        clusters, alpha, n_q, n_sessions, Q_stat, Q_p, cfg, group_name,
    )

    label_to_rank = {m: rank for rank, members in clusters for m in members}

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
            "cochrans_q": round(Q_stat, 4),
            "cochrans_q_p": round(Q_p, 6),
        })

    return fig, rows


def main():
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level (default 0.05)")
    args, _ = p.parse_known_args()

    cfg = get_config()
    data = load_question_correctness(cfg)
    dataset_name = cfg.base_dir.name
    out_dir = cfg.base_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for group_name, group_labels in [("API", cfg.api_labels), ("Interface", cfg.iface_labels)]:
        if len(group_labels) < 2:
            continue
        result = run_analysis(cfg, group_labels, data, args.alpha, group_name)
        if result is None:
            continue
        fig, rows = result
        tag = group_name.lower()
        fig_path = out_dir / f"mcnemar_rankings_{tag}.png"
        fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path}")
        all_rows.extend(rows)

    combined_labels = cfg.labels
    if len(combined_labels) >= 2:
        result = run_analysis(cfg, combined_labels, data, args.alpha, "All Models")
        if result is not None:
            fig, rows = result
            fig_path = out_dir / "mcnemar_rankings_all.png"
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fig_path}")
            all_rows = rows

    if all_rows:
        csv_path = out_dir / "mcnemar_rankings.csv"
        fields = ["condition", "mean_accuracy", "raw_rank", "corrected_rank",
                  "cluster_members", "n_questions", "cochrans_q", "cochrans_q_p"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(all_rows, key=lambda r: r["corrected_rank"]))
        print(f"Saved {csv_path}")

    print(f"\n{'='*70}")
    print(f"  {dataset_name} — McNemar Rankings (α={args.alpha})")
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
