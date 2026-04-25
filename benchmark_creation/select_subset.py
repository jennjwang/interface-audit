"""
EssenceBench Step 1: select a discriminative subset of benchmark questions.

Reads one scored CSV per model from --scored-dir (each CSV must have 'id' and
'correct' columns), filters questions by rpbis / std / accuracy / rank-similarity,
and writes the surviving questions to --output.

Also writes a <output>.calibration.json sidecar with linear coefficients to
reconstruct full-benchmark scores from subset scores:
    full_score ≈ scale * subset_score + offset

Usage:
    python select_subset.py \
        --scored-dir experiments/aa-omniscience/outputs/api \
        --queries    experiments/aa-omniscience/queries.csv \
        --output     benchmark_creation/EssenceBench/subset.csv
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm


# ── inlined from EssenceBench src/preprocess/discard_items.py ─────────────────
def filter_questions(models, question_matrix, args):
    total_scores = np.array([m["total_score"] for m in models])
    valid, removed = [], {"high_acc": [], "low_std": [], "low_rpbis": []}
    for q_idx in tqdm(range(question_matrix.shape[0]), desc="Filtering items"):
        q = question_matrix[q_idx]
        acc, std = np.mean(q), np.std(q)
        if acc > args.max_acc:
            removed["high_acc"].append((q_idx, acc)); continue
        if std < args.min_std:
            removed["low_std"].append((q_idx, std)); continue
        r = pearsonr(q, total_scores)[0]
        if abs(r) < args.min_rpbis:
            removed["low_rpbis"].append((q_idx, r)); continue
        valid.append(q_idx)
    return valid, removed


# ── inlined from EssenceBench src/preprocess/similarity.py ───────────────────
def calculate_rank_similarity(score_matrix, threshold=0.9):
    similarities = np.corrcoef(score_matrix)
    remove = set()
    for i in range(len(similarities)):
        for j in range(i + 1, len(similarities)):
            if similarities[i, j] > threshold:
                remove.add(i)
    return list(remove)


def load_scored_csvs(scored_dir: Path):
    """
    Discover all CSVs in scored_dir, align on question id, drop questions
    with any missing label. Returns (model_names, question_ids, matrix).
    matrix shape: (n_questions, n_models), dtype int8.
    """
    csv_files = sorted(scored_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {scored_dir}")

    frames = {}
    for f in csv_files:
        df = pd.read_csv(f)
        if "id" not in df.columns or "correct" not in df.columns:
            print(f"  Skipping {f.name}: missing 'id' or 'correct' column")
            continue
        frames[f.stem] = df.set_index("id")["correct"]

    if not frames:
        raise ValueError("No valid scored CSVs found.")

    all_ids = sorted(set.intersection(*[set(s.index) for s in frames.values()]))
    nan_ids = set()
    for series in frames.values():
        nan_ids.update(series.reindex(all_ids)[series.reindex(all_ids).isna()].index.tolist())
    if nan_ids:
        print(f"Dropping {len(nan_ids)} questions with missing labels across any model")
    question_ids = [qid for qid in all_ids if qid not in nan_ids]

    model_names = list(frames.keys())
    matrix = np.array(
        [[int(frames[m].loc[qid]) for qid in question_ids] for m in model_names],
        dtype=np.int8,
    ).T  # (n_questions, n_models)

    return model_names, question_ids, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored-dir", required=True,
                        help="Directory of scored CSVs (one per model, with 'id' and 'correct' columns)")
    parser.add_argument("--queries", required=True,
                        help="CSV with question metadata (must have 'id' column)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path for the selected subset")
    parser.add_argument("--min-rpbis", type=float, default=0.05)
    parser.add_argument("--min-std",   type=float, default=0.001)
    parser.add_argument("--max-acc",   type=float, default=0.95)
    # Default 1.0 effectively disables rank-sim (no correlation exceeds 1.0 strictly).
    # Lower values (e.g. 0.9) aggressively prune redundant questions — good for ranking
    # studies but hurts score reconstruction by discarding useful calibration signal.
    parser.add_argument("--rank-sim-threshold", type=float, default=1.0)
    args = parser.parse_args()

    scored_dir = Path(args.scored_dir)
    queries_df = pd.read_csv(args.queries)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading scored CSVs from {scored_dir} ...")
    model_names, question_ids, matrix = load_scored_csvs(scored_dir)
    n_total = len(question_ids)

    models = [
        {"model_name": name, "answers": matrix[:, i].tolist(), "total_score": int(matrix[:, i].sum())}
        for i, name in enumerate(model_names)
    ]

    print(f"\n{n_total} questions x {len(model_names)} models")
    print("Model accuracies:")
    for m in models:
        print(f"  {m['model_name']}: {m['total_score']/n_total:.3f}")

    filter_args = SimpleNamespace(
        min_rpbis=args.min_rpbis, min_std=args.min_std,
        max_acc=args.max_acc, show_removed=100,
    )
    print(f"\nFiltering: |rpbis| >= {args.min_rpbis}, std >= {args.min_std}, acc <= {args.max_acc}")
    valid_indices, removed = filter_questions(models, matrix, filter_args)
    for reason, items in removed.items():
        if items:
            print(f"  Removed {len(items)} ({reason})")
    n_post_filter = len(valid_indices)
    print(f"After rpbis/std/acc filter: {n_post_filter} questions remain")

    if args.rank_sim_threshold < 1.0:
        removed_rank = calculate_rank_similarity(matrix, threshold=args.rank_sim_threshold)
        valid_indices = [i for i in valid_indices if i not in removed_rank]
        n_rank_removed = n_post_filter - len(valid_indices)
        print(f"After rank-similarity filter (threshold={args.rank_sim_threshold}): {len(valid_indices)} questions remain")
    else:
        n_rank_removed = 0
        print(f"Rank-similarity filter disabled (threshold={args.rank_sim_threshold})")

    total_scores = np.array([m["total_score"] for m in models])
    rpbis_map = {}
    for i, qid in enumerate(question_ids):
        r, _ = pearsonr(matrix[i].astype(float), total_scores.astype(float))
        rpbis_map[qid] = round(float(r), 4) if not np.isnan(r) else 0.0

    selected_ids = [question_ids[i] for i in valid_indices]
    selected_df = queries_df[queries_df["id"].isin(selected_ids)].copy()
    selected_df["rpbis"] = selected_df["id"].map(rpbis_map)
    selected_df = selected_df.sort_values("rpbis", ascending=False)

    selected_df.to_csv(output_path, index=False)
    print(f"\nSelected {len(selected_df)} questions -> {output_path}")

    if {"domain", "topic"}.issubset(selected_df.columns):
        print("\nTop 10 by rpbis:")
        print(selected_df[["id", "domain", "topic", "rpbis"]].head(10).to_string(index=False))

    # ── score reconstruction calibration ──────────────────────────────────────
    n_selected = len(valid_indices)
    valid_set = set(valid_indices)
    full_scores   = matrix.sum(axis=0).astype(float)   # (n_models,)
    subset_scores = matrix[np.array(valid_indices)].sum(axis=0).astype(float) if valid_indices else np.zeros(len(model_names))

    # OLS: full_score = scale * subset_score + offset
    if n_selected > 0 and len(model_names) >= 2:
        x = subset_scores
        coeffs = np.polyfit(x, full_scores, 1)
        a_scale, b_offset = float(coeffs[0]), float(coeffs[1])
        y_pred = a_scale * x + b_offset
        ss_res = float(np.sum((full_scores - y_pred) ** 2))
        ss_tot = float(np.sum((full_scores - full_scores.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    else:
        a_scale, b_offset, r2 = float(n_total) / max(n_selected, 1), 0.0, float("nan")

    print(f"\nScore reconstruction (fit on {len(model_names)} training models):")
    print(f"  full_score ≈ {a_scale:.4f} × subset_score + {b_offset:.4f}  (R²={r2:.4f})")
    print(f"\n  {'Model':<50} {'Full':>6} {'Subset':>8} {'Predicted':>10} {'Error':>7}")
    for i, name in enumerate(model_names):
        pred = a_scale * subset_scores[i] + b_offset
        err  = pred - full_scores[i]
        print(f"  {name:<50} {int(full_scores[i]):>6} {int(subset_scores[i]):>8} {pred:>10.1f} {err:>+7.1f}")

    calibration = {
        "n_total": n_total,
        "n_selected": n_selected,
        "filters": {
            "min_rpbis": args.min_rpbis,
            "min_std":   args.min_std,
            "max_acc":   args.max_acc,
            "rank_sim_threshold": args.rank_sim_threshold,
        },
        "removed": {
            "high_acc":  len(removed["high_acc"]),
            "low_std":   len(removed["low_std"]),
            "low_rpbis": len(removed["low_rpbis"]),
            "rank_sim":  n_rank_removed,
        },
        "score_reconstruction": {
            "formula": "full_score ≈ scale * subset_score + offset",
            "scale":   round(a_scale, 6),
            "offset":  round(b_offset, 6),
            "r2":      round(r2, 6),
        },
        "training_models": [
            {
                "model":       model_names[i],
                "full_score":  int(full_scores[i]),
                "full_acc":    round(float(full_scores[i]) / n_total, 4),
                "subset_score": int(subset_scores[i]),
                "subset_acc":  round(float(subset_scores[i]) / n_selected, 4) if n_selected > 0 else 0.0,
                "predicted_full": round(a_scale * subset_scores[i] + b_offset, 2),
            }
            for i in range(len(model_names))
        ],
    }
    cal_path = output_path.with_suffix(".calibration.json")
    cal_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(f"\nCalibration saved -> {cal_path}")


if __name__ == "__main__":
    main()
