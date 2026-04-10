"""Compute Fleiss' kappa and Gwet's AC1 for answer consistency and correctness
consistency across repeated sessions for each model/condition.

Uses the generalized Fleiss' kappa that handles variable raters per subject.
Gwet's AC1 is included because Fleiss' kappa suffers from the "kappa paradox"
when one category dominates (e.g. near-ceiling accuracy makes FK(correct) ≈ 0
even with near-perfect agreement).

Answer categories span A–Z so this works for any benchmark (MMLU, TruthfulQA, etc.).

Usage:
    python fleiss_kappa.py --data-dir metabench-mmlu/data-combined
    python fleiss_kappa.py --data-dir metabench-truthfulQA/data
    python fleiss_kappa.py --data-dir metabench-arc/data --only high low
"""
from __future__ import annotations

import csv
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import get_config

ANSWER_CATEGORIES = list(string.ascii_uppercase)  # A–Z
CORRECT_CATEGORIES = ["True", "False"]


def fleiss_kappa(table: np.ndarray) -> float:
    """Generalized Fleiss' kappa allowing variable n_i raters per subject.

    table[i, j] = number of raters who assigned category j to subject i.
    Subjects with fewer than 2 raters are dropped.
    """
    n_i = table.sum(axis=1)
    mask = n_i >= 2
    table = table[mask]
    n_i = n_i[mask]
    N, k = table.shape
    if N == 0:
        return float("nan")

    P_i = (np.sum(table ** 2, axis=1) - n_i) / (n_i * (n_i - 1))
    P_bar = np.mean(P_i)

    total = n_i.sum()
    p_j = np.sum(table, axis=0) / total
    P_e = np.sum(p_j ** 2)

    if abs(1 - P_e) < 1e-10:
        return 1.0 if abs(P_bar - 1.0) < 1e-10 else float("nan")

    return float((P_bar - P_e) / (1 - P_e))


def gwet_ac1(table: np.ndarray) -> float:
    """Gwet's AC1 for multiple raters with variable n_i raters per subject.

    Uses a different chance-agreement model than Fleiss' kappa that remains
    well-behaved when category prevalence is highly skewed.

    table[i, j] = number of raters who assigned category j to subject i.
    Subjects with fewer than 2 raters are dropped.
    """
    n_i = table.sum(axis=1)
    mask = n_i >= 2
    table = table[mask]
    n_i = n_i[mask]
    N, q = table.shape
    if N == 0:
        return float("nan")

    P_i = (np.sum(table ** 2, axis=1) - n_i) / (n_i * (n_i - 1))
    P_a = np.mean(P_i)

    if q <= 1:
        return 1.0 if abs(P_a - 1.0) < 1e-10 else float("nan")

    total = n_i.sum()
    p_j = np.sum(table, axis=0) / total
    P_e = np.sum(p_j * (1 - p_j)) / (q - 1)

    if abs(1 - P_e) < 1e-10:
        return 1.0 if abs(P_a - 1.0) < 1e-10 else float("nan")

    return float((P_a - P_e) / (1 - P_e))


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_condition_data(
    cfg, source: str, filename: str, timestamps: list[str]
) -> tuple[dict[int, list[str]], dict[int, list[str]], int]:
    qid_to_runs_answer: dict[int, list[str]] = defaultdict(list)
    qid_to_runs_correct: dict[int, list[str]] = defaultdict(list)
    n_raters = 0

    for ts in timestamps:
        path = cfg.csv_path(source, filename, ts)
        if not path.exists():
            continue
        rows = load_csv_rows(path)
        n_raters += 1
        for row in rows:
            try:
                qid = int(row.get("id", ""))
            except (ValueError, TypeError):
                continue
            if not (row.get("gold_answer") or "").strip():
                continue
            answer = (row.get("answer") or "").strip().upper()
            gold = (row.get("gold_answer") or "").strip().upper()
            correct = str(answer and gold and answer == gold)
            qid_to_runs_answer[qid].append(answer if answer else "_NONE")
            qid_to_runs_correct[qid].append(correct if correct in ("True", "False") else "False")

    return qid_to_runs_answer, qid_to_runs_correct, n_raters


def compute_fleiss(
    qid_to_runs_answer: dict[int, list[str]],
    qid_to_runs_correct: dict[int, list[str]],
    qids: list[int],
) -> dict:
    if not qids:
        return {"fk_answer": float("nan"), "fk_correct": float("nan"),
                "ac1_answer": float("nan"), "ac1_correct": float("nan"),
                "n_questions": 0}

    ans_cat_idx = {c: i for i, c in enumerate(ANSWER_CATEGORIES)}
    cor_cat_idx = {c: i for i, c in enumerate(CORRECT_CATEGORIES)}

    ans_table = np.zeros((len(qids), len(ANSWER_CATEGORIES)), dtype=float)
    cor_table = np.zeros((len(qids), len(CORRECT_CATEGORIES)), dtype=float)

    for i, qid in enumerate(qids):
        for ans in qid_to_runs_answer[qid]:
            if ans in ans_cat_idx:
                ans_table[i, ans_cat_idx[ans]] += 1
        for cor in qid_to_runs_correct[qid]:
            if cor in cor_cat_idx:
                cor_table[i, cor_cat_idx[cor]] += 1

    return {
        "fk_answer": fleiss_kappa(ans_table),
        "fk_correct": fleiss_kappa(cor_table),
        "ac1_answer": gwet_ac1(ans_table),
        "ac1_correct": gwet_ac1(cor_table),
        "n_questions": len(qids),
    }


def main():
    cfg = get_config()
    timestamps = cfg.get_timestamps()
    print(f"Dataset: {cfg.base_dir.name}")
    print(f"Timestamps (runs): {len(timestamps)}\n")

    condition_data = {}
    for label, (source, filename) in cfg.models.items():
        ans, cor, n = load_condition_data(cfg, source, filename, timestamps)
        condition_data[label] = (ans, cor, n)

    active = {l: d for l, d in condition_data.items() if d[2] > 0}
    if active:
        all_qids = sorted(set.union(*(set(d[0].keys()) for d in active.values())))
    else:
        all_qids = []

    # ------------------------------------------------------------------
    # Global agreement / flip stats per condition (for summary table)
    # ------------------------------------------------------------------
    noise_stats: dict[str, dict[str, float]] = {}
    for label, (ans_map, cor_map, n) in condition_data.items():
        if n < 2 or not all_qids:
            noise_stats[label] = {"total_q": 0.0, "perfect": 0.0, "flips": 0.0}
            continue
        total_q = 0
        perfect = 0
        flips = 0
        for qid in all_qids:
            answers = ans_map.get(qid, [])
            corrects = cor_map.get(qid, [])
            if len(answers) < 2:
                continue
            ans_counts = Counter(answers)
            cor_counts = Counter(corrects)
            _, majority_ans_n = ans_counts.most_common(1)[0]
            ans_agree = majority_ans_n / len(answers)
            n_correct = cor_counts.get("True", 0)
            n_incorrect = cor_counts.get("False", 0)
            has_flips = n_correct > 0 and n_incorrect > 0

            total_q += 1
            if ans_agree == 1.0:
                perfect += 1
            if has_flips:
                flips += 1

        noise_stats[label] = {
            "total_q": float(total_q),
            "perfect": float(perfect),
            "flips": float(flips),
        }

    results = []
    header = (
        f"{'Condition':<30s}  {'FK(answer)':>10s}  {'FK(correct)':>11s}"
        f"  {'AC1(answer)':>11s}  {'AC1(correct)':>12s}  {'#Q':>4s}  {'#R':>3s}"
        f"  {'Perfect%':>9s}  {'Flip%':>7s}"
    )
    print(header)
    print("-" * len(header))
    for label in cfg.models:
        ans, cor, n = condition_data[label]
        qids = all_qids if n > 0 else []
        s = compute_fleiss(ans, cor, qids)
        results.append({"condition": label, **s, "n_raters": n})
        stats = noise_stats.get(label, {"total_q": 0.0, "perfect": 0.0, "flips": 0.0})
        if stats["total_q"] > 0:
            perfect_pct = stats["perfect"] / stats["total_q"]
            flips_pct = stats["flips"] / stats["total_q"]
        else:
            perfect_pct = 0.0
            flips_pct = 0.0
        print(
            f"{label:<30s}  {s['fk_answer']:>10.4f}  {s['fk_correct']:>11.4f}"
            f"  {s['ac1_answer']:>11.4f}  {s['ac1_correct']:>12.4f}"
            f"  {s['n_questions']:>4d}  {n:>3d}"
            f"  {perfect_pct:>8.1%}  {flips_pct:>6.1%}"
        )

    out_dir = cfg.base_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fleiss_kappa.csv"
    fieldnames = ["condition", "fk_answer", "fk_correct", "ac1_answer", "ac1_correct", "n_questions", "n_raters"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {out_path}")

    # ------------------------------------------------------------------
    # Per-item agreement analysis: locate where the noise comes from
    # ------------------------------------------------------------------
    item_rows: list[dict] = []
    for label in cfg.models:
        ans_map, cor_map, n = condition_data[label]
        if n < 2:
            continue
        for qid in all_qids:
            answers = ans_map.get(qid, [])
            corrects = cor_map.get(qid, [])
            if len(answers) < 2:
                continue
            ans_counts = Counter(answers)
            cor_counts = Counter(corrects)
            majority_ans, majority_ans_n = ans_counts.most_common(1)[0]
            ans_agree = majority_ans_n / len(answers)
            n_distinct = len(ans_counts)
            n_correct = cor_counts.get("True", 0)
            n_incorrect = cor_counts.get("False", 0)
            flips = n_correct > 0 and n_incorrect > 0

            item_rows.append({
                "condition": label,
                "question_id": qid,
                "n_runs": len(answers),
                "n_distinct_answers": n_distinct,
                "majority_answer": majority_ans,
                "majority_count": majority_ans_n,
                "answer_agreement": round(ans_agree, 4),
                "n_correct": n_correct,
                "n_incorrect": n_incorrect,
                "correctness_flips": flips,
            })

    item_path = out_dir / "item_agreement.csv"
    item_fields = [
        "condition", "question_id", "n_runs", "n_distinct_answers",
        "majority_answer", "majority_count", "answer_agreement",
        "n_correct", "n_incorrect", "correctness_flips",
    ]
    with item_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=item_fields)
        writer.writeheader()
        writer.writerows(item_rows)
    print(f"Saved per-item agreement to {item_path}")

    # Print noise summary per condition
    print(f"\n{'='*70}")
    print("  Noise source analysis (per condition)")
    print(f"{'='*70}")
    for label in cfg.models:
        rows_for = [r for r in item_rows if r["condition"] == label]
        if not rows_for:
            continue
        total_q = len(rows_for)
        perfect = sum(1 for r in rows_for if r["answer_agreement"] == 1.0)
        noisy = [r for r in rows_for if r["answer_agreement"] < 1.0]
        flippers = [r for r in rows_for if r["correctness_flips"]]

        print(f"\n  {label}")
        print(f"    Total questions:        {total_q}")
        print(f"    Perfect agreement:      {perfect} ({perfect/total_q:.1%})")
        print(f"    Any answer disagreement: {len(noisy)} ({len(noisy)/total_q:.1%})")
        print(f"    Correctness flips:      {len(flippers)} ({len(flippers)/total_q:.1%})")

        if noisy:
            agreements = [r["answer_agreement"] for r in noisy]
            print(f"    Noisy items — agreement range: [{min(agreements):.2f}, {max(agreements):.2f}]")
            worst = sorted(noisy, key=lambda r: r["answer_agreement"])[:5]
            print(f"    Worst 5 items:")
            for r in worst:
                print(
                    f"      Q{r['question_id']:>4d}: {r['n_distinct_answers']} answers, "
                    f"agree={r['answer_agreement']:.0%}, "
                    f"correct={r['n_correct']}/{r['n_runs']}, "
                    f"{'FLIPS' if r['correctness_flips'] else 'stable'}"
                )


if __name__ == "__main__":
    main()
