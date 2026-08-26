"""
Layer 2 phase 1 analysis — account-level effects on BBQ.

Loads parsed_json from a layer2 run, scores against bbq-subset-200, and
reports:
  - per-account accuracy (mean) with 95% Wilson CI
  - per-query agreement across the 3 accounts
  - chi-square test for account vs correct (account effect on accuracy)
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from scipy.stats import chi2_contingency, norm


SCRIPT_DIR = Path(__file__).resolve().parents[1]
BBQ_PATH = SCRIPT_DIR.parent / "benchmark_creation" / "results" / "bbq-subset-200.csv"

SESSION_TO_ACCOUNT = {
    "session_00": "acct1",
    "session_01": "acct2",
    "session_02": "acct3",
}


def load_ground_truth():
    truth = {}
    with open(BBQ_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth[str(row["id"])] = row["answer"].strip().upper()
    return truth


# Heuristic letter extractor: matches "answer is X", "the answer: X", "is X.",
# or a standalone first-line letter A/B/C/D. Falls back to last A/B/C/D mentioned.
ANSWER_PATTERNS = [
    re.compile(r"(?:the\s+)?answer\s+is\s*[:\-]?\s*\*?\*?([A-D])\b", re.IGNORECASE),
    re.compile(r"(?:the\s+)?answer\s*[:\-]\s*\*?\*?([A-D])\b", re.IGNORECASE),
    re.compile(r"\boption\s*\*?\*?([A-D])\b", re.IGNORECASE),
    re.compile(r"^\s*\*?\*?([A-D])[\.\)\:]", re.IGNORECASE | re.MULTILINE),
]


def extract_answer(text):
    if not text:
        return None
    for pat in ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    fallback = re.findall(r"\b([A-D])\b", text)
    return fallback[-1].upper() if fallback else None


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * (((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="parsed_json/<run_id> directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    truth = load_ground_truth()
    print(f"Loaded {len(truth)} ground-truth answers from {BBQ_PATH.name}")

    # Collect: account -> [(query_id, predicted_letter, correct_bool)]
    per_account = defaultdict(list)
    unparsed = defaultdict(int)
    missing_truth = defaultdict(int)

    for sess_dir in sorted(run_dir.iterdir()):
        if not sess_dir.is_dir() or sess_dir.name not in SESSION_TO_ACCOUNT:
            continue
        account = SESSION_TO_ACCOUNT[sess_dir.name]
        for exp_dir in sorted(sess_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            for jf in sorted(exp_dir.glob("*.json")):
                with open(jf, encoding="utf-8") as f:
                    d = json.load(f)
                qid_full = d.get("meta", {}).get("query_id", jf.stem)
                qid = qid_full.split("_run")[0]
                gt = truth.get(qid)
                if gt is None:
                    missing_truth[account] += 1
                    continue
                pred = extract_answer(d.get("ai_generated_output_text", ""))
                if pred is None:
                    unparsed[account] += 1
                per_account[account].append((qid, pred, pred == gt))

    print("\n=== Per-account accuracy (95% Wilson CI) ===")
    summary = {}
    for acct in ("acct1", "acct2", "acct3"):
        recs = per_account.get(acct, [])
        n = len(recs)
        n_correct = sum(1 for _, _, c in recs if c)
        p = n_correct / n if n else 0
        lo, hi = wilson_ci(p, n)
        print(f"  {acct}: {n_correct}/{n} = {p:.4f}  [{lo:.4f}, {hi:.4f}]   (unparsed: {unparsed[acct]}, missing-truth: {missing_truth[acct]})")
        summary[acct] = (n_correct, n, p)

    # Per-query agreement across the 3 accounts (only for queries answered by all 3)
    by_qid = defaultdict(dict)
    for acct, recs in per_account.items():
        for qid, pred, _ in recs:
            by_qid[qid][acct] = pred

    complete = {qid: votes for qid, votes in by_qid.items() if set(votes.keys()) >= {"acct1", "acct2", "acct3"}}
    print(f"\n=== Per-query agreement across 3 accounts ({len(complete)} queries with all 3 votes) ===")
    n_all_agree = sum(1 for v in complete.values() if len(set(v.values())) == 1)
    n_one_diff = sum(1 for v in complete.values() if len(set(v.values())) == 2)
    n_three_diff = sum(1 for v in complete.values() if len(set(v.values())) == 3)
    print(f"  all 3 agree:       {n_all_agree}/{len(complete)} ({100*n_all_agree/max(1,len(complete)):.1f}%)")
    print(f"  2 agree, 1 differs:{n_one_diff}/{len(complete)} ({100*n_one_diff/max(1,len(complete)):.1f}%)")
    print(f"  all 3 differ:      {n_three_diff}/{len(complete)} ({100*n_three_diff/max(1,len(complete)):.1f}%)")

    # Chi-square: account vs correct
    print("\n=== Chi-square: account x correctness ===")
    table = []
    for acct in ("acct1", "acct2", "acct3"):
        n_correct, n, _ = summary[acct]
        table.append([n_correct, n - n_correct])
    chi2, p_val, dof, exp = chi2_contingency(table)
    print(f"  contingency table (rows = acct1/2/3, cols = correct/incorrect):")
    for acct, row in zip(("acct1", "acct2", "acct3"), table):
        print(f"    {acct}: correct={row[0]}, incorrect={row[1]}")
    print(f"  chi2 = {chi2:.4f}, p = {p_val:.4f}, dof = {dof}")
    if p_val < 0.05:
        print("  -> account effect is statistically significant at α=0.05")
    else:
        print("  -> failed to reject null: accounts not significantly different")

    # Pairwise account proportion-difference 95% CIs (normal approx)
    print("\n=== Pairwise account proportion differences (95% CI) ===")
    accts = ("acct1", "acct2", "acct3")
    for i, a in enumerate(accts):
        for b in accts[i + 1:]:
            nc_a, n_a, p_a = summary[a]
            nc_b, n_b, p_b = summary[b]
            diff = p_a - p_b
            se = ((p_a * (1 - p_a) / n_a) + (p_b * (1 - p_b) / n_b)) ** 0.5
            lo, hi = diff - 1.96 * se, diff + 1.96 * se
            print(f"  {a} - {b}: {diff:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
