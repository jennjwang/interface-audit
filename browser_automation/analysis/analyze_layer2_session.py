"""
Layer 2 phase 2 analysis — session-level variance within accounts on BBQ.

Treats each run dir as one "session" for the same 3 accounts.
Decomposes variance into:
  - between-account (phase 1 result, included as reference)
  - between-session within-account (new: does the same account vary across sessions?)
  - per-query agreement across sessions (same account, different sessions)

Usage:
  python analyze_layer2_session.py --provider Claude \\
      run_dir1 run_dir2 [run_dir3 ...]

  To merge multiple dirs into one session (e.g. partial + resume), join
  with a colon:
      run_dir_main:run_dir_resume run_dir2
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
BBQ_PATH = SCRIPT_DIR.parent / "benchmark_creation" / "results" / "bbq-subset-200.csv"

SESSION_TO_ACCOUNT = {
    "session_00": "acct1",
    "session_01": "acct2",
    "session_02": "acct3",
}

ANSWER_PATTERNS = [
    re.compile(r"(?:the\s+)?answer\s+is\s*[:\-]?\s*\*?\*?([A-D])\b", re.IGNORECASE),
    re.compile(r"(?:the\s+)?answer\s*[:\-]\s*\*?\*?([A-D])\b", re.IGNORECASE),
    re.compile(r"\boption\s*\*?\*?([A-D])\b", re.IGNORECASE),
    re.compile(r"^\s*\*?\*?([A-D])[\.\)\:]", re.IGNORECASE | re.MULTILINE),
]

def load_ground_truth():
    truth = {}
    with open(BBQ_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth[str(row["id"])] = row["answer"].strip().upper()
    return truth

def extract_answer(text):
    if not text:
        return None
    for pat in ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    fallback = re.findall(r"\b([A-D])\b", text)
    return fallback[-1].upper() if fallback else None

def load_run(run_dir, truth):
    """Returns {account: {qid: (pred, correct)}}"""
    result = defaultdict(dict)
    run_dir = Path(run_dir)
    for sess_dir in sorted(run_dir.iterdir()):
        if not sess_dir.is_dir() or sess_dir.name not in SESSION_TO_ACCOUNT:
            continue
        acct = SESSION_TO_ACCOUNT[sess_dir.name]
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
                    continue
                pred = extract_answer(d.get("ai_generated_output_text", ""))
                if qid not in result[acct]:  # keep first occurrence
                    result[acct][qid] = (pred, pred == gt if pred else False)
    return result

def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * (((p*(1-p) + z*z/(4*n))/n)**0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="")
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    truth = load_ground_truth()
    print(f"Loaded {len(truth)} ground-truth answers.\n")

    # Load all sessions; each arg may be "dir1:dir2" to merge partial+resume
    runs = []
    for i, rd_spec in enumerate(args.run_dirs):
        parts = rd_spec.split(":")
        # Merge: load first dir, then overlay with subsequent dirs
        merged = load_run(parts[0], truth)
        for extra in parts[1:]:
            extra_data = load_run(extra, truth)
            for acct in extra_data:
                merged[acct].update(extra_data[acct])
        runs.append(merged)
        label = "+".join(Path(p).name[:22] for p in parts)
        print(f"Session {i+1}: {label}")
        for acct in ("acct1","acct2","acct3"):
            n = len(merged[acct])
            nc = sum(1 for _,c in merged[acct].values() if c)
            print(f"  {acct}: {nc}/{n}")
    print()

    n_sessions = len(runs)
    accounts = ("acct1", "acct2", "acct3")

    # Per-(account, session) accuracy
    print("=== Accuracy matrix (rows=accounts, cols=sessions) ===")
    acc_matrix = {}
    header = "         " + "  ".join(f"sess{i+1:02d}" for i in range(n_sessions)) + "   mean    CI95"
    print(header)
    for acct in accounts:
        row_accs = []
        for run in runs:
            recs = run[acct]
            n = len(recs)
            nc = sum(1 for _,c in recs.values() if c)
            row_accs.append(nc/n if n else 0)
        mean_acc = sum(row_accs)/len(row_accs)
        lo, hi = wilson_ci(mean_acc, 200*n_sessions)
        acc_str = "  ".join(f"{a:.4f}" for a in row_accs)
        print(f"  {acct}:  {acc_str}   {mean_acc:.4f}  [{lo:.4f},{hi:.4f}]")
        acc_matrix[acct] = row_accs

    # Within-account session variance
    print("\n=== Within-account session differences (session2 - session1) ===")
    within_diffs = []
    for acct in accounts:
        accs = acc_matrix[acct]
        if n_sessions >= 2:
            diffs = [accs[j] - accs[0] for j in range(1, n_sessions)]
            for j, d in enumerate(diffs):
                print(f"  {acct}: sess{j+2:02d}-sess01 = {d:+.4f}")
                within_diffs.append(abs(d))
    if within_diffs:
        print(f"\n  Mean |within-account session diff|: {sum(within_diffs)/len(within_diffs):.4f}")
        print(f"  Max  |within-account session diff|: {max(within_diffs):.4f}")

    # Between-account variance (reference from phase 1)
    print("\n=== Between-account variance (reference) ===")
    acct_means = [sum(acc_matrix[a])/n_sessions for a in accounts]
    grand_mean = sum(acct_means)/3
    between_acct_var = sum((m - grand_mean)**2 for m in acct_means) / 3
    print(f"  Account means: " + ", ".join(f"{m:.4f}" for m in acct_means))
    print(f"  Grand mean: {grand_mean:.4f}")
    print(f"  Between-account variance (σ²): {between_acct_var:.6f}  (SD: {between_acct_var**0.5:.4f})")

    within_session_var = sum(
        (acc_matrix[a][j] - sum(acc_matrix[a])/n_sessions)**2
        for a in accounts for j in range(n_sessions)
    ) / (3 * n_sessions)
    print(f"  Within-account session variance (σ²): {within_session_var:.6f}  (SD: {within_session_var**0.5:.4f})")

    if between_acct_var + within_session_var > 0:
        icc_acct = between_acct_var / (between_acct_var + within_session_var)
        print(f"\n  ICC_account (fraction of variance between accounts): {icc_acct:.4f}")
        print(f"  ICC_session (fraction within-account, between sessions): {1-icc_acct:.4f}")

    # Per-query agreement across sessions, within each account
    print("\n=== Per-query session agreement (same account, different sessions) ===")
    for acct in accounts:
        all_qids = set(runs[0][acct].keys())
        for run in runs[1:]:
            all_qids &= set(run[acct].keys())
        n_q = len(all_qids)
        agree = sum(
            1 for qid in all_qids
            if len({runs[i][acct][qid][0] for i in range(n_sessions) if qid in runs[i][acct]}) == 1
        )
        print(f"  {acct}: {agree}/{n_q} queries same answer across sessions ({100*agree/max(1,n_q):.1f}%)")

if __name__ == "__main__":
    main()
