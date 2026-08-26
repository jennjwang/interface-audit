"""Score HellaSwag responses across a sampling sweep.

Walks ``data/api/sweep-*-hellaswag/<cell>/session_00/metabench-hellaswag/*.api.json``
for the four (or 8) HellaSwag sweep run-ids, extracts the predicted
multiple-choice letter (same regex set as paper/tables/compute_test_retest.py),
compares to the gold answer in
``experiments/metabench/metabench-hellaswag/queries/queries.csv``, and writes:

  - per-cell accuracy + test-retest CSV (one row per grid point)

Outputs:
  automated-scraper/outputs/sweep_hellaswag/sweep_hellaswag_per_cell.csv
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ANSWER_KEY = BASE_DIR.parent / "experiments" / "metabench" / "metabench-hellaswag" / "queries" / "queries.csv"
DATA_ROOT = BASE_DIR / "data" / "api"
OUT_DIR = BASE_DIR / "outputs" / "sweep_hellaswag"

SWEEP_RUN_IDS = [
    "sweep-claude-sonnet-hellaswag",
    "sweep-claude-haiku-hellaswag",
    "sweep-gpt54-hellaswag",
    "sweep-gemini-flash-hellaswag",
    "sweep-claude-sonnet-thinking-hellaswag",
    "sweep-claude-haiku-thinking-hellaswag",
    "sweep-gpt54-thinking-hellaswag",
    "sweep-gemini-flash-thinking-hellaswag",
]

# Multi-choice answer extraction matching paper/tables/compute_test_retest.py
FINAL_LABEL_RE = re.compile(r"(?i)(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*\**\s*([ABCDEFGHI])\b")
BOLD_RE = re.compile(r"\*\*\s*([ABCDEFGHI])\s*[\.\)\*]?")
LETTER_RE = re.compile(r"(?im)(?:^|[^A-Za-z])([ABCDEFGHI])(?:[\.\)]|\b)")


def load_answer_key() -> dict[str, str]:
    key = {}
    with ANSWER_KEY.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key[str(row["id"])] = row["answer"].strip().upper()
    return key


def extract_letter(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    for r in (FINAL_LABEL_RE, BOLD_RE, LETTER_RE):
        m = r.search(s)
        if m:
            return m.group(1).upper()
    return None


AXIS_NAMES = ("temperature", "top_p", "budget_tokens", "reasoning_effort", "thinking_level")
AXIS_RE = re.compile(r"_(?P<axis>" + "|".join(AXIS_NAMES) + r")-(?P<value>[^_]+)")
MAX_TOKENS_RE = re.compile(r"_max_tokens-[^_]+")


def parse_cell_dir(name: str) -> tuple[str, str, str]:
    axis = value = ""
    m = AXIS_RE.search(name)
    if m:
        axis, value = m.group("axis"), m.group("value")
    model = name
    if m:
        model = model[: m.start()] + model[m.end():]
    model = MAX_TOKENS_RE.sub("", model)
    return model, axis, value


def score_cell(cell_dir: Path, answer_key: dict) -> dict:
    correct = total = extracted = 0
    per_run: dict[int, list[int]] = {}
    per_item: dict[str, dict[int, bool]] = {}
    run_re = re.compile(r"_run(\d+)$")
    for jf in cell_dir.rglob("*.api.json"):
        rec = json.loads(jf.read_text(encoding="utf-8"))
        text = rec.get("response_text") or ""
        if rec.get("error") or not text:
            continue
        stem = jf.stem.replace(".api", "")
        m = run_re.search(stem)
        run_idx = int(m.group(1)) if m else 0
        qid = run_re.sub("", stem)
        if qid not in answer_key:
            continue
        gold = answer_key[qid]
        predicted = extract_letter(text)
        total += 1
        per_run.setdefault(run_idx, [0, 0])
        per_run[run_idx][1] += 1
        if predicted is not None:
            extracted += 1
        is_correct = predicted == gold
        if is_correct:
            correct += 1
            per_run[run_idx][0] += 1
        per_item.setdefault(qid, {})[run_idx] = is_correct

    run_accs = [c / t for c, t in per_run.values() if t]
    if run_accs:
        mean = sum(run_accs) / len(run_accs)
        var = sum((a - mean) ** 2 for a in run_accs) / len(run_accs)
        std = var ** 0.5
    else:
        mean = std = 0.0

    n_runs = len(per_run)
    if n_runs >= 2:
        agreements = []
        for qid, runs in per_item.items():
            if len(runs) < n_runs:
                continue
            k = sum(1 for v in runs.values() if v)
            K = n_runs
            a_i = (k * (k - 1) + (K - k) * (K - k - 1)) / (K * (K - 1))
            agreements.append(a_i)
        retest = sum(agreements) / len(agreements) if agreements else 0.0
        n_items_full = len(agreements)
    else:
        retest = 0.0
        n_items_full = 0

    return {
        "n_total": total,
        "n_extracted": extracted,
        "n_correct": correct,
        "accuracy": correct / total if total else 0.0,
        "extract_rate": extracted / total if total else 0.0,
        "n_runs": n_runs,
        "run_accs": run_accs,
        "run_acc_mean": mean,
        "run_acc_std": std,
        "retest_pct": retest,
        "n_items_full": n_items_full,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    answer_key = load_answer_key()
    print(f"Loaded {len(answer_key)} HellaSwag queries")

    cell_rows: list[dict] = []
    for run_id in SWEEP_RUN_IDS:
        run_dir = DATA_ROOT / run_id
        if not run_dir.exists():
            print(f"  skip {run_id}: missing")
            continue
        for cell_dir in sorted(run_dir.iterdir()):
            if not cell_dir.is_dir() or cell_dir.name.startswith("_"):
                continue
            model, axis, value = parse_cell_dir(cell_dir.name)
            res = score_cell(cell_dir, answer_key)
            cell_rows.append({
                "run_id": run_id,
                "model": model,
                "axis": axis,
                "value": value,
                "n_total": res["n_total"],
                "n_extracted": res["n_extracted"],
                "n_correct": res["n_correct"],
                "n_runs": res["n_runs"],
                "accuracy": round(res["accuracy"], 4),
                "run_acc_mean": round(res["run_acc_mean"], 4),
                "run_acc_std": round(res["run_acc_std"], 4),
                "run_accs": "|".join(f"{a:.4f}" for a in res["run_accs"]),
                "extract_rate": round(res["extract_rate"], 4),
                "retest_pct": round(res["retest_pct"], 4),
                "n_items_full": res["n_items_full"],
            })
            run_str = "/".join(f"{a*100:.1f}" for a in res["run_accs"])
            print(f"  {model}  {axis}={value}  acc={res['accuracy']:.1%}  R={res['retest_pct']:.1%}  "
                  f"runs=[{run_str}]  n={res['n_total']}")

    cell_csv = OUT_DIR / "sweep_hellaswag_per_cell.csv"
    if cell_rows:
        with cell_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cell_rows)
        print(f"\nWrote {len(cell_rows)} per-cell rows → {cell_csv}")


if __name__ == "__main__":
    main()
