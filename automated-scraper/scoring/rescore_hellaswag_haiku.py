"""Re-score Haiku HellaSwag interface responses with a context-aware LLM judge.

The original judge in parse.py only receives the response text with no question
context. For Haiku's HellaSwag interface runs, the system prompt causes the
model to treat the multi-paragraph prompt as a document to summarize or clarify,
sometimes creating lettered clarifying menus (A. Looking for the answer...) before
eventually stating the correct answer. The no-context judge picks the letter from
the clarifying menu rather than the actual answer.

This script re-judges all haiku HellaSwag interface CSVs using a context-aware
prompt that includes the question (just the trailing MC question + options),
explicitly instructing the judge to focus on the model's final answer to the
actual question and ignore clarifying menus or document summaries.

After re-scoring the CSVs, it also updates:
  experiments/metabench/openllm_leaderboard/plots/coverage_by_run_claude.csv

Usage:
  python automated-scraper/rescore_hellaswag_haiku.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parent

load_dotenv(BASE / ".env")

HELLASWAG_DIR = REPO / "experiments" / "metabench" / "metabench-hellaswag"
QUERIES_CSV   = HELLASWAG_DIR / "queries" / "queries.csv"
IFACE_DIR     = HELLASWAG_DIR / "data-claude" / "interface"
PARSED_DIR    = HELLASWAG_DIR / "data-claude" / "parsed_json"
COVERAGE_CSV  = REPO / "experiments" / "metabench" / "openllm_leaderboard" / "plots" / "coverage_by_run_claude.csv"

# Only re-score the May 2026 haiku runs (the reruns used in the paper).
HAIKU_RUN_TIMESTAMPS = [
    "2026-05-14_01-20-44-hellaswag",
    "2026-05-14_10-57-37-hellaswag",
    "2026-05-14_12-26-15-hellaswag",
    "2026-05-14_13-32-50-hellaswag",
    "2026-05-14_14-42-08-hellaswag",
    "2026-05-24_00-37-27-hellaswag",
    "2026-05-24_00-50-03-hellaswag",
    "2026-05-24_00-50-33-hellaswag",
    "2026-05-24_01-42-03-hellaswag",
    "2026-05-24_01-57-31-hellaswag",
    "2026-05-24_02-06-55-hellaswag",
]

JUDGE_SYSTEM = (
    "You are extracting the answer letter from a model's response to a "
    "multiple-choice question. The model's response may contain meta-commentary, "
    "document summaries, clarifying menus, or unrelated content — focus ONLY on "
    "which answer letter (A, B, C, or D) the model ultimately selected for the "
    "actual question. Output only a single uppercase letter A, B, C, or D. "
    "If the model did not select an answer to the question, output NONE."
)

WORKERS = 16


def load_queries() -> dict[str, dict]:
    gold: dict[str, dict] = {}
    with open(QUERIES_CSV) as f:
        for r in csv.DictReader(f):
            gold[str(r["id"])] = {"query": r["query"], "answer": r["answer"]}
    return gold


def question_tail(query: str, chars: int = 700) -> str:
    """Return the trailing portion of a HellaSwag query (actual scenario + options)."""
    return query[-chars:].strip()


def judge_one(client: OpenAI, qid: str, query: str, response: str) -> str | None:
    tail = question_tail(query)
    user_msg = f"Question (end of prompt):\n{tail}\n\nModel's response:\n{response}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_completion_tokens=4,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
        if raw in ("A", "B", "C", "D"):
            return raw
        if raw == "NONE":
            return None
        # Sometimes the judge returns e.g. "D." or "D:"
        for ch in raw:
            if ch in "ABCD":
                return ch
        return None
    except Exception as exc:
        print(f"  judge error qid={qid}: {exc}")
        return None


def rescore_csv(csv_path: Path, parsed_ts_dir: Path, gold: dict[str, dict],
                client: OpenAI, dry_run: bool) -> tuple[int, int]:
    """Re-judge every row. Returns (n_correct, n_total)."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Build judge inputs
    judge_inputs: list[tuple[str, str, str]] = []  # (qid, query, response)
    for row in rows:
        qid = str(row.get("id", "")).strip()
        if qid not in gold:
            continue
        path_str = row.get("path", "")
        if not path_str:
            continue
        json_path = Path(path_str)
        if not json_path.exists():
            continue
        try:
            d = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        response = d.get("ai_generated_output_text") or d.get("response_text") or ""
        if response:
            judge_inputs.append((qid, gold[qid]["query"], response))

    # Run judge in parallel
    verdicts: dict[str, str | None] = {}
    if judge_inputs:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {
                ex.submit(judge_one, client, qid, query, resp): qid
                for qid, query, resp in judge_inputs
            }
            done = 0
            for fut in as_completed(futs):
                qid = futs[fut]
                verdicts[qid] = fut.result()
                done += 1
                if done % 20 == 0 or done == len(futs):
                    print(f"    judged {done}/{len(futs)}", flush=True)

    # Update rows
    n_correct = n_total = 0
    for row in rows:
        qid = str(row.get("id", "")).strip()
        if qid not in verdicts:
            continue
        letter = verdicts[qid]
        gold_ans = (gold.get(qid, {}).get("answer") or "").strip().upper()
        row["answer_llm"] = letter or ""
        if letter:
            row["answer"] = letter
            row["correct"] = str(letter == gold_ans)
            n_total += 1
            if letter == gold_ans:
                n_correct += 1
        else:
            row["answer"] = ""
            row["correct"] = "False"
            n_total += 1

    acc = n_correct / n_total if n_total else 0
    print(f"  → {csv_path.name}: {n_correct}/{n_total} = {acc:.1%}")

    if not dry_run:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    return n_correct, n_total


def update_coverage(ts_accuracies: dict[str, tuple[int, int]], dry_run: bool) -> None:
    """Patch or append Interface: Haiku rows in coverage_by_run_claude.csv."""
    with open(COVERAGE_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    existing_ts = {
        row["timestamp"]
        for row in rows
        if row.get("dataset") == "metabench-hellaswag"
        and row.get("condition") == "Interface: Haiku"
    }
    max_run_index = max(
        (int(r["run_index"]) for r in rows
         if r.get("dataset") == "metabench-hellaswag" and r["run_index"].isdigit()),
        default=0,
    )

    updated = added = 0
    for row in rows:
        if (row.get("dataset") == "metabench-hellaswag"
                and row.get("condition") == "Interface: Haiku"
                and row.get("timestamp") in ts_accuracies):
            ts = row["timestamp"]
            n_correct, n_total = ts_accuracies[ts]
            acc = n_correct / n_total if n_total else 0
            row["accuracy"] = str(acc)
            row["answered"] = str(n_total)
            row["total"] = str(n_total)
            updated += 1
            print(f"  coverage update: {ts}: {n_correct}/{n_total} = {acc:.4f}")

    for ts, (n_correct, n_total) in sorted(ts_accuracies.items()):
        if ts in existing_ts:
            continue
        max_run_index += 1
        acc = n_correct / n_total if n_total else 0
        rows.append({
            "dataset":      "metabench-hellaswag",
            "run_index":    str(max_run_index),
            "timestamp":    ts,
            "complete_run": "True",
            "condition":    "Interface: Haiku",
            "accuracy":     str(acc),
            "answered":     str(n_total),
            "total":        str(n_total),
        })
        added += 1
        print(f"  coverage add:    {ts}: {n_correct}/{n_total} = {acc:.4f}")

    print(f"  Updated {updated}, added {added} rows in coverage_by_run_claude.csv")
    if not dry_run:
        with open(COVERAGE_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_KEY not set")
    client = OpenAI(api_key=api_key, timeout=60)

    gold = load_queries()
    print(f"Loaded {len(gold)} HellaSwag questions")

    ts_accuracies: dict[str, tuple[int, int]] = {}

    for ts in HAIKU_RUN_TIMESTAMPS:
        iface_csv = IFACE_DIR / ts / "session_00" / "haiku.csv"
        parsed_ts_dir = PARSED_DIR / ts / "session_00" / "haiku"
        if not iface_csv.exists():
            print(f"  Missing: {iface_csv}")
            continue
        print(f"\nRe-scoring {ts}...")
        n_correct, n_total = rescore_csv(iface_csv, parsed_ts_dir, gold, client, args.dry_run)
        ts_accuracies[ts] = (n_correct, n_total)

    print("\nUpdating coverage_by_run_claude.csv...")
    update_coverage(ts_accuracies, args.dry_run)

    if args.dry_run:
        print("\n(dry-run: no files written)")
    else:
        print("\nDone. Next: rerun downstream scripts to update appendix tables and figures.")


if __name__ == "__main__":
    main()
