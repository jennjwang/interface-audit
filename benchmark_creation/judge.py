"""
Score benchmark runs using LLM judging.

Reads raw API JSON responses from --api-dir/<run-id>/<model>/*.api.json,
judges against gold answers in --queries, writes one scored CSV per model
to --output-dir/<model>.csv. Skips models whose output CSV already exists.

Usage:
    python judge.py \
        --api-dir  experiments/aa-omniscience/api \
        --queries  benchmark_creation/queries/aa-omniscience.csv \
        --output-dir experiments/aa-omniscience/outputs/api \
        --run-id   2026-04-23_19-14-53
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT  = Path(__file__).resolve().parent.parent
SCRAPER_DIR = REPO_ROOT / "automated-scraper"
load_dotenv(SCRAPER_DIR / ".env")

JUDGE_SYSTEM = (
    "You are a strict expert grader. "
    "Given a question, a gold answer, and a model's response, "
    "output 1 if the response is correct, or 0 if it is incorrect.\n\n"
    "Rules:\n"
    "- The response is correct if it contains or clearly implies the gold answer.\n"
    "- Numerical answers must match to reasonable precision.\n"
    "- Minor wording differences are OK if the meaning is the same.\n"
    "- Output only the digit 1 or 0."
)

CSV_FIELDS = ["query_id", "id", "source", "response", "gold_answer", "correct", "api_params", "path"]


def load_gold(queries_path: Path) -> dict[int, dict]:
    gold: dict[int, dict] = {}
    with queries_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                qid = int(row["id"])
            except (KeyError, ValueError):
                continue
            gold[qid] = {"query": row.get("query", ""), "answer": row.get("answer", "")}
    return gold


def load_api_dir(model_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(model_dir.glob("*.api.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_id = d.get("query_id", f.stem.replace(".api", ""))
        try:
            qid = int(str(raw_id).split("_run")[0])
        except ValueError:
            qid = None
        rows.append({
            "path":      str(f),
            "query_id":  raw_id,
            "id":        qid,
            "source":    "api",
            "response":  d.get("response_text", "") or "",
            "api_params": json.dumps(d.get("api_params"), sort_keys=True)
                          if isinstance(d.get("api_params"), dict) else "",
        })
    return rows


def judge_one(client: OpenAI, query: str, gold: str, response: str) -> str | None:
    if not response or not gold:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": (
                    f'Question: "{query}"\n\nGold Answer: "{gold}"\n\nModel Response: "{response}"'
                )},
            ],
            max_completion_tokens=4,
        )
        raw = resp.choices[0].message.content.strip()
        for ch in raw:
            if ch in ("0", "1"):
                return ch
        return None
    except Exception as exc:
        print(f"  LLM judge error: {exc}", file=sys.stderr)
        return None


def judge_one_regex(gold: str, response: str) -> str | None:
    """Regex judge: check if response starts with the gold token (e.g. NTA/YTA)."""
    if not response or not gold:
        return None
    prefix = response.strip().upper()[:5]
    gold_upper = gold.strip().upper()
    return "1" if gold_upper in prefix else "0"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-dir",    required=True, help="Root dir containing <run-id>/<model>/ subdirs")
    parser.add_argument("--queries",    required=True, help="CSV with id, query, answer columns")
    parser.add_argument("--output-dir", required=True, help="Dir to write scored CSVs")
    parser.add_argument("--run-id",     required=True, help="Run timestamp to score")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--regex-judge", action="store_true",
                        help="Score by regex match on response prefix instead of LLM judge")
    args = parser.parse_args()

    api_dir    = Path(args.api_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir    = api_dir / args.run_id

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    gold = load_gold(Path(args.queries))
    print(f"Loaded {len(gold)} gold answers from {args.queries}")

    client = None
    if args.regex_judge:
        print("Regex judging enabled (prefix match on gold token).")
    elif not args.dry_run:
        api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key, timeout=30)
            print("LLM judging enabled (gpt-4o-mini).")
        else:
            print("No OPENAI_KEY — dry-run mode.", file=sys.stderr)

    model_dirs = [d for d in sorted(run_dir.iterdir()) if d.is_dir()]
    print(f"Found {len(model_dirs)} model dir(s) in {run_dir.name}")

    for model_dir in model_dirs:
        out_csv = output_dir / f"{model_dir.name}.csv"
        if out_csv.exists() and out_csv.stat().st_size > 0:
            print(f"  Skip (exists): {model_dir.name}")
            continue

        rows = load_api_dir(model_dir)
        if not rows:
            print(f"  No responses in {model_dir.name}")
            continue

        correct_count = 0
        for row in rows:
            qid        = row["id"]
            gold_entry = gold.get(qid) if qid is not None else None
            gold_answer = gold_entry["answer"] if gold_entry else ""
            query       = gold_entry["query"]  if gold_entry else ""
            row["gold_answer"] = gold_answer

            if args.regex_judge:
                verdict = judge_one_regex(gold_answer, row["response"])
                row["correct"] = verdict
                if verdict == "1":
                    correct_count += 1
            elif client and gold_answer:
                verdict = judge_one(client, query, gold_answer, row["response"])
                row["correct"] = verdict
                if verdict == "1":
                    correct_count += 1
            else:
                row["correct"] = ""

        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        scored = sum(1 for r in rows if r["correct"] in ("0", "1"))
        if scored:
            print(f"  {model_dir.name}: {correct_count}/{scored} correct ({correct_count/scored:.1%})  → {out_csv.name}")
        else:
            print(f"  {model_dir.name}: {len(rows)} rows (unscored)  → {out_csv.name}")


if __name__ == "__main__":
    main()
