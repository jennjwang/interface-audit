"""LLM-judge scoring for metabench with-prompt outputs.

Uses gpt-4o-mini as a judge to extract the answer from each model response and
compare against the gold answer. Mirrors the approach used to produce the
existing per-session scored CSVs (`correct` column).

Usage:
  python automated-scraper/judge_metabench.py \
      --run-id batch-system-prompt-metabench \
      --workers 32
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

METABENCH = {
    "metabench_mmlu":       BASE.parent / "experiments/metabench/metabench-mmlu/queries/queries.csv",
    "metabench_arc":        BASE.parent / "experiments/metabench/metabench-arc/queries/queries.csv",
    "metabench_gsm8k":      BASE.parent / "experiments/metabench/metabench-gsm8k/queries/queries.csv",
    "metabench_hellaswag":  BASE.parent / "experiments/metabench/metabench-hellaswag/queries/queries.csv",
    "metabench_truthfulqa": BASE.parent / "experiments/metabench/metabench-truthfulQA/queries/queries.csv",
    "metabench_winogrande": BASE.parent / "experiments/metabench/metabench-winogrande/queries/queries.csv",
}

JUDGE_SYSTEM = (
    "You are a strict expert grader for multiple-choice and numeric-answer benchmarks. "
    "Given a question, a gold answer, and a model's response, output 1 if the response is correct, "
    "or 0 if it is incorrect.\n\n"
    "Rules:\n"
    "- For multiple-choice questions, the response is correct if it ultimately selects the same letter "
    "  (or option text) as the gold answer.\n"
    "- For numeric questions (GSM8K), the response is correct if the final numeric answer matches the "
    "  gold to reasonable precision (commas and trailing decimals are equivalent).\n"
    "- The response may contain reasoning; focus on its FINAL answer.\n"
    "- Output only the digit 1 or 0."
)


def load_gold(path: Path) -> dict[str, dict]:
    gold = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            qid = str(r["id"])
            gold[qid] = {"query": r["query"], "answer": r["answer"]}
    return gold


def gsm_gold_number(raw_answer: str) -> str:
    """Extract the number after '####' for gsm8k gold answers."""
    import re
    m = re.search(r"####\s*(-?\d[\d,.]*)", raw_answer or "")
    return m.group(1).strip() if m else (raw_answer or "").strip()


def judge_one(client: OpenAI, query: str, gold: str, response: str) -> str | None:
    if not response or not gold:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": (
                    f'Question:\n{query}\n\n'
                    f'Gold Answer: {gold}\n\n'
                    f'Model Response:\n{response}'
                )},
            ],
            max_completion_tokens=4,
        )
        raw = (resp.choices[0].message.content or "").strip()
        for ch in raw:
            if ch in ("0", "1"):
                return ch
        return None
    except Exception as exc:
        return None


def score_cell(client: OpenAI, cell_dir: Path, bench: str, gold: dict[str, dict],
               out_path: Path, workers: int) -> None:
    is_gsm = bench == "metabench_gsm8k"

    # Load responses from api.json files. Key = full query_id (e.g., "1_run0"),
    # value = (base_qid, response_text). Supports multi-run.
    items: list[tuple[str, str, str]] = []  # (run_qid, base_qid, response)
    for f in cell_dir.glob("*.api.json"):
        try: d = json.loads(f.read_text())
        except: continue
        full_qid = str(d.get("query_id",""))
        base_qid = full_qid.split("_run")[0]
        items.append((full_qid, base_qid, d.get("response_text") or ""))

    if not items:
        return
    items.sort(key=lambda t: (int(t[1]) if t[1].isdigit() else t[1], t[0]))

    # Build judge inputs (one per (qid, run))
    rows = []
    judge_inputs = []
    for full_qid, base_qid, resp in items:
        if base_qid not in gold:
            continue
        g_query = gold[base_qid]["query"]
        g_ans = gold[base_qid]["answer"]
        g_ans_simplified = gsm_gold_number(g_ans) if is_gsm else g_ans
        rows.append({
            "id": base_qid,
            "run_qid": full_qid,
            "gold_answer": g_ans_simplified,
            "response": resp,
            "correct": "",
        })
        if resp:
            judge_inputs.append((full_qid, g_query, g_ans_simplified, resp))

    verdicts: dict[str,str|None] = {}
    if judge_inputs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(judge_one, client, q, ans, r): rqid for (rqid, q, ans, r) in judge_inputs}
            done = 0
            total = len(futs)
            for fut in as_completed(futs):
                rqid = futs[fut]
                verdicts[rqid] = fut.result()
                done += 1
                if done % 100 == 0 or done == total:
                    print(f"    judged {done}/{total}", flush=True)

    correct = scored = 0
    for r in rows:
        v = verdicts.get(r["run_qid"])
        if v in ("0","1"):
            r["correct"] = "True" if v == "1" else "False"
            scored += 1
            if v == "1": correct += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id","run_qid","gold_answer","response","correct"])
        w.writeheader(); w.writerows(rows)

    acc = correct/scored if scored else 0
    print(f"  → {out_path.name}: {correct}/{scored} = {acc:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_KEY/OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, timeout=60)

    run_dir = BASE / "data" / "api" / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "judged")
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_per_bench = {b: load_gold(p) for b, p in METABENCH.items()}

    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir(): continue
        if model_dir.name.startswith("_"): continue
        for bench in METABENCH:
            cell = model_dir / "session_00" / bench
            if not cell.exists() or not any(cell.glob("*.api.json")): continue
            out_csv = out_dir / f"{model_dir.name}__{bench}.csv"
            if out_csv.exists() and out_csv.stat().st_size > 0:
                print(f"  Skip: {out_csv.name} (exists)")
                continue
            print(f"\nJudging {model_dir.name} / {bench}...")
            score_cell(client, cell, bench, gold_per_bench[bench], out_csv, args.workers)


if __name__ == "__main__":
    main()
