"""Unified scoring pipeline for all benchmarks.

Reads raw response JSONs, extracts answers (regex + optional LLM judge),
compares to a gold answer key, and writes scored CSVs.

Supports:
  - Multiple-choice letter extraction (A-I) for metabench, BBQ, etc.
  - Numeric answer extraction for GSM8K
  - LLM-judged free-form scoring for AA-Omniscience, SimpleQA, etc.

Usage:
  # Score metabench (regex-only, no API key needed)
  python browser_automation/scoring/score.py \\
      --run-dir data/api/batch-system-prompt-metabench \\
      --answer-key data/answer_keys/queries.csv \\
      --task mc

  # Score AA-Omniscience (LLM judge)
  python browser_automation/scoring/score.py \\
      --run-dir data/api/batch-aa-5runs \\
      --answer-key data/answer_keys/aa-omniscience-subset-200.csv \\
      --task judge

  # Score GSM8K
  python browser_automation/scoring/score.py \\
      --run-dir data/api/batch-metabench \\
      --answer-key data/answer_keys/queries.csv \\
      --task gsm8k --bench metabench_gsm8k
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parents[1]
load_dotenv(BASE / ".env")

# ═══════════════════════════════════════════════════════════════════════════
# Letter extraction (multiple-choice)
# ═══════════════════════════════════════════════════════════════════════════

# Broad MC patterns (A-I): metabench, hellaswag, mmlu, truthfulqa, etc.
MC_FINAL_LABEL = re.compile(
    r"(?i)(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*\**\s*([ABCDEFGHI])\b"
)
MC_BOLD = re.compile(r"\*\*\s*([ABCDEFGHI])\s*[\.\)\*]?")
MC_GENERIC = re.compile(r"(?im)(?:^|[^A-Za-z])([ABCDEFGHI])(?:[\.\)]|\b)")

# BBQ-specific patterns (A-C only)
BBQ_PATTERNS = [
    re.compile(r"\*\*([ABC])\.\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\*\*([ABC])\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:answer is|answer:)\s*\**([ABC])\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^([ABC])\.", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^([ABC])\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\b([ABC])\.", re.IGNORECASE | re.MULTILINE),
]


def extract_letter(text: str, bbq: bool = False) -> str | None:
    """Extract a multiple-choice letter from model response text.

    Args:
        text: Model response text.
        bbq: If True, use BBQ-specific patterns (A-C only).
    """
    if not text:
        return None
    s = text.strip()
    if bbq:
        for pat in BBQ_PATTERNS:
            m = pat.search(s)
            if m:
                return m.group(1).upper()
        return None
    for r in (MC_FINAL_LABEL, MC_BOLD, MC_GENERIC):
        m = r.search(s)
        if m:
            return m.group(1).upper()
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Numeric extraction (GSM8K)
# ═══════════════════════════════════════════════════════════════════════════

GSM_EXPLICIT = [
    re.compile(r"####\s*(-?\d[\d,\.]*)"),
    re.compile(r"\\boxed\{\s*(-?\d[\d,\.]*)"),
    re.compile(
        r"(?i)(?:final\s+answer|the\s+answer\s+is|answer\s+is)"
        r"\s*[:=]?\s*\$?\s*(-?\d[\d,\.]*)"
    ),
]
GSM_NUM_TOKEN = re.compile(r"(?<![\d\.,])-?\d[\d,\.]*")


def normalize_number(s: str) -> str:
    """Normalize a numeric string: strip $, commas, trailing .0."""
    s = s.strip().lstrip("+").replace("$", "").replace(",", "").replace("%", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract_number(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K-style response."""
    if not text:
        return None
    for r in GSM_EXPLICIT:
        ms = list(r.finditer(text))
        if ms:
            return normalize_number(ms[-1].group(1))
    nums = GSM_NUM_TOKEN.findall(text)
    return normalize_number(nums[-1]) if nums else None


# ═══════════════════════════════════════════════════════════════════════════
# LLM judge
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM = (
    "You are a strict expert grader. "
    "Given a question, a gold answer, and a model's response, "
    "output 1 if the response is correct, or 0 if it is incorrect.\n\n"
    "Rules:\n"
    "- The response is correct if it contains or clearly implies the gold answer.\n"
    "- Numerical answers must match to reasonable precision.\n"
    "- Minor wording differences or alternate spellings are OK if the meaning is the same.\n"
    "- If the model declines to answer, output 0.\n"
    "- Output only the digit 1 or 0."
)

MC_JUDGE_SYSTEM = (
    "You are extracting the answer letter (A–I) from a model's response to a "
    "multiple-choice question.\n"
    "1. If the model clearly selected ONE option — either by stating the "
    "letter or by paraphrasing the text of one option — output that uppercase letter.\n"
    "2. If the model did not answer the actual question (asked for clarification, "
    "refused, summarized, or wrote free text not selecting any option) — output NONE.\n"
    "3. If the model gave a ranked list of multiple letters, output NONE.\n"
    "4. Ignore letters in clarifying menus or explanations of wrong options. "
    "Focus on the FINAL chosen answer.\n\n"
    "Output ONLY the single letter or the word NONE — nothing else."
)


def judge_freeform(client: OpenAI, query: str, gold: str, response: str) -> str | None:
    """Use LLM to judge whether a free-form response matches the gold answer.

    Returns '1' (correct) or '0' (incorrect), or None on error.
    """
    if not response or not gold:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": (
                    f'Question: "{query}"\n\n'
                    f'Gold Answer: "{gold}"\n\n'
                    f'Model Response: "{response}"'
                )},
            ],
            max_completion_tokens=4,
        )
        raw = (resp.choices[0].message.content or "").strip()
        for ch in raw:
            if ch in ("0", "1"):
                return ch
        return None
    except Exception:
        return None


def judge_mc_context(client: OpenAI, question_tail: str, response: str) -> str | None:
    """Context-aware MC letter extraction via LLM.

    Includes the question text so the judge can distinguish the actual answer
    from clarifying menus or document summaries.

    Returns a letter (A-I), or None if no clear answer.
    """
    if not response:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": MC_JUDGE_SYSTEM},
                {"role": "user", "content": (
                    f"Question (tail):\n{question_tail}\n\n"
                    f"Model response:\n{response}"
                )},
            ],
            max_completion_tokens=4,
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
        if raw == "NONE":
            return None
        if len(raw) == 1 and raw in "ABCDEFGHI":
            return raw
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Common I/O
# ═══════════════════════════════════════════════════════════════════════════

def load_gold(path: Path) -> dict[str, dict]:
    """Load an answer key CSV. Returns {id: {answer, query, ...}}."""
    gold = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = str(row.get("id", "")).strip()
            if not qid:
                continue
            gold[qid] = {
                "answer": (row.get("answer") or "").strip(),
                "query": (row.get("query") or "").strip(),
                "category": (row.get("category") or "").strip(),
            }
    return gold


def read_responses(folder: Path) -> list[dict]:
    """Read all JSON response files from a folder.

    Returns list of {qid, response, source} dicts.
    """
    items = []
    for f in sorted(folder.rglob("*.json")):
        if f.name.endswith(".meta.json") or f.name == "parse.json":
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        # Determine source and extract response text
        if "response_text" in d:
            text = d.get("response_text") or ""
            source = "api"
        elif "ai_generated_output_text" in d:
            text = d.get("ai_generated_output_text") or ""
            source = "interface"
        else:
            continue
        # Extract query ID
        full_qid = str(
            d.get("query_id")
            or (d.get("meta") or {}).get("query_id")
            or f.stem.replace(".api", "")
        )
        base_qid = full_qid.split("_run")[0]
        items.append({
            "full_qid": full_qid,
            "qid": base_qid,
            "response": text,
            "source": source,
        })
    items.sort(key=lambda t: (int(t["qid"]) if t["qid"].isdigit() else t["qid"], t["full_qid"]))
    return items


def write_scored_csv(rows: list[dict], path: Path) -> None:
    """Write scored results to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Scoring tasks
# ═══════════════════════════════════════════════════════════════════════════

def score_mc(items: list[dict], gold: dict[str, dict], bbq: bool = False) -> list[dict]:
    """Score multiple-choice responses using regex extraction."""
    rows = []
    for item in items:
        g = gold.get(item["qid"])
        if not g:
            continue
        pred = extract_letter(item["response"], bbq=bbq)
        gold_ans = g["answer"].upper()
        correct = ""
        if pred is not None:
            correct = "True" if pred == gold_ans else "False"
        rows.append({
            "id": item["qid"],
            "run_qid": item["full_qid"],
            "gold_answer": g["answer"],
            "answer": pred or "",
            "correct": correct,
        })
    return rows


def score_gsm8k(items: list[dict], gold: dict[str, dict]) -> list[dict]:
    """Score GSM8K responses using numeric extraction."""
    rows = []
    for item in items:
        g = gold.get(item["qid"])
        if not g:
            continue
        # Extract gold number from #### marker
        gold_raw = g["answer"]
        m = re.search(r"####\s*(-?\d[\d,\.]*)", gold_raw)
        gold_num = normalize_number(m.group(1)) if m else normalize_number(gold_raw)
        pred = extract_number(item["response"])
        correct = ""
        if pred is not None:
            correct = "True" if pred == gold_num else "False"
        rows.append({
            "id": item["qid"],
            "run_qid": item["full_qid"],
            "gold_answer": gold_num,
            "answer": pred or "",
            "correct": correct,
        })
    return rows


def score_judge(
    items: list[dict],
    gold: dict[str, dict],
    client: OpenAI,
    workers: int = 32,
) -> list[dict]:
    """Score responses using LLM judge (free-form answers)."""
    to_judge = []
    rows = []
    for item in items:
        g = gold.get(item["qid"])
        if not g:
            continue
        row = {
            "id": item["qid"],
            "run_qid": item["full_qid"],
            "gold_answer": g["answer"],
            "response": item["response"],
            "correct": "",
        }
        rows.append(row)
        if item["response"]:
            to_judge.append((item["full_qid"], g["query"], g["answer"], item["response"]))

    if to_judge:
        verdicts = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(judge_freeform, client, q, ans, resp): rqid
                for (rqid, q, ans, resp) in to_judge
            }
            done = 0
            for fut in as_completed(futs):
                verdicts[futs[fut]] = fut.result()
                done += 1
                if done % 200 == 0 or done == len(futs):
                    print(f"  judged {done}/{len(futs)}", flush=True)
        for r in rows:
            v = verdicts.get(r["run_qid"])
            if v in ("0", "1"):
                r["correct"] = "True" if v == "1" else "False"

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Score model responses against a gold answer key."
    )
    ap.add_argument("--run-dir", required=True, help="Directory containing response JSONs")
    ap.add_argument("--answer-key", required=True, help="Path to answer key CSV (id, answer)")
    ap.add_argument(
        "--task",
        choices=["mc", "bbq", "gsm8k", "judge"],
        default="mc",
        help="Scoring method: mc (letter extraction), bbq (A-C only), gsm8k (numeric), judge (LLM)",
    )
    ap.add_argument("--bench", default=None, help="Benchmark name filter (score only this subdirectory)")
    ap.add_argument("--output", default=None, help="Output CSV path (default: <run-dir>/scored.csv)")
    ap.add_argument("--workers", type=int, default=32, help="Parallel workers for LLM judge")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    gold = load_gold(Path(args.answer_key))
    print(f"Loaded {len(gold)} gold answers from {args.answer_key}")

    client = None
    if args.task == "judge":
        client = OpenAI(
            api_key=os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY"),
            timeout=60,
        )

    # Find response folders
    if args.bench:
        # Score a specific benchmark subdirectory
        folders = sorted(run_dir.rglob(args.bench))
        folders = [f for f in folders if f.is_dir()]
    else:
        folders = [run_dir]

    for folder in folders:
        items = read_responses(folder)
        if not items:
            continue

        if args.task == "mc":
            rows = score_mc(items, gold)
        elif args.task == "bbq":
            rows = score_mc(items, gold, bbq=True)
        elif args.task == "gsm8k":
            rows = score_gsm8k(items, gold)
        elif args.task == "judge":
            rows = score_judge(items, gold, client, workers=args.workers)
        else:
            raise ValueError(f"Unknown task: {args.task}")

        if args.output:
            out_path = Path(args.output)
        else:
            out_path = folder / "scored.csv"

        write_scored_csv(rows, out_path)
        correct = sum(1 for r in rows if r["correct"] == "True")
        scored = sum(1 for r in rows if r["correct"] in ("True", "False"))
        acc = correct / scored if scored else 0
        print(f"  {out_path}: {correct}/{scored} = {acc:.1%}")


if __name__ == "__main__":
    main()
