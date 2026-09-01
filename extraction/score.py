"""Unified extraction and scoring pipeline: raw response JSON -> scored CSV.

Reads JSON response files (API or interface-scraped), extracts the model's
answer via regex + optional LLM judge (GPT-4o-mini), compares to a gold
answer key, and writes a scored CSV.

Supports:
  - Multiple-choice letter extraction (A-I) for metabench, HellaSwag, etc.
  - BBQ letter extraction (A-C)
  - Numeric answer extraction for GSM8K
  - LLM-judged free-form scoring for AA-Omniscience, SimpleQA, etc.
  - Context-aware MC judging (includes question text to avoid spurious letters)
  - HLE (Humanity's Last Exam) with mixed MC + free-form

Usage:
  # Score metabench MC (regex-only, no API key needed)
  python extraction/score.py \\
      --data-root data/metabench-arc/claude-haiku/api/run_0/responses \\
      --answer-key data/answer_keys/queries.csv --task mc

  # Score BBQ (A-C letter extraction)
  python extraction/score.py \\
      --data-root data/bbq/claude-haiku/api/run_0/responses \\
      --answer-key data/answer_keys/bbq-subset-200.csv --task bbq

  # Score AA-Omniscience (LLM judge, needs OPENAI_KEY)
  python extraction/score.py \\
      --data-root data/aa-omniscience/claude-haiku/api/run_0/responses \\
      --answer-key data/answer_keys/aa-omniscience-subset-200.csv --task judge

  # Score GSM8K (numeric extraction)
  python extraction/score.py \\
      --data-root data/metabench-gsm8k/claude-haiku/api/run_0/responses \\
      --answer-key data/answer_keys/queries.csv --task gsm8k
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_DIR = REPO_ROOT / "data"


# ═══════════════════════════════════════════════════════════════════════════
# Letter extraction (multiple-choice)
# ═══════════════════════════════════════════════════════════════════════════

# Broad MC patterns (A-I): metabench, hellaswag, mmlu, truthfulqa, etc.
MC_FINAL_LABEL = re.compile(
    r"(?i)(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*\**\s*([ABCDEFGHI])\b"
)
MC_BOLD = re.compile(r"\*\*\s*([ABCDEFGHI])\s*[\.\)\*]?")
MC_GENERIC = re.compile(r"(?im)(?:^|[^A-Za-z])([ABCDEFGHI])(?:[\.\)]|\b)")

# Interface-aware patterns from parse.py (lowercase matching)
MC_INTERFACE_PATTERNS = [
    r"(?:correct )?answer\s*(?:is|:)\s*\**([a-z])(?=[^a-z]|$)",
    r"\u2705\s*answer[:\s]+([a-z])(?=[^a-z]|$)",
    r"the (?:correct )?answer\b.*?\bis\s*:?\s*\**([a-z])(?=[^a-z]|$)",
    r"(?:^|\n)\s*([a-z])\)\s*$",
    r"(?:^|\n)\s*([a-z])\.\s",
    r"(?:^|\n)\s*([a-z])\s*$",
]

# BBQ-specific patterns (A-C only)
BBQ_PATTERNS = [
    re.compile(r"\*\*([ABC])\.\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\*\*([ABC])\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:answer is|answer:)\s*\**([ABC])\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^([ABC])\.", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^([ABC])\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\b([ABC])\.", re.IGNORECASE | re.MULTILINE),
]

DEFAULT_VALID_LETTERS = "ABCDEFGHI"



def extract_letter(text: str, task: str = "mc") -> str | None:
    """Extract a multiple-choice letter from model response text.

    Args:
        text: Model response text.
        task: "mc" for broad A-I, "bbq" for A-C only.
    """
    if not text:
        return None
    s = text.strip()

    if task == "bbq":
        for pat in BBQ_PATTERNS:
            m = pat.search(s)
            if m:
                return m.group(1).upper()
        return None

    # Standard MC: try compiled patterns first, then interface-aware patterns
    for r in (MC_FINAL_LABEL, MC_BOLD, MC_GENERIC):
        m = r.search(s)
        if m:
            return m.group(1).upper()

    # Fallback: interface-aware patterns
    text_lower = s.lower()
    for pattern in MC_INTERFACE_PATTERNS:
        match = re.search(pattern, text_lower, re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Last resort: last occurrence of "answer/correct ... <letter>"
    all_matches = re.findall(r"(?:answer|correct)\s*(?:is|:)\s*\**([a-z])(?=[^a-z]|$)", text_lower)
    if all_matches:
        return all_matches[-1].upper()

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
    "You are extracting the answer letter (A\u2013I) from a model's response to a "
    "multiple-choice question.\n"
    "1. If the model clearly selected ONE option \u2014 either by stating the "
    "letter or by paraphrasing the text of one option \u2014 output that uppercase letter.\n"
    "2. If the model did not answer the actual question (asked for clarification, "
    "refused, summarized, or wrote free text not selecting any option) \u2014 output NONE.\n"
    "3. If the model gave a ranked list of multiple letters, output NONE.\n"
    "4. Ignore letters in clarifying menus or explanations of wrong options. "
    "Focus on the FINAL chosen answer.\n\n"
    "Output ONLY the single letter or the word NONE \u2014 nothing else."
)

MC_EXTRACT_SYSTEM = (
    "Extract the final answer choice letter ({first}\u2013{last}) from the given text. "
    "Respond with ONLY a single uppercase letter ({letters}). "
    "If no clear answer is found, respond with 'NONE'."
)

GSM_EXTRACT_SYSTEM = (
    "Extract the final numeric answer from the given math solution. "
    "Respond with ONLY the number (e.g. 42, 3.5, 1500). "
    "Remove any dollar signs, commas, or units. "
    "If no clear answer is found, respond with 'NONE'."
)

HLE_JUDGE_SYSTEM = (
    "You are a strict expert grader for a very hard exam. "
    "Given a question, a gold answer, and a model's extracted final answer, "
    "output 1 if the model answer is correct, or 0 if it is incorrect.\n\n"
    "Rules:\n"
    "- The model answer must match the gold answer exactly or be mathematically equivalent.\n"
    "- Partial answers (e.g. one element of a required set) are INCORRECT (0).\n"
    "- Numerical answers are correct only if they match to reasonable precision.\n"
    "- Do NOT give credit for the right method with a wrong answer.\n"
    "- Output only the digit 1 or 0."
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
        if len(raw) == 1 and raw in DEFAULT_VALID_LETTERS:
            return raw
        return None
    except Exception:
        return None


def llm_extract_mc(client: OpenAI, text: str,
                   valid_letters: str = DEFAULT_VALID_LETTERS) -> str | None:
    """Use LLM to extract a multiple-choice letter (no question context)."""
    if not client or not text:
        return None
    first, last = valid_letters[0], valid_letters[-1]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": MC_EXTRACT_SYSTEM.format(
                    first=first, last=last, letters=", ".join(valid_letters))},
                {"role": "user", "content": text},
            ],
            max_completion_tokens=5,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer if answer in set(valid_letters) else None
    except Exception:
        return None


def llm_extract_number(client: OpenAI, text: str) -> str | None:
    """Use LLM to extract a numeric answer from a GSM8K-style response."""
    if not client or not text:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": GSM_EXTRACT_SYSTEM},
                {"role": "user", "content": text},
            ],
            max_completion_tokens=20,
        )
        answer = resp.choices[0].message.content.strip()
        return None if answer.upper() == "NONE" else normalize_number(answer)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

VALID_ANSWERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_gold(path: Path, task: str = "mc") -> dict[str, dict]:
    """Load an answer key CSV. Returns {id: {answer, query, category, ...}}.

    For GSM8K, extracts the numeric answer from the #### marker.
    """
    gold = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = str(row.get("id", "")).strip()
            if not qid:
                continue
            raw_answer = (row.get("answer") or "").strip()
            if not raw_answer:
                continue

            if task == "gsm8k":
                m = re.search(r"####\s*(-?\d[\d,\.]*)", raw_answer)
                answer = normalize_number(m.group(1)) if m else normalize_number(raw_answer)
            elif task == "hle":
                answer = raw_answer
            else:
                answer = raw_answer.upper() if raw_answer.upper() in VALID_ANSWERS else raw_answer

            gold[qid] = {
                "answer": answer,
                "query": (row.get("query") or "").strip(),
                "category": (row.get("category") or "").strip(),
                "answer_type": (row.get("answer_type") or "").strip(),
            }
    return gold


def read_responses(folder: Path) -> list[dict]:
    """Read all JSON response files from a folder.

    Returns list of {qid, full_qid, response, source} dicts.
    """
    items = []
    for f in sorted(folder.rglob("*.json")):
        if f.name.endswith(".meta.json") or f.name == "parse.json":
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if "response_text" in d:
            text = d.get("response_text") or ""
            source = "api"
        elif "ai_generated_output_text" in d:
            text = d.get("ai_generated_output_text") or ""
            source = "interface"
        else:
            continue
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
            "path": str(f),
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
# Scoring pipelines
# ═══════════════════════════════════════════════════════════════════════════

def score_mc(items: list[dict], gold: dict[str, dict], task: str = "mc",
             client: OpenAI | None = None) -> list[dict]:
    """Score multiple-choice responses (regex + optional LLM fallback)."""
    rows = []
    for item in items:
        g = gold.get(item["qid"])
        if not g:
            continue
        gold_ans = g["answer"].upper()

        # Try regex first, then LLM fallback
        regex = extract_letter(item["response"], task=task)
        llm = llm_extract_mc(client, item["response"]) if client and not regex else None
        pred = regex or llm

        correct = ""
        if pred is not None:
            correct = "True" if pred == gold_ans else "False"
        rows.append({
            "id": item["qid"],
            "run_qid": item["full_qid"],
            "source": item["source"],
            "answer": pred or "",
            "answer_regex": regex or "",
            "answer_llm": llm or "",
            "gold_answer": g["answer"],
            "correct": correct,
        })
    return rows


def score_gsm8k(items: list[dict], gold: dict[str, dict],
                client: OpenAI | None = None) -> list[dict]:
    """Score GSM8K responses (numeric regex + optional LLM fallback)."""
    rows = []
    for item in items:
        g = gold.get(item["qid"])
        if not g:
            continue
        gold_num = g["answer"]
        regex = extract_number(item["response"])
        llm = llm_extract_number(client, item["response"]) if client and not regex else None
        pred = regex or llm

        correct = ""
        if pred is not None:
            correct = "True" if pred == gold_num else "False"
        rows.append({
            "id": item["qid"],
            "run_qid": item["full_qid"],
            "source": item["source"],
            "answer": pred or "",
            "answer_regex": regex or "",
            "answer_llm": llm or "",
            "gold_answer": gold_num,
            "correct": correct,
        })
    return rows


def score_judge(items: list[dict], gold: dict[str, dict],
                client: OpenAI, workers: int = 32) -> list[dict]:
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
            "source": item["source"],
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
        description="Extract answers from raw JSON responses and write scored CSVs."
    )
    ap.add_argument(
        "--data-root",
        default=str(DATA_DIR),
        help="Root directory containing JSON response files (default: data/)",
    )
    ap.add_argument(
        "--answer-key",
        required=True,
        help="Path to answer key CSV (must have 'id' and 'answer' columns).",
    )
    ap.add_argument(
        "--task",
        choices=["mc", "bbq", "gsm8k", "judge"],
        default="mc",
        help="Scoring method: mc (letter A-I), bbq (letter A-C), gsm8k (numeric), judge (LLM)",
    )
    ap.add_argument("--output", default=None, help="Output CSV path (default: <data-root>/scored.csv)")
    ap.add_argument("--workers", type=int, default=32, help="Parallel workers for LLM judge")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (Path.cwd() / data_root).resolve()

    gold = load_gold(Path(args.answer_key), task=args.task)
    print(f"Loaded {len(gold)} gold answers from {args.answer_key}")

    client = None
    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key, timeout=60)
        print("LLM extraction enabled")
    elif args.task == "judge":
        raise SystemExit("--task judge requires OPENAI_KEY or OPENAI_API_KEY env var")
    else:
        print("LLM extraction disabled (no OPENAI_KEY)")

    items = read_responses(data_root)
    if not items:
        print(f"No response JSONs found under {data_root}")
        return

    if args.task in ("mc", "bbq"):
        rows = score_mc(items, gold, task=args.task, client=client)
    elif args.task == "gsm8k":
        rows = score_gsm8k(items, gold, client=client)
    elif args.task == "judge":
        rows = score_judge(items, gold, client, workers=args.workers)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    out_path = Path(args.output) if args.output else (data_root / "scored.csv")
    write_scored_csv(rows, out_path)

    correct = sum(1 for r in rows if r["correct"] == "True")
    scored = sum(1 for r in rows if r["correct"] in ("True", "False"))
    acc = correct / scored if scored else 0
    print(f"Wrote {out_path}: {correct}/{scored} = {acc:.1%}")


if __name__ == "__main__":
    main()
