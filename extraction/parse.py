"""Unified extraction pipeline: raw response JSON -> scored CSV.

Reads JSON response files (API or interface-scraped), extracts the model's
answer via regex + optional LLM judge (GPT-4o-mini), compares to a gold
answer key, and writes a scored CSV per folder.

Usage:
    python extraction/parse.py --data-root data/metabench-arc/claude-haiku/api/run_0/responses \\
        --answer-key data/answer_keys/queries.csv --all-runs

    python extraction/parse.py --data-root data/metabench-arc \\
        --all-runs --answer-key data/answer_keys/queries.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_DIR = REPO_ROOT / "data"

DEFAULT_VALID_LETTERS = "ABCDEFGHI"
VALID_ANSWERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def extract_answer_interface_prefix(text: str) -> str | None:
    """Extract answer from short interface responses.

    Handles two patterns common in scraped interface data:
    1. 'Claude responded: D\\nD' — Claude prefix with single letter
    2. 'D' or 'D\\n...' — bare single letter on first line (ChatGPT/Gemini)

    The LLM extractor sometimes returns wrong letters for these short
    responses, so we extract directly when the pattern is clear.
    """
    if not text:
        return None
    first_line = text.strip().split("\n")[0].strip()
    # Pattern 1: "Claude responded: D"
    m = re.search(r"(?:Claude responded:)\s*([A-Za-z])\s*$",
                  first_line, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Pattern 2: bare single letter (possibly with period)
    m = re.match(r"^([A-Za-z])\.?\s*$", first_line)
    if m:
        return m.group(1).upper()
    return None


def extract_answer_regex_multiple_choice(text: str) -> str | None:
    if not text:
        return None
    text_lower = text.lower()

    patterns = [
        # "answer is A" / "answer: A" — require the letter to be followed by
        # a non-letter (period, paren, space, comma, EOL) so we don't grab
        # letters from English words like "is", "for", "the".
        r"(?:correct )?answer\s*(?:is|:)\s*\**([a-z])(?=[^a-z]|$)",
        r"✅\s*answer[:\s]+([a-z])(?=[^a-z]|$)",
        # "the answer (to ...) is A" — allow intervening words
        r"the (?:correct )?answer\b.*?\bis\s*:?\s*\**([a-z])(?=[^a-z]|$)",
        # Line-start patterns
        r"(?:^|\n)\s*([a-z])\)\s*$",         # "A)" on its own line
        r"(?:^|\n)\s*([a-z])\.\s",           # "A. " at line start (require space after period)
        r"(?:^|\n)\s*([a-z])\s*$",           # bare "A" on its own line
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower, re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Fallback: last occurrence of "answer/correct ... <letter>"
    all_matches = re.findall(r"(?:answer|correct)\s*(?:is|:)\s*\**([a-z])(?=[^a-z]|$)", text_lower)
    if all_matches:
        return all_matches[-1].upper()

    return None


def _normalize_numeric(s: str) -> str:
    """Normalize a numeric string: strip $, commas, trailing .0, leading +."""
    s = s.strip().lstrip("+").replace("$", "").replace(",", "").replace("%", "")
    # Normalize trailing .0 (e.g. '3.0' -> '3')
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def extract_answer_regex_gsm8k(text: str) -> str | None:
    """Extract final numeric/string answer from GSM8K-style solutions.

    Primary pattern: a final line like '#### 200'. Fallback: last number in text.
    """
    if not text:
        return None
    # Prefer explicit '#### answer' marker if present.
    marker_match = re.search(r"####\s*([^\n#]+)", text)
    if marker_match:
        return _normalize_numeric(marker_match.group(1).strip())

    # Fallback: last integer/decimal in the text (handles comma-separated numbers).
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return _normalize_numeric(numbers[-1])
    return None


def extract_answer_llm_multiple_choice(text: str, client: OpenAI | None = None,
                                       valid_letters: str = DEFAULT_VALID_LETTERS) -> str | None:
    if client is None or not text:
        return None
    first, last = valid_letters[0], valid_letters[-1]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Extract the final answer choice letter ({first}–{last}) from the given text. "
                        f"Respond with ONLY a single uppercase letter ({', '.join(valid_letters)}). "
                        "If no clear answer is found, respond with 'NONE'."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_completion_tokens=5,
        )
        answer = response.choices[0].message.content.strip().upper()
        if answer in set(valid_letters):
            return answer
        return None
    except Exception as exc:
        print(f"  LLM extraction error: {exc}")
        return None


def extract_final_answer_hle(text: str, client: OpenAI | None = None) -> str | None:
    """Extract just the final answer from a long HLE model response."""
    if client is None or not text:
        return None
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract only the final answer from the model response. "
                        "Output just the answer itself — a number, letter, expression, or short phrase. "
                        "Do not include any explanation or punctuation beyond the answer."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_completion_tokens=50,
        )
        return response.choices[0].message.content.strip() or None
    except Exception as exc:
        print(f"  LLM extraction error: {exc}")
        return None


def extract_answer_llm_hle(text: str, gold: str, query: str, client: OpenAI | None = None) -> str | None:
    """Use LLM to judge whether a free-form HLE response matches the gold answer. Returns '1' or '0'."""
    if client is None or not text or not gold:
        return None
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict expert grader for a very hard exam. "
                        "Given a question, a gold answer, and a model's extracted final answer, "
                        "output 1 if the model answer is correct, or 0 if it is incorrect.\n\n"
                        "Rules:\n"
                        "- The model answer must match the gold answer exactly or be mathematically equivalent.\n"
                        "- Partial answers (e.g. one element of a required set) are INCORRECT (0).\n"
                        "- Numerical answers are correct only if they match to reasonable precision.\n"
                        "- Do NOT give credit for the right method with a wrong answer.\n"
                        "- Output only the digit 1 or 0."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'Question: "{query}"\n\n'
                        f'Gold Answer: "{gold}"\n\n'
                        f'Model Answer: "{text}"'
                    ),
                },
            ],
            max_completion_tokens=4,
        )
        raw = response.choices[0].message.content.strip()
        for ch in raw:
            if ch in ("0", "1"):
                return ch
        return None
    except Exception as exc:
        print(f"  LLM extraction error: {exc}")
        return None


def extract_answer_llm_gsm8k(text: str, client: OpenAI | None = None) -> str | None:
    """Use LLM to extract the final numeric answer from a GSM8K-style response."""
    if client is None or not text:
        return None
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the final numeric answer from the given math solution. "
                        "Respond with ONLY the number (e.g. 42, 3.5, 1500). "
                        "Remove any dollar signs, commas, or units. "
                        "If no clear answer is found, respond with 'NONE'."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_completion_tokens=20,
        )
        answer = response.choices[0].message.content.strip()
        if answer.upper() == "NONE":
            return None
        return _normalize_numeric(answer)
    except Exception as exc:
        print(f"  LLM extraction error: {exc}")
        return None


def load_text(record: Dict[str, Any]) -> Tuple[str, str]:
    if "response_text" in record:
        return record.get("response_text") or "", "api"
    if "ai_generated_output_text" in record:
        return record.get("ai_generated_output_text") or "", "interface"
    return "", "unknown"


def extract_query_id(record: Dict[str, Any], path: Path) -> str:
    return (
        record.get("query_id")
        or (record.get("meta") or {}).get("query_id")
        or path.stem.replace(".api", "")
    )


def parse_folder(root: Path, client: OpenAI | None = None, task: str = "mc",
                 answer_key: dict | None = None) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "parse.json":
            continue
        if path.name.endswith(".meta.json"):
            # raw_html metadata files are not model outputs
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        text, source = load_text(record)
        _hle_judge = None
        answer_type = "mc"
        if task == "gsm8k":
            answer_regex = extract_answer_regex_gsm8k(text)
            answer_llm = extract_answer_llm_gsm8k(text, client=client)
            answer = answer_llm or answer_regex
        elif task == "hle":
            query_id = extract_query_id(record, path)
            qid = parse_id_from_query_id(query_id)
            gold_entry = (answer_key or {}).get(qid) if qid is not None else None
            gold = gold_entry["answer"] if isinstance(gold_entry, dict) else ""
            query = gold_entry["query"] if isinstance(gold_entry, dict) else ""
            answer_type = gold_entry["answer_type"] if isinstance(gold_entry, dict) else "exactMatch"
            if answer_type == "multipleChoice":
                answer_regex = extract_answer_regex_multiple_choice(text)
                answer_llm = extract_answer_llm_multiple_choice(text, client=client)
                answer = answer_llm or answer_regex
            else:
                answer_regex = None
                answer_llm = None
                answer = extract_final_answer_hle(text, client=client)
                # Store judge verdict separately for correct computation below
                _hle_judge = extract_answer_llm_hle(text, gold, query, client=client)
        else:
            # Try interface prefix extraction first ("Claude responded: D")
            answer_prefix = extract_answer_interface_prefix(text)
            answer_regex = extract_answer_regex_multiple_choice(text)
            answer_llm = extract_answer_llm_multiple_choice(text, client=client)
            answer = answer_prefix or answer_llm or answer_regex
        api_params = record.get("api_params")
        api_params_str = json.dumps(api_params, sort_keys=True) if isinstance(api_params, dict) else ""
        results.append(
            {
                "path": str(path),
                "query_id": extract_query_id(record, path),
                "source": source,
                "answer": answer,
                "answer_regex": answer_regex,
                "answer_llm": answer_llm,
                "_hle_judge": _hle_judge if task == "hle" and answer_type != "multipleChoice" else None,
                "api_params": api_params_str,
            }
        )
    return results


def write_csv(rows: list[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "id",
                "source",
                "answer",
                "answer_regex",
                "answer_llm",
                "gold_answer",
                "correct",
                "api_params",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _has_parseable_files(folder: Path) -> bool:
    for p in folder.glob("*.json"):
        if p.name == "parse.json":
            continue
        if p.name.endswith(".meta.json"):
            continue
        return True
    return False


def _is_data_folder(folder: Path) -> bool:
    if folder.name == "csv":
        return False
    return _has_parseable_files(folder)


def _find_parseable_folders(root: Path) -> list[Path]:
    candidates = []
    for folder in root.rglob("*"):
        if folder.is_dir() and _is_data_folder(folder):
            candidates.append(folder)
    return sorted(candidates)



def load_answer_key(path: Path, task: str = "mc") -> dict[int, str | dict]:
    if not path.exists():
        print(f"Answer key not found: {path}")
        return {}
    key: dict[int, str | dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                qid = int((row.get("id") or "").strip())
            except Exception:
                continue

            raw_answer = (row.get("answer") or "").strip()
            if not raw_answer:
                continue

            if task == "gsm8k":
                # GSM8K answers are embedded in a solution text, typically ending with '#### 200'.
                extracted = extract_answer_regex_gsm8k(raw_answer)
                key[qid] = extracted or raw_answer
            elif task == "hle":
                answer_type = (row.get("answer_type") or "exactMatch").strip()
                query = (row.get("query") or "").strip()
                key[qid] = {"answer": raw_answer, "answer_type": answer_type, "query": query}
            else:
                ans = raw_answer.upper()
                if ans in VALID_ANSWERS:
                    key[qid] = ans
    return key


def parse_id_from_query_id(query_id: str) -> int | None:
    if not query_id:
        return None
    match = re.match(r"(\d+)_", query_id)
    if match:
        return int(match.group(1))
    return None


def parse_id_from_row(row: Dict[str, Any]) -> int | None:
    qid = parse_id_from_query_id(row.get("query_id") or "")
    if qid is not None:
        return qid
    return parse_id_from_query_id(str(row.get("id") or ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract answers from raw JSON responses and write scored CSVs."
    )
    parser.add_argument(
        "--data-root",
        default=str(DATA_DIR),
        help="Root directory containing JSON response files (default: data/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV directory (default: same as --data-root)",
    )
    parser.add_argument(
        "--answer-key",
        required=True,
        help="Path to answer key CSV (must have 'id' and 'answer' columns).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        default=True,
        help="Parse all runs under data-root (default: True).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (Path.cwd() / data_root).resolve()

    client = None
    if os.environ.get("OPENAI_KEY"):
        client = OpenAI(api_key=os.environ.get("OPENAI_KEY"), timeout=30)
        print("LLM extraction enabled")
    else:
        print("LLM extraction disabled (no OPENAI_KEY)")

    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()
    else:
        output_dir = data_root
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_key_path = Path(args.answer_key)
    if not answer_key_path.is_absolute():
        answer_key_path = (Path.cwd() / answer_key_path).resolve()

    # Detect task type from answer key path.
    key_path_lower = str(answer_key_path).lower()
    if "gsm8k" in key_path_lower:
        task = "gsm8k"
        print(f"Detected GSM8K-style task from answer key: {answer_key_path}")
    elif "hle" in key_path_lower:
        task = "hle"
        print(f"Detected HLE-style task from answer key: {answer_key_path}")
    else:
        task = "mc"

    answer_key = load_answer_key(answer_key_path, task=task)

    if _has_parseable_files(data_root):
        folders = [data_root]
    else:
        folders = _find_parseable_folders(data_root)

    for folder in folders:
        try:
            rel = folder.relative_to(data_root)
            parts = list(rel.parts)
            if parts and parts[0] == "parsed_json":
                parts[0] = "interface"
            norm_rel = Path(*parts) if parts else rel
            output_path = output_dir / norm_rel.parent / f"{norm_rel.name}.csv"
        except ValueError:
            output_path = output_dir / f"{folder.name}.csv"

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"Skipping {folder.name}: already parsed at {output_path}")
            continue

        rows = parse_folder(folder, client=client, task=task, answer_key=answer_key)
        for row in rows:
            qid = parse_id_from_row(row)
            row["id"] = qid if qid is not None else ""
            gold_entry = answer_key.get(qid) if qid is not None else None
            if task == "hle":
                gold = gold_entry["answer"] if isinstance(gold_entry, dict) else ""
                row["gold_answer"] = gold
                answer_type = gold_entry["answer_type"] if isinstance(gold_entry, dict) else "exactMatch"
                judge = row.pop("_hle_judge", None)
                if answer_type == "multipleChoice":
                    row["correct"] = bool(row.get("answer") and gold and row["answer"] == gold)
                else:
                    row["correct"] = judge == "1"
            else:
                row.pop("_hle_judge", None)
                gold = gold_entry
                row["gold_answer"] = gold or ""
                row["correct"] = bool(
                    row.get("answer") and gold and _normalize_numeric(row["answer"]) == _normalize_numeric(gold)
                ) if task == "gsm8k" else bool(row.get("answer") and gold and row["answer"] == gold)
        write_csv(rows, output_path)

        extracted = sum(1 for r in rows if r.get("answer"))
        discrepancies = sum(
            1
            for r in rows
            if r.get("answer_regex")
            and r.get("answer_llm")
            and r.get("answer_regex") != r.get("answer_llm")
        )
        print(f"Parsed {len(rows)} json files in {folder.name}, extracted {extracted} answers.")
        print(f"Discrepancies (regex vs LLM): {discrepancies}")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
