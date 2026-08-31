"""Apply manual extraction fixes to specific May-24 hellaswag CSV rows.

Source: human annotation of extractor_visualizer_hellaswag.html. Items where the
extractor missed an obvious letter get the correct letter; items where the
model didn't answer the question (asked for clarification, gave an ordering,
gave a free-text answer) are marked non-extractable.

Idempotent: re-running rewrites the same fields.
"""
from __future__ import annotations
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HELLA = REPO / "experiments" / "metabench" / "metabench-hellaswag"

# (csv_path, {id: extracted_letter_or_None})
# None means "model did not answer the question" → non-extractable.
FIXES: list[tuple[Path, dict[str, str | None]]] = [
    (HELLA / "data-claude/api/2026-05-24_00-37-27-hellaswag/haiku/claude-haiku-4-5-20251001.csv",
     {"13": "D",   # "The answer is **D...**"
      "23": "A"}), # "**The answer is A.**"

    (HELLA / "data-chatgpt/interface/2026-05-24_00-45-48-hellaswag/session_00/instant.csv",
     {"26": "C",   # ends "Answer: C"
      "33": "B",   # "Answer: B."
      "48": "B",   # "Answer: B"
      "50": "B",   # "Answer: B"
      "67": "B",   # "Answer: B"
      "72": "A"}), # "Answer: A."

    (HELLA / "data-chatgpt/interface/2026-05-24_00-43-29-hellaswag/session_01/thinking.csv",
     {"46": "B"}), # "Answer: B"

    (HELLA / "data-chatgpt/api/2026-05-24_00-45-48-hellaswag/session_03/gpt-5.3-chat-latest.csv",
     {"20": None, # model asked the user for clarification
      "54": None, # gave an ordering "C, D, A, B"
      "64": None, # gave an ordering "B, C, D, A"
      "79": None, # "The correct order is: D, C, A, B."
      "81": None, # gave an ordering "A, D, B, C"
      "88": None, # free-text about "How to play jenga" section
      "93": None}), # free-text about "How to create a looped updo" section
]


def apply_fixes(csv_path: Path, fixes: dict[str, str | None]) -> None:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    touched = 0
    for row in rows:
        rid = (row.get("id") or "").strip()
        if rid not in fixes:
            continue
        new_letter = fixes[rid]
        gold = (row.get("gold_answer") or "").strip().upper()
        if new_letter is None:
            row["answer"] = ""
            row["answer_llm"] = ""
            row["correct"] = "False"
            row["_has_answer"] = "False"
        else:
            row["answer"] = new_letter
            row["answer_llm"] = new_letter
            row["correct"] = str(new_letter.upper() == gold)
            row["_has_answer"] = "True"
        touched += 1

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    rel = csv_path.relative_to(REPO)
    print(f"  patched {touched}/{len(fixes)} rows in {rel}")


def main() -> None:
    for csv_path, fixes in FIXES:
        if not csv_path.exists():
            print(f"  MISSING {csv_path}")
            continue
        apply_fixes(csv_path, fixes)
    print("Done.")


if __name__ == "__main__":
    main()
