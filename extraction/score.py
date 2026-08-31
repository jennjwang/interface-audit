"""Score tinymmlu CSV outputs and report grouped accuracy."""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_DIR = DATA_DIR / "csv"
ANSWER_KEY_PATH = BASE_DIR.parent.parent / "automated-scraper" / "queries" / "tinymmlu.csv"


VALID_ANSWERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _normalize_numeric(s: str) -> str:
    """Normalize a numeric string: strip $, commas, trailing .0, leading +."""
    s = s.strip().lstrip("+").replace("$", "").replace(",", "").replace("%", "")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def _answers_match(ans: str, gold: str) -> bool:
    """Compare answers, using numeric normalization for non-letter answers."""
    if ans == gold:
        return True
    if gold.upper() in VALID_ANSWERS:
        return ans.upper() == gold.upper()
    return _normalize_numeric(ans) == _normalize_numeric(gold)



def _extract_gsm8k_answer(text: str) -> str | None:
    """Extract numeric answer from GSM8K-style solution text (#### marker or last number)."""
    import re
    if not text:
        return None
    marker = re.search(r"####\s*([^\n#]+)", text)
    if marker:
        return _normalize_numeric(marker.group(1).strip())
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return _normalize_numeric(numbers[-1])
    return None


def load_answer_key(path: Path) -> dict[int, str]:
    if not path.exists():
        print(f"Answer key not found: {path}")
        return {}
    is_gsm8k = "gsm8k" in str(path).lower()
    key: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                qid = int(row.get("id", "").strip())
            except Exception:
                continue
            raw_answer = (row.get("answer") or "").strip()
            if not raw_answer:
                continue
            if is_gsm8k:
                extracted = _extract_gsm8k_answer(raw_answer)
                if extracted:
                    key[qid] = extracted
            else:
                ans = raw_answer.upper()
                if ans in VALID_ANSWERS:
                    key[qid] = ans
    return key


def parse_id_from_query_id(query_id: str) -> int | None:
    if not query_id:
        return None
    parts = query_id.split("_", 1)
    if not parts:
        return None
    try:
        return int(parts[0])
    except Exception:
        return None


def parse_id_from_row(row: Dict[str, Any]) -> int | None:
    qid = parse_id_from_query_id(row.get("query_id") or "")
    if qid is not None:
        return qid
    return parse_id_from_query_id(str(row.get("id") or ""))


PREFERRED_CATEGORIES = [
    "gpt-5.2-chat-latest",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
]


def _slugify_category(category: str) -> str:
    return (
        category.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )


def _ordered_categories(categories: set[str]) -> list[str]:
    ordered = [c for c in PREFERRED_CATEGORIES if c in categories]
    remainder = sorted(c for c in categories if c not in set(PREFERRED_CATEGORIES))
    return ordered + remainder


def _categorize_csv(csv_path: Path, csv_root: Path) -> str | None:
    try:
        rel = csv_path.relative_to(csv_root)
        parts = rel.parts
    except Exception:
        parts = csv_path.parts

    for idx, part in enumerate(parts):
        if part.lower() == "api" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def score_existing_csvs(csv_root: Path, answer_key: dict[int, str]) -> None:
    if not csv_root.exists():
        print(f"CSV directory not found: {csv_root}")
        return

    csv_files = sorted(csv_root.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found under {csv_root}")
        return

    # Skip derived scoring outputs to avoid re-scoring them.
    skip_names = {
        "score_per_question.csv",
        "score_joint_variance_var_gt0.csv",
        "score_per_question_gpt-52-chat-latest_api.csv",
        "score_per_question_interface_gpt-52_temporary_chat.csv",
        "score_per_question_interface_gpt-52non-temporary.csv",
    }

    grouped: dict[str, list[tuple[Path, int, int]]] = {}
    per_question: dict[str, dict[int, list[int]]] = {}

    for csv_path in csv_files:
        if csv_path.name in skip_names or "score_per_question" in {p.lower() for p in csv_path.parts}:
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        if not rows:
            continue

        for row in rows:
            qid = parse_id_from_row(row)
            gold = answer_key.get(qid) if qid is not None else None
            row["id"] = qid if qid is not None else row.get("id", "")
            row["gold_answer"] = gold or ""
            answer_llm = (row.get("answer_llm") or "").strip()
            answer_regex = (row.get("answer_regex") or "").strip()
            ans = answer_llm or (row.get("answer") or "").strip() or answer_regex
            row["answer"] = ans
            has_answer = bool(ans)
            row["_has_answer"] = has_answer
            if has_answer and gold:
                row["correct"] = _answers_match(ans, gold)
            elif gold:
                row["correct"] = False
            else:
                row["correct"] = ""

        for col in ("id", "gold_answer", "correct", "_has_answer"):
            if col not in fieldnames:
                fieldnames.append(col)

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        correct = sum(1 for r in rows if r.get("_has_answer") and str(r.get("correct")) == "True")
        total = sum(1 for r in rows if r.get("_has_answer"))
        print(f"Scored {csv_path}: {correct}/{total} correct")

        category = _categorize_csv(csv_path, csv_root)
        if category:
            grouped.setdefault(category, []).append((csv_path, correct, total))
            bucket = per_question.setdefault(category, {})
            for row in rows:
                if not row.get("_has_answer"):
                    continue
                qid = parse_id_from_row(row)
                if qid is None:
                    continue
                score = 1 if str(row.get("correct")) == "True" else 0
                bucket.setdefault(qid, []).append(score)

    if grouped:
        print("\nOrganized scores:")
        for category in _ordered_categories(set(grouped.keys())):
            entries = grouped.get(category, [])
            if not entries:
                continue
            total_correct = sum(item[1] for item in entries)
            total_items = sum(item[2] for item in entries)
            accuracy = (total_correct / total_items) if total_items else 0.0
            print(f"- {category}: {total_correct}/{total_items} ({accuracy:.1%})")
            for csv_path, correct, total in entries:
                print(f"  {csv_path}: {correct}/{total}")

    if per_question:
        print("\nPer-question stats (avg and variance):")
        for category in _ordered_categories(set(per_question.keys())):
            questions = per_question.get(category, {})
            if not questions:
                continue
            print(f"- {category}")
            for qid in sorted(questions):
                scores = questions[qid]
                avg = sum(scores) / len(scores)
                var = statistics.pvariance(scores) if len(scores) > 1 else 0.0
                print(f"  {qid}: avg={avg:.3f}, var={var:.3f}, n={len(scores)}")

    return per_question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-root",
        default=str(CSV_DIR),
        help="Root directory to search for CSVs (default: data/csv)",
    )
    parser.add_argument(
        "--answer-key",
        default=str(ANSWER_KEY_PATH),
        help="Path to tinymmlu answer key CSV",
    )
    parser.add_argument(
        "--stats-out",
        default=str(CSV_DIR / "score_per_question"),
        help=(
            "Output directory or CSV prefix for per-question stats "
            "(default: data/csv/score_per_question)"
        ),
    )
    args = parser.parse_args()

    csv_root = Path(args.csv_root)
    if not csv_root.is_absolute():
        csv_root = BASE_DIR / csv_root

    answer_key_path = Path(args.answer_key)
    if not answer_key_path.is_absolute():
        answer_key_path = BASE_DIR / answer_key_path

    answer_key = load_answer_key(answer_key_path)
    per_question = score_existing_csvs(csv_root, answer_key)

    if per_question:
        stats_out = Path(args.stats_out)
        if not stats_out.is_absolute():
            stats_out = BASE_DIR / stats_out

        if stats_out.suffix == ".csv":
            out_dir = stats_out.parent
            prefix = stats_out.stem
        else:
            out_dir = stats_out
            prefix = "score_per_question"

        out_dir.mkdir(parents=True, exist_ok=True)

        for category in _ordered_categories(set(per_question.keys())):
            questions = per_question.get(category, {})
            if not questions:
                continue
            rows = []
            for qid in sorted(questions):
                scores = questions[qid]
                avg = sum(scores) / len(scores)
                var = statistics.pvariance(scores) if len(scores) > 1 else 0.0
                rows.append(
                    {
                        "category": category,
                        "question_id": qid,
                        "avg": avg,
                        "variance": var,
                        "n": len(scores),
                    }
                )

            rows.sort(key=lambda r: (-r["variance"], r["question_id"]))

            slug = _slugify_category(category)
            output_path = out_dir / f"{prefix}_{slug}.csv"
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["category", "question_id", "avg", "variance", "n"],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            "category": row["category"],
                            "question_id": row["question_id"],
                            "avg": f"{row['avg']:.6f}",
                            "variance": f"{row['variance']:.6f}",
                            "n": row["n"],
                        }
                    )
            print(f"\nWrote per-question stats CSV: {output_path}")

        categories = _ordered_categories(set(per_question.keys()))
        category_keys = {category: _slugify_category(category) for category in categories}
        all_qids = set()
        for category in categories:
            all_qids.update(per_question.get(category, {}).keys())

        joint_rows_var_gt0 = []
        for qid in sorted(all_qids):
            row = {"question_id": qid}
            has_variance = False
            for category in categories:
                key = category_keys[category]
                scores = per_question.get(category, {}).get(qid)
                if scores:
                    avg = sum(scores) / len(scores)
                    var = statistics.pvariance(scores) if len(scores) > 1 else 0.0
                    n = len(scores)
                    row[f"avg_{key}"] = f"{avg:.6f}"
                    row[f"variance_{key}"] = f"{var:.6f}"
                    row[f"n_{key}"] = n
                    if var > 0:
                        has_variance = True
                else:
                    row[f"avg_{key}"] = ""
                    row[f"variance_{key}"] = ""
                    row[f"n_{key}"] = ""
            if has_variance:
                joint_rows_var_gt0.append(row)

        joint_var_path = out_dir / "score_joint_variance_var_gt0.csv"
        joint_var_fields = ["question_id", "category", "variance", "avg", "n"]
        flattened_rows = []
        for row in joint_rows_var_gt0:
            qid = row["question_id"]
            for category in categories:
                key = category_keys[category]
                var = row.get(f"variance_{key}")
                if not var:
                    continue
                try:
                    if float(var) <= 0:
                        continue
                except Exception:
                    continue
                flattened_rows.append(
                    {
                        "question_id": qid,
                        "category": category,
                        "variance": var,
                        "avg": row.get(f"avg_{key}", ""),
                        "n": row.get(f"n_{key}", ""),
                    }
                )
        flattened_rows.sort(key=lambda r: (r["question_id"], r["category"]))
        with joint_var_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=joint_var_fields)
            writer.writeheader()
            writer.writerows(flattened_rows)
        print(f"Wrote joint variance CSV (variance > 0): {joint_var_path}")


if __name__ == "__main__":
    main()
