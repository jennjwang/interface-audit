"""Score metabench subsets across all (model, benchmark) cells under a run_id.

Usage:
  python automated-scraper/score_metabench.py --run-id batch-system-prompt-metabench

Prints a per-(model, benchmark) accuracy table and dumps a CSV summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_API = BASE / "data" / "api"

METABENCH = {
    "metabench_mmlu":        BASE.parent / "experiments/metabench/metabench-mmlu/queries/queries.csv",
    "metabench_arc":         BASE.parent / "experiments/metabench/metabench-arc/queries/queries.csv",
    "metabench_gsm8k":       BASE.parent / "experiments/metabench/metabench-gsm8k/queries/queries.csv",
    "metabench_hellaswag":   BASE.parent / "experiments/metabench/metabench-hellaswag/queries/queries.csv",
    "metabench_truthfulqa":  BASE.parent / "experiments/metabench/metabench-truthfulQA/queries/queries.csv",
    "metabench_winogrande":  BASE.parent / "experiments/metabench/metabench-winogrande/queries/queries.csv",
}

# Letter-extraction benchmarks. truthfulQA goes up to I (9 options).
LETTER_RE = re.compile(r"(?im)(?:^|[^A-Za-z])([ABCDEFGHI])(?:[\.\)]|\b)")
FINAL_LABEL_RE = re.compile(r"(?i)(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*\**\s*([ABCDEFGHI])\b")
BOLD_RE = re.compile(r"\*\*\s*([ABCDEFGHI])\s*[\.\)\*]?")
def extract_letter(text: str) -> str | None:
    if not text:
        return None
    for r in (FINAL_LABEL_RE, BOLD_RE, LETTER_RE):
        m = r.search(text.strip())
        if m:
            return m.group(1).upper()
    return None


# gsm8k: gold ends with "#### NUMBER"; predictions vary.
GSM_GOLD_RE = re.compile(r"####\s*(-?\d[\d,\.]*)")
GSM_EXPLICIT_REGEXES = [
    re.compile(r"####\s*(-?\d[\d,\.]*)"),
    re.compile(r"\\boxed\{\s*(-?\d[\d,\.]*)"),
    re.compile(r"(?i)(?:final\s+answer|the\s+answer\s+is|answer\s+is)\s*[:=]?\s*\$?\s*(-?\d[\d,\.]*)"),
]
# Match a number that is NOT followed by another digit/comma/dot (so we don't
# clip the leading digit off a multi-digit number).
GSM_NUM_TOKEN = re.compile(r"(?<![\d\.,])-?\d[\d,\.]*")
def _num(s: str) -> float | None:
    if s is None: return None
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except Exception:
        return None
def extract_number(text: str) -> float | None:
    """Get the final numeric answer from a gsm8k response.

    Strategy: prefer explicit markers (#### N, \\boxed{N}, "the answer is N");
    otherwise take the LAST number that appears in the response.
    """
    if not text: return None
    for r in GSM_EXPLICIT_REGEXES:
        ms = list(r.finditer(text))
        if ms:
            v = _num(ms[-1].group(1))
            if v is not None: return v
    nums = GSM_NUM_TOKEN.findall(text)
    return _num(nums[-1]) if nums else None


def load_gold(path: Path) -> dict[str, str]:
    gold: dict[str, str] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            gold[str(r["id"])] = r["answer"]
    return gold


def score_cell(folder: Path, gold: dict[str, str], bench: str) -> tuple[int, int, int]:
    total = correct = unparsed = 0
    is_gsm = bench == "metabench_gsm8k"
    if is_gsm:
        gold_num = {qid: _num(GSM_GOLD_RE.search(a).group(1)) if GSM_GOLD_RE.search(a) else None
                    for qid, a in gold.items()}
    for p in folder.glob("*_run0.api.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        qid = str(d.get("query_id","")).replace("_run0","")
        if qid not in gold:
            continue
        total += 1
        if d.get("error"):
            continue
        text = d.get("response_text") or ""
        if is_gsm:
            pred = extract_number(text)
            g = gold_num.get(qid)
            if pred is None:
                unparsed += 1; continue
            if g is not None and abs(pred - g) < 1e-6:
                correct += 1
        else:
            pred = extract_letter(text)
            if pred is None:
                unparsed += 1; continue
            if pred == gold[qid].upper():
                correct += 1
    return total, correct, unparsed


def short_name(model_dir: str) -> str:
    s = model_dir.lower()
    if "claude-opus" in s: return "claude-opus-4-7"
    if "claude-sonnet" in s: return "claude-sonnet-4-6"
    if "gpt-5.3" in s: return "gpt-5.3-chat-latest"
    if "gpt-5.4" in s: return "gpt-5.4-thinking"
    if "gemini" in s: return "gemini-3-flash"
    return model_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default=None, help="Optional CSV path to write summary")
    args = ap.parse_args()

    run_dir = DATA_API / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    gold_per_bench = {b: load_gold(p) for b, p in METABENCH.items()}

    rows = []
    print(f"{'Model':22s} | " + " | ".join(b.replace('metabench_','')[:9].ljust(9) for b in METABENCH))
    print("-" * 100)
    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir(): continue
        line = f"{short_name(model_dir.name):22s} | "
        cells = []
        for bench in METABENCH:
            cell_dir = model_dir / "session_00" / bench
            if not cell_dir.exists():
                cells.append("   --   ")
                rows.append({"model": short_name(model_dir.name), "bench": bench, "n": 0, "correct": 0, "acc": ""})
                continue
            t, c, u = score_cell(cell_dir, gold_per_bench[bench], bench)
            if t == 0:
                cells.append("   --   ")
                rows.append({"model": short_name(model_dir.name), "bench": bench, "n": 0, "correct": 0, "acc": ""})
                continue
            acc = c / t
            cells.append(f"{c:3d}/{t:3d}".ljust(7) + f"{acc:>5.1%}".strip()[:5])
            rows.append({"model": short_name(model_dir.name), "bench": bench, "n": t, "correct": c, "acc": f"{acc:.4f}"})
        # Better aligned print: just acc
        accs = []
        for bench in METABENCH:
            cell_dir = model_dir / "session_00" / bench
            if not cell_dir.exists():
                accs.append("   --  ")
                continue
            t, c, u = score_cell(cell_dir, gold_per_bench[bench], bench)
            if t == 0:
                accs.append("   --  ")
            else:
                accs.append(f"{c/t:.1%}".rjust(7) + f" ({t})".ljust(2))
        print(f"{short_name(model_dir.name):22s} | " + " | ".join(a.ljust(9) for a in accs))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model","bench","n","correct","acc"])
            w.writeheader(); w.writerows(rows)
        print(f"\nSummary CSV: {out_path}")


if __name__ == "__main__":
    main()
