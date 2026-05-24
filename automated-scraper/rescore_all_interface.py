"""Re-score all interface CSVs with the context-aware LLM judge.

The original judge in parse.py only passes response text (no question context),
which causes spurious letter extraction from clarifying menus or document
summaries when models misinterpret the benchmark prompt.  This script re-runs
every item through a context-aware judge (gpt-4o-mini) using the OpenAI Batch
API — all requests submitted in one JSONL file, 50% cheaper, no thread pool.

After updating CSVs, run openllm_leaderboard.py to regenerate coverage_by_run*.csv.

Usage:
  python automated-scraper/rescore_all_interface.py
  python automated-scraper/rescore_all_interface.py --bench hellaswag --provider claude
  python automated-scraper/rescore_all_interface.py --dry-run   # no writes, just stats
  python automated-scraper/rescore_all_interface.py --poll BATCH_ID  # resume a submitted batch

Options:
  --bench       Filter to one benchmark (arc, gsm8k, hellaswag, mmlu, truthfulqa, winogrande)
  --provider    Filter to one provider   (claude, chatgpt, gemini)
  --dry-run     Count items, do not submit
  --poll ID     Skip submission; poll an already-submitted batch and write results
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parent
REPO = BASE.parent

load_dotenv(BASE / ".env")

# ── Benchmark metadata ──────────────────────────────────────────────────────

BENCH_META = {
    "arc":        {"queries": "metabench-arc/queries/queries.csv",        "task": "mc"},
    "gsm8k":      {"queries": "metabench-gsm8k/queries/queries.csv",      "task": "gsm8k"},
    "hellaswag":  {"queries": "metabench-hellaswag/queries/queries.csv",  "task": "mc"},
    "mmlu":       {"queries": "metabench-mmlu/queries/queries.csv",       "task": "mc"},
    "truthfulqa": {"queries": "metabench-truthfulQA/queries/queries.csv", "task": "mc"},
    "winogrande": {"queries": "metabench-winogrande/queries/queries.csv", "task": "mc"},
}

METABENCH = REPO / "experiments" / "metabench"

MC_JUDGE_SYSTEM = (
    "You are extracting the answer letter (A–I) from a model's response to a "
    "multiple-choice question.\n\n"
    "Output rules:\n"
    "1. If the model clearly selected ONE option — either by stating the "
    "letter (e.g. 'Answer: B') or by stating or paraphrasing the text of one "
    "of the options — output the single uppercase letter for that option.\n"
    "2. If the model did not answer the actual question — e.g. it asked the "
    "user for clarification, refused, summarized instead, or wrote free text "
    "that does not select any of the lettered options — output NONE.\n"
    "3. If the model gave a ranked or ordered list of multiple letters "
    "(e.g. 'C, D, A, B' or 'The correct order is D, C, A, B'), output NONE — "
    "an ordering is not a single selection.\n"
    "4. Ignore letters appearing in clarifying menus, document summaries, or "
    "explanations of why OTHER options are wrong. Focus on the FINAL chosen "
    "answer.\n\n"
    "Output ONLY the single letter or the word NONE — nothing else."
)

GSM_JUDGE_SYSTEM = (
    "You are extracting the final numeric answer from a model's response to a "
    "math word problem. Focus on the model's final answer, ignoring intermediate "
    "steps or clarifying questions. Output only the number (e.g. 42, 3.5, 1500) "
    "or NONE if the model gave no clear numeric answer."
)

VALID_LETTERS = set("ABCDEFGHI")

GSM_RE = re.compile(r"####\s*(-?\d[\d,\.]*)")
GSM_NUM = re.compile(r"(?<![\d\.,])-?\d[\d,\.]*")


def _normalize_num(s: str) -> str:
    s = s.strip().replace(",", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s


# ── Gold answer loading ─────────────────────────────────────────────────────

def load_gold(bench: str) -> dict[str, dict]:
    meta = BENCH_META[bench]
    path = METABENCH / meta["queries"]
    gold: dict[str, dict] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            gold[str(r["id"])] = {"query": r["query"], "answer": r["answer"]}
    return gold


# ── Batch request building ──────────────────────────────────────────────────

def make_batch_request(custom_id: str, query: str, response: str, task: str) -> dict:
    if task == "mc":
        tail = query[-700:].strip()
        user = f"Question (end of prompt):\n{tail}\n\nModel's response:\n{response}"
        body = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": MC_JUDGE_SYSTEM},
                {"role": "user",   "content": user},
            ],
            "max_completion_tokens": 4,
            "temperature": 0,
        }
    else:
        tail = query[-400:].strip()
        user = f"Math problem (end):\n{tail}\n\nModel's response:\n{response}"
        body = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": GSM_JUDGE_SYSTEM},
                {"role": "user",   "content": user},
            ],
            "max_completion_tokens": 16,
            "temperature": 0,
        }
    return {"custom_id": custom_id, "method": "POST",
            "url": "/v1/chat/completions", "body": body}


def parse_verdict(raw: str, task: str) -> str:
    raw = raw.strip()
    if task == "mc":
        raw = raw.upper()
        if len(raw) == 1 and raw in VALID_LETTERS:
            return raw
        if raw == "NONE":
            return "NONE"
        for c in raw:
            if c in VALID_LETTERS:
                return c
        return "NONE"
    else:
        if raw.upper() == "NONE" or not raw:
            return "NONE"
        return _normalize_num(raw)


def answers_match(pred: str, gold: str, task: str) -> bool:
    if pred in ("NONE", "ERR", ""):
        return False
    if task == "gsm8k":
        # Gold in queries.csv is "#### 72" format; extract the number first
        gold_m = re.search(r"####\s*(-?\d[\d,\.]*)", gold)
        gold_num = _normalize_num(gold_m.group(1)) if gold_m else _normalize_num(gold)
        return _normalize_num(pred) == gold_num
    return pred.upper() == gold.upper()


# ── Path resolution ────────────────────────────────────────────────────────

def _resolve_json_path(p: Path) -> Path | None:
    """Resolve a parsed-JSON path, handling two known reorganizations:
    1. experiments/metabench-X/ → experiments/metabench/metabench-X/
    2. .../data/parsed_json/ → .../data-{provider}/parsed_json/
       (some early ChatGPT CSVs wrote 'data/' instead of 'data-chatgpt/')
    """
    if p.exists():
        return p
    s = str(p)
    # Fix 1: metabench-X subdirectory reorganization
    fixed = re.sub(
        r"(experiments)/metabench-([^/]+)/",
        r"\1/metabench/metabench-\2/",
        s,
    )
    if fixed != s:
        fp = Path(fixed)
        if fp.exists():
            return fp
        # Fix 2: stale 'data/' prefix written before provider-specific dirs
        for provider in ("chatgpt", "gemini", "claude"):
            fp2 = Path(fixed.replace("/data/parsed_json/", f"/data-{provider}/parsed_json/"))
            if fp2 != fp and fp2.exists():
                return fp2
    return None


# ── CSV loading / writing ──────────────────────────────────────────────────

def load_csv_items(csv_path: Path, gold: dict[str, dict]) -> tuple[list, list, list]:
    """Return (fieldnames, rows, inputs) where inputs=(qid, query, response)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    inputs: list[tuple[str, str, str]] = []
    for row in rows:
        qid = str(row.get("id", "")).strip()
        if qid not in gold:
            continue
        path_str = row.get("path", "")
        if not path_str:
            continue
        jp = _resolve_json_path(Path(path_str))
        if jp is None:
            continue
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        resp = d.get("ai_generated_output_text") or d.get("response_text") or ""
        if resp:
            inputs.append((qid, gold[qid]["query"], resp))

    return fieldnames, rows, inputs


def write_csv(csv_path: Path, fieldnames: list, rows: list,
              gold: dict[str, dict], task: str, verdicts: dict[str, str],
              dry_run: bool) -> dict:
    old_correct = new_correct = old_total = new_total = 0
    for row in rows:
        qid = str(row.get("id", "")).strip()
        gold_ans = (gold.get(qid, {}).get("answer") or "").strip()
        old_letter = (row.get("answer_llm") or "").strip()
        old_ok = answers_match(old_letter, gold_ans, task)
        if old_letter:
            old_total += 1
            if old_ok:
                old_correct += 1

        if qid not in verdicts:
            continue
        new_letter = verdicts[qid]
        new_ok = answers_match(new_letter, gold_ans, task)
        new_total += 1
        if new_ok:
            new_correct += 1

        row["answer_llm"] = new_letter if new_letter not in ("NONE", "ERR") else ""
        row["answer"] = row["answer_llm"]
        row["correct"] = str(new_ok)
        row["_has_answer"] = str(bool(row["answer"]))

    if not dry_run and new_total > 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    n_none = sum(1 for v in verdicts.values() if v == "NONE")
    return {
        "old_correct": old_correct, "old_total": old_total,
        "new_correct": new_correct, "new_total": new_total,
        "n_none": n_none, "n_judged": len(verdicts),
    }


# ── Discovery ──────────────────────────────────────────────────────────────

def find_csvs(bench_filter: str | None,
              provider_filter: str | None,
              kinds: set[str],
              ts_filter: str | None = None) -> list[tuple[Path, str, str]]:
    """Return list of (csv_path, bench_slug, task).

    kinds: subset of {"interface", "api"}.
    ts_filter: optional substring; only timestamps containing it are kept.
    """
    results = []
    for bench_slug, meta in BENCH_META.items():
        if bench_filter and bench_filter.lower() != bench_slug:
            continue
        bench_dir = METABENCH / f"metabench-{bench_slug}"
        if not bench_dir.exists():
            bench_dir = METABENCH / f"metabench-{bench_slug.replace('truthfulqa', 'truthfulQA')}"
        if not bench_dir.exists():
            continue
        for provider_dir in bench_dir.iterdir():
            if not provider_dir.is_dir():
                continue
            if not provider_dir.name.startswith("data"):
                continue
            if provider_filter:
                if provider_filter.lower() not in provider_dir.name.lower():
                    continue
            for kind in sorted(kinds):
                kind_root = provider_dir / kind
                if not kind_root.exists():
                    continue
                for csv_path in sorted(kind_root.rglob("*.csv")):
                    if "score_per" in csv_path.name or "joint_var" in csv_path.name:
                        continue
                    if ts_filter:
                        # The timestamp dir is the child of kind_root.
                        try:
                            ts_dir = csv_path.relative_to(kind_root).parts[0]
                        except ValueError:
                            ts_dir = ""
                        if ts_filter not in ts_dir:
                            continue
                    results.append((csv_path, bench_slug, meta["task"]))
    return results


# Backwards-compatible alias for the previous interface-only entry point.
def find_interface_csvs(bench_filter, provider_filter):
    return find_csvs(bench_filter, provider_filter, kinds={"interface"})


# ── Batch polling / writing ─────────────────────────────────────────────────

def _retry(fn, *, what: str, max_attempts: int = 8, backoff: float = 5.0):
    """Call fn() with retries on network/connection errors. Returns fn's result."""
    from openai import APIConnectionError, APITimeoutError
    import httpx
    transient = (APIConnectionError, APITimeoutError,
                 httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError,
                 ConnectionError, OSError)
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except transient as exc:
            if attempt == max_attempts:
                raise
            delay = backoff * attempt
            print(f"  [retry {attempt}/{max_attempts}] {what}: {type(exc).__name__}: {exc}; "
                  f"sleeping {delay:.0f}s", flush=True)
            time.sleep(delay)


def poll_and_write(client: OpenAI, batch_id: str,
                   csv_meta: list, gold_cache: dict,
                   id_map: dict[str, tuple[int, str, str]],
                   dry_run: bool) -> None:
    """Poll until batch completes, then parse results and write CSVs."""
    print(f"Polling batch {batch_id}...", flush=True)
    while True:
        batch = _retry(lambda: client.batches.retrieve(batch_id),
                       what="batches.retrieve")
        counts = batch.request_counts
        print(f"  status={batch.status}  "
              f"completed={counts.completed}/{counts.total}  "
              f"failed={counts.failed}", flush=True)
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(30)

    if batch.status != "completed":
        raise SystemExit(f"Batch ended with status: {batch.status}")

    # Download and parse results
    raw = _retry(lambda: client.files.content(batch.output_file_id).text,
                 what="files.content")
    verdicts: dict[int, dict[str, str]] = {i: {} for i in range(len(csv_meta))}
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]
        csv_idx, qid, task = id_map[custom_id]
        try:
            raw_text = obj["response"]["body"]["choices"][0]["message"]["content"] or ""
        except Exception:
            raw_text = ""
        verdicts[csv_idx][qid] = parse_verdict(raw_text, task)

    # Write back per CSV
    print("Writing CSVs...", flush=True)
    totals = {"old_correct": 0, "old_total": 0, "new_correct": 0, "new_total": 0, "n_none": 0}
    for csv_idx, (csv_path, fieldnames, rows, bench_slug, task) in enumerate(csv_meta):
        gold = gold_cache[bench_slug]
        rel = csv_path.relative_to(METABENCH)
        stats = write_csv(csv_path, fieldnames, rows, gold, task,
                          verdicts[csv_idx], dry_run)
        if not stats or stats["n_judged"] == 0:
            print(f"  {rel}: (no scorable rows)")
            continue
        old_acc = stats["old_correct"] / stats["old_total"] if stats["old_total"] else 0
        new_acc = stats["new_correct"] / stats["new_total"] if stats["new_total"] else 0
        delta = new_acc - old_acc
        sign  = "+" if delta >= 0 else ""
        print(f"  {rel}:  "
              f"old={stats['old_correct']}/{stats['old_total']}={old_acc:.1%}  "
              f"new={stats['new_correct']}/{stats['new_total']}={new_acc:.1%}  "
              f"Δ={sign}{delta*100:.1f}pp  NONE={stats['n_none']}/{stats['n_judged']}")
        for k in totals:
            totals[k] += stats.get(k, 0)

    print(f"\n{'='*60}")
    print(f"TOTAL across {len(csv_meta)} CSVs:")
    old_acc = totals["old_correct"] / totals["old_total"] if totals["old_total"] else 0
    new_acc = totals["new_correct"] / totals["new_total"] if totals["new_total"] else 0
    print(f"  Old: {totals['old_correct']}/{totals['old_total']} = {old_acc:.1%}")
    print(f"  New: {totals['new_correct']}/{totals['new_total']} = {new_acc:.1%}")
    print(f"  NONE: {totals['n_none']}/{totals['new_total']}")
    if dry_run:
        print("\n(dry-run: no files written)")
    else:
        print("\nCSVs updated. Re-run openllm_leaderboard.py to regenerate coverage_by_run*.csv")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench",    default=None, help="Filter to one benchmark")
    ap.add_argument("--provider", default=None, help="Filter to one provider")
    ap.add_argument("--kinds",    default="interface",
                    help="Comma-separated subset of {interface, api}")
    ap.add_argument("--ts-filter", default=None,
                    help="Only process timestamps containing this substring (e.g. 2026-05-24)")
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--poll",     default=None, metavar="BATCH_ID",
                    help="Resume: poll existing batch ID and write results")
    args = ap.parse_args()
    kinds = {k.strip().lower() for k in args.kinds.split(",") if k.strip()}
    unknown = kinds - {"interface", "api"}
    if unknown:
        raise SystemExit(f"Unknown --kinds entry: {unknown}")

    api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_KEY not set")
    client = OpenAI(api_key=api_key, timeout=60)

    csvs = find_csvs(args.bench, args.provider, kinds, ts_filter=args.ts_filter)
    print(f"Found {len(csvs)} CSVs (kinds={sorted(kinds)}, "
          f"ts_filter={args.ts_filter!r})", flush=True)

    # Pre-load gold
    gold_cache: dict[str, dict] = {}
    for _, bench_slug, _ in csvs:
        if bench_slug not in gold_cache:
            gold_cache[bench_slug] = load_gold(bench_slug)

    # Load all CSV rows
    print("Loading CSVs...", flush=True)
    csv_meta: list[tuple[Path, list, list, str, str]] = []
    # id_map: custom_id → (csv_idx, qid, task)
    id_map: dict[str, tuple[int, str, str]] = {}
    batch_requests: list[dict] = []

    for csv_idx, (csv_path, bench_slug, task) in enumerate(csvs):
        gold = gold_cache[bench_slug]
        fieldnames, rows, inputs = load_csv_items(csv_path, gold)
        csv_meta.append((csv_path, fieldnames, rows, bench_slug, task))
        for qid, query, response in inputs:
            custom_id = f"c{csv_idx}_q{qid}"
            id_map[custom_id] = (csv_idx, qid, task)
            batch_requests.append(make_batch_request(custom_id, query, response, task))

    print(f"Total items to judge: {len(batch_requests)}", flush=True)

    if args.dry_run:
        print("(dry-run: not submitting)")
        return

    if args.poll:
        # Resume existing batch
        poll_and_write(client, args.poll, csv_meta, gold_cache, id_map, dry_run=False)
        return

    # Build JSONL and upload
    print("Uploading batch file...", flush=True)
    jsonl_bytes = "\n".join(json.dumps(r) for r in batch_requests).encode()
    batch_file = client.files.create(
        file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
        purpose="batch",
    )
    print(f"  file_id={batch_file.id}  size={len(jsonl_bytes)//1024}KB", flush=True)

    # Submit batch
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch submitted: {batch.id}", flush=True)
    print(f"  To resume if interrupted: --poll {batch.id}", flush=True)

    poll_and_write(client, batch.id, csv_meta, gold_cache, id_map, dry_run=False)


if __name__ == "__main__":
    main()
