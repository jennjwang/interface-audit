"""Judge SimpleQA responses from interface (parsed_json) and/or API sync runs.

Vendor data directories (relative to this script):
  claude_data/    – Claude  (audit_claude.py)
  chatgpt_data/   – ChatGPT (audit_chatgpt.py)
  gemini_data/    – Gemini  (audit_gemini.py)

Response paths written by the audit scripts (sync mode):
  {vendor}_data/api/{run_id}/session_00/simpleqa/*.api.json
  {vendor}_data/parsed_json/{run_id}/session_00/simpleqa/*.json

Usage:
  # Exact run_id (Claude, whose --run-id is deterministic):
  python judge_simpleqa.py --vendor claude --run-id simpleqa-claude-opus

  # Tag search (ChatGPT/Gemini use --output-tag, so run_id has a timestamp prefix):
  python judge_simpleqa.py --vendor chatgpt --tag simpleqa-gpt54-thinking
  python judge_simpleqa.py --vendor gemini  --tag simpleqa-gemini-pro

  # Score all vendors at once:
  python judge_simpleqa.py \\
    --vendor claude   --run-id simpleqa-claude-opus \\
    --vendor chatgpt  --tag simpleqa-gpt54-thinking \\
    --vendor gemini   --tag simpleqa-gemini-pro
"""
from __future__ import annotations
import argparse, csv, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parents[1]
SIMPLEQA_CSV = BASE.parent / "benchmark_creation/results/simpleqa-verified-subset-100.csv"

load_dotenv(BASE / ".env")

VENDOR_DATA = {
    "claude":   BASE / "claude_data",
    "chatgpt":  BASE / "chatgpt_data",
    "gemini":   BASE / "gemini_data",
}

BENCH = "simpleqa"

JUDGE_SYSTEM = (
    "You are a strict expert grader. "
    "Given a question, a gold answer, and a model's response, "
    "output 1 if the response is correct, or 0 if it is incorrect.\n\n"
    "Rules:\n"
    "- The response is correct if it contains or clearly implies the gold answer.\n"
    "- Numerical answers must match to reasonable precision.\n"
    "- Minor wording differences, alternate spellings, or name transliterations are OK "
    "if the meaning is clearly the same.\n"
    "- If the model says it does not know or declines to answer, output 0.\n"
    "- Output only the digit 1 or 0."
)


# ── Gold loading ───────────────────────────────────────────────────────────────

def load_gold(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out[str(r["id"])] = {
                "query":       r["query"],
                "answer":      r["answer"],
                "topic":       r.get("topic", ""),
                "answer_type": r.get("answer_type", ""),
            }
    return out


# ── Response loading ───────────────────────────────────────────────────────────

def _load_api_responses(cell: Path) -> dict[str, str]:
    """Load *.api.json files → {full_qid: response_text}."""
    out = {}
    for f in sorted(cell.glob("*.api.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        qid = str(d.get("query_id", f.stem.replace(".api", "")))
        out[qid] = d.get("response_text") or ""
    return out


def _load_interface_responses(cell: Path) -> dict[str, str]:
    """Load *.json (parsed interface) files → {full_qid: response_text}."""
    out = {}
    for f in sorted(cell.glob("*.json")):
        if f.name.endswith(".api.json"):
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        # query_id lives in meta or can be inferred from filename
        qid = str(
            (d.get("meta") or {}).get("query_id")
            or f.stem
        )
        out[qid] = d.get("ai_generated_output_text") or ""
    return out


# ── Run discovery ──────────────────────────────────────────────────────────────

def find_run_id(vendor: str, run_id: str | None, tag: str | None) -> str | None:
    """Return the matching run_id inside {vendor}_data/, or None."""
    data_dir = VENDOR_DATA[vendor]

    if run_id:
        return run_id  # caller guarantees it exists

    if not tag:
        return None

    # Search api/ and parsed_json/ for directories whose name contains the tag
    candidates: list[tuple[str, Path]] = []
    for subdir in ("api", "parsed_json"):
        base = data_dir / subdir
        if not base.exists():
            continue
        for entry in base.iterdir():
            if entry.is_dir() and tag in entry.name:
                candidates.append((entry.name, entry))

    if not candidates:
        return None
    # Pick the most-recently modified (latest timestamp prefix wins)
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][0]


# ── Judging ────────────────────────────────────────────────────────────────────

def judge_one(client: OpenAI, query: str, gold: str, response: str) -> str | None:
    if not response or not gold:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content":
                    f'Question: "{query}"\n\nGold Answer: "{gold}"\n\nModel Response: "{response}"'},
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


def score_responses(
    client: OpenAI,
    responses: dict[str, str],       # {full_qid: response_text}
    gold: dict[str, dict],
    out_path: Path,
    workers: int,
) -> None:
    rows = []
    judge_inputs = []

    for full_qid, resp in sorted(responses.items()):
        base_qid = full_qid.split("_run")[0]
        if base_qid not in gold:
            continue
        g = gold[base_qid]
        rows.append({
            "id":          base_qid,
            "run_qid":     full_qid,
            "topic":       g["topic"],
            "answer_type": g["answer_type"],
            "gold_answer": g["answer"],
            "response":    resp,
            "correct":     "",
        })
        if resp:
            judge_inputs.append((full_qid, g["query"], g["answer"], resp))

    if not rows:
        print("  No scoreable rows.")
        return

    verdicts: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(judge_one, client, q, ans, rr): rqid
            for (rqid, q, ans, rr) in judge_inputs
        }
        done = 0
        for fut in as_completed(futs):
            verdicts[futs[fut]] = fut.result()
            done += 1
            if done % 100 == 0 or done == len(futs):
                print(f"    judged {done}/{len(futs)}", flush=True)

    for r in rows:
        v = verdicts.get(r["run_qid"])
        if v in ("0", "1"):
            r["correct"] = "True" if v == "1" else "False"

    correct = sum(1 for r in rows if r["correct"] == "True")
    scored  = sum(1 for r in rows if r["correct"] in ("True", "False"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "run_qid", "topic", "answer_type", "gold_answer", "response", "correct"]
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    acc = correct / scored if scored else 0
    print(f"  → {out_path.name}: {correct}/{scored} = {acc:.1%}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor",  action="append", choices=list(VENDOR_DATA), dest="vendors",
                    help="Vendor to score (repeatable)")
    ap.add_argument("--run-id",  action="append", dest="run_ids",  default=[],
                    metavar="RUN_ID", help="Exact run_id (pairs with --vendor in order)")
    ap.add_argument("--tag",     action="append", dest="tags",     default=[],
                    metavar="TAG",    help="Partial tag search (pairs with --vendor in order)")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    if not args.vendors:
        ap.error("Provide at least one --vendor.")

    # Build (vendor, run_id_or_None, tag_or_None) triples
    runs_ids_padded = (args.run_ids + [None] * len(args.vendors))[:len(args.vendors)]
    tags_padded     = (args.tags    + [None] * len(args.vendors))[:len(args.vendors)]
    targets = list(zip(args.vendors, runs_ids_padded, tags_padded))

    gold = load_gold(SIMPLEQA_CSV)
    print(f"Loaded {len(gold)} gold items from {SIMPLEQA_CSV.name}\n")

    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY"),
        timeout=60,
    )

    for vendor, run_id, tag in targets:
        data_dir = VENDOR_DATA[vendor]
        resolved = find_run_id(vendor, run_id, tag)
        if not resolved:
            print(f"[{vendor}] No run found for run_id={run_id!r} tag={tag!r}")
            continue
        print(f"=== {vendor} / {resolved} ===")

        out_dir = data_dir / "judged"

        # ── Interface responses ────────────────────────────────────────────────
        iface_root = data_dir / "parsed_json" / resolved
        iface_files = list(iface_root.rglob("*.json")) if iface_root.exists() else []
        iface_files = [f for f in iface_files if not f.name.endswith(".api.json")]
        if iface_files:
            out_csv = out_dir / f"{resolved}__simpleqa_interface.csv"
            if out_csv.exists() and out_csv.stat().st_size > 0:
                print(f"  Skip (exists): {out_csv.name}")
            else:
                print(f"  Judging {len(iface_files)} interface responses...")
                iface_resp = {}
                for f in iface_files:
                    try:
                        d = json.loads(f.read_text())
                    except Exception:
                        continue
                    qid = str((d.get("meta") or {}).get("query_id") or f.stem)
                    iface_resp[qid] = d.get("ai_generated_output_text") or ""
                score_responses(client, iface_resp, gold, out_csv, args.workers)
        else:
            print(f"  No interface responses under {iface_root}")

        # ── API responses ──────────────────────────────────────────────────────
        api_root = data_dir / "api" / resolved
        api_files = list(api_root.rglob("*.api.json")) if api_root.exists() else []
        if api_files:
            out_csv = out_dir / f"{resolved}__simpleqa_api.csv"
            if out_csv.exists() and out_csv.stat().st_size > 0:
                print(f"  Skip (exists): {out_csv.name}")
            else:
                print(f"  Judging {len(api_files)} API responses...")
                api_resp = {}
                for f in api_files:
                    try:
                        d = json.loads(f.read_text())
                    except Exception:
                        continue
                    qid = str(d.get("query_id") or f.stem.replace(".api", ""))
                    api_resp[qid] = d.get("response_text") or ""
                score_responses(client, api_resp, gold, out_csv, args.workers)
        else:
            print(f"  No API responses at {api_cell}")

        print()


if __name__ == "__main__":
    main()
