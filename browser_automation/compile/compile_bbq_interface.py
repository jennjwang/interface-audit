"""
Compile BBQ interface-only responses from Claude and Gemini runs.

Gathers responses from parsed_json (interface) directories only,
and writes per-model CSVs to:
  /Users/jenniferwang/PhD/personalization/experiments/bbq/interface/{model}/responses.csv

Each CSV has columns: id, category, query, response
"""
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
QUERY_DIR   = SCRIPT_DIR.parent / "benchmark_creation" / "results"
OUTPUT_DIR  = Path("/Users/jenniferwang/PhD/personalization/experiments/bbq/interface")

BBQ_QUERY_FILE = QUERY_DIR / "bbq-subset-200.csv"


def load_query_map(csv_path):
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[str(row["id"])] = row
    return rows


MODELS = {
    "claude-haiku":  ("haiku",   False),
    "claude-opus":   ("opus",    False),
    "claude-sonnet": ("sonnet",  False),
    "gemini-fast":     ("fast",     True),
    "gemini-thinking": ("thinking", True),
}


def collect_interface_responses(parsed_json_root, dir_prefix, query_ids):
    """Scan all parsed_json runs and return {query_id: response_text}."""
    responses = {}
    root = Path(parsed_json_root)
    if not root.exists():
        return responses
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        for session_dir in sorted(run_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for exp_dir in sorted(session_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                # Match by prefix (e.g haiku-*, fast-*)
                if not exp_dir.name.startswith(dir_prefix):
                    continue
                for jf in sorted(exp_dir.glob("*.json")):
                    # filename is {query_id}_run0.json
                    qid = jf.stem.replace("_run0", "")
                    if qid in responses or qid not in query_ids:
                        continue
                    try:
                        d = json.loads(jf.read_text(encoding="utf-8"))
                        text = d.get("ai_generated_output_text", "").strip()
                        if text:
                            responses[qid] = text
                    except Exception:
                        continue
    return responses


def write_csv(path, query_map, responses):
    path.parent.mkdir(parents=True, exist_ok=True)
    written = missing = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "query", "response"])
        writer.writeheader()
        for qid, row in query_map.items():
            resp = responses.get(qid, "")
            if resp:
                written += 1
            else:
                missing += 1
            writer.writerow({
                "id": qid,
                "category": row.get("category", ""),
                "query": row.get("query", ""),
                "response": resp,
            })
    return written, missing


def main():
    queries = load_query_map(BBQ_QUERY_FILE)
    query_ids = set(queries.keys())

    claude_parsed = SCRIPT_DIR / "claude_data" / "parsed_json"
    gemini_parsed = SCRIPT_DIR / "gemini_data" / "parsed_json"

    for model_name, (iface_prefix, is_gemini) in MODELS.items():
        parsed_root = gemini_parsed if is_gemini else claude_parsed
        responses = collect_interface_responses(parsed_root, iface_prefix, query_ids)

        out_path = OUTPUT_DIR / model_name / "responses.csv"
        written, missing = write_csv(out_path, queries, responses)

        print(f"{model_name}: {written}/200 responses ({missing} missing) -> {out_path}")


if __name__ == "__main__":
    main()
