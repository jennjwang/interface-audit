"""
Compile BBQ API-only responses from Claude and Gemini runs.

Gathers responses from api directories only,
and writes per-model CSVs to:
  /Users/jenniferwang/PhD/personalization/experiments/bbq/api/{model}/responses.csv

Each CSV has columns: id, category, query, response
"""
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
QUERY_DIR   = SCRIPT_DIR.parent / "benchmark_creation" / "results"
OUTPUT_DIR  = Path("/Users/jenniferwang/PhD/personalization/experiments/bbq/api")

BBQ_QUERY_FILE = QUERY_DIR / "bbq-subset-200.csv"


def load_query_map(csv_path):
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[str(row["id"])] = row
    return rows


MODELS = {
    "claude-haiku":  ("claude-haiku", False),
    "claude-opus":   ("claude-opus", False),
    "claude-sonnet": ("claude-sonnet", False),
    "gemini-fast":     ("thinking_level-low", True),
    "gemini-thinking": ("thinking_level-high", True),
}


def collect_api_responses(api_root, model_substring, query_ids):
    """Scan api directory for a model and return {query_id: response_text}."""
    responses = {}
    root = Path(api_root)
    if not root.exists():
        return responses
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        for sub in sorted(run_dir.iterdir()):
            if not sub.is_dir():
                continue
            # flat: api/{run}/{model_dir}/*.api.json
            # rotate: api/{run}/{benchmark}/{model_dir}/*.api.json
            candidates = []
            if model_substring in sub.name:
                candidates.append(sub)
            else:
                for child in sub.iterdir():
                    if child.is_dir() and model_substring in child.name:
                        candidates.append(child)
            for model_dir in candidates:
                for jf in sorted(model_dir.glob("*.api.json")):
                    qid = jf.stem.replace(".api", "").replace("_run0", "")
                    if qid in responses or qid not in query_ids:
                        continue
                    try:
                        d = json.loads(jf.read_text(encoding="utf-8"))
                        text = (d.get("response_text") or "").strip()
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

    claude_api = SCRIPT_DIR / "claude_data" / "api"
    gemini_api = SCRIPT_DIR / "gemini_data" / "api"

    for model_name, (api_substr, is_gemini) in MODELS.items():
        api_root = gemini_api if is_gemini else claude_api
        responses = collect_api_responses(api_root, api_substr, query_ids)

        out_path = OUTPUT_DIR / model_name / "responses.csv"
        written, missing = write_csv(out_path, queries, responses)

        print(f"{model_name}: {written}/200 responses ({missing} missing) -> {out_path}")


if __name__ == "__main__":
    main()
