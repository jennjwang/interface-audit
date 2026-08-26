"""
Compile OG and FLIP responses from Claude and Gemini runs for moral sycophancy scoring.

Gathers responses from parsed_json (interface) and api directories across all runs,
deduplicates by query ID, and writes per-model CSVs to:
  outputs/moral_sycophancy/{model}/og.csv
  outputs/moral_sycophancy/{model}/flip.csv

Each CSV has columns: id, dataset, query, response
"""
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
QUERY_DIR   = SCRIPT_DIR.parent / "benchmark_creation" / "results"
OUTPUT_DIR  = SCRIPT_DIR / "outputs" / "moral_sycophancy"

OG_QUERY_FILE   = QUERY_DIR / "elephant-moral-og-100.csv"
FLIP_QUERY_FILE = QUERY_DIR / "elephant-moral-flip-100.csv"


def load_query_map(csv_path):
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[str(row["id"])] = row
    return rows


# Map from logical model name → (interface dir prefix, api model dir substring)
MODELS = {
    # Claude interface + API
    "claude-haiku":  ("haiku",   "claude-haiku"),
    "claude-opus":   ("opus",    "claude-opus"),
    "claude-sonnet": ("sonnet",  "claude-sonnet"),
    # Gemini interface + API
    "gemini-fast":     ("fast",     "thinking_level-low"),
    "gemini-thinking": ("thinking", "thinking_level-high"),
}


def collect_interface_responses(parsed_json_root, dir_prefix, og_ids, flip_ids):
    """Scan all parsed_json runs and return {query_id: response_text} for OG and FLIP."""
    og, flip = {}, {}
    root = Path(parsed_json_root)
    if not root.exists():
        return og, flip
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        for session_dir in sorted(run_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for exp_dir in sorted(session_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                # Match by prefix (e.g. haiku-*, fast-*)
                if not exp_dir.name.startswith(dir_prefix):
                    continue
                for jf in sorted(exp_dir.glob("*.json")):
                    # filename is {query_id}_run0.json
                    qid = jf.stem.replace("_run0", "")
                    if qid in og and qid in flip:
                        continue
                    try:
                        d = json.loads(jf.read_text(encoding="utf-8"))
                        text = d.get("ai_generated_output_text", "").strip()
                        if not text:
                            continue
                        if qid in og_ids and qid not in og:
                            og[qid] = text
                        elif qid in flip_ids and qid not in flip:
                            flip[qid] = text
                    except Exception:
                        continue
    return og, flip


def collect_api_responses(api_root, model_substring, og_ids, flip_ids):
    """Scan api directory for a model and return {query_id: response_text}."""
    og, flip = {}, {}
    root = Path(api_root)
    if not root.exists():
        return og, flip
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
                    try:
                        d = json.loads(jf.read_text(encoding="utf-8"))
                        text = (d.get("response_text") or "").strip()
                        if not text:
                            continue
                        if qid in og_ids and qid not in og:
                            og[qid] = text
                        elif qid in flip_ids and qid not in flip:
                            flip[qid] = text
                    except Exception:
                        continue
    return og, flip


def write_csv(path, query_map, responses, dataset_col="dataset"):
    path.parent.mkdir(parents=True, exist_ok=True)
    written = missing = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "dataset", "query", "response"])
        writer.writeheader()
        for qid, row in query_map.items():
            resp = responses.get(qid, "")
            if resp:
                written += 1
            else:
                missing += 1
            writer.writerow({
                "id": qid,
                "dataset": row.get("dataset", ""),
                "query": row.get("query", ""),
                "response": resp,
            })
    return written, missing


def main():
    og_queries   = load_query_map(OG_QUERY_FILE)
    flip_queries = load_query_map(FLIP_QUERY_FILE)
    og_ids   = set(og_queries.keys())
    flip_ids = set(flip_queries.keys())

    # Data roots
    claude_parsed = SCRIPT_DIR / "claude_data" / "parsed_json"
    gemini_parsed = SCRIPT_DIR / "gemini_data" / "parsed_json"
    claude_api    = SCRIPT_DIR / "claude_data" / "api"
    gemini_api    = SCRIPT_DIR / "gemini_data" / "api"

    for model_name, (iface_prefix, api_substr) in MODELS.items():
        is_gemini = model_name.startswith("gemini")
        parsed_root = gemini_parsed if is_gemini else claude_parsed
        api_root    = gemini_api    if is_gemini else claude_api

        # Collect from interface (parsed HTML)
        iog, iflip = collect_interface_responses(parsed_root, iface_prefix, og_ids, flip_ids)
        # Collect from API
        aog, aflip = collect_api_responses(api_root, api_substr, og_ids, flip_ids)

        # Merge: interface takes priority, API fills gaps
        og_resp   = {**aog,   **iog}
        flip_resp = {**aflip, **iflip}

        og_path   = OUTPUT_DIR / model_name / "og.csv"
        flip_path = OUTPUT_DIR / model_name / "flip.csv"

        ow, om = write_csv(og_path,   og_queries,   og_resp)
        fw, fm = write_csv(flip_path, flip_queries, flip_resp)

        print(f"{model_name}:")
        print(f"  OG:   {ow}/100 responses  ({om} missing) -> {og_path}")
        print(f"  FLIP: {fw}/100 responses  ({fm} missing) -> {flip_path}")


if __name__ == "__main__":
    main()
