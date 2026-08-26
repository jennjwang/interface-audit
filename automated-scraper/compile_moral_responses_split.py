"""Compile OG and FLIP responses separately for interface and API sources.

Outputs to:
  outputs/moral_sycophancy_interface/{model}/{og,flip}.csv
  outputs/moral_sycophancy_api/{model}/{og,flip}.csv
"""
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
QUERY_DIR = SCRIPT_DIR.parent / "benchmark_creation" / "results"
OUTPUT_BASE = SCRIPT_DIR / "outputs"

OG_QUERY_FILE = QUERY_DIR / "elephant-moral-og-100.csv"
FLIP_QUERY_FILE = QUERY_DIR / "elephant-moral-flip-100.csv"

MODELS = {
    "claude-haiku": ("haiku", "claude-haiku"),
    "claude-opus": ("opus", "claude-opus"),
    "claude-sonnet": ("sonnet", "claude-sonnet"),
    "gemini-fast": ("fast", "thinking_level-low"),
    "gemini-thinking": ("thinking", "thinking_level-high"),
}


def load_query_map(csv_path):
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[str(row["id"])] = row
    return rows


def collect_interface(parsed_root, dir_prefix, og_ids, flip_ids):
    og, flip = {}, {}
    if not parsed_root.exists():
        return og, flip
    for run_dir in sorted(parsed_root.iterdir()):
        if not run_dir.is_dir(): continue
        for sess in sorted(run_dir.iterdir()):
            if not sess.is_dir(): continue
            for exp in sorted(sess.iterdir()):
                if not exp.is_dir() or not exp.name.startswith(dir_prefix):
                    continue
                for jf in sorted(exp.glob("*.json")):
                    qid = jf.stem.replace("_run0", "")
                    if qid in og and qid in flip: continue
                    try:
                        d = json.loads(jf.read_text(encoding="utf-8"))
                        text = d.get("ai_generated_output_text", "").strip()
                        if not text: continue
                        if qid in og_ids and qid not in og:
                            og[qid] = text
                        elif qid in flip_ids and qid not in flip:
                            flip[qid] = text
                    except Exception: continue
    return og, flip


def collect_api(api_root, model_substring, og_ids, flip_ids):
    og, flip = {}, {}
    if not api_root.exists():
        return og, flip
    for run_dir in sorted(api_root.iterdir()):
        if not run_dir.is_dir(): continue
        for sub in sorted(run_dir.iterdir()):
            if not sub.is_dir(): continue
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
                        if not text: continue
                        if qid in og_ids and qid not in og:
                            og[qid] = text
                        elif qid in flip_ids and qid not in flip:
                            flip[qid] = text
                    except Exception: continue
    return og, flip


def write_csv(path, query_map, responses):
    path.parent.mkdir(parents=True, exist_ok=True)
    written = missing = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dataset", "query", "response"])
        w.writeheader()
        for qid, row in query_map.items():
            resp = responses.get(qid, "")
            if resp: written += 1
            else: missing += 1
            w.writerow({"id": qid, "dataset": row.get("dataset",""),
                        "query": row.get("query",""), "response": resp})
    return written, missing


def collect_api_flat(api_dir, og_ids, flip_ids):
    """Collect API responses from a flat dir of *.api.json files."""
    og, flip = {}, {}
    if not api_dir.exists():
        return og, flip
    for jf in sorted(api_dir.glob("*.api.json")):
        qid = jf.stem.replace(".api", "").replace("_run0", "")
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            text = (d.get("response_text") or "").strip()
            if not text: continue
            if qid in og_ids and qid not in og:
                og[qid] = text
            elif qid in flip_ids and qid not in flip:
                flip[qid] = text
        except Exception:
            continue
    return og, flip


def collect_interface_flat(interface_dir, og_ids, flip_ids):
    """Collect interface responses from a flat dir of *.json files."""
    og, flip = {}, {}
    if not interface_dir.exists():
        return og, flip
    for jf in sorted(interface_dir.glob("*.json")):
        meta = {}
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            meta = d.get("meta", {}) if isinstance(d.get("meta"), dict) else {}
            raw_id = meta.get("query_id", jf.stem)
            qid = str(raw_id).replace("_run0", "")
            text = (d.get("ai_generated_output_text") or "").strip()
            if not text: continue
            if qid in og_ids and qid not in og:
                og[qid] = text
            elif qid in flip_ids and qid not in flip:
                flip[qid] = text
        except Exception:
            continue
    return og, flip


def main():
    og_q = load_query_map(OG_QUERY_FILE)
    flip_q = load_query_map(FLIP_QUERY_FILE)
    og_ids, flip_ids = set(og_q), set(flip_q)

    EXP_ELEPHANT = SCRIPT_DIR.parent / "experiments" / "elephant" / "data"
    claude_p = SCRIPT_DIR / "claude_data" / "parsed_json"
    gemini_p = SCRIPT_DIR / "gemini_data" / "parsed_json"
    claude_a = SCRIPT_DIR / "claude_data" / "api"
    gemini_a = SCRIPT_DIR / "gemini_data" / "api"

    print(f"\n{'Model':<22} {'Interface OG':>13} {'Interface FLIP':>15} {'API OG':>10} {'API FLIP':>10}")
    print("-" * 74)

    for model, (prefix, api_sub) in MODELS.items():
        is_gemini = model.startswith("gemini")
        parsed = gemini_p if is_gemini else claude_p
        api = gemini_a if is_gemini else claude_a

        iog, iflip = collect_interface(parsed, prefix, og_ids, flip_ids)
        aog, aflip = collect_api(api, api_sub, og_ids, flip_ids)

        iog_w, _ = write_csv(OUTPUT_BASE / "moral_sycophancy_interface" / model / "og.csv", og_q, iog)
        ifl_w, _ = write_csv(OUTPUT_BASE / "moral_sycophancy_interface" / model / "flip.csv", flip_q, iflip)
        aog_w, _ = write_csv(OUTPUT_BASE / "moral_sycophancy_api" / model / "og.csv", og_q, aog)
        afl_w, _ = write_csv(OUTPUT_BASE / "moral_sycophancy_api" / model / "flip.csv", flip_q, aflip)
        print(f"{model:<22} {iog_w:>13} {ifl_w:>15} {aog_w:>10} {afl_w:>10}")

    # Claude + Gemini + ChatGPT: data is in experiments/elephant/data/
    new_models = {
        "claude-haiku": (
            EXP_ELEPHANT / "claude/api/claude-haiku-4-5-20251001_web_search-False",
            EXP_ELEPHANT / "claude/interface/haiku",
        ),
        "claude-opus": (
            EXP_ELEPHANT / "claude/api/claude-opus-4-7_web_search-False",
            EXP_ELEPHANT / "claude/interface/opus",
        ),
        "claude-sonnet": (
            EXP_ELEPHANT / "claude/api/claude-sonnet-4-6_web_search-False",
            EXP_ELEPHANT / "claude/interface/sonnet",
        ),
        "gemini-fast": (
            EXP_ELEPHANT / "gemini/api/gemini-3-flash-preview_thinking_level-low_web_search-False",
            EXP_ELEPHANT / "gemini/interface/fast",
        ),
        "gemini-thinking": (
            EXP_ELEPHANT / "gemini/api/gemini-3-flash-preview_thinking_level-high_web_search-False",
            EXP_ELEPHANT / "gemini/interface/thinking",
        ),
        "chatgpt-instant": (
            EXP_ELEPHANT / "chatgpt/api/gpt-5.3-chat-latest_web_search-False",
            EXP_ELEPHANT / "chatgpt/interface/gpt-5-3-instant",
        ),
        "chatgpt-thinking": (
            EXP_ELEPHANT / "chatgpt/api/gpt-5.4-2026-03-05_reasoning_effort-high_web_search-False",
            EXP_ELEPHANT / "chatgpt/interface/gpt-5-4-thinking",
        ),
    }
    for model, (api_dir, iface_dir) in new_models.items():
        aog, aflip = collect_api_flat(api_dir, og_ids, flip_ids)
        iog, iflip = collect_interface_flat(iface_dir, og_ids, flip_ids)

        iog_w, _ = write_csv(OUTPUT_BASE / "moral_sycophancy_interface" / model / "og.csv", og_q, iog)
        ifl_w, _ = write_csv(OUTPUT_BASE / "moral_sycophancy_interface" / model / "flip.csv", flip_q, iflip)
        aog_w, _ = write_csv(OUTPUT_BASE / "moral_sycophancy_api" / model / "og.csv", og_q, aog)
        afl_w, _ = write_csv(OUTPUT_BASE / "moral_sycophancy_api" / model / "flip.csv", flip_q, aflip)
        print(f"{model:<22} {iog_w:>13} {ifl_w:>15} {aog_w:>10} {afl_w:>10}")


if __name__ == "__main__":
    main()
