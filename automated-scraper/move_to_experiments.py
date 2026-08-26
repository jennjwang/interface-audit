"""Move completed scraper data into the experiments/<benchmark>/data/<provider>/ tree.

Run this AFTER scraping is complete (or as a checkpoint) to consolidate data
from the scraper's flat run-timestamp layout into the canonical per-experiment
layout used downstream:

    experiments/<benchmark>/data/<provider>/
      api/<model_dir>/<id>_run0.api.json          (newest timestamp wins on conflict)
      interface/<short_name>/<id>_run0.json
      raw_html/<short_name>/<id>_run0.html (+ .meta.json)

Source layout from scrapers:
    automated-scraper/<provider>_data/api/<run>/.../<model_dir>/<id>_run0.api.json
    automated-scraper/<provider>_data/parsed_json/<run>/session_*/<exp>-<bench>/<id>_run0.json
    automated-scraper/<provider>_data/raw_html/<run>/session_*/<exp>-<bench>/<id>_run0.html

Usage:
    python move_to_experiments.py                  # all providers, all benchmarks
    python move_to_experiments.py --providers chatgpt
    python move_to_experiments.py --benchmarks bbq elephant
    python move_to_experiments.py --dry-run        # preview without copying
"""
import argparse
import csv
import shutil
from pathlib import Path

SCRAPER_DIR = Path(__file__).parent
EXP_DIR = SCRAPER_DIR.parent / "experiments"
QUERY_DIR = SCRAPER_DIR.parent / "benchmark_creation" / "results"

# Map benchmark short → (query CSV stem, target experiment dir name)
BENCHMARKS = {
    "bbq":            ("bbq-subset-200", "bbq"),
    "aa-omniscience": ("aa-omniscience-subset-200", "aa-omniscience"),
    "elephant-og":    ("elephant-moral-og-100", "elephant"),
    "elephant-flip":  ("elephant-moral-flip-100", "elephant"),
}

# Per-provider model mapping: (interface_short_name, api_model_dir_substring)
PROVIDER_MODELS = {
    "claude": [
        ("haiku",   "claude-haiku-4-5-20251001_web_search-False"),
        ("opus",    "claude-opus-4-7_web_search-False"),
        ("sonnet",  "claude-sonnet-4-6_web_search-False"),
    ],
    "gemini": [
        ("fast",     "gemini-3-flash-preview_thinking_level-low_web_search-False"),
        ("thinking", "gemini-3-flash-preview_thinking_level-high_web_search-False"),
    ],
    "chatgpt": [
        ("gpt-5-3-instant",  "gpt-5.3-chat-latest_web_search-False"),
        ("gpt-5-4-thinking", "gpt-5.4-2026-03-05_reasoning_effort-high_web_search-False"),
    ],
}


def load_query_map(query_stem):
    """Return dict {id: first_80_chars_of_query} for content validation."""
    qf = QUERY_DIR / f"{query_stem}.csv"
    if not qf.exists():
        return {}
    return {r["id"]: r["query"][:80] for r in csv.DictReader(open(qf))}


def load_query_ids(query_stem):
    return set(load_query_map(query_stem).keys())


def _content_matches(prompt_text, query_map, qid):
    """Verify the prompt actually contains the expected query (avoids ID collisions).

    HTML rendering eats markdown syntax — blank lines collapse (\\n\\n → \\n) and
    backticks/code-spans disappear. We normalize whitespace and strip backticks
    on both sides before comparing.

    Fallback: when the parser couldn't extract a prompt (long queries get rendered
    as file attachments and have no inline text), trust the dir_prefix+filename.
    """
    if qid not in query_map:
        return False

    def _norm(s):
        return " ".join((s or "").replace("`", "").split())

    normalized_prompt = _norm(prompt_text)
    if len(normalized_prompt) < 20:
        return True  # parser failed to extract — fall back to path-based trust
    return _norm(query_map[qid]) in normalized_prompt


def find_api_files(scraper_dir, model_substring, query_map):
    """Yield (qid, src_path) for API files in this provider that match the benchmark.
    Iterates oldest→newest run so caller can let later overwrite earlier.
    Validates query content against query_map to avoid ID collisions.
    """
    import json as _json
    api_root = scraper_dir / "api"
    if not api_root.exists():
        return
    for ts in sorted(api_root.iterdir()):
        if not ts.is_dir(): continue
        for sub in ts.iterdir():
            if not sub.is_dir(): continue
            candidates = []
            if sub.name == model_substring:
                candidates.append(sub)
            else:
                for child in sub.iterdir():
                    if child.is_dir() and child.name == model_substring:
                        candidates.append(child)
            for model_dir in candidates:
                for f in model_dir.glob("*.api.json"):
                    qid = f.stem.replace(".api", "").replace("_run0", "")
                    if qid not in query_map:
                        continue
                    try:
                        d = _json.load(open(f))
                        if _content_matches(d.get("prompt", ""), query_map, qid):
                            yield qid, f
                    except Exception:
                        continue


def find_iface_files(scraper_dir, dir_prefix, query_map, ext="*.json", section="parsed_json"):
    """Yield (qid, src_path) for interface files matching dir_prefix.
    For parsed_json: validate via query_parsed field.
    For raw_html: rely on dir_prefix + meta.json query_id (HTML content check too expensive).
    """
    import json as _json
    root = scraper_dir / section
    if not root.exists():
        return
    for ts in sorted(root.iterdir()):
        if not ts.is_dir(): continue
        for sess in ts.iterdir():
            if not sess.is_dir(): continue
            for exp in sess.iterdir():
                if not exp.is_dir(): continue
                if not exp.name.startswith(dir_prefix):
                    continue
                for f in exp.glob(ext):
                    if f.name.endswith(".meta.json"):
                        continue
                    qid = f.stem.replace("_run0", "")
                    if qid not in query_map:
                        continue
                    if section == "parsed_json":
                        try:
                            d = _json.load(open(f))
                            if _content_matches(d.get("query_parsed", ""), query_map, qid):
                                yield qid, f
                        except Exception:
                            continue
                    else:
                        # For raw_html: skip if a sibling parsed_json exists and fails content check
                        # (skip the cross-file lookup; just trust dir_prefix here)
                        yield qid, f


def copy_files(pairs, dst_dir, label, dry_run=False):
    """pairs is iterable of (qid, src_path). Newest wins (caller orders oldest→newest)."""
    if dry_run:
        count = 0
        for qid, src in pairs:
            count += 1
        print(f"  {label}: {count} files would be copied -> {dst_dir}")
        return count

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for qid, src in pairs:
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        # Also copy .meta.json sibling for raw_html
        if src.suffix == ".html":
            meta = src.with_suffix(".meta.json")
            if meta.exists():
                shutil.copy2(meta, dst_dir / meta.name)
        count += 1
    print(f"  {label}: copied {count} files -> {dst_dir}")
    return count


def move_provider_benchmark(provider, benchmark_key, dry_run=False):
    """Move data for one (provider, benchmark) into the experiments tree."""
    if benchmark_key not in BENCHMARKS:
        return
    query_stem, exp_name = BENCHMARKS[benchmark_key]
    query_map = load_query_map(query_stem)
    if not query_map:
        print(f"  Skip {benchmark_key}: no query file at {query_stem}.csv")
        return

    scraper_dir = SCRAPER_DIR / f"{provider}_data"
    if not scraper_dir.exists():
        return

    target_root = EXP_DIR / exp_name / "data" / provider
    print(f"\n=== {provider} / {benchmark_key} ({len(query_map)} target items) ===")

    for short_name, api_dir_substring in PROVIDER_MODELS[provider]:
        # API files
        api_pairs = list(find_api_files(scraper_dir, api_dir_substring, query_map))
        api_dst = target_root / "api" / api_dir_substring
        copy_files(api_pairs, api_dst, f"{short_name} API", dry_run)

        # Interface (parsed_json)
        iface_pairs = list(find_iface_files(scraper_dir, short_name, query_map,
                                            ext="*.json", section="parsed_json"))
        iface_dst = target_root / "interface" / short_name
        copy_files(iface_pairs, iface_dst, f"{short_name} interface", dry_run)

        # raw_html — only copy items whose parsed_json passed content check
        valid_qids = {qid for qid, _ in iface_pairs}
        html_pairs = [
            (qid, f) for qid, f in find_iface_files(scraper_dir, short_name, query_map,
                                                     ext="*.html", section="raw_html")
            if qid in valid_qids
        ]
        html_dst = target_root / "raw_html" / short_name
        copy_files(html_pairs, html_dst, f"{short_name} raw_html", dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", default=list(PROVIDER_MODELS.keys()))
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS.keys()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for provider in args.providers:
        for bench in args.benchmarks:
            move_provider_benchmark(provider, bench, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
