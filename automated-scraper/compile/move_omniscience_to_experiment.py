"""
Move aa-omniscience data from automated-scraper/{claude,gemini}_data/ into the
experiment directory at experiments/aa-omniscience/data/{provider}/, matching
the existing chatgpt layout:

  data/{provider}/
    api/{full_model_dir}/{id}_run0.api.json        (flat, merged across timestamps)
    interface/{short_name}/{id}_run0.json          (flat, merged)
    raw_html/{short_name}/{id}_run0.html + .meta.json (flat, merged)

Newer timestamps win when query_id appears in multiple runs (e.g., resumes).
"""
import csv
import shutil
from pathlib import Path

SCRAPER_DIR = Path(__file__).resolve().parents[1]
EXP_DIR = SCRAPER_DIR.parent / "experiments" / "aa-omniscience" / "data"

OMNI_QUERY_FILE = SCRAPER_DIR.parent / "benchmark_creation" / "results" / "aa-omniscience-subset-200.csv"

# Load aa-omniscience IDs to filter
omni_ids = set()
with open(OMNI_QUERY_FILE) as f:
    for row in csv.DictReader(f):
        omni_ids.add(row["id"])
print(f"aa-omniscience subset: {len(omni_ids)} IDs")


# Claude config: 4 timestamps containing aa-omniscience runs
CLAUDE_TIMESTAMPS = [
    "2026-05-04_10-10-43",  # initial run, 35 queries
    "2026-05-04_13-01-30",  # resume #1, 1 query
    "2026-05-04_13-18-26",  # resume #2, 100 queries
    "2026-05-04_16-45-12",  # resume #3, 63 queries
]

# Map: short_name → (api_model_dir_substring, interface_dir_prefix)
CLAUDE_MODELS = {
    "claude-haiku":  ("claude-haiku-4-5-20251001_web_search-False", "haiku"),
    "claude-opus":   ("claude-opus-4-7_web_search-False",            "opus"),
    "claude-sonnet": ("claude-sonnet-4-6_web_search-False",          "sonnet"),
}


def copy_file(src, dst, replace=True):
    """Copy src to dst, creating parent dirs. Newer overwrites."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not replace:
        return False
    shutil.copy2(src, dst)
    return True


def collect_api_files(timestamps, model_dir_name, scraper_provider):
    """Yield (qid, src_path) for API files matching aa-omniscience IDs.
    Walks timestamps OLDEST→NEWEST so caller can let later overwrite earlier.
    """
    api_root = SCRAPER_DIR / f"{scraper_provider}_data" / "api"
    for ts in timestamps:
        ts_dir = api_root / ts
        if not ts_dir.exists():
            continue
        # Could be flat: api/{ts}/{model_dir}/*.api.json
        # Or nested: api/{ts}/{benchmark}/{model_dir}/*.api.json
        candidates = []
        for sub in ts_dir.iterdir():
            if not sub.is_dir():
                continue
            if sub.name == model_dir_name:
                candidates.append(sub)
            else:
                # nested
                for child in sub.iterdir():
                    if child.is_dir() and child.name == model_dir_name:
                        candidates.append(child)
        for model_dir in candidates:
            for f in model_dir.glob("*.api.json"):
                qid = f.stem.replace(".api", "").replace("_run0", "")
                if qid in omni_ids:
                    yield qid, f


def collect_interface_files(timestamps, dir_prefix, scraper_provider, ext_dir):
    """Yield (qid, src_path) and meta if exists. ext_dir is 'parsed_json' or 'raw_html'.
    Walks timestamps OLDEST→NEWEST so caller can let later overwrite earlier.
    """
    root = SCRAPER_DIR / f"{scraper_provider}_data" / ext_dir
    for ts in timestamps:
        ts_dir = root / ts
        if not ts_dir.exists():
            continue
        for sess in ts_dir.iterdir():
            if not sess.is_dir():
                continue
            for exp in sess.iterdir():
                if not exp.is_dir():
                    continue
                if not exp.name.startswith(dir_prefix):
                    continue
                file_ext = "*.json" if ext_dir == "parsed_json" else "*.html"
                for f in exp.glob(file_ext):
                    if f.name.endswith(".meta.json"):
                        continue
                    qid = f.stem.replace("_run0", "")
                    if qid in omni_ids:
                        yield qid, f


def main():
    print("\n=== Moving Claude aa-omniscience data ===\n")

    for short_name, (api_model_dir, iface_prefix) in CLAUDE_MODELS.items():
        # API
        api_dst_dir = EXP_DIR / "claude" / "api" / api_model_dir
        api_count = 0
        for qid, src in collect_api_files(CLAUDE_TIMESTAMPS, api_model_dir, "claude"):
            dst = api_dst_dir / src.name
            copy_file(src, dst, replace=True)  # newer overwrites
            api_count += 1
        unique_ids = len({f.stem.replace(".api","").replace("_run0","") for f in api_dst_dir.glob("*.api.json")}) if api_dst_dir.exists() else 0
        print(f"  {short_name} API:       {unique_ids}/200 (copied {api_count} files)")

        # Interface (parsed_json)
        iface_dst_dir = EXP_DIR / "claude" / "interface" / short_name
        iface_count = 0
        for qid, src in collect_interface_files(CLAUDE_TIMESTAMPS, iface_prefix, "claude", "parsed_json"):
            dst = iface_dst_dir / src.name
            copy_file(src, dst, replace=True)
            iface_count += 1
        unique_iface = len(list(iface_dst_dir.glob("*.json"))) if iface_dst_dir.exists() else 0
        print(f"  {short_name} interface: {unique_iface}/200 (copied {iface_count} files)")

        # raw_html (.html + .meta.json)
        html_dst_dir = EXP_DIR / "claude" / "raw_html" / short_name
        html_count = 0
        for qid, src in collect_interface_files(CLAUDE_TIMESTAMPS, iface_prefix, "claude", "raw_html"):
            dst = html_dst_dir / src.name
            copy_file(src, dst, replace=True)
            # also meta
            meta_src = src.with_suffix(".meta.json")
            if meta_src.exists():
                copy_file(meta_src, html_dst_dir / meta_src.name, replace=True)
            html_count += 1
        unique_html = len(list(html_dst_dir.glob("*.html"))) if html_dst_dir.exists() else 0
        print(f"  {short_name} raw_html:  {unique_html}/200 (copied {html_count} files)")
        print()


if __name__ == "__main__":
    main()
