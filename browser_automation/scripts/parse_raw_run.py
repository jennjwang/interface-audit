#!/usr/bin/env python3
"""
Parse a raw HTML scrape run and extract MCQ answers into a CSV.

Uses regex instead of BeautifulSoup (the HTML files are ~400KB each
and BS4 gets OOM-killed on thousands of them).

Usage:
  # Parse a single run:
  python parse_raw_run.py <RUN_DIR>
  python parse_raw_run.py  # merge-all (default)

  # Merge and parse ALL runs (deduplicates personas across runs):
  python parse_raw_run.py --merge-all
  python parse_raw_run.py --merge-all --move-done

Output:
  Single run: results/interface/eval_chatgpt-scraper_<run_id>.csv
  Merged:     results/interface/merged_all_runs.csv
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRAPER_ROOT = BASE_DIR.parent
REPO_ROOT = SCRAPER_ROOT.parent
QUERIES_DIR = REPO_ROOT / "personaMem" / "queries"
DONE_DIR = REPO_ROOT / "personaMem" / "done"
DEFAULT_RAW_ROOT = SCRAPER_ROOT / "data" / "raw_html"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
INTERFACE_RESULTS_DIR = REPO_ROOT / "results" / "interface"
DEFAULT_GEMINI_OUTPUT_ROOT = REPO_ROOT / "experiments" / "metabench-mmlu" / "data-gemini" / "parsed_json"

# ── Persona/query helpers ───────────────────────────────────────────────────


def _extract_persona_id(value: str) -> str | None:
    m = re.search(r"persona(\d+)", value)
    return m.group(1) if m else None


def _query_priority(path: Path) -> tuple[int, str]:
    s = str(path)
    if "queries_interface_sampled_mod_filtered" in s:
        return (0, s)
    if "queries_interface_sampled_mod" in s:
        return (1, s)
    if "queries_interface_sampled" in s:
        return (2, s)
    if "queries_interface" in s:
        return (3, s)
    if "queries_api_sampled" in s:
        return (4, s)
    if "queries_api" in s:
        return (5, s)
    if "done" in s:
        return (6, s)
    return (7, s)


def _choose_query_file(candidates: list[Path]) -> Path:
    return sorted(candidates, key=_query_priority)[0]


def _load_persona_mcqs(search_dirs: list[Path]) -> dict[str, dict[str, dict]]:
    by_pid: dict[str, list[Path]] = {}
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for qf in sorted(search_dir.glob("persona*-queries*.json")):
            pid = _extract_persona_id(qf.name)
            if not pid:
                continue
            by_pid.setdefault(pid, []).append(qf)

    persona_items: dict[str, dict[str, dict]] = {}
    for pid, files in by_pid.items():
        qf = _choose_query_file(files)
        items = json.loads(qf.read_text(encoding="utf-8"))
        persona_items[pid] = {}
        for item in items:
            iid = item.get("id", "")
            if "_mcq" in iid.lower() and item.get("correct_answer"):
                persona_items[pid][iid] = item
    return persona_items


def _build_search_dirs() -> list[Path]:
    """Collect query directories across personaMem/queries* trees."""
    search_dirs: list[Path] = [DONE_DIR]
    queries_parent = REPO_ROOT / "personaMem"
    if not queries_parent.exists():
        return search_dirs

    for subdir in queries_parent.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name == "queries":
            # include nested query folders under personaMem/queries
            for nested in subdir.iterdir():
                if nested.is_dir():
                    search_dirs.append(nested)
        elif subdir.name.startswith("queries"):
            search_dirs.append(subdir)
    return search_dirs

# ── Regex-based extraction (avoids BS4 memory issues) ─────────────────────


def _extract_user_text_regex(html: str) -> str | None:
    """Extract the last user message text from raw HTML."""
    # Find the last user message block
    user_pattern = re.compile(
        r'<div[^>]*data-message-author-role="user"[^>]*>', re.DOTALL
    )
    matches = list(user_pattern.finditer(html))
    if not matches:
        return None
    last_match = matches[-1]
    rest = html[last_match.end():]
    # User text is in a whitespace-pre-wrap div
    wrap_match = re.search(
        r'<div\s+class="[^"]*\bwhitespace-pre-wrap\b[^"]*"[^>]*>(.*?)</div>',
        rest, re.DOTALL,
    )
    if not wrap_match:
        return None
    import html as html_mod
    raw = wrap_match.group(1)
    raw = re.sub(r'<br[^>]*/?>', '\n', raw)
    raw = re.sub(r'<[^>]+>', '', raw)
    return html_mod.unescape(raw).strip()


def _extract_assistant_text_regex(html: str) -> tuple[str | None, str | None]:
    """Extract the last assistant message text and model_slug from raw HTML using regex.

    Looks for the last `data-message-author-role="assistant"` block, then
    extracts the content of the inner ``markdown`` div up to its closing
    ``</div></div></div></div>`` boundary.
    """
    # Find all assistant message blocks
    assistant_pattern = re.compile(
        r'<div[^>]*data-message-author-role="assistant"[^>]*>', re.DOTALL
    )
    matches = list(assistant_pattern.finditer(html))
    if not matches:
        return None, None

    last_match = matches[-1]
    tag = last_match.group(0)

    # Extract model slug from the opening tag
    slug_m = re.search(r'data-message-model-slug="([^"]+)"', tag)
    model_slug = slug_m.group(1) if slug_m else None

    rest = html[last_match.end():]

    # The markdown div structure is:
    #   <div class="markdown prose ...">
    #     <p>...</p>
    #     <p>...</p>
    #   </div></div></div></div>
    # We capture everything between the markdown opening tag and the
    # first occurrence of </div></div></div></div> which closes the
    # message container.
    md_match = re.search(
        r'<div\s+class="[^"]*\bmarkdown\b[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>\s*</div>',
        rest, re.DOTALL,
    )

    if not md_match:
        # Fallback: the page was captured before the markdown div rendered.
        # The live-region assertive element often already has the response text.
        lr_match = re.search(
            r'id="live-region-assertive"[^>]*>(.*?)</div>',
            html, re.DOTALL,
        )
        if lr_match:
            import html as html_mod
            raw = lr_match.group(1)
            # Strip "ChatGPT says: " prefix if present
            raw = re.sub(r'^ChatGPT says:\s*', '', raw.strip())
            raw = re.sub(r'<[^>]+>', '', raw)
            text = html_mod.unescape(raw).strip()
            return (text or None), model_slug
        return None, model_slug

    raw_html_content = md_match.group(1)

    # Remove <script> and <style> blocks before stripping tags
    clean = re.sub(r'<script[^>]*>.*?</script>', '', raw_html_content, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.DOTALL | re.IGNORECASE)

    # Convert <br> and block elements to newlines, then strip all tags
    clean = re.sub(r'<br[^>]*/?>', '\n', clean)
    clean = re.sub(r'</p>\s*<p[^>]*>', '\n\n', clean)
    clean = re.sub(r'<[^>]+>', '', clean)

    # Decode HTML entities
    import html as html_mod
    clean = html_mod.unescape(clean)

    # Clean up whitespace
    lines = [' '.join(line.split()) for line in clean.split('\n')]
    text = '\n'.join(line for line in lines if line).strip()

    return text, model_slug


# ── Answer extraction (same logic as api_baseline.py) ─────────────────────


def extract_answer(model_response: str, correct_label: str) -> tuple[bool, str]:
    correct = correct_label.lower().strip("() ")
    text = model_response.strip()
    if "<final_answer>" in text:
        parts = text.split("<final_answer>")
        after = parts[-1]
        if after.endswith("</final_answer>"):
            after = after[: -len("</final_answer>")]
        after = after.strip()
        # Fall back to pre-tag text when post-tag content is empty
        text = after if after else parts[-2].strip()
    text = text.strip()

    def _options(s):
        s = s.lower()
        parens = re.findall(r"\(([a-d])\)", s)
        return set(parens) if parens else set(re.findall(r"\b([a-d])\b", s))

    pred = _options(text)
    if pred == {correct}:
        return True, text
    full_pred = _options(model_response)
    if full_pred == {correct}:
        return True, text
    return False, text


def _resolve_dir(path_value: str | None, default: Path) -> Path:
    if not path_value:
        return default
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_run(raw_root: Path, results_dir: Path, move_done: bool = False) -> int:
    run_id = raw_root.name
    print(f"Parsing run: {raw_root}")
    print(f"Run ID: {run_id}")

    # Load persona MCQ lookup (search queries/ subdirs and done/)
    print("Loading persona query files...")
    # Search all persona query directories
    search_dirs = _build_search_dirs()
    persona_items = _load_persona_mcqs(search_dirs)
    print(f"Loaded MCQs for {len(persona_items)} personas")

    # Collect MCQ HTML files (matches *_mcq_run0.html or *_mcq_*_run0.html)
    html_files = sorted(raw_root.rglob("*_mcq*_run0.html"))
    print(f"Found {len(html_files)} MCQ HTML files")

    rows = []
    no_match = 0
    parse_fail = 0

    for i, html_path in enumerate(html_files, 1):
        if i % 200 == 0:
            print(f"  ... processed {i}/{len(html_files)}")

        stem = html_path.stem
        base_id = stem.replace("_run0", "")
        parent_name = html_path.parent.name
        persona_id = _extract_persona_id(parent_name)
        if not persona_id:
            no_match += 1
            continue

        session = "unknown"
        for part in html_path.parts:
            if part.startswith("session_"):
                session = part
                break

        item = (persona_items.get(persona_id) or {}).get(base_id)
        if not item:
            no_match += 1
            continue

        correct_answer = item["correct_answer"]

        try:
            html_content = html_path.read_text(encoding="utf-8", errors="ignore")
            assistant_text, model_slug = _extract_assistant_text_regex(html_content)
        except Exception:
            parse_fail += 1
            continue

        if not assistant_text:
            parse_fail += 1
            continue

        # Load meta for model_slug fallback and sent_at
        meta_path = html_path.with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        model_slug = model_slug or meta.get("model_slug", "")

        is_correct, prediction = extract_answer(assistant_text, correct_answer)
        rows.append({
            "persona_id": persona_id,
            "item_id": base_id,
            "session": session,
            "category": item.get("category", ""),
            "preference": item.get("preference", ""),
            "pref_type": item.get("pref_type", ""),
            "topic_pref": item.get("topic_pref", ""),
            "updated": item.get("updated", ""),
            "prev_pref": item.get("prev_pref", ""),
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "model_prediction": (prediction or "")[:500],
            "model_slug": model_slug,
            "model_response_len": len(assistant_text),
            "sent_at": meta.get("sent_at", ""),
        })

    if not rows:
        print("No MCQ rows extracted.")
        return 1

    # Write CSV
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"eval_chatgpt-scraper_{run_id}.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Summary
    total = len(rows)
    correct = sum(1 for r in rows if r["is_correct"])
    print(f"\n{'=' * 60}")
    print(f"Parsed: {total} MCQs  (no_match: {no_match}, parse_fail: {parse_fail})")
    print(f"Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

    by_persona = defaultdict(lambda: {"t": 0, "c": 0})
    for r in rows:
        by_persona[r["persona_id"]]["t"] += 1
        if r["is_correct"]:
            by_persona[r["persona_id"]]["c"] += 1
    print(f"\nPer-persona ({len(by_persona)} personas):")
    for pid in sorted(by_persona.keys(), key=lambda x: int(x)):
        d = by_persona[pid]
        a = 100 * d["c"] / d["t"] if d["t"] else 0
        print(f"  Persona {pid:>4}: {d['c']:>3}/{d['t']:<3} ({a:.1f}%)")

    by_session = defaultdict(lambda: {"t": 0, "c": 0})
    for r in rows:
        by_session[r["session"]]["t"] += 1
        if r["is_correct"]:
            by_session[r["session"]]["c"] += 1
    print(f"\nPer-session:")
    for sid in sorted(by_session.keys()):
        d = by_session[sid]
        a = 100 * d["c"] / d["t"] if d["t"] else 0
        print(f"  {sid}: {d['c']:>3}/{d['t']:<3} ({a:.1f}%)")

    print(f"\nOutput: {out_csv}")
    print(f"{'=' * 60}")

    # Move parsed personas to done/
    if move_done:
        print()
        move_to_done(raw_root)

    return 0


# ── Main ──────────────────────────────────────────────────────────────────


def parse_to_json(raw_root: Path) -> int:
    """Parse all HTML files in a run directory into individual JSON files
    (same format as audit.py's parse_saved_htmls: query_parsed, ai_generated_output_text, model_slug, meta).

    Output goes to data/parsed_json/<run_id>/.
    """
    import html as html_mod  # noqa: F811

    run_id = raw_root.name
    parsed_root = SCRAPER_ROOT / "data" / "parsed_json" / run_id
    print(f"Parsing HTMLs to JSON: {raw_root}")
    print(f"Output dir: {parsed_root}")

    html_files = sorted(raw_root.rglob("*.html"))
    if not html_files:
        print("No HTML files found.")
        return 1

    parsed_count = 0
    skipped = 0
    failed = 0

    for i, html_path in enumerate(html_files, 1):
        if i % 500 == 0:
            print(f"  ... processed {i}/{len(html_files)}")

        rel_path = html_path.relative_to(raw_root)
        output_path = parsed_root / rel_path.with_suffix(".json")
        if output_path.exists():
            skipped += 1
            continue

        meta_path = html_path.with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        try:
            html_content = html_path.read_text(encoding="utf-8", errors="ignore")
            user_text = _extract_user_text_regex(html_content) or ""
            assistant_text, model_slug = _extract_assistant_text_regex(html_content)
            assistant_text = assistant_text or ""
        except Exception:
            failed += 1
            continue

        if not user_text and not assistant_text:
            failed += 1
            continue

        model_slug = model_slug or meta.get("model_slug")

        parsed = {
            "query_parsed": user_text,
            "ai_generated_output_text": assistant_text,
        }
        if model_slug:
            parsed["model_slug"] = model_slug
        if meta:
            parsed["meta"] = meta

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
        parsed_count += 1

    print(f"Parsed {parsed_count} HTML files to {parsed_root} (skipped {skipped}, failed {failed})")
    return 0


def move_to_done(raw_root: Path) -> int:
    """Move persona query files from queries/ to done/ for personas that have scraped data."""
    import shutil

    scraped_ids: set[str] = set()
    for d in raw_root.rglob("persona*-queries"):
        if d.is_dir():
            pid = _extract_persona_id(d.name)
            if pid:
                scraped_ids.add(pid)

    if not scraped_ids:
        print("No persona directories found in run.")
        return 0

    # Search all persona query directories
    queries_parent = REPO_ROOT / "personaMem"
    search_dirs = []
    if queries_parent.exists():
        for subdir in queries_parent.iterdir():
            if subdir.is_dir() and subdir.name.startswith("queries"):
                search_dirs.append(subdir)

    DONE_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    already = 0

    for pid in sorted(scraped_ids, key=int):
        dst = DONE_DIR / f"persona{pid}-queries.json"
        if dst.exists():
            already += 1
            continue

        # Search for the source file in all query directories
        src = None
        for search_dir in search_dirs:
            candidates = list(search_dir.glob(f"persona{pid}-queries*.json"))
            if candidates:
                src = _choose_query_file(candidates)
                break

        if src:
            shutil.move(str(src), str(dst))
            moved += 1

    print(f"Moved {moved} persona query files to {DONE_DIR.relative_to(REPO_ROOT)}/")
    if already:
        print(f"  ({already} already in done/)")

    # Count remaining files across all query directories
    remaining = 0
    for search_dir in search_dirs:
        remaining += len(list(search_dir.glob("persona*-queries*.json")))
    print(f"Remaining in queries*/: {remaining}")
    return 0


def find_all_runs(raw_html_root: Path) -> list[Path]:
    """Find all run directories sorted by timestamp."""
    runs = []
    for d in raw_html_root.iterdir():
        if d.is_dir() and d.name.startswith("20"):
            runs.append(d)
    return sorted(runs, key=lambda p: p.name)


def find_personas_in_run(run_dir: Path) -> dict[str, list[Path]]:
    """Find all persona directories in a run, grouped by persona ID.

    Returns: {persona_id: [persona_dir1, persona_dir2, ...]}
    """
    personas = defaultdict(list)
    for session_dir in run_dir.iterdir():
        if not session_dir.is_dir() or not session_dir.name.startswith("session_"):
            continue
        for persona_dir in session_dir.iterdir():
            if persona_dir.is_dir() and persona_dir.name.startswith("persona"):
                pid = _extract_persona_id(persona_dir.name)
                if not pid:
                    continue
                personas[pid].append(persona_dir)
    return personas


def merge_all_runs(
    raw_root: Path,
    results_dir: Path,
    move_done: bool = False,
    compare_queries_dir: Path | None = None,
) -> int:
    """Merge and parse all runs in raw_root, deduplicating personas."""
    if not raw_root.exists():
        print(f"ERROR: {raw_root} does not exist")
        return 1

    # Find all runs
    runs = find_all_runs(raw_root)
    if not runs:
        print(f"No runs found in {DEFAULT_RAW_ROOT}")
        return 1

    print(f"Found {len(runs)} runs under {raw_root}:")
    for run in runs:
        print(f"  {run.name}")

    # Build a map of persona_id -> (run_dir, persona_dirs)
    # Keep the latest run for each persona
    persona_sources: dict[str, tuple[Path, list[Path]]] = {}

    for run_dir in runs:
        personas = find_personas_in_run(run_dir)
        for pid, persona_dirs in personas.items():
            # Keep this run if we haven't seen this persona yet, or if newer
            if pid not in persona_sources:
                persona_sources[pid] = (run_dir, persona_dirs)
            else:
                old_run, _ = persona_sources[pid]
                if run_dir.name > old_run.name:  # Later timestamp
                    persona_sources[pid] = (run_dir, persona_dirs)

    print(f"\nFound {len(persona_sources)} unique personas (after deduplication)")

    # Load persona MCQs (used for parsing)
    print("Loading persona query files...")
    search_dirs = _build_search_dirs()
    persona_items = _load_persona_mcqs(search_dirs)
    print(f"Loaded MCQs for {len(persona_items)} personas")

    # Optional: load comparison MCQs for completeness check
    compare_items = None
    if compare_queries_dir:
        if not compare_queries_dir.exists():
            print(f"WARNING: comparison dir not found: {compare_queries_dir}")
        else:
            compare_items = _load_persona_mcqs([compare_queries_dir])
            print(
                f"Loaded comparison MCQs for {len(compare_items)} personas "
                f"from {compare_queries_dir}"
            )

    # Parse all HTML files
    rows = []
    total_files = 0
    no_match = 0
    parse_fail = 0

    for pid in sorted(persona_sources.keys(), key=int):
        run_dir, persona_dirs = persona_sources[pid]
        run_id = run_dir.name

        for persona_dir in persona_dirs:
            session = persona_dir.parent.name
            html_files = sorted(persona_dir.glob("*_mcq*_run0.html"))
            total_files += len(html_files)

            for html_path in html_files:
                stem = html_path.stem
                base_id = stem.replace("_run0", "")

                item = (persona_items.get(pid) or {}).get(base_id)
                if not item:
                    no_match += 1
                    continue

                html_content = html_path.read_text(encoding="utf-8", errors="ignore")
                user_text = _extract_user_text_regex(html_content)
                assistant_text, model_slug = _extract_assistant_text_regex(html_content)

                if not assistant_text:
                    parse_fail += 1
                    continue

                # Load meta
                meta_path = html_path.with_suffix(".meta.json")
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                model_slug = model_slug or meta.get("model_slug", "")
                sent_at = meta.get("sent_at", "")
                correct = item.get("correct_answer", "")
                is_correct, parsed = extract_answer(assistant_text, correct)

                rows.append({
                    "run_id": run_id,
                    "session": session,
                    "persona_id": pid,
                    "query_id": base_id,
                    "correct_answer": correct,
                    "model_response": assistant_text[:500],
                    "parsed_answer": parsed,
                    "is_correct": int(is_correct),
                    "model_slug": model_slug,
                    "sent_at": sent_at,
                })

    print(f"\n{'=' * 60}")
    print(f"Parsed: {len(rows)} MCQs from {total_files} HTML files")
    print(f"  no_match: {no_match}, parse_fail: {parse_fail}")

    if not rows:
        print("No rows to write!")
        return 1

    # Calculate accuracy
    correct_count = sum(r["is_correct"] for r in rows)
    accuracy = correct_count / len(rows) * 100 if rows else 0
    print(f"Overall accuracy: {correct_count}/{len(rows)} ({accuracy:.1f}%)")

    # Per-persona stats
    persona_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        pid = row["persona_id"]
        persona_stats[pid]["total"] += 1
        persona_stats[pid]["correct"] += row["is_correct"]

    print(f"\nPer-persona ({len(persona_stats)} personas):")
    for pid in sorted(persona_stats.keys(), key=int):
        stats = persona_stats[pid]
        pct = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  Persona {pid:>4}: {stats['correct']:>3}/{stats['total']:<3} ({pct:>5.1f}%)")

    # Completeness check: compare parsed vs expected MCQs
    print(f"\nCompleteness check:")
    incomplete_personas = []
    expected_items = compare_items if compare_items is not None else persona_items
    for pid in sorted(persona_stats.keys(), key=int):
        parsed_count = persona_stats[pid]["total"]
        expected_count = len(expected_items.get(pid, {}))
        if parsed_count < expected_count:
            incomplete_personas.append(pid)
            print(f"  ⚠ Persona {pid:>4}: {parsed_count}/{expected_count} MCQs parsed (missing {expected_count - parsed_count})")
        elif parsed_count > expected_count:
            print(f"  ⚠ Persona {pid:>4}: {parsed_count}/{expected_count} MCQs parsed (extra {parsed_count - expected_count})")
        else:
            print(f"  ✓ Persona {pid:>4}: {parsed_count}/{expected_count} MCQs complete")

    if incomplete_personas:
        print(f"\n⚠ {len(incomplete_personas)} personas have incomplete data")
    else:
        print(f"\n✓ All personas have complete MCQ data")

    # Write CSV
    output_path = results_dir / "merged_all_runs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOutput: {output_path}")
    print(f"{'=' * 60}\n")

    # Move to done if requested
    if move_done:
        scraped_ids = set(persona_sources.keys())
        moved = _move_personas_to_done(scraped_ids)
        print(f"Moved {moved} personas to done/")

    return 0


def _move_personas_to_done(persona_ids: set[str]) -> int:
    """Move persona query files to done/ directory."""
    import shutil

    queries_parent = REPO_ROOT / "personaMem"
    search_dirs = []
    if queries_parent.exists():
        for subdir in queries_parent.iterdir():
            if subdir.is_dir() and subdir.name.startswith("queries"):
                search_dirs.append(subdir)

    DONE_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    already = 0

    for pid in sorted(persona_ids, key=int):
        dst = DONE_DIR / f"persona{pid}-queries.json"
        if dst.exists():
            already += 1
            continue

        src = None
        for search_dir in search_dirs:
            candidates = list(search_dir.glob(f"persona{pid}-queries*.json"))
            if candidates:
                src = _choose_query_file(candidates)
                break

        if src:
            shutil.move(str(src), str(dst))
            moved += 1

    print(f"Moved {moved} persona query files to {DONE_DIR.relative_to(REPO_ROOT)}/")
    if already:
        print(f"  ({already} already in done/)")

    remaining = 0
    for search_dir in search_dirs:
        remaining += len(list(search_dir.glob("persona*-queries*.json")))
    print(f"Remaining in queries*/: {remaining}")

    return moved


# ── Gemini HTML extraction ─────────────────────────────────────────────────


def _gemini_extract_response_text(html: str) -> str | None:
    """Extract the model response text from Gemini HTML.

    Targets the last <div class="markdown markdown-main-panel ..."> element.
    """
    import html as html_mod
    pattern = re.compile(
        r'<div[^>]*class="[^"]*\bmarkdown\b[^"]*\bmarkdown-main-panel\b[^"]*"[^>]*>',
        re.DOTALL,
    )
    matches = list(pattern.finditer(html))
    if not matches:
        return None

    last = matches[-1]
    chunk = html[last.end(): last.end() + 32768]

    # Extract code blocks and their outputs from the full chunk before
    # trimming.  Gemini wraps code in <pre><code>...</code></pre> and
    # execution output in <code data-test-id="code-output-...">...</code>.
    code_parts = []
    seen_code = set()
    for m in re.finditer(r'<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>', chunk, re.DOTALL):
        code_text = re.sub(r'<[^>]+>', '', m.group(1))
        code_text = html_mod.unescape(code_text).strip()
        if code_text and code_text not in seen_code:
            seen_code.add(code_text)
            code_parts.append(code_text)
    for m in re.finditer(r'<code[^>]*data-test-id="code-output[^"]*"[^>]*>(.*?)</code>', chunk, re.DOTALL):
        out_text = re.sub(r'<[^>]+>', '', m.group(1))
        out_text = html_mod.unescape(out_text).strip()
        if out_text and out_text not in seen_code:
            seen_code.add(out_text)
            code_parts.append(out_text)

    # Trim at the first non-content element that closes the response block.
    end_match = re.search(
        r'<mat-icon|<copy-button|response-container-footer|sensitive-memories-banner',
        chunk,
    )
    if end_match:
        before = chunk[:end_match.start()]
        last_close = max(
            before.rfind('</p>'), before.rfind('</li>'),
            before.rfind('</ol>'), before.rfind('</ul>'),
        )
        chunk = before[:last_close + 4] if last_close >= 0 else before

    chunk = re.sub(r'<script[^>]*>.*?</script>', '', chunk, flags=re.DOTALL | re.IGNORECASE)
    chunk = re.sub(r'<style[^>]*>.*?</style>', '', chunk, flags=re.DOTALL | re.IGNORECASE)
    chunk = re.sub(r'</p>\s*<p[^>]*>', '\n\n', chunk)
    chunk = re.sub(r'<br[^>]*/?>', '\n', chunk)
    chunk = re.sub(r'<[^>]+>', '', chunk)

    text = html_mod.unescape(chunk)
    lines = [' '.join(line.split()) for line in text.split('\n')]
    prose = '\n'.join(line for line in lines if line).strip()

    # If prose extraction yielded nothing useful (e.g. just a language
    # label like "Python"), fall back to extracted code content.
    if code_parts and (not prose or prose in (
        'Python', 'python', 'Javascript', 'javascript', 'Java', 'java',
        'C', 'C++', 'c++', 'R', 'r', 'Go', 'go', 'Rust', 'rust',
    )):
        prose = '\n\n'.join(code_parts)

    return prose or None


def _gemini_extract_user_text(html: str) -> str | None:
    """Extract the user query text from Gemini HTML."""
    import html as html_mod
    m = re.search(
        r'<div[^>]*class="[^"]*\bquery-text\b[^"]*"[^>]*>(.*?)</div>',
        html, re.DOTALL,
    )
    if not m:
        return None
    raw = re.sub(r'<[^>]+>', '', m.group(1))
    text = html_mod.unescape(raw).strip()
    # Gemini prefixes displayed text with "You said "
    text = re.sub(r'^You said\s+', '', text)
    return text or None


def parse_gemini_run(run_dir: Path, output_root: Path) -> int:
    """Parse Gemini raw HTML files into individual JSON files.

    Output format mirrors the ChatGPT parsed_json format:
      { "query_parsed": ..., "ai_generated_output_text": ..., "model_slug": ..., "meta": ... }

    Files are read from:
      <run_dir>/session_<N>/<model_name>/<query_id>_run0.html

    Output goes to:
      <output_root>/<run_id>/session_<N>/<model_name>/<query_id>_run0.json
    """
    run_id = run_dir.name
    parsed_root = output_root / run_id
    print(f"Parsing: {run_dir}")
    print(f"Output:  {parsed_root}")

    html_files = []
    for session_dir in sorted(run_dir.iterdir()):
        if not session_dir.is_dir() or not session_dir.name.startswith("session_"):
            continue
        for model_dir in sorted(session_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for html_path in sorted(model_dir.glob("*_run0.html")):
                html_files.append((session_dir.name, model_dir.name, html_path))

    if not html_files:
        print("No HTML files found.")
        return 1

    parsed = skipped = failed = 0

    for _session, model_name, html_path in html_files:
        rel = html_path.relative_to(run_dir)
        out_path = parsed_root / rel.with_suffix(".json")

        if out_path.exists():
            skipped += 1
            continue

        meta_path = html_path.with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        try:
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            user_text = _gemini_extract_user_text(html) or ""
            response_text = _gemini_extract_response_text(html) or ""
        except Exception:
            failed += 1
            continue

        if not user_text and not response_text:
            failed += 1
            continue

        record = {
            "query_parsed": user_text,
            "ai_generated_output_text": response_text,
            "model_slug": meta.get("model_slug", model_name),
            "meta": meta,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        parsed += 1

    print(f"Done: {parsed} parsed, {skipped} skipped, {failed} failed")
    return 0


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Parse a raw HTML scrape run.")
    parser.add_argument("run_dir", nargs="?", default=None, help="Path to run directory")
    parser.add_argument(
        "--raw-root",
        default=None,
        help=f"Root directory containing run folders (default: {DEFAULT_RAW_ROOT})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output individual parsed JSON files (like audit.py parse_saved_htmls)",
    )
    parser.add_argument(
        "--move-done",
        action="store_true",
        help="Move parsed personas from queries/ to done/",
    )
    parser.add_argument(
        "--merge-all",
        action="store_true",
        help="Merge and parse ALL runs in data/raw_html/ (deduplicates personas)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory to write per-run CSVs (default: results/interface)",
    )
    parser.add_argument(
        "--compare-queries-dir",
        default=None,
        help="Directory of persona queries to use for completeness comparison",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Parse Gemini HTML runs (outputs JSON, not CSV)",
    )
    parser.add_argument(
        "--gemini-output",
        default=None,
        help=f"Output root for Gemini JSON (default: {DEFAULT_GEMINI_OUTPUT_ROOT})",
    )
    args = parser.parse_args()

    # ── Gemini mode ───────────────────────────────────────────────────────────
    if args.gemini:
        raw_root_base = _resolve_dir(args.raw_root, DEFAULT_RAW_ROOT)
        gemini_output = Path(args.gemini_output) if args.gemini_output else DEFAULT_GEMINI_OUTPUT_ROOT

        if args.merge_all:
            runs = sorted(
                [d for d in raw_root_base.iterdir() if d.is_dir() and d.name[:4].isdigit()],
                key=lambda p: p.name,
            )
            if not runs:
                print(f"ERROR: no run directories found in {raw_root_base}")
                return 1
            for run_dir in runs:
                parse_gemini_run(run_dir, gemini_output)
                print()
            return 0

        if args.run_dir:
            return parse_gemini_run(Path(args.run_dir), gemini_output)

        runs = sorted(
            [d for d in raw_root_base.iterdir() if d.is_dir() and d.name[:4].isdigit()],
            key=lambda p: p.name,
        )
        if not runs:
            print(f"ERROR: no run directories found in {raw_root_base}")
            return 1
        run_dir = runs[-1]
        print(f"Using latest run: {run_dir.name}")
        return parse_gemini_run(run_dir, gemini_output)

    # ── ChatGPT / Claude mode (original behaviour) ────────────────────────────
    results_dir = _resolve_dir(args.results_dir, INTERFACE_RESULTS_DIR)
    raw_root_base = _resolve_dir(args.raw_root, DEFAULT_RAW_ROOT)
    compare_queries_dir = _resolve_dir(args.compare_queries_dir, QUERIES_DIR) if args.compare_queries_dir else None

    # Default to merge-all unless an explicit run_dir is provided.
    if args.merge_all or args.run_dir is None:
        return merge_all_runs(
            raw_root_base,
            results_dir,
            move_done=args.move_done,
            compare_queries_dir=compare_queries_dir,
        )

    raw_root = Path(args.run_dir) if args.run_dir else None
    if not raw_root or not raw_root.is_dir():
        if raw_root_base.is_dir():
            runs = sorted(raw_root_base.iterdir(), key=lambda p: p.name, reverse=True)
            raw_root = runs[0] if runs else None
        if not raw_root or not raw_root.is_dir():
            print("Usage: python parse_raw_run.py <RUN_DIR> [--json] [--merge-all]")
            return 1

    # If --json only (no CSV), produce parsed JSON files and optionally move done
    if args.json and not args.move_done:
        return parse_to_json(raw_root)

    # If --json with --move-done
    if args.json:
        rc = parse_to_json(raw_root)
        if rc != 0:
            return rc
        print()
        return move_to_done(raw_root)

    rc = parse_run(raw_root, results_dir, move_done=args.move_done)
    if rc != 0:
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
