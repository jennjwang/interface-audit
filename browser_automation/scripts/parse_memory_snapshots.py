#!/usr/bin/env python3
import argparse
import csv
import html
import re
from collections import Counter
from pathlib import Path


def find_switch(text: str, label: str):
    patterns = [
        rf'aria-label=\"{re.escape(label)}\"[^>]*aria-checked=\"(true|false)\"',
        rf'aria-checked=\"(true|false)\"[^>]*aria-label=\"{re.escape(label)}\"',
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.group(1).lower() == "true"
    return None


def textarea_value(text: str, name: str):
    m = re.search(
        rf'name=\"{re.escape(name)}\"[^>]*>(.*?)</textarea>',
        text,
        flags=re.I | re.S,
    )
    if not m:
        return None
    val = html.unescape(m.group(1))
    val = re.sub(r"\s+", " ", val).strip()
    return val or None


def occupation_span(text: str):
    m = re.search(
        r'<span[^>]*>([^<]+)</span>\s*<div><textarea[^>]*name=\"role_user_message\"',
        text,
        flags=re.I | re.S,
    )
    if not m:
        return None
    val = html.unescape(m.group(1))
    val = re.sub(r"\s+", " ", val).strip()
    return val or None


def extract_memory_items(text: str):
    """Extract memory items from the memory modal table HTML.

    The structure is: <tbody><div class="group ..."><div><div class="whitespace-pre-wrap">MEMORY</div></div></div></tbody>
    """
    import re
    memories = []

    # Look for divs with whitespace-pre-wrap class inside tbody
    # Pattern: <div class="... whitespace-pre-wrap ...">MEMORY TEXT</div>
    wrap_pattern = r'<div[^>]*whitespace-pre-wrap[^>]*>(.*?)</div>'
    matches = re.findall(wrap_pattern, text, flags=re.DOTALL | re.I)

    for match in matches:
        # Clean HTML entities and tags
        memory_text = html.unescape(match)
        memory_text = re.sub(r'<[^>]+>', '', memory_text)  # Remove any nested tags
        memory_text = re.sub(r'\s+', ' ', memory_text).strip()
        if memory_text:
            memories.append(memory_text)

    return memories


def parse_dir(root: Path):
    files = sorted(root.rglob("*.html"))
    rows = []
    for p in files:
        text = p.read_text(errors="ignore")
        persona_name, persona_id = infer_persona_from_path(p)

        # Determine file type: "memories_" prefix = memory modal, "dialog_" = settings
        is_memory_modal = p.name.startswith("memories_")

        if is_memory_modal:
            # Parse memory modal HTML - extract actual memory items
            memories = extract_memory_items(text)
            row = {
                "persona_name": persona_name,
                "persona_id": persona_id,
                "file": str(p.relative_to(root)),
                "memory_count": len(memories),
                "memories": "; ".join(memories) if memories else None,
                # Settings fields are not present in memory modal
                "reference_saved_memories": None,
                "reference_chat_history": None,
                "reference_record_history": None,
                "name_user_message": None,
                "role_user_message": None,
                "role_user_span": None,
                "other_user_message": None,
                "traits_model_message": None,
            }
        else:
            # Parse settings dialog HTML (old behavior)
            row = {
                "persona_name": persona_name,
                "persona_id": persona_id,
                "file": str(p.relative_to(root)),
                "memory_count": None,
                "memories": None,
                "reference_saved_memories": find_switch(text, "Reference saved memories"),
                "reference_chat_history": find_switch(text, "Reference chat history"),
                "reference_record_history": find_switch(text, "Reference record history"),
                "name_user_message": textarea_value(text, "name_user_message"),
                "role_user_message": textarea_value(text, "role_user_message"),
                "role_user_span": occupation_span(text),
                "other_user_message": textarea_value(text, "other_user_message"),
                "traits_model_message": textarea_value(text, "traits_model_message"),
            }
        rows.append(row)
    return rows


def infer_persona_from_path(path: Path):
    m = re.search(r"persona(\d+)", str(path), flags=re.I)
    if not m:
        return None, None
    persona_id = m.group(1)
    return f"persona{persona_id}", persona_id


def summarize(rows):
    def summary(key):
        return Counter(r[key] for r in rows)

    return {
        "persona_name": summary("persona_name"),
        "persona_id": summary("persona_id"),
        "reference_saved_memories": summary("reference_saved_memories"),
        "reference_chat_history": summary("reference_chat_history"),
        "reference_record_history": summary("reference_record_history"),
        "name_user_message": summary("name_user_message"),
        "role_user_message": summary("role_user_message"),
        "role_user_span": summary("role_user_span"),
        "other_user_message": summary("other_user_message"),
        "traits_model_message": summary("traits_model_message"),
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Parse ChatGPT memory snapshot HTML.")
    parser.add_argument(
        "--root",
        default="automated-scraper/data/memory_snapshots",
        help="Root directory containing memory snapshot HTML (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="automated-scraper/data/memory_snapshots/parsed_memory_snapshots.csv",
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary counts to stdout",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = parse_dir(root)
    write_csv(Path(args.out), rows)

    if args.print_summary:
        summary = summarize(rows)
        print(f"files {len(rows)}")
        for k, v in summary.items():
            print(f"{k} {dict(v)}")


if __name__ == "__main__":
    raise SystemExit(main())
