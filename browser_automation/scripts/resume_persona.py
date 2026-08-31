#!/usr/bin/env python3
"""Create a resume file for a partially completed persona.

Usage:
  python resume_persona.py --persona 235 --start-index 59 --log-file data/logs/.../session_01.log
  python resume_persona.py --persona 235 --log-file data/logs/.../session_01.log  # auto-detect
"""
import argparse
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
QUERIES_PARENT = REPO_ROOT / "personaMem"


def find_last_processed(log_file: Path, persona_id: str) -> int | None:
    """Find the last processed query index from the log file."""
    pattern = re.compile(rf"\[(\d+)/\d+\] Processing {persona_id}_")
    last_index = None

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                last_index = int(match.group(1))

    return last_index


def create_resume_file(persona_id: str, start_index: int, output_dir: Path | None = None, source_path: Path | None = None):
    """Create a resume query file starting from start_index."""

    # Find the original query file
    if source_path and source_path.exists():
        source_file = source_path
    else:
        source_file = None
        # Search in priority order: interface_sampled, api_sampled, then others
        search_order = ["queries_interface_sampled", "queries_api_sampled"]

        # Add all other queries_* directories
        for subdir in sorted(QUERIES_PARENT.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("queries") and subdir.name not in search_order:
                search_order.append(subdir.name)

        for dirname in search_order:
            candidate = QUERIES_PARENT / dirname / f"persona{persona_id}-queries.json"
            if candidate.exists():
                source_file = candidate
                break

    if not source_file:
        print(f"ERROR: Could not find persona{persona_id}-queries.json in personaMem/queries*/")
        return None

    print(f"Source file: {source_file}")

    # Load queries
    queries = json.loads(source_file.read_text(encoding='utf-8'))
    print(f"Total queries: {len(queries)}")
    print(f"Resuming from index: {start_index} (1-indexed)")

    # Create resume queries (start_index is 1-indexed, so we skip first start_index-1 items)
    resume_queries = queries[start_index:]
    print(f"Resume queries: {len(resume_queries)}")

    if not resume_queries:
        print("No remaining queries to process!")
        return None

    # Create output file
    if output_dir is None:
        output_dir = source_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"persona{persona_id}-queries-resume.json"
    output_file.write_text(json.dumps(resume_queries, indent=2), encoding='utf-8')

    print(f"\nCreated resume file: {output_file}")
    print(f"  Queries: {len(resume_queries)}")
    print(f"  First query: {resume_queries[0].get('id', 'unknown')}")
    print(f"  Last query: {resume_queries[-1].get('id', 'unknown')}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Create resume query file for partially completed persona")
    parser.add_argument("--persona", required=True, help="Persona ID (e.g., 235)")
    parser.add_argument("--start-index", type=int, help="Index to start from (1-indexed, e.g., 60)")
    parser.add_argument("--log-file", help="Path to log file to auto-detect last processed query")
    parser.add_argument("--output-dir", help="Output directory (default: same as source)")
    parser.add_argument("--source-file", help="Explicit path to source query file")

    args = parser.parse_args()

    start_index = args.start_index
    source_path = Path(args.source_file) if args.source_file else None

    # Auto-detect from log file if provided
    if args.log_file and not start_index:
        log_path = Path(args.log_file)
        if not log_path.exists():
            print(f"ERROR: Log file not found: {log_path}")
            return 1

        last_index = find_last_processed(log_path, args.persona)
        if last_index is None:
            print(f"ERROR: Could not find any processed queries for persona {args.persona} in log")
            return 1

        print(f"Auto-detected: Last processed query index = {last_index}")
        start_index = last_index + 1

    if not start_index:
        print("ERROR: Must provide --start-index or --log-file")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else None
    result = create_resume_file(args.persona, start_index - 1, output_dir, source_path)  # Convert to 0-indexed

    return 0 if result else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
