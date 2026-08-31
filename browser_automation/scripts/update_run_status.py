"""Regenerate automated-scraper/run_status.md from scraper data on disk.

After the 2026-05-10 reorganization, scraper data lives at:
  experiments/{bench_dir}/data/{provider}/scraper_runs/raw_html/{timestamp}/session_X/{model}-{bench}/...
"""
import csv
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
SCRAPER_DIR = SCRIPT_DIR.parent
EXP_ROOT = SCRAPER_DIR.parent / "experiments"
BENCH_DIR = SCRAPER_DIR.parent / "benchmark_creation" / "results"

# (benchmark_file_stem, experiment_dir_under_experiments/)
BENCHES = [
    ("aa-omniscience-subset-200", "aa-omniscience"),
    ("bbq-subset-200", "bbq"),
    ("elephant-moral-flip-100", "elephant"),
    ("elephant-moral-og-100", "elephant"),
]

PROVIDERS = {
    "claude":  ["sonnet", "opus", "haiku"],
    "chatgpt": ["gpt-5-4-thinking", "gpt-5-3-instant"],
    "gemini":  ["fast", "thinking"],
}


def load_ids(bench_file):
    with open(BENCH_DIR / f"{bench_file}.csv") as f:
        return {row["id"] for row in csv.DictReader(f)}


def get_session_data(bench_dir, provider, model, bench_file):
    """Walk experiments/{bench_dir}/data/{provider}/scraper_runs/raw_html/."""
    sessions = []
    root = EXP_ROOT / bench_dir / "data" / provider / "scraper_runs" / "raw_html"
    if not root.exists():
        return sessions
    for ts_dir in sorted(root.iterdir()):
        if not ts_dir.is_dir():
            continue
        for sess_dir in ts_dir.iterdir():
            if not sess_dir.is_dir():
                continue
            for inner_dir in sess_dir.iterdir():
                if not inner_dir.is_dir():
                    continue
                if not (inner_dir.name.startswith(f"{model}-") and inner_dir.name.endswith(bench_file)):
                    continue
                ids = {f.stem.split("_run")[0] for f in inner_dir.glob("*.html")}
                if ids:
                    sessions.append({
                        "ts": ts_dir.name,
                        "is_resume": "resume-cache" in inner_dir.name,
                        "ids": ids,
                        "n": len(ids),
                    })
    return sessions


def group_sessions(sessions):
    groups = []
    for s in sessions:
        if s["is_resume"] and groups:
            groups[-1]["resumes"].append(s)
        else:
            groups.append({"main": s, "resumes": []})
    return groups


def main():
    out = []
    out.append("# Scraper Run Status\n")
    out.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
    out.append("Each row = one logical pass. Resume sessions are folded into the prior non-resume session.")
    out.append("- `[complete]` = full coverage (all queries, no overlap)")
    out.append("- `[partial]` = combined union < benchmark size")
    out.append("- `[dup N]` = resume re-ran N queries that were already in the prior session")
    out.append("- **Complete** column = number of complete passes for that model+benchmark\n")

    for prov, models in PROVIDERS.items():
        out.append(f"## {prov}\n")
        for model in models:
            out.append(f"### {model}\n")
            out.append("| Benchmark | Complete | Sessions |")
            out.append("|---|---|---|")
            for bench_file, bench_dir in BENCHES:
                full_size = len(load_ids(bench_file))
                groups = group_sessions(get_session_data(bench_dir, prov, model, bench_file))

                cells = []
                complete_count = 0
                for g in groups:
                    main = g["main"]
                    if main["is_resume"]:
                        cells.append(f"{main['ts']}: {main['n']} (resume) [partial]")
                        continue

                    if not g["resumes"]:
                        if main["n"] == full_size:
                            complete_count += 1
                            marker = " [complete]"
                        else:
                            marker = " [partial]"
                        cells.append(f"{main['ts']}: **{main['n']}**{marker}")
                    else:
                        all_ids = main["ids"].copy()
                        breakdown_parts = [str(main["n"])]
                        overlap_count = 0
                        for r in g["resumes"]:
                            overlap_count += len(all_ids & r["ids"])
                            all_ids = all_ids | r["ids"]
                            breakdown_parts.append(f"{r['n']} (resume)")
                        union_size = len(all_ids)

                        breakdown = " + ".join(breakdown_parts)
                        if overlap_count > 0:
                            marker = f" [dup {overlap_count}]"
                        elif union_size == full_size:
                            complete_count += 1
                            marker = " [complete]"
                        else:
                            marker = " [partial]"
                        cells.append(f"{main['ts']}: {breakdown} = **{union_size}**{marker}")

                short_bench = bench_file.replace("-subset-200", "").replace("-100", "").replace("-moral", "")
                sessions_md = "<br>".join(cells) if cells else "(none)"
                out.append(f"| {short_bench} | **{complete_count}** | {sessions_md} |")
            out.append("")
        out.append("")

    out_path = SCRAPER_DIR / "run_status.md"
    out_path.write_text("\n".join(out))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
