"""Per-cell extraction rates (API and Interface) for the appendix capability table.

For each (model, benchmark) cell, takes the latest 5 runs whose extraction
rate is >= 0.90 on the relevant side, then reports the mean extraction rate.
This matches the run-selection logic used to produce appendix_full_capability.csv.

Inputs:
  experiments/metabench/openllm_leaderboard/plots/coverage_by_run{,_claude,_gemini}.csv
    for ARC / GSM8K / HellaSwag / MMLU / TruthfulQA / WinoGrande
  experiments/plots/intersected_comparison.csv
    for BBQ / AA-Omniscience / Elephant Flip

Output:
  paper/tables/appendix_full_capability_extractability.csv
  (also prints a human-readable table)
"""
import csv
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
PLOTS = REPO / "experiments" / "plots"
LB_PLOTS = REPO / "experiments" / "metabench" / "openllm_leaderboard" / "plots"
OUT_CSV = THIS.parent / "appendix_extractability.csv"

EXTRACT_MIN = 0.90
MAX_RUNS = 5

# (label, metabench API condition, metabench Iface condition,
#         intersected provider, intersected model slug)
MODELS = [
    ("GPT 5.3 Instant",   "API: GPT 5.3 Chat (Instant)", "Interface: Instant",  "chatgpt", "gpt-5-3-instant"),
    ("GPT 5.4 Thinking",  "API: GPT 5.4 Reasoning High", "Interface: Thinking", "chatgpt", "gpt-5-4-thinking"),
    ("Claude Haiku",      "API: Claude Haiku 4.5",       "Interface: Haiku",    "claude",  "haiku"),
    ("Claude Opus",       "API: Claude Opus 4.6",        "Interface: Opus",     "claude",  "opus"),
    ("Claude Sonnet",     "API: Claude Sonnet 4.6",      "Interface: Sonnet",   "claude",  "sonnet"),
    ("Gemini 3 Thinking", "API: Gemini 3 Flash (High)",  "Interface: Thinking", "gemini",  "thinking"),
    ("Gemini 3 Fast",     "API: Gemini 3 Flash (Low)",   "Interface: Fast",     "gemini",  "fast"),
]

# (panel label, source kind, key in csv)
PANELS = [
    ("ARC",            "metabench",   "metabench-arc"),
    ("GSM8K",          "metabench",   "metabench-gsm8k"),
    ("HellaSwag",      "metabench",   "metabench-hellaswag"),
    ("MMLU",           "metabench",   "metabench-mmlu"),
    ("TruthfulQA",     "metabench",   "metabench-truthfulqa"),
    ("WinoGrande",     "metabench",   "metabench-winogrande"),
    ("BBQ",            "intersected", "bbq"),
    ("AA-Omniscience", "intersected", "aa-omniscience"),
    ("Elephant Flip",  "intersected", "elephant-flip"),
]


def load_metabench():
    """Return {(condition, dataset_lower): [(timestamp, extraction)]}."""
    runs = defaultdict(list)
    for fname in ("coverage_by_run.csv", "coverage_by_run_claude.csv", "coverage_by_run_gemini.csv"):
        path = LB_PLOTS / fname
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["complete_run"] != "True":
                    continue
                total = int(r["total"])
                if total == 0:
                    continue
                ext = int(r["answered"]) / total
                if ext < EXTRACT_MIN:
                    continue
                runs[(r["condition"], r["dataset"].lower())].append((r["timestamp"], ext))
    return runs


def load_intersected():
    """Return {(provider, model, benchmark, side): [(run_index, extraction)]}."""
    runs = defaultdict(list)
    path = PLOTS / "intersected_comparison.csv"
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            n = int(r["n_intersected"])
            if n == 0:
                continue
            ife = int(r["ifc_extracted"]) / n
            apie = int(r["api_extracted"]) / n
            if ife < EXTRACT_MIN or apie < EXTRACT_MIN:
                continue
            run_idx = int(r["run"])
            runs[(r["provider"], r["model"], r["benchmark"], "ifc")].append((run_idx, ife))
            runs[(r["provider"], r["model"], r["benchmark"], "api")].append((run_idx, apie))
    return runs


def mean_latest_n(seq, n=MAX_RUNS):
    if not seq:
        return None, 0
    seq = sorted(seq, key=lambda x: x[0])[-n:]
    vals = [v for _, v in seq]
    return sum(vals) / len(vals), len(vals)


def main():
    meta = load_metabench()
    inter = load_intersected()

    header = f"{'Model':<22} {'Benchmark':<16} {'API ext%':>8} {'Iface ext%':>10} {'API n':>5} {'Ifc n':>5}"
    print(header)
    print("-" * len(header))

    out_rows = []
    for mlabel, api_cond, ifc_cond, prov, mshort in MODELS:
        for plabel, kind, key in PANELS:
            if kind == "metabench":
                api_seq = meta.get((api_cond, key), [])
                ifc_seq = meta.get((ifc_cond, key), [])
            else:
                api_seq = inter.get((prov, mshort, key, "api"), [])
                ifc_seq = inter.get((prov, mshort, key, "ifc"), [])

            api_mean, api_n = mean_latest_n(api_seq)
            ifc_mean, ifc_n = mean_latest_n(ifc_seq)

            api_str = f"{api_mean * 100:.1f}" if api_mean is not None else "—"
            ifc_str = f"{ifc_mean * 100:.1f}" if ifc_mean is not None else "—"
            print(f"{mlabel:<22} {plabel:<16} {api_str:>8} {ifc_str:>10} {api_n:>5} {ifc_n:>5}")
            out_rows.append({
                "model": mlabel, "benchmark": plabel,
                "api_extract_pct": f"{api_mean * 100:.2f}" if api_mean is not None else "",
                "interface_extract_pct": f"{ifc_mean * 100:.2f}" if ifc_mean is not None else "",
                "api_runs": api_n, "interface_runs": ifc_n,
            })
        print()

    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "model", "benchmark", "api_extract_pct", "interface_extract_pct",
            "api_runs", "interface_runs",
        ])
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {OUT_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
