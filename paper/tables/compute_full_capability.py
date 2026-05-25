"""Build paper/tables/appendix_full_capability.csv from source CSVs.

For each (model, benchmark) cell, picks the latest 5 runs whose
extraction rate is >= 0.90 on the relevant side, pairs API and Interface
by timestamp / run-index, then reports paired accuracy and the paired
standard error of the API - Interface gap.

Selection / extraction-threshold logic matches
paper/tables/compute_extractability.py.

Inputs:
  experiments/metabench/openllm_leaderboard/plots/coverage_by_run{,_claude,_gemini}.csv
    for ARC / GSM8K / HellaSwag / MMLU / TruthfulQA / WinoGrande
  experiments/plots/intersected_comparison.csv
    for BBQ / AA-Omniscience / Elephant Flip

Output:
  paper/tables/appendix_full_capability.csv  (or --out <path>)

Schema:
  model, benchmark, N, runs, api_pct, interface_pct, diff_pp, se_pp

Accuracy convention:
  - metabench: uses the `accuracy` column from coverage_by_run, which
    openllm_leaderboard.py emits as correct/answered (per data_audit.txt
    item 6).
  - intersected: api_correct/api_extracted, ifc_correct/ifc_extracted —
    mirroring the same correct/answered denominator.

Paired SE: sample std dev of per-run (api - ifc) differences divided by
sqrt(n_runs). With <2 runs the SE is 0.0.

IMPORTANT — input drift:
  The hand-curated appendix_full_capability.csv was produced when the
  source CSVs were in an older state. Re-running this producer against
  the current CSVs will yield numerically different values for several
  cells, because:
    - coverage_by_run*.csv has not been regenerated since the May-24
      hellaswag re-runs, the GSM8K 4-run cap, or the correct/answered
      change (see data_audit.txt items 5, 6, 7).
    - intersected_comparison.csv is currently still the regex-scored
      BBQ output; item 8 changed BBQ to LLM-judged and the cached
      regenerated CSV has not been refreshed.
  See data_audit.txt "PENDING REGENERATIONS" for the full list.

  This script is the canonical producer; running it after regenerating
  the source CSVs will reproduce the appendix table from scratch.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
LB_PLOTS = REPO / "experiments" / "metabench" / "openllm_leaderboard" / "plots"
PLOTS = REPO / "experiments" / "plots"
OUT_CSV = THIS.parent / "appendix_full_capability.csv"

EXTRACT_MIN = 0.90
MAX_RUNS = 5

MODELS = [
    ("GPT 5.3 Instant",   "API: GPT 5.3 Chat (Instant)", "Interface: Instant",  "chatgpt", "gpt-5-3-instant"),
    ("GPT 5.4 Thinking",  "API: GPT 5.4 Reasoning High", "Interface: Thinking", "chatgpt", "gpt-5-4-thinking"),
    ("Claude Haiku",      "API: Claude Haiku 4.5",       "Interface: Haiku",    "claude",  "haiku"),
    ("Claude Opus",       "API: Claude Opus 4.6",        "Interface: Opus",     "claude",  "opus"),
    ("Claude Sonnet",     "API: Claude Sonnet 4.6",      "Interface: Sonnet",   "claude",  "sonnet"),
    ("Gemini 3 Thinking", "API: Gemini 3 Flash (High)",  "Interface: Thinking", "gemini",  "thinking"),
    ("Gemini 3 Fast",     "API: Gemini 3 Flash (Low)",   "Interface: Fast",     "gemini",  "fast"),
]

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
    """{(condition, dataset_lower): [(timestamp, accuracy, answered, total), ...]}"""
    by_cell = defaultdict(list)
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
                answered = int(r["answered"])
                ext = answered / total
                if ext < EXTRACT_MIN:
                    continue
                acc = float(r["accuracy"])
                by_cell[(r["condition"], r["dataset"].lower())].append(
                    (r["timestamp"], acc, answered, total))
    return by_cell


def load_intersected():
    """{(provider, model, benchmark): [(run_idx, api_acc, ifc_acc, n_intersected), ...]}

    Only includes runs where BOTH api and ifc have extraction >= 0.90.
    """
    by_cell = defaultdict(list)
    path = PLOTS / "intersected_comparison.csv"
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            n = int(r["n_intersected"])
            if n == 0:
                continue
            ifc_ext = int(r["ifc_extracted"])
            api_ext = int(r["api_extracted"])
            if ifc_ext / n < EXTRACT_MIN or api_ext / n < EXTRACT_MIN:
                continue
            ifc_acc = int(r["ifc_correct"]) / ifc_ext
            api_acc = int(r["api_correct"]) / api_ext
            by_cell[(r["provider"], r["model"], r["benchmark"])].append(
                (int(r["run"]), api_acc, ifc_acc, n))
    return by_cell


def take_latest(seq, n=MAX_RUNS):
    """Sort by first element (timestamp or run idx) ascending, keep last n."""
    return sorted(seq, key=lambda x: x[0])[-n:]


def paired_stats(api_vals, ifc_vals):
    """Return (api_pct, ifc_pct, diff_pp, se_pp) given paired per-run accuracies."""
    api_pct = mean(api_vals) * 100
    ifc_pct = mean(ifc_vals) * 100
    diffs = [a - i for a, i in zip(api_vals, ifc_vals)]
    diff_pp = mean(diffs) * 100
    if len(diffs) < 2:
        se_pp = 0.0
    else:
        se_pp = stdev(diffs) * 100 / math.sqrt(len(diffs))
    return api_pct, ifc_pct, diff_pp, se_pp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_CSV),
                    help="Output CSV path (default: paper/tables/appendix_full_capability.csv)")
    args = ap.parse_args()
    out_path = Path(args.out)

    meta = load_metabench()
    inter = load_intersected()

    out_rows = []
    print(f"{'Model':<20} {'Benchmark':<16} {'N':>4} {'runs':>4} "
          f"{'API%':>6} {'Ifc%':>6} {'Δpp':>7} {'SEpp':>5}")
    print("-" * 78)

    for label, api_cond, ifc_cond, prov, slug in MODELS:
        for panel_label, kind, key in PANELS:
            if kind == "metabench":
                api_seq = take_latest(meta.get((api_cond, key), []))
                ifc_seq = take_latest(meta.get((ifc_cond, key), []))
                # Pair by timestamp.
                api_by_ts = {ts: (acc, answered, total) for ts, acc, answered, total in api_seq}
                ifc_by_ts = {ts: (acc, answered, total) for ts, acc, answered, total in ifc_seq}
                paired_ts = sorted(set(api_by_ts) & set(ifc_by_ts))
                paired_ts = paired_ts[-MAX_RUNS:]
                if not paired_ts:
                    continue
                api_vals = [api_by_ts[t][0] for t in paired_ts]
                ifc_vals = [ifc_by_ts[t][0] for t in paired_ts]
                # N from the mode of `total` across paired runs (should be constant).
                N = api_by_ts[paired_ts[-1]][2]
            else:
                seq = take_latest(inter.get((prov, slug, key), []))
                if not seq:
                    continue
                api_vals = [api_acc for _, api_acc, _, _ in seq]
                ifc_vals = [ifc_acc for _, _, ifc_acc, _ in seq]
                N = seq[-1][3]

            api_pct, ifc_pct, diff_pp, se_pp = paired_stats(api_vals, ifc_vals)
            print(f"{label:<20} {panel_label:<16} {N:>4} {len(api_vals):>4} "
                  f"{api_pct:>6.1f} {ifc_pct:>6.1f} {diff_pp:>+7.1f} {se_pp:>5.1f}")
            out_rows.append({
                "model": label,
                "benchmark": panel_label,
                "N": N,
                "runs": len(api_vals),
                "api_pct": f"{api_pct:.1f}",
                "interface_pct": f"{ifc_pct:.1f}",
                "diff_pp": f"{diff_pp:.1f}",
                "se_pp": f"{se_pp:.1f}",
            })

    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "model", "benchmark", "N", "runs",
            "api_pct", "interface_pct", "diff_pp", "se_pp",
        ])
        w.writeheader()
        w.writerows(out_rows)
    try:
        rel = out_path.relative_to(REPO)
    except ValueError:
        rel = out_path
    print(f"\nWrote {rel}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
