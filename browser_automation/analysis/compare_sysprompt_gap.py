"""Three-way comparison: API Baseline vs API+SysPrompt vs Interface.

Tests whether injecting the leaked system prompt explains the API↔Interface gap.

Outputs:
  data/api/sysprompt_gap_analysis.csv  — per-cell summary with effect sizes
  printed table

Usage:
  python automated-scraper/compare_sysprompt_gap.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, NormalDist
from scipy import stats  # for paired t-test

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parent
EXPERIMENTS = REPO / "experiments" / "metabench"
DATA = BASE / "data" / "api"

# ---------------------------------------------------------------------------
# Model definitions: (model_display, bench_key, api_csv, iface_csv, iface_session,
#                      provider_dir, sysprompt_prefix)
# ---------------------------------------------------------------------------
# For metabench: per-date-dir is one run; use all date dirs as runs.
# For aa-bbq: use aa_bbq_existing_5run_summary.json for api/interface.

METABENCH_SUBSETS = {
    "arc":         "metabench-arc",
    "gsm8k":       "metabench-gsm8k",
    "hellaswag":   "metabench-hellaswag",
    "mmlu":        "metabench-mmlu",
    "truthfulqa":  "metabench-truthfulQA",
    "winogrande":  "metabench-winogrande",
}

# Focal models: (display, provider_dir, api_csv_stem, iface_filename, iface_session_idx,
#                sysprompt_model_prefix, alt_api_csv_stem, alt_api_subdir)
# alt_api_csv_stem: fallback CSV name if primary not found (e.g., gpt-5.4 used two naming schemes)
# alt_api_subdir: required subdir prefix for alt (e.g., "session_01/thinking") to avoid wrong variant
FOCAL = [
    ("gpt-5.3",          "data-chatgpt", "gpt-5.3-chat-latest",                          "instant",  0, "gpt-5.3-chat-latest_system_prompt",              None,                        None),
    ("gpt-5.4",          "data-chatgpt", "gpt-5.4-2026-03-05_reasoning_effort-high",      "thinking", 1, "gpt-5.4-2026-03-05_reasoning_effort-high_system_prompt", "gpt-5.4-2026-03-05", "thinking"),
    ("claude-opus-4.6",  "data-claude",  "claude-opus-4-6",                               "opus",     1, "claude-opus-4-6_system_prompt",                  None,                        None),
    ("claude-sonnet-4.6","data-claude",  "claude-sonnet-4-6",                             "sonnet",   2, "claude-sonnet-4-6_system_prompt",                None,                        None),
    ("gemini-3-flash",   "data-gemini",  "gemini-3-flash-preview_thinking_level-low",     "fast",     0, "gemini-3-flash-preview_system_prompt",           None,                        None),
]

# aa-omni/bbq summary JSON keys (as in aa_bbq_existing_5run_summary.json)
AA_BBQ_API_KEYS = {
    "gpt-5.3":           ("GPT 5.3",           "GPT 5.3 bbq"),
    "gpt-5.4":           ("GPT 5.4",           "GPT 5.4 bbq"),
    "claude-opus-4.7":   ("Claude Opus 4-7",   "Claude Opus 4-7 bbq"),
    "claude-sonnet-4.6": ("Claude Sonnet 4-6", "Claude Sonnet 4-6 bbq"),
    "gemini-3-flash":    ("Gemini 3 Flash",     "Gemini bbq"),
}
AA_BBQ_IFACE_KEYS = {
    "gpt-5.3":           ("GPT 5.3 interface",   "GPT 5.3 bbq iface"),
    "gpt-5.4":           ("GPT 5.4 interface",   "GPT 5.4 bbq iface"),
    "claude-opus-4.7":   ("Opus interface",      "Opus bbq iface"),
    "claude-sonnet-4.6": ("Sonnet interface",    "Sonnet bbq iface"),
    "gemini-3-flash":    ("Gemini interface",    "Gemini bbq iface"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def acc_of_csv(path: Path) -> float | None:
    """Return accuracy from a scored CSV (correct column)."""
    correct = total = 0
    try:
        with path.open() as f:
            for r in csv.DictReader(f):
                v = r.get("correct", "").strip().lower()
                if v in ("true", "1"): correct += 1; total += 1
                elif v in ("false", "0"): total += 1
    except Exception:
        return None
    return correct / total if total else None


def cohen_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = ((len(a)-1)*stdev(a)**2 + (len(b)-1)*stdev(b)**2) / (len(a)+len(b)-2)
    return (mean(a) - mean(b)) / (pooled**0.5 + 1e-12)


def t_pval(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return p


def fmt(vals: list[float]) -> str:
    if not vals: return "  ---  "
    m = mean(vals)*100
    s = stdev(vals)*100 if len(vals) > 1 else 0
    return f"{m:.1f}±{s:.1f}({len(vals)})"


# ---------------------------------------------------------------------------
# Load +SysPrompt per-run data
# ---------------------------------------------------------------------------

def load_sysprompt() -> dict[tuple[str,str], list[float]]:
    """Returns {(model_prefix_match, bench): [run_acc, ...]}"""
    out: dict = defaultdict(list)
    for run_id in ["batch-system-prompt-aa-bbq-5runs", "batch-system-prompt-metabench-5runs"]:
        d = DATA / run_id / "judged"
        if not d.exists(): continue
        for f in sorted(d.glob("*.csv")):
            stem = f.stem
            dir_part = stem.split("_system_prompt")[0]
            bench_raw = stem.rsplit("__", 1)[-1]
            bench = bench_raw.replace("metabench_","").replace("aa_omniscience","aa-omniscience")

            run_stats: dict[str, list[bool]] = defaultdict(list)
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    run_str = r["run_qid"].rsplit("_run",1)[-1]
                    v = r.get("correct","").strip().lower()
                    if v in ("true","1"): run_stats[run_str].append(True)
                    elif v in ("false","0"): run_stats[run_str].append(False)
            for run_str, vals in run_stats.items():
                out[(dir_part, bench)].append(sum(vals)/len(vals) if vals else 0)
    return dict(out)


# ---------------------------------------------------------------------------
# Load metabench baseline (API) and Interface per-date-dir accuracies
# ---------------------------------------------------------------------------

def load_metabench_baseline(model_dir: str, api_csv_stem: str, iface_filename_hint: str,
                             iface_session_idx: int, alt_csv_stem: str | None = None,
                             alt_subdir: str | None = None) -> dict[str, dict[str, list[float]]]:
    """
    Returns {"api": [acc_per_run], "interface": [acc_per_run]} for each bench.
    """
    result: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for bench_key, bench_dir in METABENCH_SUBSETS.items():
        bench_path = EXPERIMENTS / bench_dir / model_dir

        # --- API ---
        api_dir = bench_path / "api"
        if api_dir.exists():
            for date_dir in sorted(api_dir.iterdir()):
                if not date_dir.is_dir(): continue
                matches = list(date_dir.rglob(f"{api_csv_stem}.csv"))
                if not matches and alt_csv_stem:
                    # For models with alternate naming (e.g. gpt-5.4), search in specific subdir
                    if alt_subdir:
                        for sub in date_dir.rglob(alt_subdir):
                            matches = list(sub.glob(f"{alt_csv_stem}.csv"))
                            if matches: break
                    else:
                        matches = list(date_dir.rglob(f"{alt_csv_stem}.csv"))
                for m in matches:
                    a = acc_of_csv(m)
                    if a is not None:
                        result[bench_key]["api"].append(a)
                        break  # one per date_dir

        # --- Interface ---
        iface_dir = bench_path / "interface"
        if iface_dir.exists():
            for date_dir in sorted(iface_dir.iterdir()):
                if not date_dir.is_dir(): continue
                # session dirs are session_00, session_01, ...
                sess_dir = date_dir / f"session_{iface_session_idx:02d}"
                if not sess_dir.exists(): continue
                # find any CSV in this session dir
                csvs = list(sess_dir.glob("*.csv"))
                # For cases where filename is the generic bench name, just pick any;
                # for named sessions (e.g. opus.csv), prefer matching hint
                picked = None
                for f in csvs:
                    if iface_filename_hint in f.stem:
                        picked = f; break
                if picked is None and csvs:
                    picked = csvs[0]
                if picked:
                    a = acc_of_csv(picked)
                    if a is not None:
                        result[bench_key]["interface"].append(a)

    return result


# ---------------------------------------------------------------------------
# Load aa-omni / bbq baseline from existing JSON summary
# ---------------------------------------------------------------------------

def load_aa_bbq_baseline() -> dict[tuple[str,str,str], list[float]]:
    """Returns {(model, bench, condition): [acc_per_run]}"""
    j = json.loads((DATA / "aa_bbq_existing_5run_summary.json").read_text())
    # key format: "display|bench|source"
    out = {}
    for raw_key, vals in j.items():
        parts = raw_key.split("|")
        if len(parts) == 3:
            out[tuple(parts)] = vals
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sysprompt = load_sysprompt()
    aa_bbq_base = load_aa_bbq_baseline()

    rows = []
    print(f"\n{'Model':20s} {'Bench':12s} {'API Baseline':18s} {'+ SysPrompt':18s} {'Interface':18s} {'Gap→Remain':14s} {'Expl%':6s} p(API↔If) p(SP↔If)")
    print("-"*128)

    for display, model_dir, api_csv_stem, iface_hint, iface_sess_idx, sp_prefix, alt_csv, alt_sub in FOCAL:
        metabench_data = load_metabench_baseline(model_dir, api_csv_stem, iface_hint, iface_sess_idx, alt_csv, alt_sub)

        # Collect +SysPrompt data keyed by partial prefix match
        # For claude-opus: metabench used opus-4.6, aa-bbq used opus-4.7 (different model runs)
        sp_model_stem = sp_prefix.split("_system_prompt")[0]
        sp_aa_bbq_stems = [sp_model_stem]
        if display == "claude-opus-4.6":
            sp_aa_bbq_stems.append("claude-opus-4-7")  # aa-bbq runs used this model

        sp_data: dict[str, list[float]] = {}
        for (k_model, k_bench), accs in sysprompt.items():
            if k_bench in ("aa-omniscience", "bbq"):
                if any(k_model.startswith(s) for s in sp_aa_bbq_stems):
                    sp_data[k_bench] = accs
            else:
                if k_model.startswith(sp_model_stem):
                    sp_data[k_bench] = accs

        # metabench benches
        for bench_key in METABENCH_SUBSETS:
            api_accs = metabench_data[bench_key].get("api", [])
            iface_accs = metabench_data[bench_key].get("interface", [])
            sp_accs = sp_data.get(bench_key, [])

            if not api_accs and not sp_accs: continue

            api_m = mean(api_accs)*100 if api_accs else float("nan")
            sp_m  = mean(sp_accs)*100  if sp_accs  else float("nan")
            iface_m = mean(iface_accs)*100 if iface_accs else float("nan")

            gap_orig   = iface_m - api_m
            gap_remain = iface_m - sp_m
            # pct gap closed: how much of |gap_orig| was |closed| by SP
            # positive = SP moved API closer to Interface; negative = SP moved away
            expl_pct = (1 - abs(gap_remain)/abs(gap_orig))*100 if gap_orig != 0 else float("nan")
            p_api_sp  = t_pval(api_accs, sp_accs)   if api_accs and sp_accs   else float("nan")
            p_api_if  = t_pval(api_accs, iface_accs) if api_accs and iface_accs else float("nan")
            p_sp_if   = t_pval(sp_accs,  iface_accs) if sp_accs  and iface_accs else float("nan")

            rows.append({
                "model": display, "benchmark": bench_key,
                "api_mean": round(api_m,2), "api_n": len(api_accs),
                "api_sd": round(stdev(api_accs)*100,2) if len(api_accs)>1 else 0,
                "sp_mean": round(sp_m,2), "sp_n": len(sp_accs),
                "sp_sd": round(stdev(sp_accs)*100,2) if len(sp_accs)>1 else 0,
                "iface_mean": round(iface_m,2) if iface_m==iface_m else "",
                "iface_n": len(iface_accs),
                "iface_sd": round(stdev(iface_accs)*100,2) if len(iface_accs)>1 else 0,
                "gap_orig": round(gap_orig,2),
                "gap_remain": round(gap_remain,2),
                "pct_explained": round(expl_pct,1) if expl_pct==expl_pct else "",
                "p_api_vs_sp": round(p_api_sp,4) if p_api_sp==p_api_sp else "",
                "p_api_vs_iface": round(p_api_if,4) if p_api_if==p_api_if else "",
                "p_sp_vs_iface": round(p_sp_if,4) if p_sp_if==p_sp_if else "",
            })
            p_if  = f"{p_api_if:.3f}" if p_api_if==p_api_if else "  ---"
            p_sif = f"{p_sp_if:.3f}"  if p_sp_if==p_sp_if   else "  ---"
            print(f"{display:20s} {bench_key:12s} {fmt(api_accs):18s} {fmt(sp_accs):18s} {fmt(iface_accs):18s} "
                  f"{gap_orig:+.1f}→{gap_remain:+.1f}{'   ':4s} {expl_pct:5.0f}%  {p_if}  {p_sif}")

        # aa-omniscience and bbq from JSON summary
        aa_bbq_map = {
            "gpt-5.3":           ("GPT 5.3", "GPT 5.3 bbq", "GPT 5.3 interface", "GPT 5.3 bbq iface"),
            "gpt-5.4":           ("GPT 5.4", "GPT 5.4 bbq", "GPT 5.4 interface", "GPT 5.4 bbq iface"),
            "claude-opus-4.6":   ("Claude Opus 4-7", "Claude Opus 4-7 bbq", "Opus interface", "Opus bbq iface"),
            "claude-sonnet-4.6": ("Claude Sonnet 4-6", "Claude Sonnet 4-6 bbq", "Sonnet interface", "Sonnet bbq iface"),
            "gemini-3-flash":    ("Gemini 3 Flash", "Gemini bbq", "Gemini interface", "Gemini bbq iface"),
        }.get(display, None)

        # For opus, the +SysPrompt aa-bbq was run with opus-4.7 (different model from baseline opus-4.6)
        # Map accordingly
        sp_aa_prefix = sp_prefix.split("_system_prompt")[0]
        # aa-bbq sysprompt uses claude-opus-4-7 for claude-opus, fix prefix
        if display == "claude-opus-4.6":
            sp_aa_prefix = "claude-opus-4-7"

        if aa_bbq_map:
            aa_api_key, bbq_api_key, aa_iface_key, bbq_iface_key = aa_bbq_map
            for bench_name, api_key, iface_key in [
                ("aa-omniscience", aa_api_key, aa_iface_key),
                ("bbq",            bbq_api_key, bbq_iface_key),
            ]:
                api_accs  = aa_bbq_base.get((api_key,  bench_name[:2] if bench_name=="aa-omniscience" else "bbq", "api"), [])
                iface_accs = aa_bbq_base.get((iface_key, bench_name[:2] if bench_name=="aa-omniscience" else "bbq", "interface"), [])
                sp_accs    = sp_data.get(bench_name, [])

                if not api_accs and not sp_accs: continue

                api_m   = mean(api_accs)*100   if api_accs   else float("nan")
                sp_m    = mean(sp_accs)*100    if sp_accs    else float("nan")
                iface_m = mean(iface_accs)*100 if iface_accs else float("nan")
                gap_orig   = iface_m - api_m
                gap_remain = iface_m - sp_m
                expl_pct  = (1 - abs(gap_remain)/abs(gap_orig))*100 if gap_orig and gap_orig==gap_orig else float("nan")
                p_api_sp  = t_pval(api_accs, sp_accs)    if api_accs and sp_accs   else float("nan")
                p_api_if  = t_pval(api_accs, iface_accs) if api_accs and iface_accs else float("nan")
                p_sp_if   = t_pval(sp_accs,  iface_accs) if sp_accs  and iface_accs else float("nan")

                rows.append({
                    "model": display, "benchmark": bench_name,
                    "api_mean": round(api_m,2), "api_n": len(api_accs),
                    "api_sd": round(stdev(api_accs)*100,2) if len(api_accs)>1 else 0,
                    "sp_mean": round(sp_m,2), "sp_n": len(sp_accs),
                    "sp_sd": round(stdev(sp_accs)*100,2) if len(sp_accs)>1 else 0,
                    "iface_mean": round(iface_m,2) if iface_m==iface_m else "",
                    "iface_n": len(iface_accs),
                    "iface_sd": round(stdev(iface_accs)*100,2) if len(iface_accs)>1 else 0,
                    "gap_orig": round(gap_orig,2) if gap_orig==gap_orig else "",
                    "gap_remain": round(gap_remain,2) if gap_remain==gap_remain else "",
                    "pct_explained": round(expl_pct,1) if expl_pct==expl_pct else "",
                    "p_api_vs_sp": round(p_api_sp,4) if p_api_sp==p_api_sp else "",
                    "p_api_vs_iface": round(p_api_if,4) if p_api_if==p_api_if else "",
                    "p_sp_vs_iface": round(p_sp_if,4) if p_sp_if==p_sp_if else "",
                })
                p_if  = f"{p_api_if:.3f}" if p_api_if==p_api_if else "  ---"
                p_sif = f"{p_sp_if:.3f}"  if p_sp_if==p_sp_if   else "  ---"
                print(f"{display:20s} {bench_name:12s} {fmt(api_accs):18s} {fmt(sp_accs):18s} {fmt(iface_accs):18s} "
                      f"{gap_orig:+.1f}→{gap_remain:+.1f}{'   ':4s} {expl_pct:5.0f}%  {p_if}  {p_sif}")
        print()

    out = DATA / "sysprompt_gap_analysis.csv"
    fieldnames = ["model","benchmark","api_mean","api_sd","api_n",
                  "sp_mean","sp_sd","sp_n","iface_mean","iface_sd","iface_n",
                  "gap_orig","gap_remain","pct_explained",
                  "p_api_vs_sp","p_api_vs_iface","p_sp_vs_iface"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
