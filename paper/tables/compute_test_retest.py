"""Test-retest outcome agreement R^s per (model, benchmark, surface).

For each item with K paired runs on surface s, let k be the number of runs
where the model got it right. Convention varies by data source to match the
original table:
  * Metabench (ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande): K=5
    always; failed extraction counts as incorrect (binary outcome = "extracted
    AND correct").
  * Intersected (BBQ, AA-Omniscience, Elephant Flip): K = number of *extracted*
    runs per item; items with K_ext<2 on either side are dropped.

The per-item agreement is

    a_i^s = ( k*(k-1) + (K-k)*(K-k-1) ) / ( K*(K-1) )

R^s = mean(a_i^s over items). Paired SE on ΔR = R^API − R^Iface is
std(a_i^API − a_i^Iface) / sqrt(N).

Sources:
  - Metabench benchmarks (ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande):
    coverage_by_run{,_claude,_gemini}.csv picks the 5 first runs per side that
    pass extraction ≥0.90; per-item correctness is read from the scored CSVs
    under experiments/metabench/metabench-<bench>/data-<provider>/{api,interface}/...
    For HellaSwag rerun runs (May 14+) where the API side has no scored CSV,
    the *.api.json files are scored on the fly with the regex used elsewhere.

  - Intersected benchmarks (BBQ, AA-Omniscience, Elephant Flip):
    experiments/plots/per_query.csv (has api_correct/ifc_correct per
    (provider, model, benchmark, run, qid)).

N is the number of items present on both sides with K>=2 — items entirely
missing from a side are dropped. Within those items, K is fixed at the number
of kept runs for the side; failed-extraction runs are treated as incorrect.

Output:
  paper/tables/appendix_test_retest_agreement.csv  (overwrites in place)
  prints a human-readable table.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
PLOTS = REPO / "experiments" / "plots"
LB_PLOTS = REPO / "experiments" / "metabench" / "openllm_leaderboard" / "plots"
METABENCH_ROOT = REPO / "experiments" / "metabench"
OUT_CSV = THIS.parent / "appendix_test_retest_agreement.csv"

EXTRACT_MIN = 0.90
MAX_RUNS = 5

# (label, api condition (cov csv), iface condition (cov csv),
#         api stem patterns (ordered: primary first, fallback after; matches by substring),
#         iface session CSV stem prefix (also used as slug_hint to disambiguate API path),
#         intersected provider, intersected model slug)
MODELS = [
    ("GPT 5.3 Instant",   "API: GPT 5.3 Chat (Instant)", "Interface: Instant",
        ("gpt-5.3-chat-latest",),                                                "instant",
        "chatgpt", "gpt-5-3-instant"),
    ("GPT 5.4 Thinking",  "API: GPT 5.4 Reasoning High", "Interface: Thinking",
        ("gpt-5.4-2026-03-05_reasoning_effort-high", "gpt-5.4-2026-03-05"),     "thinking",
        "chatgpt", "gpt-5-4-thinking"),
    ("Claude Haiku",      "API: Claude Haiku 4.5",       "Interface: Haiku",
        ("claude-haiku-4-5-20251001",),                                          "haiku",
        "claude",  "haiku"),
    ("Claude Opus",       "API: Claude Opus 4.6",        "Interface: Opus",
        ("claude-opus-4-6",),                                                    "opus",
        "claude",  "opus"),
    ("Claude Sonnet",     "API: Claude Sonnet 4.6",      "Interface: Sonnet",
        ("claude-sonnet-4-6",),                                                  "sonnet",
        "claude",  "sonnet"),
    ("Gemini 3 Thinking", "API: Gemini 3 Flash (High)",  "Interface: Thinking",
        ("thinking_level-high",),                                                "thinking",
        "gemini",  "thinking"),
    ("Gemini 3 Fast",     "API: Gemini 3 Flash (Low)",   "Interface: Fast",
        ("thinking_level-low",),                                                 "fast",
        "gemini",  "fast"),
]

# (label, source kind, key)
# For metabench: key = (disk dir name, lowercased coverage_by_run dataset name)
PANELS = [
    ("ARC",            "metabench",   ("metabench-arc",        "metabench-arc")),
    ("GSM8K",          "metabench",   ("metabench-gsm8k",      "metabench-gsm8k")),
    ("HellaSwag",      "metabench",   ("metabench-hellaswag",  "metabench-hellaswag")),
    ("MMLU",           "metabench",   ("metabench-mmlu",       "metabench-mmlu")),
    ("TruthfulQA",     "metabench",   ("metabench-truthfulQA", "metabench-truthfulqa")),
    ("WinoGrande",     "metabench",   ("metabench-winogrande", "metabench-winogrande")),
    ("BBQ",            "intersected", "bbq"),
    ("AA-Omniscience", "intersected", "aa-omniscience"),
    ("Elephant Flip",  "intersected", "elephant-flip"),
]

# Map condition string to provider data dir (for finding per-item CSVs).
PROVIDER_BY_COND = {
    "chatgpt": ("data-chatgpt", "GPT", "Instant", "Thinking", "Auto"),
    "claude":  ("data-claude",  "Claude"),
    "gemini":  ("data-gemini",  "Gemini"),
}

# ---- API answer extraction for May 14+ HellaSwag JSONs ----
FINAL_LABEL_RE = re.compile(r"(?i)(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*\**\s*([ABCDEFGHI])\b")
BOLD_RE = re.compile(r"\*\*\s*([ABCDEFGHI])\s*[\.\)\*]?")
LETTER_RE = re.compile(r"(?im)(?:^|[^A-Za-z])([ABCDEFGHI])(?:[\.\)]|\b)")


def extract_letter(text):
    if not text:
        return None
    s = text.strip()
    for r in (FINAL_LABEL_RE, BOLD_RE, LETTER_RE):
        m = r.search(s)
        if m:
            return m.group(1).upper()
    return None


def parse_qid(qid):
    m = re.match(r"^(\d+)_run\d+$", qid)
    return int(m.group(1)) if m else None


# ---- coverage_by_run loading: pick 5 latest qualifying (timestamp, side) pairs ----
PROV_FILES = {
    "chatgpt": "coverage_by_run.csv",
    "claude": "coverage_by_run_claude.csv",
    "gemini": "coverage_by_run_gemini.csv",
}


def load_kept_runs():
    """Return {(provider, condition, dataset_lower): [timestamps]} kept (latest 5 with ext>=0.90)."""
    raw = defaultdict(list)
    for prov, fname in PROV_FILES.items():
        path = LB_PLOTS / fname
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["complete_run"] != "True":
                    continue
                tot = int(r["total"])
                if tot == 0 or int(r["answered"]) / tot < EXTRACT_MIN:
                    continue
                raw[(prov, r["condition"], r["dataset"].lower())].append(r["timestamp"])
    kept = {}
    for k, ts_list in raw.items():
        # First MAX_RUNS by timestamp order (matches the original capability table convention)
        ts_list = sorted(ts_list)[:MAX_RUNS]
        kept[k] = ts_list
    return kept


# ---- Metabench per-item scoring ----
def short_ts(name):
    s = name.split("-hellaswag")[0]
    for bench in ("-arc", "-gsm8k", "-mmlu", "-truthfulqa", "-truthfulQA",
                  "-winogrande"):
        s = s.split(bench)[0]
    for suf in ("-gemini", "-claude"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


# Fallback session index by model slug, used when CSV stems don't disambiguate
# (e.g. Claude GSM8K's session_00/01/02/metabench-gsm8k.csv all share the same name).
SESSION_BY_SLUG = {
    "haiku": 0, "opus": 1, "sonnet": 2,
    "instant": 0, "thinking": 1, "auto": 2,
    "fast": 0,
}


def find_iface_csv(run_dir, iface_slug):
    """Find the interface scored CSV for a given model slug under a run dir."""
    # 1) Try stem match within each session
    for session in sorted(run_dir.iterdir()):
        if not session.is_dir():
            continue
        for cand in session.glob("*.csv"):
            stem = cand.stem.lower()
            if stem == iface_slug or stem.startswith(iface_slug + "-") or stem.startswith(iface_slug + "_"):
                return cand
    # 2) Try stem match anywhere
    for cand in run_dir.rglob("*.csv"):
        stem = cand.stem.lower()
        if stem == iface_slug or stem.startswith(iface_slug + "-") or stem.startswith(iface_slug + "_"):
            return cand
    # 3) Fallback: use session order. iface_slug → expected session index.
    if iface_slug == "thinking" and (run_dir / "session_01").exists():
        # Gemini layout: session_01=thinking; Claude has no thinking → handled by 1)
        pass
    idx = SESSION_BY_SLUG.get(iface_slug)
    if idx is not None:
        sess = run_dir / f"session_{idx:02d}"
        if sess.is_dir():
            csvs = list(sess.glob("*.csv"))
            if csvs:
                return csvs[0]
    return None


def find_api_correctness(api_run_dir, api_patterns, gold, slug_hint=None):
    """Return {item_id: (extracted, correct)} for the API side.

    api_patterns: ordered tuple of substring patterns. The first pattern with any
    matching CSV wins (so primary names are preferred over fallbacks). If multiple
    CSVs match a pattern, prefer paths whose components include slug_hint.
    """
    for pat in api_patterns:
        candidates = [p for p in api_run_dir.rglob("*.csv")
                      if pat.lower() in p.stem.lower()]
        if not candidates:
            continue
        if slug_hint and len(candidates) > 1:
            slug = slug_hint.lower()
            hinted = [p for p in candidates if any(part.lower() == slug for part in p.parts)]
            if hinted:
                candidates = hinted
        return per_item_from_csv(candidates[0])
    # Fall back to scoring JSONs by the primary pattern
    primary = api_patterns[0]
    for jf in api_run_dir.rglob("*.api.json"):
        parent = jf.parent
        if primary.lower() in parent.name.lower():
            return per_item_from_api_jsons(parent, gold)
    return None


def per_item_from_csv(csv_path):
    """Return {item: (extracted: bool, correct: bool)} from a pre-scored CSV."""
    out = {}
    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            item = parse_qid(r.get("query_id", ""))
            if item is None:
                continue
            has = r.get("_has_answer")
            if has is not None:
                ext = has.strip().lower() == "true"
            else:
                ext = bool((r.get("answer") or "").strip())
            corr = (r.get("correct") or "").strip().lower() == "true"
            out[item] = (ext, corr)
    return out


def per_item_from_api_jsons(api_dir, gold):
    """Return {item: (extracted, correct)} scoring raw API JSONs."""
    out = {}
    for jf in api_dir.glob("*.api.json"):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        item = parse_qid(data.get("query_id", ""))
        if item is None or item not in gold:
            continue
        pred = extract_letter(data.get("response_text") or "")
        ext = pred is not None
        corr = bool(ext and pred == gold[item])
        out[item] = (ext, corr)
    return out


def load_gold_for_bench(bench_dir_name):
    queries_csv = METABENCH_ROOT / bench_dir_name / "queries" / "queries.csv"
    if not queries_csv.exists():
        return {}
    gold = {}
    with queries_csv.open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                gold[int(r["id"])] = (r.get("answer") or "").strip().upper()
            except (KeyError, ValueError):
                continue
    return gold


def collect_metabench_per_item(bench_dir_name, provider_dir, iface_slug, api_patterns, kept_timestamps_iface, kept_timestamps_api):
    """Return ({item: [bool, ...]} iface, {item: [bool, ...]} api) across kept runs."""
    bench_root = METABENCH_ROOT / bench_dir_name
    iface_root = bench_root / provider_dir / "interface"
    api_root = bench_root / provider_dir / "api"

    gold = load_gold_for_bench(bench_dir_name)

    def find_by_short_ts(root, target_ts_set):
        out = {}
        if not root.exists():
            return out
        for d in root.iterdir():
            if not d.is_dir():
                continue
            ts = short_ts(d.name)
            if ts in target_ts_set:
                out.setdefault(ts, d)
        return out

    iface_target = {short_ts(t) for t in kept_timestamps_iface}
    api_target = {short_ts(t) for t in kept_timestamps_api}

    iface_dirs = find_by_short_ts(iface_root, iface_target)
    api_dirs = find_by_short_ts(api_root, api_target)

    # item -> [(extracted, correct), ...] per kept run, in fixed order.
    # compute_cell will (a) drop items with <2 extracted runs on either side
    # and (b) use (ext AND correct) as the binary outcome for the rest.
    iface_items = {it: [] for it in gold}
    api_items = {it: [] for it in gold}

    iface_ts_sorted = sorted(iface_target)
    api_ts_sorted = sorted(api_target)

    for ts in iface_ts_sorted:
        d = iface_dirs.get(ts)
        csv_path = find_iface_csv(d, iface_slug) if d else None
        per = per_item_from_csv(csv_path) if csv_path else {}
        for it in iface_items:
            iface_items[it].append(per.get(it, (False, False)))

    for ts in api_ts_sorted:
        d = api_dirs.get(ts)
        per = find_api_correctness(d, api_patterns, gold, slug_hint=iface_slug) if d else None
        per = per or {}
        for it in api_items:
            api_items[it].append(per.get(it, (False, False)))

    return iface_items, api_items


# ---- Intersected per-item from per_query.csv ----
_INTER_CACHE = None


def load_intersected_per_query():
    """Drop items where any kept run has ifc_extracted=0 or api_extracted=0."""
    global _INTER_CACHE
    if _INTER_CACHE is not None:
        return _INTER_CACHE
    # {(provider, model, benchmark): {item: [(run, ext_api, corr_api, ext_ifc, corr_ifc), ...]}}
    raw = defaultdict(lambda: defaultdict(list))
    path = PLOTS / "per_query.csv"
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["provider"], r["model"], r["benchmark"])
            try:
                qid = int(r["qid"])
            except ValueError:
                continue
            raw[key][qid].append((
                int(r["run"]),
                bool(int(r["api_extracted"])),
                bool(int(r["api_correct"])),
                bool(int(r["ifc_extracted"])),
                bool(int(r["ifc_correct"])),
            ))

    # Store [(extracted, correct), ...] per run; compute_cell handles filtering.
    fixed = {}
    for k, items in raw.items():
        fitems = {}
        for qid, rows in items.items():
            rows = sorted(rows)[:MAX_RUNS]   # first MAX_RUNS runs by index
            if len(rows) < 2:
                continue
            fitems[qid] = {
                "api": [(r[1], r[2]) for r in rows],
                "ifc": [(r[3], r[4]) for r in rows],
            }
        fixed[k] = fitems
    _INTER_CACHE = fixed
    return fixed


# ---- Test-retest math ----
def per_item_agreement(corr_list):
    """Probability two repeated runs concord. corr_list = [bool, ...]."""
    K = len(corr_list)
    if K < 2:
        return None
    k = sum(corr_list)
    return (k * (k - 1) + (K - k) * (K - k - 1)) / (K * (K - 1))


def compute_cell(api_items, ifc_items, failure_mode="incorrect"):
    """api_items / ifc_items: {item: [(extracted, correct), ...]} per kept run.

    failure_mode controls how failed extractions are treated:
      "incorrect": K is the number of kept runs; non-extracted runs contribute
        a False outcome (treats "didn't answer" the same as "wrong answer").
      "ignore":   K per item is the number of *extracted* runs for that side;
        items with K_ext<2 on either side are dropped (no pair to compare).
    """
    shared = sorted(set(api_items) & set(ifc_items))
    a_api, a_ifc, diffs = [], [], []
    for it in shared:
        api_runs = api_items[it]
        ifc_runs = ifc_items[it]
        if failure_mode == "incorrect":
            api_outcomes = [bool(ext and corr) for ext, corr in api_runs]
            ifc_outcomes = [bool(ext and corr) for ext, corr in ifc_runs]
        else:  # "ignore"
            api_outcomes = [bool(corr) for ext, corr in api_runs if ext]
            ifc_outcomes = [bool(corr) for ext, corr in ifc_runs if ext]
            if len(api_outcomes) < 2 or len(ifc_outcomes) < 2:
                continue
        a = per_item_agreement(api_outcomes)
        b = per_item_agreement(ifc_outcomes)
        if a is None or b is None:
            continue
        a_api.append(a)
        a_ifc.append(b)
        diffs.append(a - b)
    if not diffs:
        return None
    n = len(diffs)
    return {
        "N": n,
        "runs": MAX_RUNS,
        "R_api": float(np.mean(a_api)) * 100,
        "R_ifc": float(np.mean(a_ifc)) * 100,
        "diff": float(np.mean(diffs)) * 100,
        "se": float(np.std(diffs, ddof=1) / np.sqrt(n)) * 100 if n > 1 else 0.0,
    }


def main():
    kept = load_kept_runs()
    out_rows = []

    header = f"{'Model':<22} {'Benchmark':<16} {'N':>3} {'R_api':>6} {'R_ifc':>6} {'diff':>7} {'se':>5}"
    print(header)
    print("-" * len(header))

    for (mlabel, api_cond, ifc_cond, api_patterns, iface_slug, prov_inter, mslug_inter) in MODELS:
        # provider dir for metabench is data-<prov>
        if prov_inter == "chatgpt":
            prov_dir = "data-chatgpt"
        elif prov_inter == "claude":
            prov_dir = "data-claude"
        else:
            prov_dir = "data-gemini"

        for (plabel, kind, key) in PANELS:
            if kind == "metabench":
                bench_dir_name, dataset_lower = key
                iface_ts = kept.get((prov_inter, ifc_cond, dataset_lower), [])
                api_ts = kept.get((prov_inter, api_cond, dataset_lower), [])
                iface_items, api_items = collect_metabench_per_item(
                    bench_dir_name, prov_dir, iface_slug, api_patterns, iface_ts, api_ts,
                )
                failure_mode = "incorrect"
            else:
                ipq = load_intersected_per_query()
                item_map = ipq.get((prov_inter, mslug_inter, key), {})
                api_items = {qid: v["api"] for qid, v in item_map.items()}
                iface_items = {qid: v["ifc"] for qid, v in item_map.items()}
                failure_mode = "ignore"

            cell = compute_cell(api_items, iface_items, failure_mode=failure_mode)
            if cell is None:
                print(f"{mlabel:<22} {plabel:<16} {'—':>3} {'—':>6} {'—':>6} {'—':>7} {'—':>5}")
                out_rows.append({
                    "model": mlabel, "benchmark": plabel,
                    "N": "", "runs": "", "api_retest_pct": "",
                    "interface_retest_pct": "", "diff_pp": "", "se_pp": "",
                })
                continue
            print(f"{mlabel:<22} {plabel:<16} {cell['N']:>3} "
                  f"{cell['R_api']:>6.1f} {cell['R_ifc']:>6.1f} "
                  f"{cell['diff']:>+7.1f} {cell['se']:>5.1f}")
            out_rows.append({
                "model": mlabel, "benchmark": plabel,
                "N": cell["N"], "runs": cell["runs"],
                "api_retest_pct": f"{cell['R_api']:.1f}",
                "interface_retest_pct": f"{cell['R_ifc']:.1f}",
                "diff_pp": f"{cell['diff']:.1f}",
                "se_pp": f"{cell['se']:.1f}",
            })
        print()

    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "model", "benchmark", "N", "runs",
            "api_retest_pct", "interface_retest_pct", "diff_pp", "se_pp",
        ])
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {OUT_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
