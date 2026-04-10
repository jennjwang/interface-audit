"""Analyze whether interface accuracy variation correlates with time-of-day.

Reads coverage_by_run CSVs for all providers and extracts hour from the
timestamp column, then computes correlation between hour and accuracy.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent / "openllm_leaderboard" / "plots"

PROVIDER_CSVS = {
    "ChatGPT": PLOTS_DIR / "coverage_by_run.csv",
    "Claude":  PLOTS_DIR / "coverage_by_run_claude.csv",
    "Gemini":  PLOTS_DIR / "coverage_by_run_gemini.csv",
}

# Time buckets: night (0-5), morning (6-11), afternoon (12-17), evening (18-23)
TIME_BUCKETS = {
    "night (0–5h)":     (0, 5),
    "morning (6–11h)":  (6, 11),
    "afternoon (12–17h)": (12, 17),
    "evening (18–23h)": (18, 23),
}


def parse_hour(timestamp: str) -> int | None:
    """Extract hour from timestamp like '2026-03-07_13-09-25-hellaswag'."""
    m = re.match(r"\d{4}-\d{2}-\d{2}_(\d{2})-\d{2}-\d{2}", timestamp)
    if m:
        return int(m.group(1))
    return None


def bucket_label(hour: int) -> str:
    for label, (lo, hi) in TIME_BUCKETS.items():
        if lo <= hour <= hi:
            return label
    return "unknown"


def load_all_rows() -> list[dict]:
    """Load rows from all provider CSVs, tagging each with provider."""
    all_rows = []
    for provider, csv_path in PROVIDER_CSVS.items():
        if not csv_path.exists():
            print(f"  Skipping {provider}: {csv_path} not found")
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["provider"] = provider
                all_rows.append(r)
    return all_rows


def analyze_provider(rows: list[dict], provider: str) -> None:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    by_cond_dataset: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for r in rows:
        cond = (r.get("condition") or "").strip()
        dataset = (r.get("dataset") or "").strip()
        ts = (r.get("timestamp") or "").strip()
        hour = parse_hour(ts)
        try:
            acc = float(r["accuracy"])
        except (TypeError, ValueError, KeyError):
            continue
        if not cond or hour is None:
            continue
        entry = {"hour": hour, "accuracy": acc, "dataset": dataset, "timestamp": ts,
                 "bucket": bucket_label(hour)}
        by_cond[cond].append(entry)
        by_cond_dataset[(cond, dataset)].append(entry)

    # ── Correlation per condition ──
    print(f"\n  {'Condition':<35s}  {'r':>7s}  {'p':>8s}  {'n':>4s}  {'hour_range':>12s}  {'acc_range':>15s}")
    print(f"  {'-' * 87}")

    api_hours, api_accs = [], []
    iface_hours, iface_accs = [], []

    for cond in sorted(by_cond.keys()):
        entries = by_cond[cond]
        hours = np.array([e["hour"] for e in entries])
        accs = np.array([e["accuracy"] for e in entries])
        if len(hours) < 3:
            print(f"  {cond:<35s}  (n={len(hours)}, too few)")
            continue
        r_val, p_val = stats.pearsonr(hours, accs)
        sig = " *" if p_val < 0.05 else ""
        print(f"  {cond:<35s}  {r_val:>7.3f}  {p_val:>8.4f}  {len(hours):>4d}  "
              f"{hours.min():>2d}–{hours.max():<2d}       "
              f"{accs.min():.3f}–{accs.max():.3f}{sig}")

        if cond.startswith("API"):
            api_hours.extend(hours)
            api_accs.extend(accs)
        else:
            iface_hours.extend(hours)
            iface_accs.extend(accs)

    # Aggregate
    print()
    if len(api_hours) >= 3:
        r, p = stats.pearsonr(api_hours, api_accs)
        sig = " *" if p < 0.05 else ""
        print(f"  Aggregate API:        r={r:.3f}, p={p:.4f}, n={len(api_hours)}{sig}")
    if len(iface_hours) >= 3:
        r, p = stats.pearsonr(iface_hours, iface_accs)
        sig = " *" if p < 0.05 else ""
        print(f"  Aggregate Interface:  r={r:.3f}, p={p:.4f}, n={len(iface_hours)}{sig}")

    # ── Time bucket breakdown (Interface only) ──
    print(f"\n  Time bucket breakdown (Interface conditions):")
    print(f"  {'Condition':<30s}", end="")
    for bl in TIME_BUCKETS:
        print(f"  {bl:>20s}", end="")
    print(f"  {'ANOVA p':>10s}")
    print(f"  {'-' * 120}")

    for cond in sorted(by_cond.keys()):
        if not cond.startswith("Interface"):
            continue
        entries = by_cond[cond]
        if len(entries) < 4:
            continue
        bucket_accs = defaultdict(list)
        for e in entries:
            bucket_accs[e["bucket"]].append(e["accuracy"])

        print(f"  {cond:<30s}", end="")
        group_lists = []
        for bl in TIME_BUCKETS:
            vals = bucket_accs.get(bl, [])
            if vals:
                print(f"  {np.mean(vals):.3f}±{np.std(vals):.3f} n={len(vals):>2d}", end="")
                group_lists.append(vals)
            else:
                print(f"  {'—':>20s}", end="")

        # One-way ANOVA across buckets (if ≥2 groups with ≥2 obs)
        valid_groups = [g for g in group_lists if len(g) >= 2]
        if len(valid_groups) >= 2:
            f_stat, anova_p = stats.f_oneway(*valid_groups)
            sig = " *" if anova_p < 0.05 else ""
            print(f"  {anova_p:>8.4f}{sig}", end="")
        print()

    # ── Per-benchmark breakdown (Interface only) ──
    print(f"\n  Per-benchmark (Interface only):")
    datasets = sorted(set(r.get("dataset", "") for r in rows))
    for dataset in datasets:
        has_data = False
        for cond in sorted(by_cond.keys()):
            if not cond.startswith("Interface"):
                continue
            entries = by_cond_dataset.get((cond, dataset), [])
            if len(entries) >= 3:
                has_data = True
                break
        if not has_data:
            continue
        print(f"    {dataset}")
        for cond in sorted(by_cond.keys()):
            if not cond.startswith("Interface"):
                continue
            entries = by_cond_dataset.get((cond, dataset), [])
            if len(entries) < 3:
                continue
            hours = np.array([e["hour"] for e in entries])
            accs = np.array([e["accuracy"] for e in entries])
            r_val, p_val = stats.pearsonr(hours, accs)
            sig = " *" if p_val < 0.05 else ""
            print(f"      {cond:<30s}  r={r_val:>7.3f}  p={p_val:.4f}  n={len(hours)}{sig}")

    # ── Run-order analysis ──
    print(f"\n  Run-order vs accuracy:")
    by_cond_run: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        cond = (r.get("condition") or "").strip()
        try:
            run_idx = int(r["run_index"])
            acc = float(r["accuracy"])
        except (TypeError, ValueError, KeyError):
            continue
        if cond:
            by_cond_run[cond].append((run_idx, acc))

    print(f"  {'Condition':<35s}  {'r':>7s}  {'p':>8s}  {'n':>4s}")
    print(f"  {'-' * 60}")
    for cond in sorted(by_cond_run.keys()):
        pairs = by_cond_run[cond]
        if len(pairs) < 3:
            continue
        runs = np.array([p[0] for p in pairs])
        accs = np.array([p[1] for p in pairs])
        r_val, p_val = stats.pearsonr(runs, accs)
        sig = " *" if p_val < 0.05 else ""
        print(f"  {cond:<35s}  {r_val:>7.3f}  {p_val:>8.4f}  {len(pairs):>4d}{sig}")


def main():
    all_rows = load_all_rows()

    for provider in PROVIDER_CSVS:
        provider_rows = [r for r in all_rows if r.get("provider") == provider]
        if not provider_rows:
            continue
        print(f"\n{'=' * 80}")
        print(f"  {provider.upper()} (n={len(provider_rows)} rows)")
        print(f"{'=' * 80}")
        analyze_provider(provider_rows, provider)

    # ── Cross-provider summary ──
    print(f"\n{'=' * 80}")
    print(f"  CROSS-PROVIDER SUMMARY")
    print(f"{'=' * 80}")

    for provider in PROVIDER_CSVS:
        provider_rows = [r for r in all_rows if r.get("provider") == provider]
        iface_entries = []
        for r in provider_rows:
            cond = (r.get("condition") or "").strip()
            if not cond.startswith("Interface"):
                continue
            ts = (r.get("timestamp") or "").strip()
            hour = parse_hour(ts)
            try:
                acc = float(r["accuracy"])
            except (TypeError, ValueError, KeyError):
                continue
            if hour is not None:
                iface_entries.append({"hour": hour, "accuracy": acc})

        if len(iface_entries) < 3:
            continue
        hours = np.array([e["hour"] for e in iface_entries])
        accs = np.array([e["accuracy"] for e in iface_entries])
        r_val, p_val = stats.pearsonr(hours, accs)
        acc_std = np.std(accs)
        sig = " *" if p_val < 0.05 else ""
        print(f"  {provider:<10s}  Interface: r={r_val:.3f}, p={p_val:.4f}, n={len(hours)}, "
              f"acc_std={acc_std:.4f}{sig}")

        # Bucket means
        bucket_means = {}
        for bl, (lo, hi) in TIME_BUCKETS.items():
            mask = (hours >= lo) & (hours <= hi)
            if mask.sum() > 0:
                bucket_means[bl] = (accs[mask].mean(), mask.sum())
        for bl, (m, n) in bucket_means.items():
            print(f"    {bl}: {m:.3f} (n={n})")


if __name__ == "__main__":
    main()
