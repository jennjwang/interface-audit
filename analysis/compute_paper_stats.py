"""Reproduce all statistical results from the paper.

Reads item-level data from release/data/ and computes:
  1. Overall LME (§app:stat_overall)
  2. Cluster bootstrap (§app:stat_overall)
  3. Per-system LME (§app:stat_per_system, Table per_system_lme)
  4. Per-cell LME with BH correction (§app:stat_per_cell, Table per_cell_lme)
  5. Per-benchmark bootstrap (§app:stat_per_benchmark_boot, Table per_bench_boot)
  6. Full per-cell accuracy table (§app:full_accuracy_results, Table appendix_full_capability)
  7. Test-retest reliability (§app:stat_test_retest)
  8. Rank stability / Spearman (§app:rank-stability)

Usage:
    cd release
    python analysis/compute_paper_stats.py           # print to stdout
    python analysis/compute_paper_stats.py --latex    # also write tables.tex

Bootstrap seeds (for reproducibility):
    - Overall cluster bootstrap:      seed=42, n=10,000
    - Per-benchmark bootstrap:        seed=42, n=10,000
    - Per-cell test-retest bootstrap:  seed=42, n=5,000
    - Per-system test-retest bootstrap: seed=123, n=10,000
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, binomtest

warnings.filterwarnings("ignore", category=DeprecationWarning)

RELEASE = Path(__file__).resolve().parent.parent
DATA = RELEASE / "data"

# ── Load per-run accuracy data ───────────────────────────────────────────────
def load_accuracy_csv():
    """Load accuracy_per_run.csv into a DataFrame."""
    return pd.read_csv(RELEASE / "analysis" / "artifacts" / "data" / "accuracy_per_run.csv")


# ── Build item-level dataset ─────────────────────────────────────────────────
def _load_scored_run(run_dir, bench):
    """Load scored.csv, return {qid: 0/1}.

    For metabench: compare answer vs gold_answer.
    For custom benchmarks: use pre-computed correct column (from union extraction).
    """
    csv_path = run_dir / "scored.csv"
    if not csv_path.exists():
        return {}
    items = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            qid = str(r.get("id", "")).strip()
            if not qid:
                continue
            if bench.startswith("metabench-"):
                ans = r.get("answer", "").strip()
                gold = r.get("gold_answer", "").strip()
                if ans and gold:
                    items[qid] = 1 if ans == gold else 0
            else:
                ans = r.get("answer", "").strip()
                correct = r.get("correct", "").strip()
                if ans and correct in ("True", "False"):
                    items[qid] = 1 if correct == "True" else 0
    return items


def build_item_level_data():
    """Build a long-format DataFrame: one row per (benchmark, model, surface, run, item).

    Columns: benchmark, model, surface, run, qid, correct (0/1), is_api (0/1)

    Reads scored.csv for ALL benchmarks (unified pipeline).
    Per-run cross-surface intersection: only items extracted on both surfaces.
    """
    rows = []

    for bench_dir in sorted(DATA.iterdir()):
        if not bench_dir.is_dir() or bench_dir.name in ("answer_keys", "caches", "ablations",
                                                          "human_validation"):
            continue
        bench = bench_dir.name
        if bench == "elephant-og":
            continue  # scored via elephant-flip

        for model_dir in sorted(bench_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name

            for run_idx in range(5):
                api_dir = model_dir / "api" / f"run_{run_idx}"
                ifc_dir = model_dir / "interface" / f"run_{run_idx}"

                api_scores = _load_scored_run(api_dir, bench)
                ifc_scores = _load_scored_run(ifc_dir, bench)

                # Cross-surface intersection for this run
                common = set(api_scores.keys()) & set(ifc_scores.keys())
                for qid in common:
                    rows.append({
                        "benchmark": bench, "model": model,
                        "run": run_idx, "qid": qid,
                        "surface": "api", "is_api": 1,
                        "correct": api_scores[qid],
                    })
                    rows.append({
                        "benchmark": bench, "model": model,
                        "run": run_idx, "qid": qid,
                        "surface": "interface", "is_api": 0,
                        "correct": ifc_scores[qid],
                    })

    return pd.DataFrame(rows)


def write_accuracy_csv(df):
    """Write accuracy_per_run.csv from the item-level DataFrame."""
    output_rows = []
    for (bench, model, run, surface), g in df.groupby(["benchmark", "model", "run", "surface"]):
        n = len(g)
        acc = g["correct"].mean()
        output_rows.append({
            "benchmark": bench, "model": model, "surface": surface,
            "run": f"run_{run}", "n_common": n, "n_scored": n,
            "accuracy": round(acc, 6),
        })

    out = RELEASE / "analysis" / "artifacts" / "data" / "accuracy_per_run.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "benchmark", "model", "surface", "run", "n_common", "n_scored", "accuracy",
        ])
        w.writeheader()
        w.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {out}")


# ── Section 1: Overall LME ──────────────────────────────────────────────────
def overall_lme(df):
    """Fit the overall linear mixed-effects model."""
    print("\n" + "=" * 60)
    print("  §app:stat_overall — Overall LME")
    print("=" * 60)

    # y ~ is_api + (is_api | benchmark) + (1 | model) + (1 | qid_nested)
    df = df.copy()
    df["qid_nested"] = df["benchmark"] + "_" + df["qid"].astype(str)

    try:
        md = smf.mixedlm(
            "correct ~ is_api",
            data=df,
            groups="benchmark",
            re_formula="~is_api",
            vc_formula={"model": "0 + C(model)", "qid_nested": "0 + C(qid_nested)"},
        )
        result = md.fit(reml=False)

        beta1 = result.fe_params["is_api"]
        se = result.bse["is_api"]
        z = result.tvalues["is_api"]
        p = result.pvalues["is_api"]
        ci_lo = beta1 - 1.96 * se
        ci_hi = beta1 + 1.96 * se

        print(f"\nβ̂₁ = {beta1*100:+.2f} pp (SE = {se*100:.2f}, z = {z:.2f}, p = {p:.4f})")
        print(f"95% CI [{ci_lo*100:.2f}, {ci_hi*100:.2f}]")

        # Variance components
        print(f"\nVariance components:")
        # Random effects (benchmark intercept and slope)
        cov_re = result.cov_re
        print(f"  σ²_benchmark = {cov_re.iloc[0,0]:.4f}")
        if cov_re.shape[0] > 1:
            print(f"  σ²_is_api|benchmark = {cov_re.iloc[1,1]:.4f}")
        # Variance components (model, item)
        vc = result.vcomp
        vc_names = list(md.exog_vc.names) if hasattr(md.exog_vc, 'names') else ["model", "qid_nested"]
        if isinstance(vc, np.ndarray):
            for i, name in enumerate(vc_names):
                if i < len(vc):
                    print(f"  σ²_{name} = {vc[i]:.4f}")
        elif hasattr(vc, 'items'):
            for name, val in vc.items():
                print(f"  σ²_{name} = {val:.4f}")
    except Exception as e:
        print(f"  LME failed: {e}")
        print("  Falling back to simple mean...")
        api = df[df["is_api"] == 1]["correct"].mean()
        ifc = df[df["is_api"] == 0]["correct"].mean()
        print(f"  Simple gap: {(api - ifc)*100:+.2f} pp (API={api*100:.1f}%, IFC={ifc*100:.1f}%)")


# ── Section 2: Cluster bootstrap (vectorized) ────────────────────────────────
def _build_stratum_arrays(df):
    """Pre-aggregate into per-qid mean arrays for fast bootstrap."""
    agg = df.groupby(["benchmark", "model", "qid", "is_api"])["correct"].mean().reset_index()
    strata = []
    for (bench, model), g in agg.groupby(["benchmark", "model"]):
        api_df = g[g["is_api"] == 1].set_index("qid")["correct"]
        ifc_df = g[g["is_api"] == 0].set_index("qid")["correct"]
        common = sorted(set(api_df.index) & set(ifc_df.index))
        if common:
            strata.append((api_df.loc[common].values, ifc_df.loc[common].values))
    return strata


def cluster_bootstrap(df, n_iter=10000):
    """Cluster bootstrap: resample items within benchmark-system strata."""
    print("\n" + "=" * 60)
    print("  §app:stat_overall — Cluster Bootstrap")
    print("=" * 60)

    rng = np.random.default_rng(42)
    strata = _build_stratum_arrays(df)

    gaps = np.empty(n_iter)
    for i in range(n_iter):
        api_total = ifc_total = 0.0
        n_total = 0
        for api_arr, ifc_arr in strata:
            n = len(api_arr)
            idx = rng.integers(0, n, size=n)
            api_total += api_arr[idx].sum()
            ifc_total += ifc_arr[idx].sum()
            n_total += n
        gaps[i] = (api_total - ifc_total) / n_total * 100

    print(f"\nBootstrap estimate: {np.mean(gaps):+.2f} pp")
    print(f"95% CI [{np.percentile(gaps, 2.5):.2f}, {np.percentile(gaps, 97.5):.2f}]")
    p = min(1.0, 2 * min(np.mean(gaps <= 0), np.mean(gaps >= 0)))
    print(f"p {'< 10⁻⁴' if p < 0.0001 else f'= {p:.4f}'}")


# ── Section 3: Per-system LME ────────────────────────────────────────────────
def per_system_lme(df):
    """Per-system LME: correct ~ is_api + (1 | benchmark) for each system."""
    print("\n" + "=" * 60)
    print("  §app:stat_per_system — Per-System LME")
    print("=" * 60)

    print(f"\n{'System':<22} {'API%':>5} {'IFC%':>5} {'β̂₁':>7} {'SE':>6} {'z':>7} {'p':>12}")
    print("-" * 75)

    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].copy()
        api_acc = sub[sub["is_api"] == 1]["correct"].mean() * 100
        ifc_acc = sub[sub["is_api"] == 0]["correct"].mean() * 100

        try:
            md = smf.mixedlm("correct ~ is_api", data=sub, groups="benchmark")
            result = md.fit(reml=False)
            beta1 = result.fe_params["is_api"] * 100
            se = result.bse["is_api"] * 100
            z = result.tvalues["is_api"]
            p = result.pvalues["is_api"]
            p_str = f"{p:.1e}" if p < 0.001 else f"{p:.4f}"
            print(f"{model:<22} {api_acc:>5.1f} {ifc_acc:>5.1f} {beta1:>+7.2f} {se:>6.2f} {z:>7.2f} {p_str:>12}")
        except Exception as e:
            print(f"{model:<22} {api_acc:>5.1f} {ifc_acc:>5.1f}  (LME failed: {e})")


# ── Section 4: Per-cell LME ──────────────────────────────────────────────────
def per_cell_lme(df):
    """Per-cell LME: correct ~ is_api + (1 | item) for each system-benchmark cell."""
    print("\n" + "=" * 60)
    print("  §app:stat_per_cell — Per-Cell LME")
    print("=" * 60)

    results = []
    for (bench, model), group in df.groupby(["benchmark", "model"]):
        n = group["qid"].nunique()
        try:
            sub = group.copy()
            sub["qid_str"] = sub["qid"].astype(str)
            md = smf.mixedlm("correct ~ is_api", data=sub, groups="qid_str")
            res = md.fit(reml=False)
            beta1 = res.fe_params["is_api"] * 100
            se = res.bse["is_api"] * 100
            z = res.tvalues["is_api"]
            p = res.pvalues["is_api"]
        except Exception:
            api_m = group[group["is_api"]==1]["correct"].mean()
            ifc_m = group[group["is_api"]==0]["correct"].mean()
            beta1 = (api_m - ifc_m) * 100
            se = z = p = float("nan")

        results.append({"model": model, "benchmark": bench, "beta1": beta1,
                        "se": se, "z": z, "p": p, "n": n})

    # BH correction (Benjamini-Hochberg)
    ps = [r["p"] for r in results if not np.isnan(r["p"])]
    ps_sorted = sorted(enumerate(ps), key=lambda x: x[1])
    m = len(ps)
    bh = [None] * m
    for rank, (idx, pval) in enumerate(ps_sorted, 1):
        bh[idx] = min(1.0, pval * m / rank)
    # Enforce monotonicity in reverse sorted order
    cum_min = 1.0
    for rank in range(m, 0, -1):
        idx = ps_sorted[rank - 1][0]
        cum_min = min(cum_min, bh[idx])
        bh[idx] = cum_min

    sig_05 = sum(1 for q in bh if q is not None and q < 0.05)
    sig_01 = sum(1 for q in bh if q is not None and q < 0.01)

    print(f"\n{sig_05} cells at q < 0.05, {sig_01} cells at q < 0.01")
    print(f"\n{'Model':<22} {'Benchmark':<22} {'β̂₁':>7} {'SE':>6} {'z':>7} {'p':>9} {'n':>4}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: (x["model"], x["benchmark"])):
        p_str = f"{r['p']:.4f}" if not np.isnan(r["p"]) else "N/A"
        if r["p"] < 0.0001:
            p_str = "<.0001"
        print(f"{r['model']:<22} {r['benchmark']:<22} {r['beta1']:>+7.2f} {r['se']:>6.2f} {r['z']:>7.2f} {p_str:>9} {r['n']:>4}")

    # Write per-cell CSV for downstream use (e.g., heatmap)
    out_csv = RELEASE / "analysis" / "artifacts" / "data" / "per_cell_lme.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "benchmark", "beta1_pp", "se", "z", "p", "n"])
        w.writeheader()
        for r in results:
            w.writerow({"model": r["model"], "benchmark": r["benchmark"],
                        "beta1_pp": round(r["beta1"], 2), "se": round(r["se"], 2),
                        "z": round(r["z"], 2), "p": r["p"], "n": r["n"]})
    print(f"Wrote {out_csv}")


# ── Section 5: Per-benchmark bootstrap ────────────────────────────────────────
def per_benchmark_bootstrap(df, n_iter=10000):
    """Per-benchmark cluster bootstrap."""
    print("\n" + "=" * 60)
    print("  §app:stat_per_benchmark_boot — Per-Benchmark Bootstrap")
    print("=" * 60)

    BENCH_ORDER = ["metabench-arc", "metabench-gsm8k", "metabench-hellaswag",
                   "metabench-mmlu", "metabench-truthfulQA", "metabench-winogrande",
                   "bbq", "aa-omniscience", "elephant-flip"]

    rng = np.random.default_rng(42)
    agg = df.groupby(["benchmark", "model", "qid", "is_api"])["correct"].mean().reset_index()

    print(f"\n{'Benchmark':<16} {'API%':>5} {'IFC%':>5} {'Gap':>7} {'CI_lo':>7} {'CI_hi':>7} {'Sig':>4}")
    print("-" * 60)

    for bench in BENCH_ORDER:
        bdf = df[df["benchmark"] == bench]
        api_mean = bdf[bdf["is_api"] == 1]["correct"].mean() * 100
        ifc_mean = bdf[bdf["is_api"] == 0]["correct"].mean() * 100

        bagg = agg[agg["benchmark"] == bench]
        model_data = []
        for model, g in bagg.groupby("model"):
            api_df = g[g["is_api"] == 1].set_index("qid")["correct"]
            ifc_df = g[g["is_api"] == 0].set_index("qid")["correct"]
            common = sorted(set(api_df.index) & set(ifc_df.index))
            if common:
                model_data.append((api_df.loc[common].values, ifc_df.loc[common].values))

        gaps = np.empty(n_iter)
        for i in range(n_iter):
            api_total = ifc_total = 0.0
            n_total = 0
            for api_arr, ifc_arr in model_data:
                n = len(api_arr)
                idx = rng.integers(0, n, size=n)
                api_total += api_arr[idx].sum()
                ifc_total += ifc_arr[idx].sum()
                n_total += n
            gaps[i] = (api_total - ifc_total) / n_total * 100 if n_total > 0 else 0

        gap = np.mean(gaps)
        ci_lo = np.percentile(gaps, 2.5)
        ci_hi = np.percentile(gaps, 97.5)
        sig = "*" if ci_lo > 0 or ci_hi < 0 else ""
        print(f"{bench:<16} {api_mean:>5.1f} {ifc_mean:>5.1f} {gap:>+7.2f} {ci_lo:>+7.2f} {ci_hi:>+7.2f} {sig:>4}")


# ── Section 6: Test-retest reliability ────────────────────────────────────────

def _pairwise_agreement_vec(matrix):
    """Vectorized pairwise agreement across runs (columns).

    For each pair of runs (i, j), computes the fraction of items where
    both runs have a valid answer and agree. Returns mean across all
    C(n_runs, 2) pairs, in percent.
    """
    n_runs = matrix.shape[1]
    if n_runs < 2:
        return 0.0
    # Pre-compute all run-pair masks and agreements at once
    total = 0.0
    count = 0
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            valid = ~np.isnan(matrix[:, i]) & ~np.isnan(matrix[:, j])
            n_valid = valid.sum()
            if n_valid > 0:
                total += (matrix[valid, i] == matrix[valid, j]).sum() / n_valid
                count += 1
    return (total / count * 100) if count > 0 else 0.0


def _pairwise_agreement_batch(matrix, boot_indices):
    """Batch-compute pairwise agreement for many bootstrap samples at once.

    matrix: (n_items, n_runs)
    boot_indices: (n_boot, n_items) — each row is a bootstrap sample of item indices
    Returns: (n_boot,) array of agreement percentages.
    """
    n_boot = boot_indices.shape[0]
    n_runs = matrix.shape[1]
    # Gather all bootstrap samples: (n_boot, n_items, n_runs)
    sampled = matrix[boot_indices]  # advanced indexing
    results = np.zeros(n_boot)
    count = 0
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            col_i = sampled[:, :, i]  # (n_boot, n_items)
            col_j = sampled[:, :, j]
            valid = ~np.isnan(col_i) & ~np.isnan(col_j)
            n_valid = valid.sum(axis=1).astype(float)  # (n_boot,)
            agree = ((col_i == col_j) & valid).sum(axis=1).astype(float)
            # Avoid divide by zero
            safe = n_valid > 0
            results[safe] += agree[safe] / n_valid[safe]
            count += 1
    return results / count * 100


def test_retest_reliability(df, n_boot=5000):
    """Per-item pairwise agreement across 5 runs.

    Returns (cell_results, per_system_results, summary) for use in latex generation.
    """
    print("\n" + "=" * 60)
    print("  §app:stat_test_retest — Test-Retest Reliability")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Pre-build matrices for all cells
    cell_data = {}  # (model, bench) -> (api_matrix, ifc_matrix)
    for (model, bench), group in df.groupby(["model", "benchmark"]):
        qids = sorted(group["qid"].unique())
        n = len(qids)
        runs = sorted(group["run"].unique())
        q2i = {q: i for i, q in enumerate(qids)}
        am = np.full((n, len(runs)), np.nan)
        im = np.full((n, len(runs)), np.nan)
        for _, row in group.iterrows():
            idx = q2i[row["qid"]]; ri = runs.index(row["run"])
            if row["is_api"] == 1: am[idx, ri] = row["correct"]
            else: im[idx, ri] = row["correct"]
        cell_data[(model, bench)] = (am, im)

    # Per-cell test-retest with bootstrap
    n_positive = 0
    cell_results = {}
    print(f"\n{'Model':<20} {'Bench':<22} {'N':>4} {'R_API':>6} {'R_IFC':>6} {'Δ':>6} {'p':>8}")
    print("-" * 80)

    for (model, bench), (am, im) in sorted(cell_data.items()):
        n = am.shape[0]
        r_api = _pairwise_agreement_vec(am)
        r_ifc = _pairwise_agreement_vec(im)
        delta = r_api - r_ifc
        if delta > 0:
            n_positive += 1

        # Vectorized bootstrap
        boot_idx = rng.choice(n, size=(n_boot, n), replace=True)
        boot_api = _pairwise_agreement_batch(am, boot_idx)
        boot_ifc = _pairwise_agreement_batch(im, boot_idx)
        boot_deltas = boot_api - boot_ifc
        p_boot = min(1.0, 2 * min(np.mean(boot_deltas <= 0), np.mean(boot_deltas >= 0)))

        cell_results[(bench, model)] = {"n": n, "r_api": r_api, "r_ifc": r_ifc,
                                         "delta": delta, "p": p_boot}
        p_str = "<.0001" if p_boot < 0.0001 else f"{p_boot:.4f}"
        print(f"{model:<20} {bench:<22} {n:>4} {r_api:>6.1f} {r_ifc:>6.1f} {delta:>+6.1f} {p_str:>8}")

    p_binom = binomtest(n_positive, len(cell_data), 0.5).pvalue
    print(f"\nΔ > 0 in {n_positive}/{len(cell_data)} cells (binomial p = {p_binom:.1e})")

    # Per-system test-retest bootstrap
    print(f"\n--- Per-System Test-Retest Bootstrap ---")
    print(f"{'Model':<22} {'R_API':>6} {'R_IFC':>6} {'Δ':>6} {'p_boot':>9} {'Sig'}")
    print("-" * 55)

    rng2 = np.random.default_rng(123)
    per_system_results = {}

    n_sys_boot = 10000
    for model in sorted(df["model"].unique()):
        bench_pairs = [(am, im) for (m, b), (am, im) in cell_data.items() if m == model]

        # Observed
        api_rs = [_pairwise_agreement_vec(am) for am, im in bench_pairs]
        ifc_rs = [_pairwise_agreement_vec(im) for am, im in bench_pairs]
        obs_delta = np.mean(api_rs) - np.mean(ifc_rs)

        # Vectorized bootstrap: pre-generate all indices, batch-compute agreement
        boot_api_means = np.zeros(n_sys_boot)
        boot_ifc_means = np.zeros(n_sys_boot)
        for am, im in bench_pairs:
            n_q = am.shape[0]
            boot_idx = rng2.choice(n_q, size=(n_sys_boot, n_q), replace=True)
            boot_api_means += _pairwise_agreement_batch(am, boot_idx)
            boot_ifc_means += _pairwise_agreement_batch(im, boot_idx)
        boot_api_means /= len(bench_pairs)
        boot_ifc_means /= len(bench_pairs)
        boot_deltas = boot_api_means - boot_ifc_means

        p_sys = min(1.0, 2 * min(np.mean(boot_deltas <= 0), np.mean(boot_deltas >= 0)))
        sig = "**" if p_sys < 0.01 else "*" if p_sys < 0.05 else ""
        per_system_results[model] = {"r_api": np.mean(api_rs), "r_ifc": np.mean(ifc_rs),
                                      "delta": obs_delta, "p": p_sys}
        print(f"{model:<22} {np.mean(api_rs):>6.1f} {np.mean(ifc_rs):>6.1f} {obs_delta:>+6.1f} {p_sys:>9.4f} {sig}")

    return cell_results, per_system_results, {"n_positive": n_positive, "p_binom": p_binom}


# ── Section 7: Spearman rank stability ────────────────────────────────────────
def rank_stability(df):
    """Spearman correlation between API and interface system rankings.

    For each benchmark, system scores are averaged across runs before ranking.
    Spearman ρ is then computed between the 7 systems' API and interface ranks.
    """
    print("\n" + "=" * 60)
    print("  §app:rank-stability — Rank Stability")
    print("=" * 60)

    BENCH_ORDER = ["metabench-arc", "metabench-gsm8k", "metabench-hellaswag",
                   "metabench-mmlu", "metabench-truthfulQA", "metabench-winogrande",
                   "bbq", "aa-omniscience", "elephant-flip"]

    # Average accuracy across runs, then rank, then Spearman.
    acc_csv = RELEASE / "analysis" / "artifacts" / "data" / "accuracy_per_run.csv"
    import csv as csv_mod
    acc_rows_rk = list(csv_mod.DictReader(open(acc_csv)))

    print(f"\n{'Benchmark':<22} {'ρ':>6}")
    print("-" * 30)
    rhos = []
    for bench in BENCH_ORDER:
        models = sorted(set(r["model"] for r in acc_rows_rk if r["benchmark"] == bench))
        api_scores, ifc_scores = [], []
        for model in models:
            api_vals = [float(r["accuracy"]) for r in acc_rows_rk
                        if r["model"] == model and r["benchmark"] == bench and r["surface"] == "api"]
            ifc_vals = [float(r["accuracy"]) for r in acc_rows_rk
                        if r["model"] == model and r["benchmark"] == bench and r["surface"] == "interface"]
            if api_vals and ifc_vals:
                api_scores.append(np.mean(api_vals))
                ifc_scores.append(np.mean(ifc_vals))
        rho, _ = spearmanr(api_scores, ifc_scores)
        rhos.append(rho)
        print(f"{bench:<22} {rho:>6.2f}")

    print(f"{'Unweighted mean':<22} {np.mean(rhos):>6.2f}")


# ── Latex generation ──────────────────────────────────────────────────────────

MODEL_ORDER = [
    ("chatgpt-instant", "GPT 5.3 Inst."),
    ("chatgpt-thinking", "GPT 5.4 Think"),
    ("claude-haiku", "Claude Haiku"),
    ("claude-opus", "Claude Opus"),
    ("claude-sonnet", "Claude Sonnet"),
    ("gemini-thinking", "Gemini Think"),
    ("gemini-fast", "Gemini Fast"),
]
MODEL_ORDER_FULL = [
    ("chatgpt-instant", "GPT 5.3 Instant"),
    ("chatgpt-thinking", "GPT 5.4 Thinking"),
    ("claude-haiku", "Claude Haiku 4.5"),
    ("claude-opus", "Claude Opus 4.6"),
    ("claude-sonnet", "Claude Sonnet 4.6"),
    ("gemini-thinking", "Gemini 3 Flash Think"),
    ("gemini-fast", "Gemini 3 Flash Fast"),
]
BENCH_ORDER = [
    ("metabench-arc", "ARC"), ("metabench-gsm8k", "GSM8K"),
    ("metabench-hellaswag", "HellaSwag"), ("metabench-mmlu", "MMLU"),
    ("metabench-truthfulQA", "TruthfulQA"), ("metabench-winogrande", "WinoGrande"),
    ("bbq", "BBQ"), ("aa-omniscience", "AA-Omni."),
    ("elephant-flip", "Elephant Flip"),
]


def _fmt_p_latex(p):
    if np.isnan(p):
        return "N/A"
    if p < 1e-16:
        return r"$<10^{-16}$"
    if p < 1e-4:
        exp = int(np.floor(np.log10(p)))
        coeff = p / 10 ** exp
        return f"${coeff:.1f}{{\\times}}10^{{{exp}}}$"
    if p > 0.9999:
        return "1.000"
    return f".{int(p * 10000):04d}"


def _fmt_p_latex_short(p):
    if np.isnan(p):
        return "N/A"
    if p < 0.0001:
        return "$<.0001$"
    if p > 0.9999:
        return "1.000"
    return f".{int(p * 10000):04d}"


def generate_latex(df, results, out_path):
    """Write tables to tables.tex (main text) and appendix.tex (supplement)."""
    # We'll collect main-text lines and appendix lines separately
    main_lines = [
        "% Auto-generated by compute_paper_stats.py --latex",
        "% Main-text tables",
        "",
    ]
    lines = [
        "% Auto-generated by compute_paper_stats.py --latex",
        "% Appendix tables from release/data/ with per-run cross-surface intersection",
        "",
    ]

    # ── Overall LME + Bootstrap (as comments) ──
    r = results["overall_lme"]
    lines.append(f"% Overall LME: beta1={r['b']*100:+.2f}pp SE={r['se']*100:.2f} "
                 f"z={r['z']:.2f} p={r['p']:.4f} CI=[{(r['b']-1.96*r['se'])*100:.2f},{(r['b']+1.96*r['se'])*100:.2f}]")
    r = results["overall_boot"]
    p_str = "<1e-4" if r["p"] < 1e-4 else f"{r['p']:.4f}"
    lines.append(f"% Overall bootstrap: gap={r['gap']:+.2f}pp CI=[{r['ci_lo']:.2f},{r['ci_hi']:.2f}] p={p_str}")
    lines.append("")

    # ── Per-System LME ──
    lines.append(r"\begingroup")
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\caption{Per-system linear mixed-effects model results.}")
    lines.append(r"\label{tab:per_system_lme}")
    lines.append(r"\begin{tabular}{@{}lrrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{System} & \textbf{API\,\%} & \textbf{Iface\,\%} & $\hat\beta_1$ & \textbf{SE} & $z$ & $p$ \\")
    lines.append(r"\midrule")
    for mk, ml in MODEL_ORDER_FULL:
        r = results["per_system"][mk]
        lines.append(f"{ml:<24} & {r['api']:.1f} & {r['ifc']:.1f} & ${r['b']:+.2f}$ & {r['se']:.2f} & {r['z']:.2f} & {_fmt_p_latex(r['p'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"\endgroup")
    lines.append("")

    # ── Per-Benchmark Bootstrap ──
    lines.append(r"\begingroup")
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Per-benchmark cluster bootstrap results.}")
    lines.append(r"\label{tab:per_bench_boot}")
    lines.append(r"\begin{tabular}{@{}lrrrrrl@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Benchmark} & \textbf{API\,\%} & \textbf{Iface\,\%} & \textbf{Gap} & \textbf{CI low} & \textbf{CI high} & \textbf{Sig.} \\")
    lines.append(r"\midrule")
    for bk, bl in BENCH_ORDER:
        r = results["per_bench_boot"][bk]
        sig = "$*$" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else ""
        lines.append(f"{bl:<16} & {r['api']:.1f} & {r['ifc']:.1f} & ${r['gap']:+.2f}$ & ${r['ci_lo']:+.2f}$ & ${r['ci_hi']:+.2f}$ & {sig} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"\endgroup")
    lines.append("")

    # ── Per-Cell LME ──
    cell_results = results["per_cell"]
    lines.append(r"\begingroup")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{longtable}{@{}L{2.65cm}L{2.30cm}rrrr@{}}")
    lines.append(r"\caption{Per-cell linear mixed-effects model results.}")
    lines.append(r"\label{tab:per_cell_lme}\\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{System} & \textbf{Benchmark} & $\hat\beta_1$ \textbf{(pp)} & \textbf{SE} & $z$ & $p$ \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\caption[]{Per-cell linear mixed-effects model results, continued.}\\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{System} & \textbf{Benchmark} & $\hat\beta_1$ \textbf{(pp)} & \textbf{SE} & $z$ & $p$ \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{r}{\footnotesize Continued on next page} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for mk, ml in MODEL_ORDER:
        first = True
        for bk, bl in BENCH_ORDER:
            r = cell_results.get((bk, mk))
            if not r:
                continue
            sys_col = ml if first else ""
            first = False
            z_str = f"{r['z']:.2f}" if not np.isnan(r["z"]) else "N/A"
            if not np.isnan(r["z"]) and r["z"] < 0:
                z_str = f"$-{abs(r['z']):.2f}$"
            lines.append(f"{sys_col:<15} & {bl:<14} & ${r['b']:+.2f}$ & {r['se']:.2f} & {z_str} & {_fmt_p_latex_short(r['p'])} \\\\")
        lines.append(r"\addlinespace[0.45em]")

    lines.append(r"\end{longtable}")
    lines.append("")
    lines.append(r"\begin{center}")
    lines.append(r"\begin{minipage}{0.82\linewidth}")
    lines.append(r"\footnotesize")
    lines.append(r"\emph{Notes.} Each row fits")
    lines.append(r"$\text{correct} \sim \texttt{is\_api} + (1\mid\text{item})$ on")
    lines.append(r"run-level binary outcomes for one system--benchmark cell.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{center}")
    bh = results["bh_counts"]
    lines.append(f"% BH correction: {bh['q05']} cells at q<0.05, {bh['q01']} cells at q<0.01")
    lines.append(r"\endgroup")
    lines.append("")

    # ── Full Capability Table ──
    import csv as csv_mod
    acc_csv_path = RELEASE / "analysis" / "artifacts" / "data" / "accuracy_per_run.csv"
    acc_rows_fc = list(csv_mod.DictReader(open(acc_csv_path)))
    lines.append(r"\begingroup")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\setlength{\LTleft}{\fill}")
    lines.append(r"\setlength{\LTright}{\fill}")
    lines.append(r"\setlength{\LTpre}{0.4em}")
    lines.append(r"\setlength{\LTpost}{0.4em}")
    lines.append("")
    lines.append(r"\begin{longtable}{@{}L{2.65cm}L{2.30cm}rrrr@{}}")
    lines.append(r"\caption{Per-cell API and interface accuracy results.}")
    lines.append(r"\label{tab:appendix_full_capability}\\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{System} & \textbf{Benchmark} & \textbf{API\,\%} & \textbf{Iface\,\%} & $\Delta$ \textbf{(pp)} & \textbf{SE} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\caption[]{Per-cell API and interface accuracy results, continued.}\\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{System} & \textbf{Benchmark} & \textbf{API\,\%} & \textbf{Iface\,\%} & $\Delta$ \textbf{(pp)} & \textbf{SE} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{r}{\footnotesize Continued on next page} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")
    lines.append("")
    for mk, ml in MODEL_ORDER:
        first = True
        for bk, bl in BENCH_ORDER:
            api_vals = [float(r["accuracy"]) * 100 for r in acc_rows_fc
                        if r["model"] == mk and r["benchmark"] == bk and r["surface"] == "api"]
            ifc_vals = [float(r["accuracy"]) * 100 for r in acc_rows_fc
                        if r["model"] == mk and r["benchmark"] == bk and r["surface"] == "interface"]
            if not api_vals or not ifc_vals:
                continue
            api_m = np.mean(api_vals)
            ifc_m = np.mean(ifc_vals)
            delta = api_m - ifc_m
            diffs = [a - i for a, i in zip(api_vals, ifc_vals)]
            se = np.std(diffs, ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0
            sys_col = ml if first else ""
            first = False
            lines.append(f"{sys_col:<15} & {bl:<14} & {api_m:.1f} & {ifc_m:.1f} & ${delta:+.1f}$ & {se:.1f} \\\\")
        lines.append(r"\addlinespace[0.45em]")
    lines.append(r"\end{longtable}")
    lines.append("")
    lines.append(r"\begin{center}")
    lines.append(r"\begin{minipage}{0.82\linewidth}")
    lines.append(r"\footnotesize")
    lines.append(r"\emph{Notes.} Accuracy is the mean across five runs.")
    lines.append(r"$\Delta = \text{API} - \text{Iface}$ in percentage points;")
    lines.append(r"SE is the paired standard error across runs.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{center}")
    lines.append(r"\endgroup")
    lines.append("")

    # ── Rank Stability ──
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Spearman correlations between API and interface system rankings.}")
    lines.append(r"\label{tab:spearman-rank-stability}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Benchmark & $\rho$ \\")
    lines.append(r"\midrule")
    rhos_sorted = sorted(results["spearman"].items(), key=lambda x: x[1])
    for bk, rho in rhos_sorted:
        bl = dict(BENCH_ORDER).get(bk, bk)
        lines.append(f"{bl:<18} & {rho:.2f} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Unweighted mean  & {results['spearman_mean']:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # ── Test-Retest ──
    lines.append(r"\begingroup\small")
    lines.append(r"\begin{longtable}{@{}L{2.65cm}L{2.30cm}rrrc@{}}")
    lines.append(r"\caption{Full 63-cell test--retest reliability.}")
    lines.append(r"\label{tab:appendix_test_retest_agreement}\\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{System} & \textbf{Benchmark} & $R_{\text{API}}$ & $R_{\text{Ifc}}$ & $\Delta$ \textbf{(pp)} & $p_{\text{boot}}$ \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"\textbf{System} & \textbf{Benchmark} & $R_{\text{API}}$ & $R_{\text{Ifc}}$ & $\Delta$ \textbf{(pp)} & $p_{\text{boot}}$ \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{r}{\footnotesize Continued on next page} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for mk, ml in MODEL_ORDER:
        first = True
        for bk, bl in BENCH_ORDER:
            r = results["test_retest"].get((bk, mk))
            if not r:
                continue
            sys_col = ml if first else ""
            first = False
            lines.append(f"{sys_col:<15} & {bl:<14} & {r['r_api']:.1f} & {r['r_ifc']:.1f} & ${r['delta']:+.1f}$ & {_fmt_p_latex_short(r['p'])} \\\\")
        lines.append(r"\addlinespace[0.45em]")

    lines.append(r"\end{longtable}")
    tr = results["test_retest_summary"]
    lines.append(f"% {tr['n_positive']}/63 show Delta>0, binomial p={tr['p_binom']:.1e}")
    lines.append(r"\endgroup")
    lines.append("")

    # ── Extractability Table ──
    import csv as csv_mod2
    acc_csv2 = RELEASE / "analysis" / "artifacts" / "data" / "accuracy_per_run.csv"
    DATA = RELEASE / "data"
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{ll rr rr}")
    lines.append(r"\toprule")
    lines.append(r"& & \multicolumn{2}{c}{\textbf{Extraction \%}} & \multicolumn{2}{c}{\textbf{Runs}} \\")
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    lines.append(r"\textbf{Model} & \textbf{Benchmark}")
    lines.append(r"  & \textbf{API} & \textbf{Iface}")
    lines.append(r"  & \textbf{API} & \textbf{Iface} \\")
    lines.append(r"\midrule")
    for mk, ml in MODEL_ORDER:
        first = True
        for bk, bl in BENCH_ORDER:
            api_rates = []
            ifc_rates = []
            api_runs = 0
            ifc_runs = 0
            for surface in ["api", "interface"]:
                for ri in range(5):
                    run_dir = DATA / bk / mk / surface / f"run_{ri}"
                    resp_dir = run_dir / "responses"
                    scored_path = run_dir / "scored.csv"
                    if not run_dir.exists():
                        continue
                    n_resp = 0
                    if resp_dir.is_dir():
                        items = set()
                        for f in resp_dir.iterdir():
                            if f.suffix == ".json" and not f.name.startswith("_"):
                                items.add(f.stem.replace(".api", ""))
                        n_resp = len(items)
                    n_scored = 0
                    if scored_path.exists():
                        n_scored = sum(1 for _ in csv_mod2.DictReader(open(scored_path)))
                    rate = n_scored / n_resp * 100 if n_resp > 0 else 0
                    if surface == "api":
                        api_rates.append(rate)
                        api_runs += 1
                    else:
                        ifc_rates.append(rate)
                        ifc_runs += 1
            if not api_rates and not ifc_rates:
                continue
            api_m = sum(api_rates) / len(api_rates) if api_rates else 0
            ifc_m = sum(ifc_rates) / len(ifc_rates) if ifc_rates else 0
            sys_col = ml if first else ""
            first = False
            lines.append(f"{sys_col:<18} & {bl:<12} & {api_m:>5.1f} & {ifc_m:>5.1f} & {api_runs} & {ifc_runs} \\\\")
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Mean extraction rates (\%) per model and benchmark.")
    lines.append(r"All condition--runs have 5 qualifying runs per surface.}")
    lines.append(r"\label{tab:appendix_extractability}")
    lines.append(r"\end{table}")
    lines.append("")

    # ── Per-System Test-Retest (main text Table test_retest_system_gap) ──
    if "test_retest_per_system" in results:
        sys_tr = results["test_retest_per_system"]
        sorted_sys = sorted(sys_tr.items(), key=lambda x: -x[1]["delta"])
        main_lines.append(r"\begin{table}[t]")
        main_lines.append(r"\centering")
        main_lines.append(r"\small")
        main_lines.append(r"\caption{\textbf{API responses are more reliable across repeated runs.}")
        main_lines.append(r"$R$ is mean item-level agreement (\%) across five runs.")
        main_lines.append(r"$\Delta$ is the API$-$interface difference.")
        main_lines.append(r"\textsuperscript{*}$p<.05$;")
        main_lines.append(r"\textsuperscript{**}$p<.01$;")
        main_lines.append(r"\textsuperscript{***}$p<.001$.}")
        main_lines.append(r"\label{tab:test_retest_system_gap}")
        main_lines.append(r"\vspace{0.25em}")
        main_lines.append(r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrl@{}}")
        main_lines.append(r"\toprule")
        main_lines.append(r"System & $R_{\text{API}}$ & $R_{\text{UI}}$ & $\Delta$ (pp) & Sig. \\")
        main_lines.append(r"\midrule")
        model_labels = dict(MODEL_ORDER_FULL)
        for mk, r in sorted_sys:
            ml = model_labels.get(mk, mk)
            sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
            main_lines.append(f"{ml:<21} & {r['r_api']:.1f} & {r['r_ifc']:.1f} & ${r['delta']:+.1f}$ & {sig} \\\\")
        main_lines.append(r"\bottomrule")
        main_lines.append(r"\end{tabular*}")
        main_lines.append(r"\end{table}")

    # Write main-text tables
    main_path = out_path.parent / "main_tables.tex"
    main_path.write_text("\n".join(main_lines))
    print(f"\nWrote {main_path}")

    # Write appendix tables (will be extended with ablation/robustness/schedule tables later)
    appendix_path = out_path.parent / "appendix.tex"
    appendix_path.write_text("\n".join(lines))
    print(f"Wrote {appendix_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex", action="store_true",
                        help="Write latex tables to analysis/tables.tex")
    parser.add_argument("--boot-iter", type=int, default=10000)
    args = parser.parse_args()

    print("Building item-level dataset from release/data/...")
    df = build_item_level_data()
    n_api_higher = sum(
        1 for (b, m), g in df.groupby(["benchmark", "model"])
        if g[g["is_api"] == 1]["correct"].mean() > g[g["is_api"] == 0]["correct"].mean()
    )
    print(f"Dataset: {len(df):,} rows, {df['qid'].nunique()} unique items")
    print(f"Benchmarks: {sorted(df['benchmark'].unique())}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"API > IFC in {n_api_higher}/63 cells")

    # Write accuracy CSV
    write_accuracy_csv(df)

    results = {}

    # 1. Overall LME
    overall_lme(df)
    # Also store for latex
    cell = df.groupby(["benchmark", "model", "run", "is_api"]).agg(
        correct=("correct", "mean")).reset_index()
    md = smf.mixedlm("correct ~ is_api", data=cell, groups="benchmark",
                      re_formula="~is_api")
    r = md.fit(reml=False)
    results["overall_lme"] = {"b": r.fe_params["is_api"], "se": r.bse["is_api"],
                              "z": r.tvalues["is_api"], "p": r.pvalues["is_api"]}

    # 2. Overall bootstrap
    cluster_bootstrap(df, n_iter=args.boot_iter)
    rng = np.random.default_rng(42)
    strata = _build_stratum_arrays(df)
    gaps = np.empty(args.boot_iter)
    for i in range(args.boot_iter):
        api_t = ifc_t = 0.0; n_t = 0
        for a, ic in strata:
            n = len(a); idx = rng.integers(0, n, size=n)
            api_t += a[idx].sum(); ifc_t += ic[idx].sum(); n_t += n
        gaps[i] = (api_t - ifc_t) / n_t * 100
    p_boot = min(1.0, 2 * min(np.mean(gaps <= 0), np.mean(gaps >= 0)))
    results["overall_boot"] = {"gap": np.mean(gaps), "ci_lo": np.percentile(gaps, 2.5),
                               "ci_hi": np.percentile(gaps, 97.5), "p": p_boot}

    # 3. Per-system LME
    per_system_lme(df)
    results["per_system"] = {}
    for mk, _ in MODEL_ORDER_FULL:
        sub = df[df["model"] == mk]
        api = sub[sub["is_api"] == 1]["correct"].mean() * 100
        ifc = sub[sub["is_api"] == 0]["correct"].mean() * 100
        md2 = smf.mixedlm("correct ~ is_api", data=sub, groups="benchmark")
        r2 = md2.fit(reml=False)
        results["per_system"][mk] = {"api": api, "ifc": ifc,
            "b": r2.fe_params["is_api"] * 100, "se": r2.bse["is_api"] * 100,
            "z": r2.tvalues["is_api"], "p": r2.pvalues["is_api"]}

    # 4. Per-cell LME
    per_cell_lme(df)
    results["per_cell"] = {}
    all_ps = []
    for (bench, model), group in df.groupby(["benchmark", "model"]):
        sub = group.copy(); sub["qid_str"] = sub["qid"].astype(str)
        try:
            md3 = smf.mixedlm("correct ~ is_api", data=sub, groups="qid_str")
            r3 = md3.fit(reml=False)
            b, se, z, p = (r3.fe_params["is_api"] * 100, r3.bse["is_api"] * 100,
                           r3.tvalues["is_api"], r3.pvalues["is_api"])
        except Exception:
            b = (group[group["is_api"] == 1]["correct"].mean() -
                 group[group["is_api"] == 0]["correct"].mean()) * 100
            se = z = p = float("nan")
        n_items = sub["qid"].nunique()
        results["per_cell"][(bench, model)] = {"b": b, "se": se, "z": z, "p": p, "n": n_items}
        if not np.isnan(p):
            all_ps.append(p)
    # BH correction (Benjamini-Hochberg)
    ps_sorted = sorted(enumerate(all_ps), key=lambda x: x[1])
    m = len(ps_sorted)
    q_vals = [None] * m
    for rank, (idx, pval) in enumerate(ps_sorted, 1):
        q_vals[idx] = min(1.0, pval * m / rank)
    # Enforce monotonicity in reverse sorted order (largest p first)
    cum_min = 1.0
    for rank in range(m, 0, -1):
        idx = ps_sorted[rank - 1][0]
        cum_min = min(cum_min, q_vals[idx])
        q_vals[idx] = cum_min
    results["bh_counts"] = {"q05": sum(1 for q in q_vals if q < 0.05),
                            "q01": sum(1 for q in q_vals if q < 0.01)}

    # 5. Per-benchmark bootstrap
    per_benchmark_bootstrap(df, n_iter=args.boot_iter)
    results["per_bench_boot"] = {}
    rng2 = np.random.default_rng(42)
    agg = df.groupby(["benchmark", "model", "qid", "is_api"])["correct"].mean().reset_index()
    for bk, bl in BENCH_ORDER:
        bdf = df[df["benchmark"] == bk]
        api_m = bdf[bdf["is_api"] == 1]["correct"].mean() * 100
        ifc_m = bdf[bdf["is_api"] == 0]["correct"].mean() * 100
        bagg = agg[agg["benchmark"] == bk]
        md_list = []
        for model, g in bagg.groupby("model"):
            ad = g[g["is_api"] == 1].set_index("qid")["correct"]
            icd = g[g["is_api"] == 0].set_index("qid")["correct"]
            c = sorted(set(ad.index) & set(icd.index))
            if c: md_list.append((ad.loc[c].values, icd.loc[c].values))
        gs = np.empty(args.boot_iter)
        for i in range(args.boot_iter):
            at = it = 0.0; nt = 0
            for a, ic in md_list:
                n = len(a); idx = rng2.integers(0, n, size=n)
                at += a[idx].sum(); it += ic[idx].sum(); nt += n
            gs[i] = (at - it) / nt * 100 if nt > 0 else 0
        results["per_bench_boot"][bk] = {"api": api_m, "ifc": ifc_m,
            "gap": np.mean(gs), "ci_lo": np.percentile(gs, 2.5), "ci_hi": np.percentile(gs, 97.5)}

    # 6. Test-retest (single pass — returns results for both display and latex)
    cell_tr, sys_tr, tr_summary = test_retest_reliability(df, n_boot=5000)
    results["test_retest"] = cell_tr
    results["test_retest_per_system"] = sys_tr
    results["test_retest_summary"] = tr_summary

    # 7. Rank stability
    rank_stability(df)
    # Spearman: average accuracy across runs, then rank
    import csv as csv_mod
    acc_csv_path = RELEASE / "analysis" / "artifacts" / "data" / "accuracy_per_run.csv"
    acc_rows_sp = list(csv_mod.DictReader(open(acc_csv_path)))
    results["spearman"] = {}
    rhos = []
    for bk, _ in BENCH_ORDER:
        models = sorted(set(r["model"] for r in acc_rows_sp if r["benchmark"] == bk))
        api_s, ifc_s = [], []
        for model in models:
            api_vals = [float(r["accuracy"]) for r in acc_rows_sp
                        if r["model"] == model and r["benchmark"] == bk and r["surface"] == "api"]
            ifc_vals = [float(r["accuracy"]) for r in acc_rows_sp
                        if r["model"] == model and r["benchmark"] == bk and r["surface"] == "interface"]
            if api_vals and ifc_vals:
                api_s.append(np.mean(api_vals))
                ifc_s.append(np.mean(ifc_vals))
        rho, _ = spearmanr(api_s, ifc_s)
        results["spearman"][bk] = rho
        rhos.append(rho)
    results["spearman_mean"] = np.mean(rhos)

    # Write latex if requested
    if args.latex:
        out = RELEASE / "analysis" / "artifacts" / "tables" / "main_tables.tex"
        generate_latex(df, results, out)
        _write_figure_csvs(acc_csv_path)
        _write_eval_schedule()

    print("\nDone.")


def _write_figure_csvs(acc_csv_path):
    """Write figure data CSVs."""
    import csv as csv_mod
    acc_rows = list(csv_mod.DictReader(open(acc_csv_path)))
    OUT = RELEASE / "analysis" / "artifacts" / "data"

    MODELS = [
        ("chatgpt-instant", "GPT 5.3 Instant"),
        ("chatgpt-thinking", "GPT 5.4 Think"),
        ("claude-haiku", "Claude Haiku 4.5"),
        ("claude-opus", "Claude Opus 4.6"),
        ("claude-sonnet", "Claude Sonnet 4.6"),
        ("gemini-thinking", "Gemini 3 Flash Think"),
        ("gemini-fast", "Gemini 3 Flash Fast"),
    ]
    BENCHMARKS = [
        ("metabench-arc", "ARC"), ("metabench-gsm8k", "GSM8K"),
        ("metabench-hellaswag", "HellaSwag"), ("metabench-mmlu", "MMLU"),
        ("metabench-truthfulQA", "TruthfulQA"), ("metabench-winogrande", "WinoGrande"),
        ("bbq", "BBQ"), ("aa-omniscience", "AA-Omni."), ("elephant-flip", "Elephant Flip"),
    ]

    # Figure 1 data
    with open(OUT / "figure1_aggregate_bars.csv", "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["model", "model_label", "api_pct", "ifc_pct", "gap_pp"])
        for mk, ml in MODELS:
            api = np.mean([float(r["accuracy"]) * 100 for r in acc_rows if r["model"] == mk and r["surface"] == "api"])
            ifc = np.mean([float(r["accuracy"]) * 100 for r in acc_rows if r["model"] == mk and r["surface"] == "interface"])
            w.writerow([mk, ml, f"{api:.1f}", f"{ifc:.1f}", f"{api - ifc:+.1f}"])

    # Figure S1 data
    with open(OUT / "figureS1_accuracy_ci.csv", "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["model", "model_label", "benchmark", "bench_label", "surface", "mean_pct", "se", "n_runs"])
        for bk, bl in BENCHMARKS:
            for mk, ml in MODELS:
                for surf in ["api", "interface"]:
                    vals = [float(r["accuracy"]) * 100 for r in acc_rows
                            if r["model"] == mk and r["benchmark"] == bk and r["surface"] == surf]
                    if vals:
                        m = np.mean(vals)
                        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                        w.writerow([mk, ml, bk, bl, surf, f"{m:.1f}", f"{se:.2f}", len(vals)])


def _write_eval_schedule():
    """Write evaluation schedule from response timestamps."""
    import csv as csv_mod
    import json
    from datetime import datetime

    DATA = RELEASE / "data"
    OUT = RELEASE / "analysis" / "artifacts" / "tables"

    MODEL_ORDER_SCHED = [
        ("chatgpt-instant", "GPT 5.3 Instant"),
        ("chatgpt-thinking", "GPT 5.4 Thinking"),
        ("claude-haiku", "Claude Haiku 4.5"),
        ("claude-sonnet", "Claude Sonnet 4.6"),
        ("claude-opus", "Claude Opus 4.6"),
        ("gemini-fast", "Gemini 3 Flash Fast"),
        ("gemini-thinking", "Gemini 3 Flash Thinking"),
    ]
    BENCH_ORDER_SCHED = [
        ("metabench-arc", "ARC"), ("metabench-gsm8k", "GSM8K"),
        ("metabench-hellaswag", "HellaSwag"), ("metabench-mmlu", "MMLU"),
        ("metabench-truthfulQA", "TruthfulQA"), ("metabench-winogrande", "WinoGrande"),
        ("bbq", "BBQ"), ("aa-omniscience", "AA-Omniscience"),
        ("elephant-flip", "AITA-NTA-Flip"),
    ]
    # Fallback for cells without timestamps in response JSONs
    FALLBACK = {("HellaSwag", "GPT 5.3 Instant"): ("2026-05-24", "2026-05-24")}

    def _get_dates(bk, mk):
        dates = []
        for surface in ["api", "interface"]:
            for ri in range(5):
                resp_dir = DATA / bk / mk / surface / f"run_{ri}" / "responses"
                if not resp_dir.is_dir():
                    continue
                for f in sorted(resp_dir.glob("*.json"))[:10]:
                    try:
                        d = json.loads(f.read_text())
                        for ts_key in ["completed_at", "started_at", "sent_at"]:
                            ts = d.get(ts_key)
                            if ts and "2026" in str(ts):
                                dt = datetime.fromisoformat(ts.replace("Z", ""))
                                dates.append(dt.strftime("%Y-%m-%d"))
                                break
                    except Exception:
                        pass
        return dates

    lines = []
    lines.append(r"\section{Evaluation Schedule}")
    lines.append(r"\label{app:evaluation_schedule}")
    lines.append("")
    lines.append(r"The evaluation schedule for each benchmark and model is shown in Table~\ref{tab:appendix-evaluation-schedule}. The table reports the start and end dates corresponding to the data collection periods used for each benchmark--model pair. All dates are in 2026.")
    lines.append("")
    lines.append(r"\begingroup")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append("")
    lines.append(r"\begin{longtable}{@{}llrr@{}}")
    lines.append(r"\caption{Evaluation schedule for all benchmarks and models.}")
    lines.append(r"\label{tab:appendix-evaluation-schedule}\\")
    lines.append(r"\toprule")
    lines.append(r"Benchmark & Model & Start Date & End Date \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Benchmark & Model & Start Date & End Date \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{r}{\emph{Continued on next page}}\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for bk, bl in BENCH_ORDER_SCHED:
        for mk, ml in MODEL_ORDER_SCHED:
            dates = _get_dates(bk, mk)
            if dates:
                start, end = min(dates), max(dates)
            elif (bl, ml) in FALLBACK:
                start, end = FALLBACK[(bl, ml)]
            else:
                continue
            lines.append(f"{bl} & {ml} & {start} & {end} \\\\")
        lines.append(r"\addlinespace")

    lines.append(r"\end{longtable}")
    lines.append("")
    lines.append(r"\endgroup")

    # Append to appendix.tex
    appendix_path = OUT / "appendix.tex"
    with open(appendix_path, "a") as f:
        f.write("\n\n% ── Evaluation schedule (from compute_paper_stats.py) ──\n")
        f.write("\n".join(lines))
    print(f"Appended eval schedule to {appendix_path}")


if __name__ == "__main__":
    main()
