"""Compute per-system test--retest API-interface gaps with significance.

Mirrors compute_aggregate_gap_significance.py but for the test--retest table.
For each system, reports gap, SE, z (= gap / SE), and the Wald-test p-value.
Adds a final OVERALL row that aggregates across systems using a one-sample
t-test (treating each of the 7 system-level gaps as one observation).

Inputs:
  paper/tables/aggregate_test_retest.csv
  paper/tables/appendix_test_retest_agreement.csv  (for 63-cell counts)

Outputs:
  paper/tables/aggregate_test_retest_with_significance.csv
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from scipy import stats
from scipy.stats import binomtest

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
AGG_CSV  = REPO / "paper" / "tables" / "aggregate_test_retest.csv"
CELL_CSV = REPO / "paper" / "tables" / "appendix_test_retest_agreement.csv"
OUT_CSV  = REPO / "paper" / "tables" / "aggregate_test_retest_with_significance.csv"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def main() -> None:
    if not AGG_CSV.exists():
        raise SystemExit(f"Missing input: {AGG_CSV}")

    agg = list(csv.DictReader(AGG_CSV.open()))
    rows = []

    # Per-system Wald z-test on the system-level gap.
    for r in agg:
        gap = float(r["gap"])
        se = float(r["gap_se"])
        z = gap / se if se > 0 else float("nan")
        p = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else float("nan")
        rows.append({
            "model": r["model"],
            "api_mean": r["api_mean"],
            "interface_mean": r["interface_mean"],
            "gap": round(gap, 2),
            "gap_se": round(se, 2),
            "z": round(z, 2),
            "p": fmt_p(p),
            "n_benchmarks": r["n_benchmarks"],
        })

    # OVERALL row: one-sample t-test on the 7 system-level gaps.
    # Across-system SE (method A) is what we use in the paper — generalizes
    # the claim across LLM systems rather than treating them as fixed.
    gaps = [float(r["gap"]) for r in agg]
    api_means = [float(r["api_mean"]) for r in agg]
    iface_means = [float(r["interface_mean"]) for r in agg]
    n_sys = len(gaps)
    mean_gap = statistics.mean(gaps)
    se_gap = statistics.stdev(gaps) / math.sqrt(n_sys)
    t_stat = mean_gap / se_gap if se_gap > 0 else float("nan")
    df = n_sys - 1
    p_overall = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    t_crit = stats.t.ppf(0.975, df)
    ci_lo, ci_hi = mean_gap - t_crit * se_gap, mean_gap + t_crit * se_gap

    # Also compute mean API and Interface SEs across systems (useful for prose).
    se_api = statistics.stdev(api_means) / math.sqrt(n_sys)
    se_iface = statistics.stdev(iface_means) / math.sqrt(n_sys)

    rows.append({
        "model": f"OVERALL (across-system t-test, df={df})",
        "api_mean": f"{statistics.mean(api_means):.2f} (SE={se_api:.2f})",
        "interface_mean": f"{statistics.mean(iface_means):.2f} (SE={se_iface:.2f})",
        "gap": round(mean_gap, 2),
        "gap_se": round(se_gap, 2),
        "z": round(t_stat, 2),  # t-statistic, kept in same column
        "p": fmt_p(p_overall),
        "n_benchmarks": f"{n_sys} systems  95%CI=[{ci_lo:.2f}, {ci_hi:.2f}]",
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "api_mean", "interface_mean",
                                           "gap", "gap_se", "z", "p", "n_benchmarks"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV.relative_to(REPO)}")

    # ── 63-cell summary (for the prose) ────────────────────────────────────
    if CELL_CSV.exists():
        cells = list(csv.DictReader(CELL_CSV.open()))
        n_total = len(cells)
        n_pos = sum(1 for r in cells if float(r["diff_pp"]) > 0)
        binom_p = binomtest(n_pos, n_total, p=0.5, alternative="greater").pvalue

        # BH correction
        ps = []
        for r in cells:
            d, s = float(r["diff_pp"]), float(r["se_pp"])
            z = d / s if s > 0 else 0
            ps.append(2 * (1 - stats.norm.cdf(abs(z))))
        m = len(ps)
        order = sorted(range(m), key=lambda i: ps[i])
        q = [0.0] * m
        prev = 1.0
        for rank, i in enumerate(reversed(order)):
            val = ps[i] * m / (m - rank)
            prev = min(prev, val)
            q[i] = prev

        print()
        print(f"63-cell summary:")
        print(f"  API > Interface: {n_pos}/{n_total} (binomial p={binom_p:.2e} vs null=0.5)")
        print(f"  BH q<0.05: {sum(1 for v in q if v < 0.05)}")
        print(f"  BH q<0.01: {sum(1 for v in q if v < 0.01)}")

    print()
    print(OUT_CSV.read_text())


if __name__ == "__main__":
    main()
