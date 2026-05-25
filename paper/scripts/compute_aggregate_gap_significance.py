"""Compute per-system API-interface accuracy gaps with significance.

For each system, reports gap, SE, z (= gap / SE), and the Wald-test p-value.
Adds a final OVERALL row that aggregates across systems using a one-sample
t-test (treating each of the 7 system-level gaps as one observation).

Inputs:
  paper/tables/aggregate_accuracy.csv

Outputs:
  paper/tables/aggregate_accuracy_gaps_with_significance.csv
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from scipy import stats

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
IN_CSV  = REPO / "paper" / "tables" / "aggregate_accuracy.csv"
OUT_CSV = REPO / "paper" / "tables" / "aggregate_accuracy_gaps_with_significance.csv"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(f"Missing input: {IN_CSV}")

    agg = list(csv.DictReader(IN_CSV.open()))
    rows = []

    # Per-system Wald z-test on the system-level gap.
    for r in agg:
        gap = float(r["gap"])
        se = float(r["gap_se"])
        z = gap / se if se > 0 else float("nan")
        p = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else float("nan")
        rows.append({
            "model": r["model"],
            "gap": round(gap, 2),
            "gap_se": round(se, 2),
            "z": round(z, 2),
            "p": fmt_p(p),
            "n_benchmarks": r["n_benchmarks"],
        })

    # OVERALL row: one-sample t-test treating each system gap as one observation.
    # Standard "treat models as a random sample" framing. SE here is across-system
    # SD / sqrt(n_systems), which is what we use in the paper.
    gaps = [float(r["gap"]) for r in agg]
    n_sys = len(gaps)
    mean_gap = statistics.mean(gaps)
    se = statistics.stdev(gaps) / math.sqrt(n_sys)
    t_stat = mean_gap / se if se > 0 else float("nan")
    df = n_sys - 1
    p_overall = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    rows.append({
        "model": f"OVERALL (across-system t-test, df={df})",
        "gap": round(mean_gap, 2),
        "gap_se": round(se, 2),
        "z": round(t_stat, 2),  # this is actually t-statistic, but kept in same column
        "p": fmt_p(p_overall),
        "n_benchmarks": f"{n_sys} systems",
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "gap", "gap_se", "z", "p", "n_benchmarks"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV.relative_to(REPO)}")
    print()
    print(OUT_CSV.read_text())


if __name__ == "__main__":
    main()
