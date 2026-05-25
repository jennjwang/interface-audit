"""Exact concordance fraction across all (system, benchmark) cells.

Uses the same per-item data as compute_item_correlation.py. For each cell,
counts concordant (C) and discordant (D) pairs:
  - For all item pairs (i, j) with i < j:
      a = sign(api_i - api_j),  b = sign(ifc_i - ifc_j)
      If a == 0 OR b == 0: tied (excluded from both C and D)
      Else if a * b > 0: concordant
      Else: discordant

Reports:
  - Per-cell C, D, tied, fraction concordant on non-tied pairs
  - Aggregate across all 63 cells: total C, D, fraction concordant
  - For comparison: heuristic from mean tau-b: (1 + tau_mean) / 2

Output: paper/tables/item_concordance.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))

from compute_item_correlation import _per_item_means  # noqa: E402
from compute_test_retest import (  # noqa: E402
    MODELS, PANELS,
    load_kept_runs, collect_metabench_per_item, load_intersected_per_query,
)

REPO = THIS.parents[2]
OUT_CSV = THIS.parent / "item_concordance.csv"


def count_cd(xs: list[float], ys: list[float]) -> tuple[int, int, int]:
    """Returns (concordant, discordant, tied) pair counts."""
    n = len(xs)
    C = D = T = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = xs[i] - xs[j]
            b = ys[i] - ys[j]
            if a == 0 or b == 0:
                T += 1
            elif (a > 0 and b > 0) or (a < 0 and b < 0):
                C += 1
            else:
                D += 1
    return C, D, T


def main() -> None:
    kept = load_kept_runs()
    cells = []

    total_C = total_D = total_T = 0
    taus = []

    print(f"{'Model':<22} {'Benchmark':<16} {'N':>4} {'C':>6} {'D':>6} {'Tied':>6} {'p_conc':>8}")
    print("-" * 76)

    for (mlabel, api_cond, ifc_cond, api_patterns, iface_slug, prov_inter, mslug_inter) in MODELS:
        prov_dir = {"chatgpt": "data-chatgpt", "claude": "data-claude", "gemini": "data-gemini"}[prov_inter]
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

            xs, ys, _qids = _per_item_means(api_items, iface_items, failure_mode)
            if len(xs) < 2:
                continue
            C, D, T = count_cd(xs, ys)
            tau = float("nan")
            if float(np.std(xs)) > 0 and float(np.std(ys)) > 0:
                t = kendalltau(np.array(xs), np.array(ys), variant="b")
                tau = float(t.correlation if hasattr(t, "correlation") else t[0])
            p_conc = C / (C + D) if (C + D) > 0 else float("nan")
            cells.append({
                "model": mlabel, "benchmark": plabel,
                "N": len(xs), "C": C, "D": D, "tied": T,
                "p_concordant": round(p_conc, 4) if p_conc == p_conc else "",
                "tau_b": round(tau, 4) if tau == tau else "",
            })
            total_C += C
            total_D += D
            total_T += T
            if tau == tau:
                taus.append(tau)
            print(f"{mlabel:<22} {plabel:<16} {len(xs):>4} {C:>6} {D:>6} {T:>6} {p_conc:>8.4f}")

    # ── Aggregate ──
    total_nontied = total_C + total_D
    agg_p_conc = total_C / total_nontied if total_nontied > 0 else float("nan")
    heuristic = (1 + np.mean(taus)) / 2 if taus else float("nan")

    print()
    print("=== Aggregate across all cells ===")
    print(f"  Total pairs: {total_C + total_D + total_T:,}")
    print(f"  Concordant : {total_C:,}")
    print(f"  Discordant : {total_D:,}")
    print(f"  Tied       : {total_T:,}")
    print(f"  Non-tied   : {total_nontied:,}")
    print()
    print(f"  Exact concordance fraction (C / (C+D))     = {agg_p_conc*100:.2f}%")
    print(f"  Discordance fraction (D / (C+D))           = {(1-agg_p_conc)*100:.2f}%")
    print()
    print(f"  Heuristic from mean tau-b: (1 + {np.mean(taus):.3f}) / 2 = {heuristic*100:.2f}%")
    print(f"  Difference: {(agg_p_conc - heuristic)*100:+.2f} pp")

    cells.append({
        "model": "AGGREGATE",
        "benchmark": "(all 63 cells)",
        "N": "", "C": total_C, "D": total_D, "tied": total_T,
        "p_concordant": round(agg_p_conc, 4),
        "tau_b": "",
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "benchmark", "N", "C", "D", "tied",
                                           "p_concordant", "tau_b"])
        w.writeheader()
        w.writerows(cells)
    print()
    print(f"Wrote {OUT_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
