"""Per-item correctness correlation, API vs Interface.

Uses the same 5-run data sources as appendix_full_capability.csv /
compute_test_retest.py, by importing those loaders directly:
  - Metabench: scored CSVs at the first 5 qualifying coverage_by_run timestamps
  - BBQ / AA-Omniscience / Elephant Flip: per_query.csv (5 runs per item)

For each (model, benchmark):
  per_item_api  = mean correctness across kept runs (using the same failure-mode
                  convention as compute_test_retest: "incorrect" for metabench,
                  "ignore" for intersected — drop items with K_ext<2 on either
                  side).
  per_item_iface = same for the interface side.
Pearson r is computed across items present on both sides.
Agreement % is the fraction of items where API and Interface majority-vote
(correct in >50% of kept runs) match.

Outputs (in paper/tables/):
  item_tau_heatmap.{png,pdf}             (Kendall tau-b heatmap)
  item_correlation_tau_table.tex         (LaTeX matrix table, tau-b)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, kendalltau

# Sibling import from paper/tables/compute_test_retest.py
from compute_test_retest import (  # noqa: E402
    MODELS, PANELS, MAX_RUNS,
    load_kept_runs, collect_metabench_per_item, load_intersected_per_query,
)

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
OUT_DIR = THIS.parent  # paper/tables/
HEATMAP_T = OUT_DIR / "item_tau_heatmap.png"
TAU_TEX   = OUT_DIR / "item_correlation_tau_table.tex"


def _per_item_means(api_items, ifc_items, failure_mode: str):
    """Return (xs, ys, qids) aligned per item using the same outcome rules as compute_test_retest."""
    shared = sorted(set(api_items) & set(ifc_items))
    xs, ys, qids = [], [], []
    for it in shared:
        api_runs = api_items[it]
        ifc_runs = ifc_items[it]
        if failure_mode == "incorrect":
            a = [bool(ext and corr) for ext, corr in api_runs]
            b = [bool(ext and corr) for ext, corr in ifc_runs]
        else:  # "ignore" — only use extracted runs; need ≥2 on each side
            a = [bool(corr) for ext, corr in api_runs if ext]
            b = [bool(corr) for ext, corr in ifc_runs if ext]
            if len(a) < 2 or len(b) < 2:
                continue
        if not a or not b:
            continue
        xs.append(sum(a) / len(a))
        ys.append(sum(b) / len(b))
        qids.append(it)
    return xs, ys, qids


def _r(xs, ys) -> float:
    if len(xs) < 2 or float(np.std(xs)) == 0 or float(np.std(ys)) == 0:
        return float("nan")
    return float(pearsonr(np.array(xs), np.array(ys))[0])


def _tau(xs, ys) -> float:
    """Kendall's tau-b: rank-based, robust to ties at 0/1 from binary/discrete correctness."""
    if len(xs) < 2 or float(np.std(xs)) == 0 or float(np.std(ys)) == 0:
        return float("nan")
    t = kendalltau(np.array(xs), np.array(ys), variant="b")
    return float(t.correlation if hasattr(t, "correlation") else t[0])


def _agreement(xs, ys) -> float:
    if not xs:
        return float("nan")
    a = np.array(xs) > 0.5
    b = np.array(ys) > 0.5
    return float(np.mean(a == b))


def main():
    kept = load_kept_runs()
    cells = []
    summary_by_model: dict[str, dict] = {}

    print(f"{'Model':<22} {'Benchmark':<16} {'N':>4} {'r':>6} {'tau':>6} {'agree%':>7}")
    print("-" * 67)

    for (mlabel, api_cond, ifc_cond, api_patterns, iface_slug, prov_inter, mslug_inter) in MODELS:
        prov_dir = {"chatgpt": "data-chatgpt", "claude": "data-claude", "gemini": "data-gemini"}[prov_inter]
        per_bench_xy: dict[str, tuple[list, list]] = {}

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
            r = _r(xs, ys)
            tau = _tau(xs, ys)
            agree = _agreement(xs, ys)
            n = len(xs)
            cells.append({"model": mlabel, "benchmark": plabel,
                          "N": n, "r": r, "tau": tau, "agree": agree})
            per_bench_xy[plabel] = (xs, ys)
            print(f"{mlabel:<22} {plabel:<16} {n:>4} "
                  f"{('---' if r != r else f'{r:.3f}'):>6} "
                  f"{('---' if tau != tau else f'{tau:.3f}'):>6} "
                  f"{('---' if agree != agree else f'{agree*100:.1f}'):>7}")

        # per-model summary: pool all items, also Fisher-z avg of per-bench r / tau
        all_x = sum((per_bench_xy[b][0] for b in per_bench_xy), [])
        all_y = sum((per_bench_xy[b][1] for b in per_bench_xy), [])
        r_pool = _r(all_x, all_y)
        tau_pool = _tau(all_x, all_y)
        agree_pool = _agreement(all_x, all_y)

        def _fisher_avg(vals):
            v = [x for x in vals if x == x]
            if not v: return float("nan")
            return float(np.tanh(np.mean(np.arctanh(np.clip(v, -0.9999, 0.9999)))))

        per_bench_rs   = [c["r"]   for c in cells if c["model"] == mlabel]
        per_bench_taus = [c["tau"] for c in cells if c["model"] == mlabel]
        per_bench_ags  = [c["agree"] for c in cells if c["model"] == mlabel]
        r_avg   = _fisher_avg(per_bench_rs)
        tau_avg = _fisher_avg(per_bench_taus)
        agree_avg = float(np.mean([a for a in per_bench_ags if a == a])) if per_bench_ags else float("nan")

        summary_by_model[mlabel] = {
            "r_pool": r_pool, "r_avg": r_avg,
            "tau_pool": tau_pool, "tau_avg": tau_avg,
            "agree_pool": agree_pool, "agree_avg": agree_avg,
            "n_total": len(all_x),
        }
        print()

    # ---- Build the tau-b matrix (rows = models, cols = benches + Pooled + Avg) ----
    bench_order = [p[0] for p in PANELS]
    model_order = [m[0] for m in MODELS]
    col_labels = bench_order + ["Pooled", "Avg"]

    T = np.full((len(model_order), len(col_labels)), np.nan)
    for c in cells:
        i = model_order.index(c["model"]); j = bench_order.index(c["benchmark"])
        T[i, j] = c["tau"]
    for i, m in enumerate(model_order):
        s = summary_by_model[m]
        T[i, len(bench_order)]     = s["tau_pool"]
        T[i, len(bench_order) + 1] = s["tau_avg"]
    sep = len(bench_order) - 0.5

    # ---- Kendall tau-b heatmap ----
    cmap_t = LinearSegmentedColormap.from_list("torange", ["#fdf6ed", "#f0c896", "#dd884b", "#8a3e10"])
    figT, axT = plt.subplots(figsize=(11, 4.4), dpi=150)
    imT = axT.imshow(T, cmap=cmap_t, vmin=0.0, vmax=1.0, aspect="auto")
    axT.set_xticks(range(len(col_labels))); axT.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    axT.set_yticks(range(len(model_order))); axT.set_yticklabels(model_order, fontsize=10)
    axT.axvline(sep, color="white", linewidth=2.2)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            v = T[i, j]
            if v != v: continue
            color = "white" if v > 0.55 else "#222"
            weight = "bold" if j >= len(bench_order) else "normal"
            axT.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight=weight)
    cbT = figT.colorbar(imT, ax=axT, fraction=0.03, pad=0.02); cbT.set_label("Kendall τ-b  (API ↔ Interface, item-level rank concordance)", fontsize=9)
    axT.set_title("Item-level rank concordance, API vs Interface", fontsize=12, pad=8)
    figT.tight_layout(); figT.savefig(HEATMAP_T); figT.savefig(HEATMAP_T.with_suffix(".pdf")); plt.close(figT)
    print(f"wrote {HEATMAP_T.relative_to(REPO)}")

    # ---- LaTeX tau-b matrix table ----
    with TAU_TEX.open("w") as fh:
        fh.write("% Auto-generated by paper/tables/compute_item_correlation.py — do not edit by hand.\n")
        fh.write(r"\begin{tabular}{@{}l" + "r" * (len(bench_order) + 2) + r"@{}}" + "\n")
        fh.write(r"\toprule" + "\n")
        header = "Model & " + " & ".join(bench_order) + r" & Pooled & Avg \\" + "\n"
        fh.write(header)
        fh.write(r"\midrule" + "\n")
        for i, m in enumerate(model_order):
            cells_str = []
            for j in range(len(bench_order)):
                v = T[i, j]
                cells_str.append("---" if v != v else f"{v:.2f}")
            s = summary_by_model[m]
            cells_str.append("---" if s["tau_pool"] != s["tau_pool"] else rf"\textbf{{{s['tau_pool']:.2f}}}")
            cells_str.append("---" if s["tau_avg"]  != s["tau_avg"]  else rf"\textbf{{{s['tau_avg']:.2f}}}")
            fh.write(f"{m} & " + " & ".join(cells_str) + r" \\" + "\n")
        fh.write(r"\bottomrule" + "\n")
        fh.write(r"\end{tabular}" + "\n")
    print(f"wrote {TAU_TEX.relative_to(REPO)}")


if __name__ == "__main__":
    main()
