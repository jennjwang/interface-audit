"""BH-correct significance for the system-prompt ablation table.

Reads paper/tables/appendix_system_prompt_ablation.csv. For each (system, benchmark)
cell there are two p-values:
  p_api_vs_iface : Iface − API gap (gap_orig)
  p_sp_vs_iface  : Iface − SP gap (gap_remain)

Applies Benjamini-Hochberg across all 5*8*2 = 80 hypotheses at once and outputs
the longtable body with raw p-values shown but boldface driven by BH q < 0.05.
"""
import csv
from pathlib import Path

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
CSV_PATH = REPO / "paper" / "tables" / "appendix_system_prompt_ablation.csv"

# (display label, CSV "model" value)
SYSTEMS = [
    ("GPT 5.3 Instant",  "gpt-5.3"),
    ("GPT 5.4 Thinking", "gpt-5.4"),
    ("Claude Opus",      "claude-opus-4.6"),
    ("Claude Sonnet",    "claude-sonnet-4.6"),
    ("Gemini 3 Fast",    "gemini-3-flash"),
]
# (display label, CSV "benchmark" value)
BENCHES = [
    ("ARC",            "arc"),
    ("GSM8K",          "gsm8k"),
    ("HellaSwag",      "hellaswag"),
    ("MMLU",           "mmlu"),
    ("TruthfulQA",     "truthfulqa"),
    ("WinoGrande",     "winogrande"),
    ("BBQ",            "bbq"),
    ("AA-Omniscience", "aa-omniscience"),
    ("Elephant Flip",  "elephant-flip"),
]


def bh_adjust(pvals):
    """Return BH-adjusted q-values aligned to pvals (preserves input order)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [None] * m
    cum_min = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        adj = pvals[i] * m / rank
        cum_min = min(cum_min, adj)
        q[i] = min(cum_min, 1.0)
    return q


def fmt_p(p, q):
    """4-decimal p-value, bold if q < 0.05."""
    s = f"{p:.4f}"
    return rf"\textbf{{{s}}}" if q < 0.05 else s


def main():
    rows = {(r["model"], r["benchmark"]): r for r in csv.DictReader(CSV_PATH.open())}

    # Collect every cell's two p-values, in row-major order. Skip (sys, bench)
    # combinations with no row (e.g. GPT 5.3 Instant + Elephant Flip has no SP).
    pvals = []
    flat = []   # parallel: list of (sys_idx, bench_idx, kind in {"api","sp"})
    for si, (_, mkey) in enumerate(SYSTEMS):
        for bi, (_, bkey) in enumerate(BENCHES):
            r = rows.get((mkey, bkey))
            if r is None:
                continue
            pvals.append(float(r["p_api_vs_iface"]))
            flat.append((si, bi, "api"))
            pvals.append(float(r["p_sp_vs_iface"]))
            flat.append((si, bi, "sp"))

    qvals = bh_adjust(pvals)

    # Index q-values by (si, bi, kind)
    q_by = {(si, bi, k): q for (si, bi, k), q in zip(flat, qvals)}

    n_raw_005 = sum(1 for p in pvals if p < 0.05)
    n_bh_005 = sum(1 for q in qvals if q < 0.05)
    print(f"raw p<0.05: {n_raw_005}/{len(pvals)}   BH q<0.05: {n_bh_005}/{len(pvals)}\n")

    # --- LaTeX body ---
    print(r"\begin{longtable}{@{}lrrrrr@{}}")
    n_contrasts = len(pvals)
    print(rf"\caption{{System-prompt ablation results. SP is the accuracy of the system-prompted API condition in percent. Iface--API is the interface accuracy minus the baseline API accuracy; Iface--SP is the interface accuracy minus the system-prompted API accuracy. Gaps are reported in percentage points. Bold p-values indicate $q < 0.05$ after Benjamini--Hochberg FDR correction across all {n_contrasts} contrasts.}}")
    print(r"\label{tab:system-prompt-ablation}\\")
    print()
    print(r"\toprule")
    print(r"Benchmark & SP (\%) & Iface--API $\Delta$ & Iface--SP $\Delta$ & Iface--API $p$ & Iface--SP $p$ \\")
    print(r"\midrule")
    print(r"\endfirsthead")
    print()
    print(r"\toprule")
    print(r"Benchmark & SP (\%) & Iface--API $\Delta$ & Iface--SP $\Delta$ & Iface--API $p$ & Iface--SP $p$ \\")
    print(r"\midrule")
    print(r"\endhead")
    print()
    print(r"\midrule")
    print(r"\multicolumn{6}{r}{\emph{Continued on next page}}\\")
    print(r"\endfoot")
    print()
    print(r"\bottomrule")
    print(r"\endlastfoot")
    print()
    for si, (sys_label, mkey) in enumerate(SYSTEMS):
        print(rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{sys_label}}}}} \\")
        for bi, (bench_label, bkey) in enumerate(BENCHES):
            r = rows.get((mkey, bkey))
            if r is None:
                continue
            sp = float(r["sp_mean"])
            gap_orig = float(r["gap_orig"])
            gap_remain = float(r["gap_remain"])
            p_api = float(r["p_api_vs_iface"])
            p_sp = float(r["p_sp_vs_iface"])
            q_api = q_by[(si, bi, "api")]
            q_sp = q_by[(si, bi, "sp")]
            print(f"{bench_label:<15} & {sp:>5.1f} & {gap_orig:>5.1f} & {gap_remain:>5.1f} "
                  f"& {fmt_p(p_api, q_api)} & {fmt_p(p_sp, q_sp)} \\\\")
        if si < len(SYSTEMS) - 1:
            print(r"\addlinespace")
            print()
    print()
    print(r"\end{longtable}")


if __name__ == "__main__":
    main()
