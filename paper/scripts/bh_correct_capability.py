"""BH-correct significance for the API-interface capability delta table.

Reads paper/tables/appendix_full_capability.csv. For each (system, benchmark)
cell, computes a two-sided Wald p-value from diff_pp / se_pp, then applies the
Benjamini-Hochberg procedure across all 63 cells.

Prints
  1) a per-system summary with the BH-adjusted q-value per cell, and
  2) a LaTeX row body for a delta-only table using
       \bfseries + \sym{**}   when q < 0.01
       \bfseries + \sym{*}    when q < 0.05
       plain value             otherwise
"""
import csv
from pathlib import Path

from scipy import stats

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
CSV_PATH = REPO / "paper" / "tables" / "appendix_full_capability.csv"

SYSTEMS = [
    ("GPT 5.3 Inst.",        "GPT 5.3 Instant"),
    ("GPT 5.4 Think",        "GPT 5.4 Thinking"),
    ("Claude Haiku 4.5",     "Claude Haiku"),
    ("Claude Opus 4.6",      "Claude Opus"),
    ("Claude Sonnet 4.6",    "Claude Sonnet"),
    ("Gemini 3 Flash Think", "Gemini 3 Thinking"),
    ("Gemini 3 Flash Fast",  "Gemini 3 Fast"),
]
BENCHES = ["ARC", "GSM8K", "HellaSwag", "MMLU", "TruthfulQA", "WinoGrande",
           "BBQ", "AA-Omniscience", "Elephant Flip"]


def main():
    rows = {(r["model"], r["benchmark"]): r for r in csv.DictReader(CSV_PATH.open())}

    cells = []   # [sys_idx, bench_idx, diff, se, p_raw, p_bh]
    for si, (_, mkey) in enumerate(SYSTEMS):
        for bi, bench in enumerate(BENCHES):
            r = rows[(mkey, bench)]
            d = float(r["diff_pp"])
            se = float(r["se_pp"])
            if se == 0:
                p = 1.0
            else:
                z = abs(d / se)
                p = 2 * (1 - stats.norm.cdf(z))
            cells.append([si, bi, d, se, p, None])

    # Benjamini-Hochberg across all cells
    m = len(cells)
    order = sorted(range(m), key=lambda i: cells[i][4])
    cum_min = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        adj = cells[i][4] * m / rank
        cum_min = min(cum_min, adj)
        cells[i][5] = min(cum_min, 1.0)

    n_raw_001 = sum(1 for c in cells if c[4] < 0.01)
    n_raw_005 = sum(1 for c in cells if c[4] < 0.05)
    n_bh_001 = sum(1 for c in cells if c[5] < 0.01)
    n_bh_005 = sum(1 for c in cells if c[5] < 0.05)
    print(f"raw p<0.01: {n_raw_001}   p<0.05: {n_raw_005}")
    print(f"BH  q<0.01: {n_bh_001}   q<0.05: {n_bh_005}\n")

    print(f"{'System':<22}", "  ".join(f"{b:<22}" for b in BENCHES))
    for si, (label, _) in enumerate(SYSTEMS):
        out = []
        for bi in range(len(BENCHES)):
            c = next(c for c in cells if c[0] == si and c[1] == bi)
            stars = "**" if c[5] < 0.01 else "*" if c[5] < 0.05 else ""
            out.append(f"{c[2]:+5.1f}{stars:<3} (q={c[5]:.3f})")
        print(f"{label:<22} ", "  ".join(f"{s:<22}" for s in out))

    print("\n% --- BH-corrected LaTeX rows (paste into the capability-delta table body) ---\n")
    for si, (label, _) in enumerate(SYSTEMS):
        if si in (2, 5):
            print(r"\addlinespace")
        bits = [label]
        for bi in range(len(BENCHES)):
            c = next(c for c in cells if c[0] == si and c[1] == bi)
            val = f"{c[2]:.1f}"
            if c[5] < 0.01:
                bits.append(rf"\bfseries {val} & \sym{{**}}")
            elif c[5] < 0.05:
                bits.append(rf"\bfseries {val} & \sym{{*}}")
            else:
                bits.append(f"{val} & ")
        print(" & ".join(bits) + r" \\")


if __name__ == "__main__":
    main()
