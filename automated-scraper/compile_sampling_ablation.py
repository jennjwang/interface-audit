"""Compile BBQ and HellaSwag sampling/reasoning sweeps into LaTeX tables.

Reads per-cell CSVs:
  automated-scraper/outputs/sweep_bbq/sweep_bbq_per_cell.csv
  automated-scraper/outputs/sweep_hellaswag/sweep_hellaswag_per_cell.csv

Writes one LaTeX table per benchmark:
  automated-scraper/outputs/sweep_bbq/sampling_ablation_table.tex
  automated-scraper/outputs/sweep_hellaswag/sampling_ablation_table.tex

Each cell shows accuracy ± SD on top and within-cell test-retest agreement R
(across 3 runs) below in small text. Each row adds two summary range columns.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve()

# (benchmark, per_cell csv, output tex, run-id suffix for non-thinking, suffix for thinking)
BENCHMARKS = [
    ("bbq",       THIS.parent / "outputs" / "sweep_bbq" / "sweep_bbq_per_cell.csv",
                  THIS.parent / "outputs" / "sweep_bbq" / "sampling_ablation_table.tex",
                  "bbq",       "thinking-bbq"),
    ("hellaswag", THIS.parent / "outputs" / "sweep_hellaswag" / "sweep_hellaswag_per_cell.csv",
                  THIS.parent / "outputs" / "sweep_hellaswag" / "sampling_ablation_table.tex",
                  "hellaswag", "thinking-hellaswag"),
]

# (display, model id in csv, sweep_run_id template prefix)
MODELS_TEMPLATE = [
    ("Claude Sonnet 4.6", "claude-sonnet-4-6",
        "sweep-claude-sonnet-{nt}", "sweep-claude-sonnet-{th}"),
    ("Claude Haiku 4.5",  "claude-haiku-4-5-20251001",
        "sweep-claude-haiku-{nt}",  "sweep-claude-haiku-{th}"),
    ("GPT 5.4",           "gpt-5.4-2026-03-05",
        "sweep-gpt54-{nt}",         "sweep-gpt54-{th}"),
    ("Gemini 3 Flash",    "gemini-3-flash-preview",
        "sweep-gemini-flash-{nt}",  "sweep-gemini-flash-{th}"),
]


def models_for(nt_suffix: str, th_suffix: str) -> list[tuple[str, str, str, str]]:
    return [(d, m, n.format(nt=nt_suffix), t.format(th=th_suffix)) for d, m, n, t in MODELS_TEMPLATE]

# Non-thinking axes/values to render as five columns
NONTHINK_COLS = [
    ("temperature", "0.0", r"$T=0.0$"),
    ("temperature", "0.5", r"$T=0.5$"),
    ("temperature", "0.7", r"$T=0.7$"),
    ("top_p",       "0.9", r"top-$p=0.9$"),
    ("top_p",       "0.95", r"top-$p=0.95$"),
]

# Thinking-axis values, ordered low/medium/high (also accepting numeric budget_tokens)
# Per-model axis name dictates which column-value-set to use.
THINK_AXIS_BY_MODEL = {
    "claude-sonnet-4-6":        ("budget_tokens",     [("1024","Low / 1024"),
                                                       ("4096","Medium / 4096"),
                                                       ("16384","High / 16384")]),
    "claude-haiku-4-5-20251001":("budget_tokens",     [("1024","Low / 1024"),
                                                       ("4096","Medium / 4096"),
                                                       ("16384","High / 16384")]),
    "gpt-5.4-2026-03-05":       ("reasoning_effort",  [("low","Low / 1024"),
                                                       ("medium","Medium / 4096"),
                                                       ("high","High / 16384")]),
    "gemini-3-flash-preview":   ("thinking_level",    [("low","Low / 1024"),
                                                       ("medium","Medium / 4096"),
                                                       ("high","High / 16384")]),
}


def load_per_cell(path: Path) -> dict:
    """Returns {(run_id, model, axis, value): {accuracy, run_acc_std, ...}}."""
    out = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["run_id"], r["model"], r["axis"], r["value"])
            out[key] = r
    return out


def cell_str(rec) -> str:
    """Stacked cell: accuracy ± SD on top, test-retest agreement (R) below."""
    acc = float(rec["accuracy"]) * 100
    sd  = float(rec["run_acc_std"]) * 100
    r   = float(rec.get("retest_pct", 0)) * 100
    cell_body = rf"${acc:.1f} \pm {sd:.1f}$\\\scriptsize $R\!=\!{r:.1f}$"
    return r"\makecell{" + cell_body + r"}"


def model_range_pp(rec_list) -> float:
    accs = [float(r["accuracy"]) * 100 for r in rec_list]
    return max(accs) - min(accs)


def model_retest_range(rec_list) -> tuple[float, float]:
    rs = [float(r.get("retest_pct", 0)) * 100 for r in rec_list]
    return (min(rs), max(rs)) if rs else (float("nan"), float("nan"))


def build_nonthinking_row(rows, display, mid, run_id):
    cells = []
    recs = []
    for axis, value, _label in NONTHINK_COLS:
        rec = rows.get((run_id, mid, axis, value))
        if rec is None:
            cells.append("---")
        else:
            cells.append(cell_str(rec))
            recs.append(rec)
    rng = model_range_pp(recs) if recs else float("nan")
    r_lo, r_hi = model_retest_range(recs)
    r_str = f"{r_lo:.1f}--{r_hi:.1f}" if r_lo == r_lo else "---"
    return f"{display} & " + " & ".join(cells) + f" & {rng:.1f} & {r_str}"


def build_thinking_row(rows, display, mid, run_id):
    axis, value_labels = THINK_AXIS_BY_MODEL[mid]
    cells = []
    recs = []
    for value, _label in value_labels:
        rec = rows.get((run_id, mid, axis, value))
        if rec is None:
            cells.append("---")
        else:
            cells.append(cell_str(rec))
            recs.append(rec)
    rng = model_range_pp(recs) if recs else float("nan")
    r_lo, r_hi = model_retest_range(recs)
    r_str = f"{r_lo:.1f}--{r_hi:.1f}" if r_lo == r_lo else "---"
    return f"{display} & " + " & ".join(cells) + f" & & & {rng:.1f} & {r_str}"


def build_table(rows: dict, models: list, benchmark: str) -> list[str]:
    out_lines = []
    out_lines.append(r"% Auto-generated by automated-scraper/compile_sampling_ablation.py")
    out_lines.append(rf"% Benchmark: {benchmark}")
    out_lines.append(r"\begin{tabular}{@{}lccccc r r@{}}")
    out_lines.append(r"\toprule")
    out_lines.append(r"& \multicolumn{5}{c}{Non-thinking sampling parameters} & & \\")
    out_lines.append(r"\cmidrule(lr){2-6}")
    out_lines.append(r"Model & $T=0.0$ & $T=0.5$ & $T=0.7$ & top-$p=0.9$ & top-$p=0.95$ & Acc range & $R^{\mathrm{API}}$ range \\")
    out_lines.append(r"\midrule")
    for display, mid, nt_run, _think_run in models:
        out_lines.append(build_nonthinking_row(rows, display, mid, nt_run) + r" \\")
    out_lines.append(r"\midrule")
    out_lines.append(r"& \multicolumn{4}{c}{Reasoning / thinking parameters} & & & \\")
    out_lines.append(r"\cmidrule(lr){2-5}")
    out_lines.append(r"Model & Low / 1024 & Medium / 4096 & High / 16384 & & & Acc range & $R^{\mathrm{API}}$ range \\")
    out_lines.append(r"\midrule")
    for display, mid, _nt_run, think_run in models:
        out_lines.append(build_thinking_row(rows, display, mid, think_run) + r" \\")
    out_lines.append(r"\bottomrule")
    out_lines.append(r"\end{tabular}")
    return out_lines


def main():
    for benchmark, per_cell, out_tex, nt_suffix, th_suffix in BENCHMARKS:
        if not per_cell.exists():
            print(f"skip {benchmark}: {per_cell} not found")
            continue
        print(f"\n=== {benchmark} ===")
        rows = load_per_cell(per_cell)
        models = models_for(nt_suffix, th_suffix)
        out_lines = build_table(rows, models, benchmark)
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_tex.write_text("\n".join(out_lines) + "\n")
        print(f"wrote {out_tex}")
        for line in out_lines:
            print(line)
        print()
        print(f"# {benchmark} within-model spread (max-min accuracy in pp):")
        for display, mid, nt_run, think_run in models:
            nt_recs = [rows[(nt_run, mid, ax, v)] for ax, v, _ in NONTHINK_COLS if (nt_run, mid, ax, v) in rows]
            think_axis, think_vals = THINK_AXIS_BY_MODEL[mid]
            th_recs = [rows[(think_run, mid, think_axis, v)] for v, _ in think_vals if (think_run, mid, think_axis, v) in rows]
            nt_rng = model_range_pp(nt_recs) if nt_recs else float("nan")
            th_rng = model_range_pp(th_recs) if th_recs else float("nan")
            print(f"  {display:<22} non-thinking range = {nt_rng:.1f}    reasoning range = {th_rng:.1f}")


if __name__ == "__main__":
    main()
