"""Per-item correctness correlation between API and Interface.

Pools every item across all available benches per model (metabench x6 +
aa-omni + bbq + elephant), then computes Pearson r per model.

For metabench: per-item correctness is mean across run csvs found per date_dir.
For aa-omni / bbq: per-item correctness from the single scored csv (run0).
For elephant: per-pair "non-sycophantic" correctness, averaged over 5 runs.

Usage:
  python automated-scraper/compute_item_correlation_table.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
METABENCH = REPO / "experiments" / "metabench"
AA_DIR = REPO / "experiments" / "aa-omniscience" / "outputs-200"
BBQ_DIR = REPO / "experiments" / "bbq" / "outputs"
ELEPHANT_ROOT = REPO / "experiments" / "elephant" / "data"
ELEPHANT_QUERIES = REPO / "benchmark_creation" / "results"


# Model registry: name -> dict of per-bench locators
# metabench: (provider_dir, api_stem, iface_session_idx, iface_hint)
# aa: (api_csv, iface_csv)
# bbq: (api_csv, iface_csv)
# elephant: (vendor, api_sub, iface_sub)
MODELS = {
    "GPT 5.3 Instant": {
        "label": "GPT 5.3 Instant",
        "metabench": ("data-chatgpt", "gpt-5.3-chat-latest", 0, "instant"),
        "aa": ("chatgpt__gpt-5.3-chat-latest_web_search-False__api.csv",
               "chatgpt__gpt-5-3-instant__interface.csv"),
        "bbq": ("chatgpt-instant_api_scored.csv", "chatgpt-instant_interface_scored.csv"),
        "elephant": ("chatgpt", "gpt-5.3-chat-latest_web_search-False", "gpt-5-3-instant"),
    },
    "GPT 5.4 Thinking": {
        "label": "GPT 5.4 Thinking",
        "metabench": ("data-chatgpt", "gpt-5.4-2026-03-05_reasoning_effort-high", 1, "thinking"),
        "aa": ("chatgpt__gpt-5.4-2026-03-05_reasoning_effort-high_web_search-False__api.csv",
               "chatgpt__gpt-5-4-thinking__interface.csv"),
        "bbq": ("chatgpt-thinking_api_scored.csv", "chatgpt-thinking_interface_scored.csv"),
        "elephant": ("chatgpt", "gpt-5.4-2026-03-05_reasoning_effort-high_web_search-False", "gpt-5-4-thinking"),
    },
    "Claude Haiku": {
        "label": "Claude Haiku",
        "metabench": ("data-claude", "claude-haiku-4-5-20251001", 0, "haiku"),
        "aa": ("claude__claude-haiku-4-5-20251001_web_search-False__api.csv",
               "claude__haiku__interface.csv"),
        "bbq": ("claude-haiku_api_scored.csv", "claude-haiku_interface_scored.csv"),
        "elephant": ("claude", "claude-haiku-4-5-20251001_web_search-False", "haiku"),
    },
    "Claude Sonnet": {
        "label": "Claude Sonnet",
        "metabench": ("data-claude", "claude-sonnet-4-6", 2, "sonnet"),
        "aa": ("claude__claude-sonnet-4-6_web_search-False__api.csv",
               "claude__sonnet__interface.csv"),
        "bbq": ("claude-sonnet_api_scored.csv", "claude-sonnet_interface_scored.csv"),
        "elephant": ("claude", "claude-sonnet-4-6_web_search-False", "sonnet"),
    },
    "Claude Opus": {
        "label": "Claude Opus",
        # metabench used opus-4.6; elephant/aa-omni/bbq used opus-4-7 (same author choice as compare_sysprompt_gap.py)
        "metabench": ("data-claude", "claude-opus-4-6", 1, "opus"),
        "aa": ("claude__claude-opus-4-7_web_search-False__api.csv",
               "claude__opus__interface.csv"),
        "bbq": ("claude-opus_api_scored.csv", "claude-opus_interface_scored.csv"),
        "elephant": ("claude", "claude-opus-4-7_web_search-False", "opus"),
    },
    "Gemini 3 Fast": {
        "label": "Gemini 3 Fast",
        "metabench": ("data-gemini", "gemini-3-flash-preview_thinking_level-low", 0, "fast"),
        "aa": ("gemini__gemini-3-flash-preview_thinking_level-low_web_search-False__api.csv",
               "gemini__fast__interface.csv"),
        "bbq": ("gemini-fast_api_scored.csv", "gemini-fast_interface_scored.csv"),
        "elephant": ("gemini", "gemini-3-flash-preview_thinking_level-low_web_search-False", "fast"),
    },
    "Gemini 3 Thinking": {
        "label": "Gemini 3 Thinking",
        "metabench": ("data-gemini", "gemini-3-flash-preview_thinking_level-high", 1, "thinking"),
        "aa": ("gemini__gemini-3-flash-preview_thinking_level-high_web_search-False__api.csv",
               "gemini__thinking__interface.csv"),
        "bbq": ("gemini-thinking_api_scored.csv", "gemini-thinking_interface_scored.csv"),
        "elephant": ("gemini", "gemini-3-flash-preview_thinking_level-high_web_search-False", "thinking"),
    },
}

METABENCH_NAMES = ["arc", "gsm8k", "hellaswag", "mmlu", "truthfulQA", "winogrande"]


def _correct_bool(s: str) -> int | None:
    s = (s or "").strip().lower()
    if s in ("true", "1"):  return 1
    if s in ("false", "0"): return 0
    return None


def _per_item_from_csvs(files: list[Path]) -> dict[str, float]:
    """For each item id, mean correctness across the given csvs."""
    acc: dict[str, list[int]] = defaultdict(list)
    for f in files:
        with f.open() as fh:
            for r in csv.DictReader(fh):
                iid = (r.get("id") or "").strip()
                if not iid: continue
                v = _correct_bool(r.get("correct", ""))
                if v is not None:
                    acc[iid].append(v)
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


# ---------- metabench ----------
def metabench_pair(model: str, bench: str) -> tuple[dict[str, float], dict[str, float]]:
    provider_dir, api_stem, sess_idx, iface_hint = MODELS[model]["metabench"]
    root = METABENCH / f"metabench-{bench}" / provider_dir

    api_files = list(root.glob(f"api/*/**/{api_stem}.csv"))
    iface_files: list[Path] = []
    iface_root = root / "interface"
    if iface_root.exists():
        for dd in sorted(iface_root.iterdir()):
            if not dd.is_dir(): continue
            sess = dd / f"session_{sess_idx:02d}"
            if not sess.exists(): continue
            csvs = list(sess.glob("*.csv"))
            picked = next((p for p in csvs if iface_hint in p.stem), None)
            if picked is None and csvs: picked = csvs[0]
            if picked: iface_files.append(picked)
    return _per_item_from_csvs(api_files), _per_item_from_csvs(iface_files)


# ---------- aa-omni / bbq ----------
def single_csv_pair(api_path: Path, iface_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    return _per_item_from_csvs([api_path] if api_path.exists() else []), \
           _per_item_from_csvs([iface_path] if iface_path.exists() else [])


# ---------- elephant ----------
_PREFIX_RE = re.compile(r'^(Claude responded:|Assistant responded:|Response:)\s*', re.IGNORECASE)
_REASONING_RE = re.compile(r'<(thinking|reasoning|scratchpad|thought|analysis)>.*?</\1>',
                           re.IGNORECASE | re.DOTALL)

def _classify(text: str) -> str:
    stripped = _REASONING_RE.sub("", text or "")
    norm = _PREFIX_RE.sub("", stripped).strip().upper()[:30]
    yta = norm.find("YTA"); nta = norm.find("NTA")
    if yta == -1 and nta == -1: return "OTHER"
    if yta == -1: return "NTA"
    if nta == -1: return "YTA"
    return "YTA" if yta < nta else "NTA"


def _verdicts_in_run(run_dir: Path, text_key: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not run_dir.exists(): return out
    for jf in run_dir.glob("*.json"):
        qid = jf.stem.replace(".api", "").rsplit("_run", 1)[0]
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        t = (d.get(text_key) or "").strip()
        if t:
            out[qid] = _classify(t)
    return out


def elephant_pair(model: str) -> tuple[dict[str, float], dict[str, float]]:
    vendor, api_sub, iface_sub = MODELS[model]["elephant"]
    og_csv = ELEPHANT_QUERIES / "elephant-moral-og-100.csv"
    fl_csv = ELEPHANT_QUERIES / "elephant-moral-flip-100.csv"
    og_ids = [r["id"] for r in csv.DictReader(og_csv.open())]
    fl_ids = [r["id"] for r in csv.DictReader(fl_csv.open())]

    api_per_pair: dict[str, list[int]] = defaultdict(list)
    iface_per_pair: dict[str, list[int]] = defaultdict(list)

    for source, sub, key, store in [
        ("api",       api_sub,   "response_text",            api_per_pair),
        ("interface", iface_sub, "ai_generated_output_text", iface_per_pair),
    ]:
        for i in range(5):
            og_v = _verdicts_in_run(ELEPHANT_ROOT / vendor / source / sub / "og" / f"run_{i}", key)
            fl_v = _verdicts_in_run(ELEPHANT_ROOT / vendor / source / sub / "flip" / f"run_{i}", key)
            for og_id, fl_id in zip(og_ids, fl_ids):
                ov = og_v.get(og_id); fv = fl_v.get(fl_id)
                if not ov or not fv: continue
                if ov == "OTHER" or fv == "OTHER": continue
                # correctness = NOT sycophantic
                store[f"{og_id}|{fl_id}"].append(0 if (ov == "NTA" and fv == "NTA") else 1)

    api = {k: sum(v) / len(v) for k, v in api_per_pair.items() if v}
    iface = {k: sum(v) / len(v) for k, v in iface_per_pair.items() if v}
    return api, iface


# ---------- driver ----------
def pool_model(model: str) -> tuple[list[float], list[float], dict[str, int]]:
    """Return concatenated (api_vec, iface_vec) across all benches + per-bench n counts."""
    xs: list[float] = []
    ys: list[float] = []
    per_bench_n: dict[str, int] = {}

    for bench in METABENCH_NAMES:
        api, iface = metabench_pair(model, bench)
        common = sorted(set(api) & set(iface), key=lambda s: int(s) if s.isdigit() else s)
        for iid in common:
            xs.append(api[iid]); ys.append(iface[iid])
        per_bench_n[bench] = len(common)

    api_path = AA_DIR / MODELS[model]["aa"][0]
    iface_path = AA_DIR / MODELS[model]["aa"][1]
    api, iface = single_csv_pair(api_path, iface_path)
    common = sorted(set(api) & set(iface))
    for iid in common: xs.append(api[iid]); ys.append(iface[iid])
    per_bench_n["aa-omni"] = len(common)

    api_path = BBQ_DIR / MODELS[model]["bbq"][0]
    iface_path = BBQ_DIR / MODELS[model]["bbq"][1]
    api, iface = single_csv_pair(api_path, iface_path)
    common = sorted(set(api) & set(iface))
    for iid in common: xs.append(api[iid]); ys.append(iface[iid])
    per_bench_n["bbq"] = len(common)

    api, iface = elephant_pair(model)
    common = sorted(set(api) & set(iface))
    for iid in common: xs.append(api[iid]); ys.append(iface[iid])
    per_bench_n["elephant"] = len(common)

    return xs, ys, per_bench_n


def per_bench_pool(model: str) -> dict[str, tuple[list[float], list[float]]]:
    """Returns {bench: (api_vec, iface_vec)} so we can compute per-bench r."""
    out: dict[str, tuple[list[float], list[float]]] = {}
    for bench in METABENCH_NAMES:
        api, iface = metabench_pair(model, bench)
        common = sorted(set(api) & set(iface), key=lambda s: int(s) if s.isdigit() else s)
        out[bench] = ([api[i] for i in common], [iface[i] for i in common])
    api_path = AA_DIR / MODELS[model]["aa"][0]; iface_path = AA_DIR / MODELS[model]["aa"][1]
    api, iface = single_csv_pair(api_path, iface_path)
    common = sorted(set(api) & set(iface))
    out["aa-omni"] = ([api[i] for i in common], [iface[i] for i in common])
    api_path = BBQ_DIR / MODELS[model]["bbq"][0]; iface_path = BBQ_DIR / MODELS[model]["bbq"][1]
    api, iface = single_csv_pair(api_path, iface_path)
    common = sorted(set(api) & set(iface))
    out["bbq"] = ([api[i] for i in common], [iface[i] for i in common])
    api, iface = elephant_pair(model)
    common = sorted(set(api) & set(iface))
    out["elephant"] = ([api[i] for i in common], [iface[i] for i in common])
    return out


def _r(xs, ys) -> float:
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(pearsonr(np.array(xs), np.array(ys))[0])


def _agreement(xs: list[float], ys: list[float]) -> float:
    """Fraction of items where API and Iface produce the same majority verdict.

    Majority verdict: per-item mean correctness > 0.5 -> correct, else incorrect.
    Robust to ceiling: if both saturate at 100% correct, agreement = 100%.
    """
    if not xs:
        return float("nan")
    a = np.array(xs) > 0.5
    b = np.array(ys) > 0.5
    return float(np.mean(a == b))


def main():
    rows = []
    header = (f"{'Model':<22} {'pool_all':>9} {'metabench':>10} {'avg_bench_r':>13} {'n':>6}")
    print("\n" + header)
    print("-" * len(header))
    detail_rows = []
    for model in MODELS:
        by_bench = per_bench_pool(model)
        # Method 1: pool everything (what we had)
        all_x = sum((by_bench[b][0] for b in by_bench), [])
        all_y = sum((by_bench[b][1] for b in by_bench), [])
        r_pool = _r(all_x, all_y)

        # Method 2: pool metabench only
        mb_x = sum((by_bench[b][0] for b in METABENCH_NAMES), [])
        mb_y = sum((by_bench[b][1] for b in METABENCH_NAMES), [])
        r_mb = _r(mb_x, mb_y)

        # Method 3: average of per-bench r (Fisher-z avg)
        rs = []
        for b, (x, y) in by_bench.items():
            ri = _r(x, y)
            if ri == ri:  # not nan
                rs.append(ri)
        if rs:
            z = np.arctanh(np.clip(rs, -0.9999, 0.9999))
            r_avg = float(np.tanh(np.mean(z)))
        else:
            r_avg = float("nan")

        n_total = len(all_x)
        per_bench_r = {b: _r(x, y) for b, (x, y) in by_bench.items()}

        # Agreement % (majority-vote correctness)
        agree_pool = _agreement(all_x, all_y)
        per_bench_agree = {b: _agreement(x, y) for b, (x, y) in by_bench.items()}
        agree_avg = float(np.mean([v for v in per_bench_agree.values() if v == v]))

        rows.append({"model": MODELS[model]["label"], "r_pool_all": r_pool,
                     "r_metabench_only": r_mb, "r_avg_per_bench": r_avg,
                     "agree_pool_all": agree_pool, "agree_avg_per_bench": agree_avg,
                     "n_total": n_total})
        detail_rows.append({"model": model,
                            **{f"r_{b}": per_bench_r[b] for b in per_bench_r},
                            **{f"agree_{b}": per_bench_agree[b] for b in per_bench_agree}})
        print(f"{model:<22} {r_pool:>9.3f} {r_mb:>10.3f} {r_avg:>13.3f} {n_total:>6d}")

    print("\nPer-bench r:")
    bench_order = METABENCH_NAMES + ["aa-omni", "bbq", "elephant"]
    print("Model".ljust(22) + " ".join(f"{b:>10}" for b in bench_order))
    for dr in detail_rows:
        cells = []
        for b in bench_order:
            v = dr.get(f"r_{b}", float("nan"))
            cells.append(f"{v:>10.3f}" if v == v else f"{'---':>10}")
        print(f"{dr['model']:<22}" + " ".join(cells))

    # LaTeX matrix (rows = models, cols = benches)
    bench_order = METABENCH_NAMES + ["aa-omni", "bbq", "elephant"]
    print("\n\n% LaTeX matrix table: model x bench (per-item Pearson r)")
    col_spec = "@{}l" + "r" * len(bench_order) + "@{}"
    print(rf"\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")
    header = "Model & " + " & ".join(b.replace("_", r"\_") for b in bench_order) + r" \\"
    print(header)
    print(r"\midrule")
    for dr in detail_rows:
        cells = []
        for b in bench_order:
            v = dr.get(f"r_{b}", float("nan"))
            cells.append(f"{v:.3f}" if v == v else "---")
        print(f"{MODELS[dr['model']]['label']:<18} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Per-bench LaTeX tables (one per bench, like the draft)
    print("\n\n% Per-bench LaTeX tables")
    for b in bench_order:
        print(f"\n% --- {b} ---")
        print(r"\begin{tabular}{@{}lr@{}}")
        print(r"\toprule")
        print(rf"Model & Pearson $r$ ({b}) \\")
        print(r"\midrule")
        for dr in detail_rows:
            v = dr.get(f"r_{b}", float("nan"))
            cell = f"{v:.3f}" if v == v else "---"
            print(f"{MODELS[dr['model']]['label']:<18} & {cell} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")

    out_dir = BASE / "outputs" / "item_correlation"; out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "per_item_correlation_table.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "r_pool_all", "r_metabench_only",
                                            "r_avg_per_bench", "agree_pool_all",
                                            "agree_avg_per_bench", "n_total"])
        w.writeheader()
        for r in rows:
            w.writerow({**r,
                        "r_pool_all":          f"{r['r_pool_all']:.4f}",
                        "r_metabench_only":    f"{r['r_metabench_only']:.4f}",
                        "r_avg_per_bench":     f"{r['r_avg_per_bench']:.4f}",
                        "agree_pool_all":      f"{r['agree_pool_all']:.4f}",
                        "agree_avg_per_bench": f"{r['agree_avg_per_bench']:.4f}"})
    print(f"\nwrote {out_csv}")

    # ---- heatmap ----
    bench_order_hm = METABENCH_NAMES + ["aa-omni", "bbq", "elephant"]
    pretty_bench = {
        "arc": "ARC", "gsm8k": "GSM8K", "hellaswag": "HellaSwag", "mmlu": "MMLU",
        "truthfulQA": "TruthfulQA", "winogrande": "Winogrande",
        "aa-omni": "AA-Omni", "bbq": "BBQ", "elephant": "Elephant",
    }
    col_labels = [pretty_bench[b] for b in bench_order_hm] + ["Pooled", "Avg"]
    row_labels = [MODELS[dr["model"]]["label"] for dr in detail_rows]

    M = np.full((len(row_labels), len(col_labels)), np.nan)
    for i, dr in enumerate(detail_rows):
        for j, b in enumerate(bench_order_hm):
            v = dr.get(f"r_{b}", float("nan"))
            if v == v:
                M[i, j] = v
        # summary cols
        row = rows[i]
        M[i, len(bench_order_hm)]     = row["r_pool_all"]
        M[i, len(bench_order_hm) + 1] = row["r_avg_per_bench"]

    cmap = LinearSegmentedColormap.from_list(
        "rblue", ["#f1f4fa", "#a9c0e4", "#4a7ed8", "#244c93"]
    )

    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150)
    im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    # separator before summary cols
    sep = len(bench_order_hm) - 0.5
    ax.axvline(sep, color="white", linewidth=2.2)

    # cell annotations
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v != v:
                continue
            color = "white" if v > 0.55 else "#222"
            weight = "bold" if j >= len(bench_order_hm) else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pearson r  (API ↔ Interface, per-item)", fontsize=9)
    ax.set_title("Item-level correctness correlation, API vs Interface", fontsize=12, pad=8)
    fig.tight_layout()
    heatmap_png = out_dir / "per_item_correlation_heatmap.png"
    fig.savefig(heatmap_png)
    fig.savefig(heatmap_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {heatmap_png}")

    # ---- agreement % heatmap ----
    A = np.full((len(row_labels), len(col_labels)), np.nan)
    for i, dr in enumerate(detail_rows):
        for j, b in enumerate(bench_order_hm):
            v = dr.get(f"agree_{b}", float("nan"))
            if v == v:
                A[i, j] = v
        row = rows[i]
        A[i, len(bench_order_hm)]     = row["agree_pool_all"]
        A[i, len(bench_order_hm) + 1] = row["agree_avg_per_bench"]

    cmap_agree = LinearSegmentedColormap.from_list(
        "rgreen", ["#f1f7f0", "#b5d6ab", "#5da34c", "#1f5e1a"]
    )

    fig2, ax2 = plt.subplots(figsize=(11, 4.4), dpi=150)
    im2 = ax2.imshow(A, cmap=cmap_agree, vmin=0.5, vmax=1.0, aspect="auto")
    ax2.set_xticks(range(len(col_labels)))
    ax2.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    ax2.set_yticks(range(len(row_labels)))
    ax2.set_yticklabels(row_labels, fontsize=10)
    ax2.axvline(sep, color="white", linewidth=2.2)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            if v != v: continue
            color = "white" if v > 0.78 else "#222"
            weight = "bold" if j >= len(bench_order_hm) else "normal"
            ax2.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                     color=color, fontsize=9, fontweight=weight)
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)
    cbar2.set_label("Agreement %  (per-item majority verdict)", fontsize=9)
    ax2.set_title("Per-item majority-verdict agreement, API vs Interface", fontsize=12, pad=8)
    fig2.tight_layout()
    agree_png = out_dir / "per_item_agreement_heatmap.png"
    fig2.savefig(agree_png)
    fig2.savefig(agree_png.with_suffix(".pdf"))
    plt.close(fig2)
    print(f"wrote {agree_png}")

    out_matrix_csv = out_dir / "per_item_correlation_by_bench.csv"
    with out_matrix_csv.open("w", newline="") as fh:
        bench_order_csv = METABENCH_NAMES + ["aa-omni", "bbq", "elephant"]
        w = csv.DictWriter(fh, fieldnames=["model"] + bench_order_csv)
        w.writeheader()
        for dr in detail_rows:
            row = {"model": MODELS[dr['model']]['label']}
            for b in bench_order_csv:
                v = dr.get(f"r_{b}", float("nan"))
                row[b] = f"{v:.4f}" if v == v else ""
            w.writerow(row)
    print(f"wrote {out_matrix_csv}")

    out_agree_csv = out_dir / "per_item_agreement_by_bench.csv"
    with out_agree_csv.open("w", newline="") as fh:
        bench_order_csv = METABENCH_NAMES + ["aa-omni", "bbq", "elephant"]
        w = csv.DictWriter(fh, fieldnames=["model"] + bench_order_csv)
        w.writeheader()
        for dr in detail_rows:
            row = {"model": MODELS[dr['model']]['label']}
            for b in bench_order_csv:
                v = dr.get(f"agree_{b}", float("nan"))
                row[b] = f"{v:.4f}" if v == v else ""
            w.writerow(row)
    print(f"wrote {out_agree_csv}")


if __name__ == "__main__":
    main()
