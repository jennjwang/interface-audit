"""Plot time-of-day vs accuracy for all providers, API vs Interface.

Computes accuracy as correct/answered (filtering out non-extractable questions)
rather than correct/total, to avoid conflating scraping failures with model errors.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent / "openllm_leaderboard" / "plots"
OUT_DIR = PLOTS_DIR

PROVIDER_CSVS = {
    "ChatGPT": PLOTS_DIR / "coverage_by_run.csv",
    "Claude":  PLOTS_DIR / "coverage_by_run_claude.csv",
    "Gemini":  PLOTS_DIR / "coverage_by_run_gemini.csv",
}

TIME_BUCKETS = [
    ("Night\n(0–5h)",     0, 5),
    ("Morning\n(6–11h)",  6, 11),
    ("Afternoon\n(12–17h)", 12, 17),
    ("Evening\n(18–23h)", 18, 23),
]

API_COLOR = "#4A90D9"
IFACE_COLOR = "#E07B53"


def parse_hour(timestamp: str) -> int | None:
    m = re.match(r"\d{4}-\d{2}-\d{2}_(\d{2})-\d{2}-\d{2}", timestamp)
    return int(m.group(1)) if m else None


def bucket_idx(hour: int) -> int:
    for i, (_, lo, hi) in enumerate(TIME_BUCKETS):
        if lo <= hour <= hi:
            return i
    return -1


def filtered_accuracy(accuracy: float, answered: int, total: int) -> float | None:
    """Recompute accuracy as correct/answered, filtering non-extractable items.

    Original accuracy = correct/total (from scorable rows with gold answers).
    We want correct/answered instead.
    correct = accuracy * total, so filtered = (accuracy * total) / answered.
    Returns None if answered == 0.
    """
    if answered <= 0:
        return None
    correct = accuracy * total
    return correct / answered


def load_provider(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ts = (r.get("timestamp") or "").strip()
            hour = parse_hour(ts)
            try:
                raw_acc = float(r["accuracy"])
                answered = int(r["answered"])
                total = int(r["total"])
            except (TypeError, ValueError, KeyError):
                continue
            if hour is None:
                continue
            acc = filtered_accuracy(raw_acc, answered, total)
            if acc is None:
                continue
            rows.append({
                "condition": (r.get("condition") or "").strip(),
                "hour": hour,
                "accuracy": acc,
                "raw_accuracy": raw_acc,
                "answered": answered,
                "total": total,
                "extract_rate": answered / total if total else 0,
                "dataset": (r.get("dataset") or "").strip(),
                "bucket": bucket_idx(hour),
            })
    return rows


def plot_scatter_all_providers():
    """Scatter: hour vs accuracy, one subplot per provider, API vs Interface color."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, (provider, csv_path) in zip(axes, PROVIDER_CSVS.items()):
        rows = load_provider(csv_path)
        if not rows:
            ax.set_title(provider + "\n(no data)")
            continue

        api_h, api_a = [], []
        iface_h, iface_a = [], []
        for r in rows:
            if r["condition"].startswith("API"):
                api_h.append(r["hour"])
                api_a.append(r["accuracy"])
            else:
                iface_h.append(r["hour"])
                iface_a.append(r["accuracy"])

        ax.scatter(api_h, api_a, c=API_COLOR, alpha=0.5, s=30, label="API", zorder=3)
        ax.scatter(iface_h, iface_a, c=IFACE_COLOR, alpha=0.5, s=30, label="Interface", zorder=3)

        for hours, accs, color, label in [
            (api_h, api_a, API_COLOR, "API"),
            (iface_h, iface_a, IFACE_COLOR, "Interface"),
        ]:
            if len(hours) >= 3:
                h = np.array(hours)
                a = np.array(accs)
                slope, intercept, r_val, p_val, _ = stats.linregress(h, a)
                x_line = np.linspace(0, 23, 100)
                ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2,
                        alpha=0.8, linestyle="--")
                sig = "*" if p_val < 0.05 else ""
                y_pos = 0.02 if label == "API" else 0.08
                ax.text(0.02, y_pos,
                        f"{label}: r={r_val:.2f}, p={p_val:.3f}{sig}",
                        transform=ax.transAxes, fontsize=8, color=color,
                        verticalalignment="bottom")

        ax.set_title(provider, fontsize=13, fontweight="bold")
        ax.set_xlabel("Hour of day")
        ax.set_xlim(-1, 24)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy (correct / answered)")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)

    y_min = min(ax.get_ylim()[0] for ax in axes)
    for ax in axes:
        ax.set_ylim(max(0, y_min - 0.02), 1.02)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig.suptitle("Accuracy vs Time of Day — filtered (correct/answered only)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "time_of_day_scatter.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_bucket_bars():
    """Grouped bar chart: time buckets, API vs Interface, one subplot per provider."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    bucket_labels = [b[0] for b in TIME_BUCKETS]
    x = np.arange(len(TIME_BUCKETS))
    width = 0.35

    for ax, (provider, csv_path) in zip(axes, PROVIDER_CSVS.items()):
        rows = load_provider(csv_path)
        if not rows:
            ax.set_title(provider + "\n(no data)")
            continue

        api_buckets = defaultdict(list)
        iface_buckets = defaultdict(list)
        for r in rows:
            if r["bucket"] < 0:
                continue
            if r["condition"].startswith("API"):
                api_buckets[r["bucket"]].append(r["accuracy"])
            else:
                iface_buckets[r["bucket"]].append(r["accuracy"])

        api_means = [np.mean(api_buckets[i]) if api_buckets[i] else 0 for i in range(len(TIME_BUCKETS))]
        api_sems = [np.std(api_buckets[i]) / np.sqrt(len(api_buckets[i])) if len(api_buckets[i]) > 1 else 0
                    for i in range(len(TIME_BUCKETS))]
        iface_means = [np.mean(iface_buckets[i]) if iface_buckets[i] else 0 for i in range(len(TIME_BUCKETS))]
        iface_sems = [np.std(iface_buckets[i]) / np.sqrt(len(iface_buckets[i])) if len(iface_buckets[i]) > 1 else 0
                      for i in range(len(TIME_BUCKETS))]

        bars_api = ax.bar(x - width / 2, api_means, width, yerr=api_sems, capsize=4,
                          color=API_COLOR, edgecolor="white", linewidth=1, label="API",
                          error_kw=dict(lw=1.2, capthick=1), zorder=3)
        bars_iface = ax.bar(x + width / 2, iface_means, width, yerr=iface_sems, capsize=4,
                            color=IFACE_COLOR, edgecolor="white", linewidth=1, label="Interface",
                            error_kw=dict(lw=1.2, capthick=1), zorder=3)

        for i in range(len(TIME_BUCKETS)):
            for bars, buckets in [(bars_api, api_buckets), (bars_iface, iface_buckets)]:
                n = len(buckets[i])
                if n > 0:
                    bar = bars[i]
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.015,
                            f"n={n}", ha="center", va="top", fontsize=7, color="white", alpha=0.9)

        ax.set_title(provider, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, fontsize=9)
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean Accuracy ±SE (correct/answered)")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)

    y_min = min(ax.get_ylim()[0] for ax in axes)
    for ax in axes:
        ax.set_ylim(max(0, y_min - 0.02), 1.02)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig.suptitle("Accuracy by Time-of-Day Bucket — filtered (correct/answered only)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "time_of_day_buckets.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_extraction_rate():
    """Show extraction rate (answered/total) by time bucket, Interface only."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    bucket_labels = [b[0] for b in TIME_BUCKETS]
    x = np.arange(len(TIME_BUCKETS))

    for ax, (provider, csv_path) in zip(axes, PROVIDER_CSVS.items()):
        rows = load_provider(csv_path)
        if not rows:
            ax.set_title(provider + "\n(no data)")
            continue

        iface_buckets = defaultdict(list)
        api_buckets = defaultdict(list)
        for r in rows:
            if r["bucket"] < 0:
                continue
            if r["condition"].startswith("Interface"):
                iface_buckets[r["bucket"]].append(r["extract_rate"])
            else:
                api_buckets[r["bucket"]].append(r["extract_rate"])

        width = 0.35
        api_means = [np.mean(api_buckets[i]) if api_buckets[i] else 1 for i in range(len(TIME_BUCKETS))]
        iface_means = [np.mean(iface_buckets[i]) if iface_buckets[i] else 1 for i in range(len(TIME_BUCKETS))]
        api_sems = [np.std(api_buckets[i]) / np.sqrt(len(api_buckets[i])) if len(api_buckets[i]) > 1 else 0
                    for i in range(len(TIME_BUCKETS))]
        iface_sems = [np.std(iface_buckets[i]) / np.sqrt(len(iface_buckets[i])) if len(iface_buckets[i]) > 1 else 0
                      for i in range(len(TIME_BUCKETS))]

        ax.bar(x - width / 2, api_means, width, yerr=api_sems, capsize=4,
               color=API_COLOR, edgecolor="white", label="API", zorder=3)
        ax.bar(x + width / 2, iface_means, width, yerr=iface_sems, capsize=4,
               color=IFACE_COLOR, edgecolor="white", label="Interface", zorder=3)

        ax.set_title(provider, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, fontsize=9)
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Extraction Rate (answered/total)")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    for ax in axes:
        ax.set_ylim(0.85, 1.01)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig.suptitle("Answer Extraction Rate by Time-of-Day Bucket",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "time_of_day_extraction_rate.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_interface_by_benchmark():
    """Per-benchmark scatter for Interface conditions only, one row per provider."""
    benchmarks = ["metabench-arc", "metabench-mmlu", "metabench-hellaswag",
                   "metabench-truthfulQA", "metabench-winogrande", "metabench-gsm8k"]
    providers = list(PROVIDER_CSVS.keys())

    fig, axes = plt.subplots(len(providers), len(benchmarks), figsize=(22, 10),
                             sharex=True, sharey=False)

    for row_i, (provider, csv_path) in enumerate(PROVIDER_CSVS.items()):
        rows = load_provider(csv_path)
        iface_rows = [r for r in rows if r["condition"].startswith("Interface")]

        for col_j, bench in enumerate(benchmarks):
            ax = axes[row_i][col_j]
            bench_rows = [r for r in iface_rows if r["dataset"] == bench]

            if not bench_rows:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
            else:
                conditions = sorted(set(r["condition"] for r in bench_rows))
                cmap = plt.cm.Set2
                for ci, cond in enumerate(conditions):
                    c_rows = [r for r in bench_rows if r["condition"] == cond]
                    h = [r["hour"] for r in c_rows]
                    a = [r["accuracy"] for r in c_rows]
                    short = cond.split(": ", 1)[1] if ": " in cond else cond
                    ax.scatter(h, a, s=25, alpha=0.7, label=short, color=cmap(ci), zorder=3)

                    if len(h) >= 3:
                        slope, intercept, r_val, p_val, _ = stats.linregress(h, a)
                        x_line = np.linspace(0, 23, 50)
                        ax.plot(x_line, slope * x_line + intercept, color=cmap(ci),
                                linewidth=1.5, alpha=0.6, linestyle="--")

            if row_i == 0:
                short_bench = bench.replace("metabench-", "")
                ax.set_title(short_bench, fontsize=10, fontweight="bold")
            if col_j == 0:
                ax.set_ylabel(provider + "\nAccuracy", fontsize=9)
            if row_i == len(providers) - 1:
                ax.set_xlabel("Hour", fontsize=9)

            ax.set_xlim(-1, 24)
            ax.set_xticks([0, 6, 12, 18])
            ax.grid(alpha=0.15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_i == 0 and col_j == len(benchmarks) - 1 and bench_rows:
                ax.legend(fontsize=6, loc="lower left", framealpha=0.8)

    fig.suptitle("Interface Accuracy vs Hour — filtered (correct/answered), by Provider & Benchmark",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "time_of_day_interface_by_benchmark.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_scatter_by_model():
    """Scatter: hour vs filtered accuracy, one subplot per provider, colored by model."""
    from config import CONDITION_COLORS

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, (provider, csv_path) in zip(axes, PROVIDER_CSVS.items()):
        rows = load_provider(csv_path)
        if not rows:
            ax.set_title(provider + "\n(no data)")
            continue

        # Group by model (= condition column)
        models = sorted(set(r["condition"] for r in rows))
        for model in models:
            m_rows = [r for r in rows if r["condition"] == model]
            h = [r["hour"] for r in m_rows]
            a = [r["accuracy"] for r in m_rows]
            color = CONDITION_COLORS.get(model, "#6B7280")
            short = model.split(": ", 1)[1] if ": " in model else model
            marker = "o" if model.startswith("API") else "s"

            ax.scatter(h, a, c=color, alpha=0.55, s=30, label=short,
                       marker=marker, zorder=3, edgecolors="white", linewidths=0.3)

            if len(h) >= 3:
                h_arr = np.array(h)
                a_arr = np.array(a)
                slope, intercept, r_val, p_val, _ = stats.linregress(h_arr, a_arr)
                x_line = np.linspace(0, 23, 100)
                ax.plot(x_line, slope * x_line + intercept, color=color,
                        linewidth=1.5, alpha=0.6, linestyle="--")

        ax.set_title(provider, fontsize=13, fontweight="bold")
        ax.set_xlabel("Hour of day")
        ax.set_xlim(-1, 24)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, loc="lower left", framealpha=0.9, ncol=2,
                  title="● API  ■ Interface", title_fontsize=7)

    axes[0].set_ylabel("Accuracy (correct / answered)")
    y_min = min(ax.get_ylim()[0] for ax in axes)
    for ax in axes:
        ax.set_ylim(max(0, y_min - 0.02), 1.02)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig.suptitle("Accuracy vs Time of Day — by Model (filtered, correct/answered)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "time_of_day_scatter_by_model.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary():
    """Print the numeric summary with both raw and filtered accuracy."""
    print("\n" + "=" * 90)
    print("  CROSS-PROVIDER SUMMARY: raw (correct/total) vs filtered (correct/answered)")
    print("=" * 90)

    for provider, csv_path in PROVIDER_CSVS.items():
        rows = load_provider(csv_path)
        if not rows:
            continue
        iface = [r for r in rows if r["condition"].startswith("Interface")]
        if not iface:
            continue

        print(f"\n  {provider}")
        print(f"  {'bucket':<20s}  {'raw_acc':>10s}  {'filtered_acc':>12s}  {'extract_rate':>12s}  {'n':>4s}")
        print(f"  {'-' * 65}")

        for bi, (label, lo, hi) in enumerate(TIME_BUCKETS):
            bucket_rows = [r for r in iface if r["bucket"] == bi]
            if not bucket_rows:
                continue
            raw = np.mean([r["raw_accuracy"] for r in bucket_rows])
            filt = np.mean([r["accuracy"] for r in bucket_rows])
            ext = np.mean([r["extract_rate"] for r in bucket_rows])
            n = len(bucket_rows)
            label_clean = label.replace("\n", " ")
            print(f"  {label_clean:<20s}  {raw:>9.3f}  {filt:>11.3f}  {ext:>11.3f}  {n:>4d}")

        # Correlation with filtered accuracy
        hours = np.array([r["hour"] for r in iface])
        filt_accs = np.array([r["accuracy"] for r in iface])
        raw_accs = np.array([r["raw_accuracy"] for r in iface])
        if len(hours) >= 3:
            r_raw, p_raw = stats.pearsonr(hours, raw_accs)
            r_filt, p_filt = stats.pearsonr(hours, filt_accs)
            sig_raw = " *" if p_raw < 0.05 else ""
            sig_filt = " *" if p_filt < 0.05 else ""
            print(f"  Correlation (raw):      r={r_raw:.3f}, p={p_raw:.4f}{sig_raw}")
            print(f"  Correlation (filtered): r={r_filt:.3f}, p={p_filt:.4f}{sig_filt}")


if __name__ == "__main__":
    plot_scatter_all_providers()
    plot_bucket_bars()
    plot_extraction_rate()
    plot_interface_by_benchmark()
    plot_scatter_by_model()
    print_summary()
