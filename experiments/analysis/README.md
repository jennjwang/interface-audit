# Metabench Analysis Scripts

All scripts are meant to be run from the **`experiments/`** directory
(i.e. `cd experiments && python openllm_leaderboard/openllm_leaderboard.py`), or from
within `experiments/analysis/` using relative `--data-dir` paths.

---

## Pipeline overview

```
experiments/
  metabench-*/          ← raw CSV data, one folder per benchmark
  analysis/
    config.py           ← shared model/colour config (edit model list here)
  openllm_leaderboard/        ← cross-benchmark leaderboard outputs + scripts
    openllm_leaderboard.py    ← Step 1: aggregate CSVs → leaderboard CSVs
    accuracy_table_visual.py  ← Step 2: render accuracy table PNG
    plot_overall_ranking_stability.py  ← Step 3: overall bump+freq chart
    *.csv / *.png             ← generated artifacts
  analysis/
    config.py           ← shared model/colour config (edit model list here)
    plot_rankings.py          ← per-benchmark bump chart  (split API / Iface)
    plot_accuracy.py          ← per-benchmark accuracy bar chart
    plot_question_examples.py ← per-benchmark answer heatmap
    fleiss_kappa.py           ← per-benchmark Fleiss' κ
```

---

## Step 1 — Build the overall leaderboard CSVs

```bash
cd /path/to/personalization/experiments
python openllm_leaderboard/openllm_leaderboard.py
# outputs → openllm_leaderboard/
#   run_leaderboard.csv       per-run scores + ranks
#   ranking_summary.csv       mean rank / score per condition
#   ordering_frequency.csv    how often each ordering appeared
#   fleiss_kappa.csv          inter-run consistency
#   coverage_by_run.csv       per-dataset coverage audit
```

## Step 2 — Accuracy table PNG

```bash
python openllm_leaderboard/accuracy_table_visual.py
# outputs → openllm_leaderboard/plots/accuracy_table.png
```

## Step 3 — Overall leaderboard ranking stability (API vs Interface split)

```bash
python openllm_leaderboard/plot_overall_ranking_stability.py
# outputs → openllm_leaderboard/plots/overall_ranking_stability_split.png
```

## Step 2b — Leaderboard AVG accuracy bar chart

```bash
python analysis/plot_accuracy.py --leaderboard
# outputs → openllm_leaderboard/plots/avg_accuracy_by_condition.png
```

Mimics the per-benchmark `ranking_stability.png` style: left panel = API
models, right panel = Interface presets, each with a bump chart (rank over
runs) and an ordering-frequency bar chart.

---

## Per-benchmark scripts

These all require `--data-dir`:

```bash
# bump chart — API vs Interface split
python analysis/plot_rankings.py --data-dir metabench-mmlu/data

# accuracy bar chart
python analysis/plot_accuracy.py --data-dir metabench-mmlu/data

# per-question answer heatmap  (auto-selects high-variance questions)
python analysis/plot_question_examples.py --data-dir metabench-mmlu/data \
    --answer-key metabench-mmlu/queries/metabench_mmlu_5shot.csv

# Fleiss' κ
python analysis/fleiss_kappa.py --data-dir metabench-mmlu/data
```

Outputs are saved under `metabench-<name>/plots/`.

---

## Editing model / colour config

`config.py` is the single source of truth for:

- **`DATASET_MODELS`** — which model files are used per benchmark
- **`CONDITION_COLORS`** — per-condition colours used in all plots
- **`SESSION_MAP`** — maps interface CSV filenames → session folder names
