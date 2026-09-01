# interface-audit

Code and data for **"Does the API Mirror the Interface? Comparing LLM Performance Across Access Surfaces"**.

We compare LLM accuracy when accessed via API versus web interface across 9 benchmarks and 7 model systems (Claude, GPT, Gemini). All 700 runs (10 benchmarks × 7 models × 2 surfaces × 5 runs) are included.

## Repository structure

```
interface-audit/
├── analysis/                  # Paper reproduction (3 scripts → all tables + figures)
│   ├── compute_paper_stats.py     # Main results: LME, bootstrap, test-retest, Spearman
│   ├── compute_ablation_stats.py  # Ablations: model-version, account, timestamp, sweeps
│   ├── plot_figures.py            # Figure 1 (bar chart) + Figure 3 (heatmap)
│   ├── _sweep_sp.py              # Sweep/system-prompt table generation (imported by ablation script)
│   └── artifacts/                 # Generated tables, figures, and CSVs
├── extraction/                # Answer extraction + scoring pipeline
│   └── score.py                   # JSON responses → scored CSV (regex + LLM judge)
├── browser_automation/        # Data collection infrastructure
│   ├── audit/                     # Browser scrapers (ChatGPT, Claude, Gemini; layer 1 + 2)
│   ├── runners/                   # Batch API runner + sweep YAML generator
│   ├── yamls/                     # Experiment configs
│   ├── outputs/                   # Human validation data + extractor audit trail
│   ├── api_runner.py              # Shared API client (Anthropic, OpenAI, Google SDKs)
│   └── response_cleaning.py      # Normalize scraped interface responses
├── data.zip                   # All benchmark data (Git LFS, 287 MB)
└── data/                      # Unzipped data (gitignored, 981 MB)
```

## Reproducing paper results

```bash
# Unzip data (if not already present)
unzip data.zip

# Generate all tables, figures, and statistics
python analysis/compute_paper_stats.py --latex      # ~3 min
python analysis/compute_ablation_stats.py --latex    # ~30 sec
python analysis/plot_figures.py                      # ~3 sec
```

Outputs:
- `analysis/artifacts/tables/main_tables.tex` — main-text table
- `analysis/artifacts/tables/appendix.tex` — 18 appendix tables
- `analysis/artifacts/figures/` — 2 figures (aggregate accuracy bars, capability deltas heatmap)
- `analysis/artifacts/data/` — intermediate CSVs (accuracy per run, per-cell LME, figure data)

## Data

`data.zip` contains:

| Directory | Contents |
|---|---|
| `metabench-{arc,gsm8k,hellaswag,mmlu,truthfulQA,winogrande}/` | 6 metabench subsets (420 runs) |
| `bbq/`, `aa-omniscience/`, `elephant-{flip,og}/` | 3 custom benchmarks (280 runs) |
| `answer_keys/` | Gold answer CSVs for custom benchmarks |
| `caches/` | LLM judge caches (BBQ, AA-Omni, elephant, system-prompt) |
| `human_validation/` | Annotator audit data (extractor votes + validation summary) |
| `ablations/` | System-prompt, sampling/reasoning sweeps, account variation, GPT 5.4 instant |

Each run directory contains `responses/` (raw JSON) and `scored.csv` (pre-computed scores).

## Scoring a new run

```bash
# Multiple-choice (regex extraction, no API key needed)
python extraction/score.py --data-root <json_dir> --answer-key <csv> --task mc

# BBQ (A-C letter extraction)
python extraction/score.py --data-root <json_dir> --answer-key <csv> --task bbq

# GSM8K (numeric extraction)
python extraction/score.py --data-root <json_dir> --answer-key <csv> --task gsm8k

# Free-form (LLM judge, requires OPENAI_KEY)
python extraction/score.py --data-root <json_dir> --answer-key <csv> --task judge
```
