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
│   ├── parse_raw_html.py          # Raw HTML scrapes → JSON
│   └── score.py                   # JSON responses → scored CSV (regex + LLM judge)
├── browser_automation/        # Data collection infrastructure
│   ├── audit/                     # Browser scrapers (ChatGPT, Claude, Gemini)
│   ├── runners/                   # Batch API runner + sweep YAML generator
│   ├── yamls/                     # Experiment configs (5 files)
│   ├── api_runner.py              # Shared API client (Anthropic, OpenAI, Google SDKs)
│   └── response_cleaning.py      # Normalize scraped interface responses
├── data.zip                   # All benchmark data (Git LFS, 287 MB)
└── data/                      # Unzipped data (gitignored, 981 MB)
```

## Reproducing paper results

```bash
# Unzip data (if not already present) — 287 MB zip → 966 MB on disk
unzip data.zip

# Install dependencies (numpy, pandas, scipy, statsmodels, matplotlib)
python -m venv .venv-analysis
.venv-analysis/bin/pip install -r analysis/requirements.txt

# Generate all tables, figures, and statistics
.venv-analysis/bin/python analysis/compute_paper_stats.py --latex     # ~5 min
.venv-analysis/bin/python analysis/compute_ablation_stats.py --latex  # ~1 min
.venv-analysis/bin/python analysis/plot_figures.py                    # ~5 sec
```

No API key is needed: the reproduction runs entirely off the shipped data and
the LLM-judge caches in `data/caches/`.

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
| `human_validation/` | Annotator audit data: raw extractor votes, the AA-Omniscience 100-item grader sample, consolidated vote CSVs, and the producer scripts |
| `ablations/` | System-prompt, sampling/reasoning sweeps, account variation, GPT 5.4 instant |

Each run directory contains `responses/` (raw JSON) and `scored.csv` (pre-computed scores).

## Collecting new data

Data collection lives in `browser_automation/` (see its README for details).

```bash
cd browser_automation
python -m venv .venv && .venv/bin/pip install -e .   # pyyaml, dotenv, DrissionPage, provider SDKs

# API channel — needs ANTHROPIC_KEY / OPENAI_KEY / GEMINI_API_KEY in .env
.venv/bin/python api_runner.py yamls/api_batch.yaml#6 --config-api-models --run-id <id>

# Interface channel — launches Chrome via DrissionPage; needs a logged-in profile
.venv/bin/python audit/audit_claude.py --configs yamls/interface_scraping.yaml#haiku
```

The consolidated configs hold several YAML documents each, so a selector
(`#<index>` or `#<experiment-name>`) picks which one to run; omitting it prints
the available documents. Collected runs are scored with `extraction/score.py`
below.

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
