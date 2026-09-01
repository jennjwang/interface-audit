# interface-audit

Code and data for **"API Benchmark Scores Do Not Reliably Transfer to Chatbot Platforms"**.

We compare LLM accuracy when accessed via API versus web interface across 9 benchmarks and 7 model systems (Claude, GPT, Gemini). All 700 runs are included — 10 evaluation sets × 7 models × 2 surfaces × 5 runs, where the 9 benchmarks yield 10 sets because AITA-NTA-OG and AITA-NTA-Flip are evaluated separately.

## Repository structure

```
interface-audit/
├── analysis/                  # Paper reproduction (3 scripts → all tables + figures)
│   ├── compute_paper_stats.py     # Main results: LME, bootstrap, test-retest, Spearman
│   ├── compute_ablation_stats.py  # Ablations: model-version, account, timestamp, sweeps
│   ├── plot_figures.py            # Figure 1 (bar chart) + Figure 3 (heatmap)
│   ├── _sweep_sp.py              # Sweep/system-prompt table generation (imported by ablation script)
│   ├── requirements.txt           # Dependencies for reproduction + scoring
│   └── artifacts/                 # Generated tables, figures, and CSVs
├── extraction/                # Answer extraction + scoring pipeline
│   ├── parse_raw_html.py          # Raw HTML scrapes → JSON
│   └── score.py                   # JSON responses → scored CSV (regex + LLM judge)
├── browser_automation/        # Data collection infrastructure
│   ├── audit/                     # Browser scrapers (ChatGPT, Claude, Gemini)
│   ├── runners/                   # Batch API runner + sweep YAML generator
│   ├── yamls/                     # Experiment configs (5 files)
│   ├── api_runner.py              # Shared API client (Anthropic, OpenAI, Google SDKs)
│   ├── config_loader.py           # Selects one config document from a multi-document YAML
│   └── response_cleaning.py      # Normalize scraped interface responses
└── data.zip                   # All benchmark data (Git LFS, 287 MB)
```

## Reproducing paper results

```bash
# Unzip data (if not already present) — 287 MB zip → 966 MB on disk
unzip data.zip

# Install dependencies (numpy, pandas, scipy, statsmodels, matplotlib)
pip install -r analysis/requirements.txt

# Generate all tables, figures, and statistics
python analysis/compute_paper_stats.py --latex     # ~5 min
python analysis/compute_ablation_stats.py --latex  # ~1 min
python analysis/plot_figures.py                    # ~5 sec
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
| `human_validation/` | Raw extractor-validation votes from the manual audit rounds, the AA-Omniscience 100-item grader sample, and the consolidation scripts. These are the annotation records; the per-benchmark agreement rates reported in the paper are in the appendix table, not recomputed here |
| `ablations/` | System-prompt, sampling/reasoning sweeps, account variation, GPT 5.4 instant |

Each run directory contains `responses/` (raw JSON) and `scored.csv` (pre-computed scores).

## Collecting new data

Collection lives in `browser_automation/`, which drives the two access surfaces the
paper compares: the **API** (provider SDKs, headless) and the **interface**
(a real browser session via DrissionPage). See `browser_automation/README.md`
for the module layout.

### Setup

```bash
cd browser_automation
pip install -e .
```

Prerequisites differ by channel:

| Channel | Needs |
|---|---|
| API | Provider keys in a `.env` at the repo root: `ANTHROPIC_KEY`, `OPENAI_KEY`, `GEMINI_API_KEY`. No browser. |
| Interface | Google Chrome, plus a Chrome profile signed in to the provider. Profiles live under `<vendor>_data/chrome_profiles*/` and are gitignored, so a fresh clone has none. |

### Choosing a config

The five files in `yamls/` each hold **several YAML documents**, one per
provider/model/run. Because each `--configs` entry supplies one session, append a
selector naming the document — a document index, or an experiment/api-model name
unique to it:

```bash
yamls/interface_scraping.yaml#5        # by document index
yamls/interface_scraping.yaml#haiku    # by experiment name
```

Omitting the selector on a multi-document file, or giving an ambiguous one, prints
the available documents instead of guessing. `system_prompt.yaml` is a single
document and needs no selector.

### API channel

```bash
python api_runner.py yamls/api_batch.yaml#6 --config-api-models --run-id <id>
```

Writes one JSON per query under `<out-root>/<run-id>/<model>_<params>/session_00/<experiment>/`.
Each record carries the prompt, timestamps, token usage and the response — and on
failure an `error` field (expired key, exhausted credit) rather than failing silently.
`runners/batch_runner.py` submits the same configs through the OpenAI/Anthropic
Batch APIs at roughly half the cost.

### Interface channel

```bash
# First time on a machine: sign in when Chrome opens, then leave it running
python audit/audit_claude.py --configs yamls/interface_scraping.yaml#haiku \
    --allow-manual-login
```

Useful flags: `--sessions N` (parallel browser sessions), `--profile-base <dir>`
(where Chrome profiles live), `--start-in <minutes>` (delay the start),
`--no-parse` (skip the HTML→JSON step), `--clear-memory` (wipe conversation history
first), `--seed` (shuffle order). Layer-2 variance runs attach to already-open
Chrome instances with `--attach-port-base 9222`.

A run writes raw pages to `<vendor>_data/raw_html/<run-id>/session_XX/<group>/`,
then parses them into `parsed_json/` at the end (this step needs `beautifulsoup4`,
installed above). Progress is logged to `<vendor>_data/logs/<run-id>/`.

**Pacing.** The scraper waits `soft_refresh_timeout = 90` seconds for a reply
before refreshing and re-sending, and gives up on a query after 600s. Accounts on
free tiers frequently exceed both, producing refresh/re-send loops and no captured
output; raise those limits or collect from a paid account.

### Scoring what you collect

Both channels produce JSON the scoring pipeline reads directly:

```bash
python extraction/score.py --data-root <run_dir> --answer-key data/answer_keys/bbq-subset-200.csv --task bbq
```

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
