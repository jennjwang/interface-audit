# browser_automation

Data collection infrastructure for the API vs Interface study. Collects model
responses through two channels — the **API** (SDK calls / Batch API) and the
**web interface** (browser automation via DrissionPage).

## Layout

| Path | Contents |
|---|---|
| `api_runner.py` | Shared API client (Anthropic, OpenAI, Google SDKs). Imported by audit scripts and batch runner. |
| `response_cleaning.py` | Normalizes scraped interface responses (strips provider accessibility labels). |
| `audit/` | Browser-automation scrapers, one per vendor (`audit_chatgpt.py`, `audit_claude.py`, `audit_gemini.py`). Each supports both standard and layer-2 modes (account/session/request variance via `--attach-port-base`). |
| `runners/` | `batch_runner.py` (OpenAI/Anthropic Batch API submission) and `generate_sweep_yaml.py` (expands parameter specs into experiment configs). |
| `yamls/` | Consolidated experiment configs: `interface_scraping.yaml`, `api_batch.yaml`, `sampling_sweeps.yaml`, `account_variance.yaml`, `system_prompt.yaml`. All but `system_prompt.yaml` hold **multiple YAML documents** (one per provider/model/run) — see [Selecting a config](#selecting-a-config). |

## Running

```bash
# Interface scraping (launches Chrome via DrissionPage)
python audit/audit_claude.py --configs yamls/interface_scraping.yaml#haiku

# Layer-2 mode (attach to existing Chrome sessions for account variance)
python audit/audit_chatgpt.py --configs yamls/account_variance.yaml#0 \
    --attach-port-base 9222 --profile-base <chrome_profiles_dir>

# Batch API submission (OpenAI/Anthropic, 50% cheaper)
python runners/batch_runner.py yamls/api_batch.yaml#6 --config-api-models --run-id <id>

# Generate sweep configs from parameter spec
python runners/generate_sweep_yaml.py --temperature 0,0.5,0.7 --top-p 0.9,0.95
```

### Selecting a config

Each entry passed to `--configs` supplies **one session**, so a config file that
holds several YAML documents needs a selector saying which one to use — by
document index or by an experiment / api-model name unique to that document:

```bash
yamls/interface_scraping.yaml#5        # by document index
yamls/interface_scraping.yaml#haiku    # by experiment name
```

Single-document files (`system_prompt.yaml`) load with no selector. Omitting a
selector on a multi-document file — or giving one that matches several
documents — fails with a listing of the available documents, rather than
silently running the wrong provider's config.

## Data

Run data (gitignored, multi-GB) is stored in `chatgpt_data/`, `claude_data/`,
`gemini_data/`, and `data/`. Each contains `raw_html/`, `parsed_json/`, and
`api/` subdirectories.

## Dependencies

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

Installs pyyaml, python-dotenv, DrissionPage, beautifulsoup4 and the Anthropic /
OpenAI / Google SDKs from `pyproject.toml`. beautifulsoup4 is required for the
HTML→JSON parse step that runs after each interface scrape; without it the run
collects raw HTML and prints `Parse skipped: BeautifulSoup not installed`.

## Prerequisites per channel

| Channel | Needs |
|---|---|
| **API** (`api_runner.py`, `runners/batch_runner.py`) | Provider keys in a `.env` at the repo root or above: `ANTHROPIC_KEY`, `OPENAI_KEY`, `GEMINI_API_KEY`. Runs headless — no browser. |
| **Interface** (`audit/audit_*.py`) | Google Chrome, plus a Chrome profile already signed in to the provider. Profiles live under `<vendor>_data/chrome_profiles*/` and are gitignored, so a fresh clone has none — sign in once with `--allow-manual-login` before unattended runs. |

Failed API calls are not silent: the per-query JSON records an `error` field
(e.g. an expired key or exhausted credit) alongside the prompt and timestamps.
