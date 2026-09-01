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
| `yamls/` | Consolidated experiment configs: `interface_scraping.yaml`, `api_batch.yaml`, `sampling_sweeps.yaml`, `account_variance.yaml`, `system_prompt.yaml`. |

## Running

```bash
# Interface scraping (launches Chrome via DrissionPage)
python audit/audit_claude.py --configs yamls/interface_scraping.yaml

# Layer-2 mode (attach to existing Chrome sessions for account variance)
python audit/audit_chatgpt.py --configs yamls/account_variance.yaml \
    --attach-port-base 9222 --profile-base <chrome_profiles_dir>

# Batch API submission (OpenAI/Anthropic, 50% cheaper)
python runners/batch_runner.py yamls/api_batch.yaml --config-api-models --run-id <id>

# Generate sweep configs from parameter spec
python runners/generate_sweep_yaml.py --temperature 0,0.5,0.7 --top-p 0.9,0.95
```

## Data

Run data (gitignored, multi-GB) is stored in `chatgpt_data/`, `claude_data/`,
`gemini_data/`, and `data/`. Each contains `raw_html/`, `parsed_json/`, and
`api/` subdirectories.

## Dependencies

```
pip install pyyaml python-dotenv DrissionPage anthropic openai google-genai
```
