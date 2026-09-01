# browser_automation

Data collection infrastructure for the API vs Interface study. Collects model
responses through two channels — the **API** (SDK calls / Batch API) and the
**web interface** (browser automation via DrissionPage).

## Layout

| Path | Contents |
|---|---|
| `api_runner.py` | Shared API client (Anthropic, OpenAI, Google SDKs). Imported by audit scripts and batch runner. |
| `response_cleaning.py` | Normalizes scraped interface responses (strips provider accessibility labels). |
| `config_loader.py` | Resolves a `path#selector` reference to one YAML document (`load_config`), and lists the available documents when the selector is missing or ambiguous — see [Selecting a config](#selecting-a-config). |
| `audit/` | Browser-automation scrapers, one per vendor (`audit_chatgpt.py`, `audit_claude.py`, `audit_gemini.py`). Each supports both standard and routing-variance modes (account/session/request variance via `--attach-port-base`). Alongside them, `rate_limit.py` enforces the collection protocol below — a rolling-window query cap and a cooldown after rate-limit events, with one limiter shared by all sessions of a run. |
| `runners/` | `batch_runner.py` (OpenAI/Anthropic Batch API submission) and `generate_sweep_yaml.py` (expands parameter specs into experiment configs). |
| `yamls/` | Consolidated experiment configs: `interface_scraping.yaml`, `api_batch.yaml`, `sampling_sweeps.yaml`, `account_variance.yaml`, `system_prompt.yaml`. All but `system_prompt.yaml` hold **multiple YAML documents** (one per provider/model/run) — see [Selecting a config](#selecting-a-config). |

## Running

```bash
# Interface scraping (launches Chrome via DrissionPage)
python audit/audit_claude.py --configs yamls/interface_scraping.yaml#haiku

# Routing-variance mode (attach to existing Chrome sessions for account variance)
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

Run data (gitignored, multi-GB) is written to two roots, both relative to this
directory. A fresh clone has neither — they are created on the first run.

| Root | Written by | Contains |
|---|---|---|
| `chatgpt_data/`, `claude_data/`, `gemini_data/` | `audit/audit_*.py` | `raw_html/<run-id>/session_XX/<group>/` (scraped pages), `parsed_json/<run-id>/` (HTML→JSON), `logs/<run-id>/`, plus the Chrome profile dirs `chrome_profiles_*/` and `drission_tmp/` |
| `data/` | `api_runner.py` (default `--out-root`) | `api/<run-id>/<model>_<params>/session_XX/<experiment>/` — one JSON per query |

## Dependencies

```bash
pip install -e .
```

Installs pyyaml, python-dotenv, DrissionPage, beautifulsoup4 and the Anthropic /
OpenAI / Google SDKs from `pyproject.toml`. beautifulsoup4 is required for the
HTML→JSON parse step that runs after each interface scrape; without it the run
collects raw HTML and prints `Parse skipped: BeautifulSoup not installed`.

## Prerequisites per channel

| Channel | Needs |
|---|---|
| **API** (`api_runner.py`, `runners/batch_runner.py`) | Provider keys in a `.env` at the repo root or above: `ANTHROPIC_KEY`, `OPENAI_KEY`, `GEMINI_API_KEY` — the standard SDK names (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`) are accepted as fallbacks. Runs headless — no browser. |
| **Interface** (`audit/audit_*.py`) | Google Chrome, plus a Chrome profile already signed in to the provider. Profiles live under `<vendor>_data/chrome_profiles*/` and are gitignored, so a fresh clone has none — sign in once with `--allow-manual-login` before unattended runs. |

Failed API calls are not silent: the per-query JSON records an `error` field
(e.g. an expired key or exhausted credit) alongside the prompt and timestamps.

## Responsible-Use Guidance

This software is intended to support authorized research on the behavior of deployed chatbot interfaces. Browser automation can impose costs on providers, implicate platform rules, and expose account or response data. Users are responsible for ensuring that their use complies with applicable laws, institutional requirements, and platform terms and policies.

### Authorization and access

* Use the software only with accounts and systems that you are authorized to access.
* Review the relevant platform terms and policies before beginning collection.
* Do not bypass CAPTCHAs, automation blocks, authentication requirements, paywalls, usage caps, rate limits, or other technical access controls.
* Stop the affected session if the platform displays an automation warning or otherwise indicates that automated access is prohibited.
* Seek guidance from your institution or legal counsel when the permissibility of a proposed audit is unclear.

### Rate limiting and provider burden

Configure conservative throughput limits and monitor collection continuously. Our study capped throughput at 150 queries per three-hour window and imposed a two-hour cooldown after any rate-limit event. These parameters document our protocol; they do not guarantee that the same limits are appropriate or permitted on another platform.

The audit scripts enforce these limits by default (`audit/rate_limit.py`): each query blocks until the rolling window has room, and a detected rate-limit event pauses collection for the cooldown period across every session of the run. Adjust with `--max-queries-per-window` (default 150), `--rate-limit-window` (default 10800s), and `--rate-limit-cooldown` (default 7200s). Lowering the cap is always safe; `--max-queries-per-window 0` disables it entirely and should be used only where you have confirmed that higher throughput is permitted.

Users should:

* adopt the lowest throughput sufficient for the research objective;
* avoid unnecessary retries and duplicate requests;
* pause collection after rate-limit or usage-cap notices;
* schedule collection to avoid concentrated bursts of traffic; and
* terminate collection if it appears to impair the service or impose an unexpected burden.

Model refusals should generally be retained as responses rather than repeatedly resubmitted. Retries for technical failures should be bounded and logged.

### Data minimization and privacy

Collect only the information necessary for the stated research purpose. Do not use this software to collect private conversations, personal information about other users, authentication credentials, sensitive data, or proprietary material unrelated to the audit.

Rendered pages, screenshots, HTML captures, logs, cookies, and browser profiles may contain account identifiers or other sensitive information. Store these materials securely, restrict access to authorized researchers, redact identifying information before sharing, and delete them when they are no longer required. Raw captures should not be released publicly unless they have been reviewed for sensitive or proprietary content.

### Credentials and account security

Do not commit passwords, session cookies, API keys, browser profiles, or authentication tokens to a repository. Use separate research accounts where permitted, apply least-privilege access, and protect credentials using appropriate secret-management practices. Authentication should not be transferred between users or accounts unless explicitly authorized.

### Transparency and reproducibility

Document the platform, model selection, account type, collection dates, browser and software versions, rate limits, retry behavior, and any interface changes observed during collection. Maintain logs sufficient to identify failures and verify adherence to the collection protocol. When publishing results, disclose relevant access conditions and distinguish measurements obtained through an interface from those obtained through an API.

### Prohibited uses

Do not use this software to:

* evade platform safeguards or conceal automated activity;
* gain unauthorized access to accounts, systems, or data;
* collect sensitive, personal, or proprietary information without authorization;
* conduct surveillance, harassment, manipulation, or other harmful activity;
* disrupt platform availability or impose excessive computational load; or
* misrepresent automated outputs as human activity.

This guidance does not itself grant permission to automate any platform and is not a substitute for legal, ethical, or institutional review. Researchers remain responsible for evaluating whether a particular deployment and collection protocol is appropriate.
