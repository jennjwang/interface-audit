# browser_automation

Collects model responses to benchmark items through two channels — the **API**
and the **web interface** (browser automation) — then scores and aggregates them.

## Layout

| Path | Contents |
|---|---|
| `api_runner.py` | Shared API client. Imported by `audit/*` and `runners/batch_runner.py`. |
| `audit/` | Browser-automation scrapers, one per vendor. `audit_<vendor>.py` is layer 1; `audit_<vendor>_layer2.py` adds account/session/request dimensions. |
| `runners/` | `batch_runner.py` (Batch API submission) and `generate_sweep_yaml.py` (expands a sweep spec into `yamls/` configs). |
| `scoring/` | Unified `score.py` for answer extraction and grading (MC, BBQ, GSM8K, LLM-judged). |
| `compile/` | Aggregate raw runs into per-experiment tables. |
| `analysis/` | Cross-condition analyses: sysprompt gap, layer 2 phase1/session. |
| `viz/` | Standalone HTML inspection visualizers. |
| `scripts/` | Shell drivers for layer 2 login + smoke flows, plus parsing helpers. |
| `yamls/` | Run configs for sweep, batch, seq, layer 2, and ablation experiments. |
| `outputs/` | Human validation data and extractor vote audit trail. |

Run data lives in `chatgpt_data/`, `claude_data/`, `gemini_data/`, and `data/`.
All are gitignored.

## Running

```bash
python audit/audit_claude_layer2.py --configs yamls/<config>.yaml
python runners/batch_runner.py --configs yamls/<config>.yaml
python scoring/score.py --run-dir <path> --answer-key <csv> --task mc
```
