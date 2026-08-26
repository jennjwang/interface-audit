# automated-scraper

Collects model responses to benchmark items through two channels — the **API**
and the **web interface** (browser automation) — then scores and aggregates them
into the tables under `../experiments/` and `../paper/`.

## Layout

| Path | Contents |
|---|---|
| `api_runner.py` | Shared API client. Imported as a library by `audit/*` and `runners/batch_runner.py`, so it stays at the root. |
| `audit/` | Browser-automation scrapers, one per vendor. `audit_<vendor>.py` is layer 1; `audit_<vendor>_layer2.py` adds the account/session/request dimensions. |
| `runners/` | `batch_runner.py` (Batch API submission) and `generate_sweep_yaml.py` (expands a sweep spec into `yamls/` configs). |
| `scoring/` | `score_*` answer extraction and grading, `judge_*` LLM-judged variants, `rescore_*` / `apply_*` backfills over already-collected runs. |
| `compile/` | Aggregate raw runs into per-experiment tables; `move_*` relocate collected runs into `../experiments/`. |
| `analysis/` | Cross-condition analyses: sysprompt gap, item correlation, layer 2 phase1/session. |
| `viz/` | Build the standalone HTML inspection visualizers. |
| `scripts/` | Shell drivers for the layer 2 login + smoke flows, plus small parsing helpers. |
| `yamls/` | Run configs (~130) for the sweep, batch, seq, layer 2, and ablation experiments. |
| `outputs/` | Figure generators (`.py`, committed) and their rendered output (gitignored). |

Run data lives in `chatgpt_data/`, `claude_data/`, `gemini_data/`, and `data/`.
All four are gitignored — they hold ~14 GB of `raw_html/`, `parsed_json/`,
`sync/`, `api/`, and `logs/`.

## Path convention

Every script anchors its paths to its own location, **not** the working
directory, so scripts can be invoked from anywhere:

```python
BASE = Path(__file__).resolve().parents[1]   # automated-scraper/
REPO = Path(__file__).resolve().parents[2]   # repo root
```

If you move a script between directories, the `parents[N]` index must change to
match its new depth. Scripts in `audit/` additionally prepend the
`automated-scraper` root to `sys.path` before `from api_runner import ...`.

## Running

Shell drivers `cd` to this directory themselves, so invoke them from anywhere:

```bash
scripts/login_layer2_claude.sh                      # login + parallel smoke
scripts/login_layer2_claude.sh --resume             # skip completed queries
python audit/audit_claude_layer2.py --configs yamls/<config>.yaml
python runners/batch_runner.py --configs yamls/<config>.yaml
```

Browser automation needs `DrissionPage`, which is **not** in `.venv_api` — that
venv only carries the API-side dependencies. The analysis and plotting scripts
need `pandas`/`scipy`, available on `/opt/homebrew/bin/python3.11`.

Note that most `compile_*` and `score_*` scripts ignore argv and run
immediately, overwriting their output tables — there is no `--help` or
dry-run flag.
