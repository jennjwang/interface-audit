"""Generate a one-at-a-time sampling sweep YAML for ``api_runner.py --config-api-models``.

Each axis is varied independently; every other param is left at the provider
default. So ``--temperature 0,0.5,1.0 --top-p 0.9,1.0`` produces 5 entries
(3 temperature points + 2 top_p points), not 6. This is the right shape for
the paper claim "we sweep across plausible interface settings" — it isolates
the contribution of each knob rather than confounding them.

Each entry becomes its own ``api_models`` row. ``api_runner.py`` already maps
every entry to its own output directory (the param values are embedded in the
directory name via ``_api_model_dir_name``), so the resulting sweep produces
one folder per axis × value.

Example:

    python automated-scraper/generate_sweep_yaml.py \
        --model claude-sonnet-4-6 \
        --query-file ../benchmark_creation/results/bbq-subset-200.csv \
        --exp-name bbq \
        --out yamls/sweep_claude_sonnet_bbq.yaml \
        --temperature 0.0,0.5,0.7,1.0 \
        --top-p 0.9,0.95,1.0 \
        --max-tokens 1024 \
        --runs 1

Reasoning / extended-thinking axis (a separate axis — never crossed with the
sampling axes; on Claude, ``budget_tokens`` forces temperature=1 anyway):

    --budget-tokens 1024,4096            # claude extended thinking
    --reasoning-effort low,medium,high   # openai reasoning models
    --thinking-level low,medium,high     # gemini
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


CLAUDE_TEMP_MAX = 1.0
OPENAI_TEMP_MAX = 2.0
GEMINI_TEMP_MAX = 2.0


def _provider(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("claude-"):
        return "claude"
    if m.startswith("gemini-"):
        return "gemini"
    return "openai"


def _parse_floats(s: str | None) -> list[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_ints(s: str | None) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_strs(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _validate(provider: str, temps: list[float], top_ps: list[float],
              reasoning_effort: list[str], thinking_level: list[str],
              budget_tokens: list[int]) -> None:
    if provider == "claude":
        for t in temps:
            if not 0.0 <= t <= CLAUDE_TEMP_MAX:
                raise ValueError(f"Claude temperature must be in [0, 1.0]; got {t}")
        if reasoning_effort:
            raise ValueError("reasoning_effort is OpenAI-only; use --budget-tokens for Claude")
        if thinking_level:
            raise ValueError("thinking_level is Gemini-only; use --budget-tokens for Claude")
    elif provider == "openai":
        for t in temps:
            if not 0.0 <= t <= OPENAI_TEMP_MAX:
                raise ValueError(f"OpenAI temperature must be in [0, 2.0]; got {t}")
        if budget_tokens:
            raise ValueError("budget_tokens is Claude-only; use --reasoning-effort for OpenAI")
        if thinking_level:
            raise ValueError("thinking_level is Gemini-only; use --reasoning-effort for OpenAI")
    elif provider == "gemini":
        for t in temps:
            if not 0.0 <= t <= GEMINI_TEMP_MAX:
                raise ValueError(f"Gemini temperature must be in [0, 2.0]; got {t}")
        if reasoning_effort:
            raise ValueError("reasoning_effort is OpenAI-only; use --thinking-level for Gemini")
        if budget_tokens:
            raise ValueError("budget_tokens is Claude-only; use --thinking-level for Gemini")
    for p in top_ps:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1.0]; got {p}")


def build_grid(
    model: str,
    *,
    temperatures: list[float],
    top_ps: list[float],
    max_tokens: int | None,
    reasoning_effort: list[str],
    thinking_level: list[str],
    budget_tokens: list[int],
    thinking_max_tokens: int | None,
) -> list[dict[str, Any]]:
    """Return a list of api_models entries for the cross-product."""
    provider = _provider(model)
    _validate(provider, temperatures, top_ps, reasoning_effort, thinking_level, budget_tokens)

    entries: list[dict[str, Any]] = []

    def _base() -> dict[str, Any]:
        entry: dict[str, Any] = {"model": model}
        if max_tokens is not None:
            entry["max_tokens"] = int(max_tokens)
        return entry

    for t in temperatures:
        entry = _base()
        entry["temperature"] = float(t)
        entries.append(entry)

    for p_val in top_ps:
        entry = _base()
        entry["top_p"] = float(p_val)
        entries.append(entry)

    # Reasoning / extended-thinking axis: emit as a separate grid so we don't
    # cross with temperature on Claude (which the SDK rejects anyway).
    reasoning_mt = thinking_max_tokens if thinking_max_tokens is not None else max_tokens

    for b in budget_tokens:
        entry = {"model": model, "budget_tokens": int(b)}
        if reasoning_mt is not None:
            entry["max_tokens"] = int(reasoning_mt)
        entries.append(entry)

    for r in reasoning_effort:
        entry = {"model": model, "reasoning_effort": r}
        if reasoning_mt is not None:
            entry["max_tokens"] = int(reasoning_mt)
        entries.append(entry)

    for level in thinking_level:
        entry = {"model": model, "thinking_level": level}
        if reasoning_mt is not None:
            entry["max_tokens"] = int(reasoning_mt)
        entries.append(entry)

    if not entries:
        raise ValueError("Empty grid: pass at least one axis or --max-tokens")
    return entries


def render_yaml(
    *,
    api_models: list[dict[str, Any]],
    query_file: str,
    exp_name: str,
    runs: int,
    shuffle: bool,
    seed: int | None,
) -> str:
    """Render a yaml string that matches the api_runner --config-api-models schema."""
    defaults: dict[str, Any] = {"runs": int(runs), "shuffle": bool(shuffle)}
    if seed is not None:
        defaults["seed"] = int(seed)

    config: dict[str, Any] = {
        "defaults": defaults,
        "api_models": api_models,
        "experiments": [{"name": exp_name, "query_file": query_file}],
    }
    # default_flow_style=None gives flow style for short dicts (compact grid rows)
    # while keeping the top-level keys in block style.
    return yaml.dump(config, sort_keys=False, default_flow_style=None, width=120)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="API model name (e.g., claude-sonnet-4-6)")
    p.add_argument("--out", required=True, help="Output yaml path")
    p.add_argument("--query-file", required=True, help="Benchmark CSV path (relative to automated-scraper/)")
    p.add_argument("--exp-name", default=None, help="Experiment name (default: query file stem)")
    p.add_argument("--temperature", default=None, help="Comma-separated floats")
    p.add_argument("--top-p", default=None, help="Comma-separated floats in (0, 1.0]")
    p.add_argument("--max-tokens", type=int, default=None, help="Max output tokens (fixed)")
    p.add_argument("--reasoning-effort", default=None,
                   help="OpenAI only: comma-separated, e.g., minimal,low,medium,high")
    p.add_argument("--thinking-level", default=None,
                   help="Gemini only: comma-separated, e.g., low,medium,high")
    p.add_argument("--budget-tokens", default=None,
                   help="Claude extended thinking: comma-separated ints (forces temp=1)")
    p.add_argument("--thinking-max-tokens", type=int, default=None,
                   help="Override max_tokens for reasoning entries (default: same as --max-tokens)")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    exp_name = args.exp_name or Path(args.query_file).stem

    entries = build_grid(
        args.model,
        temperatures=_parse_floats(args.temperature),
        top_ps=_parse_floats(args.top_p),
        max_tokens=args.max_tokens,
        reasoning_effort=_parse_strs(args.reasoning_effort),
        thinking_level=_parse_strs(args.thinking_level),
        budget_tokens=_parse_ints(args.budget_tokens),
        thinking_max_tokens=args.thinking_max_tokens,
    )

    yaml_text = render_yaml(
        api_models=entries,
        query_file=args.query_file,
        exp_name=exp_name,
        runs=args.runs,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_text, encoding="utf-8")

    print(f">> Wrote {len(entries)} grid entries to {out_path}")
    pkg_root = Path(__file__).resolve().parent
    try:
        cfg_arg = out_path.relative_to(pkg_root)
    except ValueError:
        cfg_arg = out_path
    print(f">> Run with: python automated-scraper/api_runner.py {cfg_arg} \\")
    print(f"       --config-api-models --run-id sweep-{exp_name}")


if __name__ == "__main__":
    main()
