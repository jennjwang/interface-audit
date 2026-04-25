"""API query runner utilities and CLI experiment runner.

This module is intentionally free of browser/DrissionPage dependencies so it can
be used standalone to generate API answers for a query CSV or for a YAML+CSV
experiment config (the former ``run_api.py`` CLI is now folded in here).

Supports:
  - OpenAI-compatible chat models via the OpenAI SDK
  - Anthropic Claude models (detected by a ``claude-`` prefix)
  - Google Gemini models (detected by a ``gemini-`` prefix)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _is_claude_model(model_name: str) -> bool:
    return (model_name or "").lower().startswith("claude-")


def _is_gemini_model(model_name: str) -> bool:
    return (model_name or "").lower().startswith("gemini-")


def _safe_id(query_id: str) -> str:
    return "".join([c for c in (query_id or "") if c.isalnum() or c in ("-", "_")]).strip()


def save_api_response(output_dir: str | Path, query_id: str, record: dict[str, Any]) -> Path:
    """Save API response data for a single query."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = _safe_id(query_id) or "unknown"
    filename = out_dir / f"{safe_id}.api.json"
    filename.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f">> Saved API response to {filename}")
    return filename


def run_api_query(
    query_id: str,
    prompt: str,
    output_dir: str | Path,
    model_name: str,
    sent_at: str | None = None,
    api_key: str | None = None,
    **api_params,
) -> dict[str, Any]:
    """Run a single API query and save the response.

    Any extra keyword arguments (temperature, top_p, max_tokens, seed, etc.)
    are forwarded directly to ``client.chat.completions.create()``.
    """
    started_at = datetime.now().isoformat()
    print(f">> Starting API query for {query_id} using {model_name}...")
    record: dict[str, Any] = {
        "query_id": query_id,
        "model": model_name,
        "prompt": prompt,
        "started_at": started_at,
    }
    # Save a copy before Gemini/Claude branches mutate api_params via .pop()
    if api_params:
        record["api_params"] = dict(api_params)
    if sent_at:
        record["sent_at"] = sent_at

    web_search: bool = bool(api_params.pop("web_search", False))

    if _is_gemini_model(model_name):
        # ── Google Gemini path ───────────────────────────────────────────────
        gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            record["error"] = "GEMINI_API_KEY/GOOGLE_API_KEY is not set."
            record["completed_at"] = datetime.now().isoformat()
            save_api_response(output_dir, query_id, record)
            return record
        try:
            from google import genai
            from google.genai import types as genai_types

            client = genai.Client(api_key=gemini_key)

            temperature = api_params.pop("temperature", None)
            max_tokens = api_params.pop("max_tokens", None)
            top_p = api_params.pop("top_p", None)
            thinking_level = api_params.pop("thinkingLevel", api_params.pop("thinking_level", None))

            config_kwargs: dict[str, Any] = {}
            if temperature is not None:
                config_kwargs["temperature"] = float(temperature)
            if max_tokens is not None:
                config_kwargs["max_output_tokens"] = int(max_tokens)
            if top_p is not None:
                config_kwargs["top_p"] = float(top_p)
            if thinking_level is not None:
                config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_level=str(thinking_level)
                )
            if web_search:
                config_kwargs["tools"] = [genai_types.Tool(
                    google_search=genai_types.GoogleSearch()
                )]
            config_kwargs.update(api_params)

            for attempt in range(5):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
                    )
                    break
                except Exception as retry_exc:
                    if "503" in str(retry_exc) or "UNAVAILABLE" in str(retry_exc):
                        wait = 10 * (2 ** attempt)
                        print(f">> Gemini 503, retrying in {wait}s (attempt {attempt+1}/5)...")
                        time.sleep(wait)
                        if attempt == 4:
                            raise
                    else:
                        raise

            content = response.text or ""
            record.update({
                "response_text": content,
                "finish_reason": str(response.candidates[0].finish_reason) if response.candidates else None,
            })
            if response.usage_metadata:
                record["usage"] = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                }
        except Exception as exc:
            record["error"] = str(exc)
    elif _is_claude_model(model_name):
        # ── Anthropic / Claude path ──────────────────────────────────────────
        anthropic_key = api_key or os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            record["error"] = "ANTHROPIC_KEY/ANTHROPIC_API_KEY is not set."
            record["completed_at"] = datetime.now().isoformat()
            save_api_response(output_dir, query_id, record)
            return record
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=anthropic_key)

            # Separate out extended-thinking params from standard ones
            create_kwargs: dict[str, Any] = {}
            thinking_budget = api_params.pop("budget_tokens", None)
            if thinking_budget is not None:
                create_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": int(thinking_budget),
                }
                # Extended thinking requires temperature=1
                api_params.pop("temperature", None)

            create_kwargs.update(api_params)

            if web_search:
                create_kwargs.setdefault("tools", [])
                create_kwargs["tools"].append({
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5,
                })
                message = client.beta.messages.create(
                    model=model_name,
                    max_tokens=create_kwargs.pop("max_tokens", 8096),
                    messages=[{"role": "user", "content": prompt}],
                    betas=["web-search-2025-03-05"],
                    **create_kwargs,
                )
            else:
                message = client.messages.create(
                    model=model_name,
                    max_tokens=create_kwargs.pop("max_tokens", 8096),
                    messages=[{"role": "user", "content": prompt}],
                    **create_kwargs,
                )

            # Concatenate all text blocks (thinking/tool blocks are skipped)
            content = "".join(
                block.text for block in message.content
                if getattr(block, "type", None) == "text"
            )
            record.update({
                "response_text": content,
                "response_id": message.id,
                "finish_reason": message.stop_reason,
            })
            if message.usage:
                record["usage"] = {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                }
        except Exception as exc:
            record["error"] = str(exc)
    else:
        # ── OpenAI path ──────────────────────────────────────────────────────
        key = api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            record["error"] = "OPENAI_KEY/OPENAI_API_KEY is not set."
            record["completed_at"] = datetime.now().isoformat()
            save_api_response(output_dir, query_id, record)
            return record
        try:
            from openai import OpenAI

            client = OpenAI(api_key=key)
            if web_search:
                # Responses API supports web_search_preview tool
                response = client.responses.create(
                    model=model_name,
                    input=prompt,
                    tools=[{"type": "web_search_preview"}],
                    **api_params,
                )
                content = response.output_text or ""
                record.update({
                    "response_text": content,
                    "response_id": getattr(response, "id", None),
                    "finish_reason": getattr(response, "status", None),
                })
                if getattr(response, "usage", None):
                    record["usage"] = response.usage.to_dict() if hasattr(response.usage, "to_dict") else vars(response.usage)
            else:
                # Reasoning models (o-series / gpt-5.4) require max_completion_tokens
                # instead of max_tokens when reasoning_effort is set.
                if "reasoning_effort" in api_params and "max_tokens" in api_params:
                    api_params["max_completion_tokens"] = api_params.pop("max_tokens")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **api_params,
                )
                content = response.choices[0].message.content or ""
                record.update({
                    "response_text": content,
                    "response_id": getattr(response, "id", None),
                    "finish_reason": response.choices[0].finish_reason,
                    "created": getattr(response, "created", None),
                })
                if getattr(response, "usage", None):
                    record["usage"] = response.usage.to_dict()
        except Exception as exc:
            record["error"] = str(exc)

    record["completed_at"] = datetime.now().isoformat()
    save_api_response(output_dir, query_id, record)
    return record


def _safe_model_dir(model: str) -> str:
    """Return a filesystem-safe directory name for a model."""
    name = (model or "").strip() or "unknown_model"
    name = name.replace(os.sep, "_").replace(" ", "_")
    return "".join([c for c in name if c.isalnum() or c in ("-", "_", ".", "@")]).strip("_") or "unknown_model"


def load_queries_from_file(query_file: str) -> list[tuple[str, str]]:
    """Load queries from a CSV file with 'id' and 'query' columns."""
    queries: list[tuple[str, str]] = []
    query_path = Path(query_file)
    if not query_path.is_absolute():
        query_path = BASE_DIR / query_path

    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    with query_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if "id" in row and "query" in row:
                queries.append((row["id"], row["query"]))
    return queries


def shuffle_queries(queries: list[tuple[str, str]], seed: int | None) -> list[tuple[str, str]]:
    """Optionally shuffle a list of (id, query) pairs."""
    if len(queries) <= 1:
        return queries
    out = list(queries)
    if seed is None:
        random.shuffle(out)
    else:
        rng = random.Random(seed)
        rng.shuffle(out)
    return out


def run_api_from_config(
    *,
    config_path: Path,
    api_model: str,
    run_id: str,
    session_id: int,
    out_root: Path,
    sleep_seconds: float,
    shuffle: bool,
    skip_existing: bool,
    api_key: str | None,
) -> None:
    """Run a full API-only experiment described by a YAML config."""
    with config_path.open("r", encoding="utf-8") as handle:
        full_config = yaml.safe_load(handle) or {}

    defaults = full_config.get("defaults", {}) or {}
    experiments = full_config.get("experiments", []) or []

    if not experiments:
        print(f"No experiments found in {config_path}")
        return

    model_dir = _safe_model_dir(api_model)
    session_dir = out_root / run_id / model_dir / f"session_{session_id:02d}"

    print(f">> API-only run_id: {run_id}")
    print(f">> API model: {api_model}")
    print(f">> Output root: {session_dir}")

    for exp in experiments:
        exp_name = exp.get("name", "experiment")
        query_file = exp.get("query_file", defaults.get("query_file"))
        if not query_file:
            print(f">> Skipping {exp_name}: missing query_file")
            continue

        runs = int(exp.get("runs", defaults.get("runs", 1)) or 1)

        seed_int = None
        if shuffle:
            seed = exp.get("seed", defaults.get("seed"))
            try:
                seed_int = int(seed) if seed is not None else None
            except Exception:
                seed_int = None

        base_queries = load_queries_from_file(str(query_file))
        queries: list[tuple[str, str]] = []
        for q_id, q_text in base_queries:
            for r in range(runs):
                queries.append((f"{q_id}_run{r}", q_text))

        if shuffle:
            queries = shuffle_queries(queries, seed_int)
            print(f">> {exp_name}: shuffled {len(queries)} queries (seed={seed_int})")
        else:
            print(f">> {exp_name}: {len(queries)} queries")

        exp_out_dir = session_dir / exp_name
        for i, (qid, qtext) in enumerate(queries, start=1):
            print(f"[{exp_name}] {i}/{len(queries)} {qid}")
            if skip_existing:
                safe_id = _safe_id(qid) or "unknown"
                out_file = exp_out_dir / f"{safe_id}.api.json"
                if out_file.exists():
                    print(f">> Skipping existing output: {out_file}")
                    continue
            sent_at = datetime.now().isoformat()
            run_api_query(
                qid,
                qtext,
                exp_out_dir,
                api_model,
                sent_at=sent_at,
                api_key=api_key,
            )
            if sleep_seconds > 0 and i < len(queries):
                time.sleep(sleep_seconds)


def _model_worker(kwargs: dict) -> None:
    run_api_from_config(**kwargs)


def main() -> None:
    """CLI entry point (was previously in run_api.py)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="exp.yaml", help="Path to audit YAML config")
    parser.add_argument(
        "--api-model",
        default="gpt-5.2-chat-latest",
        help="API model to query (default: gpt-5.2-chat-latest)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of API models to run in parallel (overrides --api-model).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id folder name (default: YYYY-MM-DD_HH-MM-SS)",
    )
    parser.add_argument(
        "--session-id",
        type=int,
        default=0,
        help="Session id used in output path (default: 0)",
    )
    parser.add_argument(
        "--out-root",
        default=str(DATA_DIR / "api"),
        help="Root directory for API outputs (default: automated-scraper/data/api)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between API queries in seconds",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle queries (default: disabled, even if config requests shuffling).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip queries whose output JSON already exists (resume a partial run).",
    )
    args = parser.parse_args()

    load_dotenv()
    # Do not preselect one global key here. Each provider branch in
    # run_api_query resolves the correct env var for its own model.
    api_key = None
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = BASE_DIR / out_root

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = [args.api_model]

    if len(models) == 1:
        run_api_from_config(
            config_path=config_path,
            api_model=models[0],
            run_id=run_id,
            session_id=int(args.session_id),
            out_root=out_root,
            sleep_seconds=float(args.sleep or 0.0),
            shuffle=bool(args.shuffle),
            skip_existing=bool(args.skip_existing),
            api_key=api_key,
        )
        return

    # Run models in parallel as separate processes to overlap latency.
    print(f">> Running {len(models)} models in parallel under run_id={run_id}")
    workers: list[mp.Process] = []
    for model in models:
        kwargs = {
            "config_path": config_path,
            "api_model": model,
            "run_id": run_id,
            "session_id": int(args.session_id),
            "out_root": out_root,
            "sleep_seconds": float(args.sleep or 0.0),
            "shuffle": bool(args.shuffle),
            "skip_existing": bool(args.skip_existing),
            "api_key": api_key,
        }
        p = mp.Process(target=_model_worker, args=(kwargs,))
        p.start()
        workers.append(p)

    exit_codes = []
    for p in workers:
        p.join()
        exit_codes.append(p.exitcode)

    if any(code not in (0, None) for code in exit_codes):
        raise SystemExit(f"One or more model workers failed: exit_codes={exit_codes}")


if __name__ == "__main__":
    main()
