"""
End-to-end pipeline: run models on a benchmark, judge responses, select subset.

Steps:
  1. Run each configured model on all questions (sequential or --batch)
  2. Run judge.py to score responses (LLM judge against gold answers)
  3. Run select_subset.py to pick discriminative questions

Batching:
  Claude and OpenAI models support their native Batch APIs (50% cheaper, no
  rate limits). Gemini models fall back to sequential.  Batches are submitted
  in parallel then polled together until all complete.

Usage:
    cd benchmark_creation
    python run_pipeline.py --config configs/aa-omniscience.yaml
    python run_pipeline.py --config configs/aa-omniscience.yaml --batch
    python run_pipeline.py --config configs/aa-omniscience.yaml --skip-run --skip-judge
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT          = Path(__file__).resolve().parent.parent
SCRAPER_DIR        = REPO_ROOT / "automated-scraper"
BENCHMARK_CREATION = Path(__file__).resolve().parent


# ── helpers ───────────────────────────────────────────────────────────────────

def model_dir_name(model_cfg: dict) -> str:
    parts = [model_cfg["model"]]
    for k, v in sorted(model_cfg.items()):
        if k != "model":
            parts.append(f"{k}-{v}")
    name = "_".join(parts)
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def save_response(out_dir: Path, run_qid: str, record: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_qid}.api.json").write_text(
        json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8"
    )


# ── sequential inference ──────────────────────────────────────────────────────

def run_models_sequential(cfg: dict, queries: list, api_dir: Path, skip_existing: bool) -> None:
    sys.path.insert(0, str(SCRAPER_DIR))
    from api_runner import run_api_query

    for model_cfg in cfg["models"]:
        model_name   = model_cfg["model"]
        extra_params = {k: v for k, v in model_cfg.items() if k != "model"}
        # GPT-5+ models require max_completion_tokens instead of max_tokens
        name_lower = model_name.lower()
        if not name_lower.startswith("claude-") and not name_lower.startswith("gemini-"):
            if "max_tokens" in extra_params:
                extra_params["max_completion_tokens"] = extra_params.pop("max_tokens")
        out_dir      = api_dir / model_dir_name(model_cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {model_name} (sequential) -> {out_dir.name} ===")

        for i, (qid, qtext) in enumerate(queries, 1):
            run_qid  = f"{qid}_run0"
            out_file = out_dir / f"{run_qid}.api.json"
            if skip_existing and out_file.exists():
                print(f"  [{i}/{len(queries)}] skip {run_qid}")
                continue
            print(f"  [{i}/{len(queries)}] {run_qid}")
            run_api_query(run_qid, qtext, str(out_dir), model_name, **extra_params)


# ── batch inference ───────────────────────────────────────────────────────────

def _submit_claude_batch(model_cfg: dict, queries: list, out_dir: Path) -> tuple[str, Path]:
    """Submit an Anthropic message batch. Returns (batch_id, out_dir)."""
    import anthropic, os
    api_key = os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    model_name   = model_cfg["model"]
    extra_params = {k: v for k, v in model_cfg.items() if k != "model"}
    web_search   = extra_params.pop("web_search", False)
    max_tokens   = extra_params.pop("max_tokens", 8096)

    tools = []
    if web_search:
        tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 5})

    requests = []
    for qid, qtext in queries:
        params = {"model": model_name, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": qtext}]}
        if tools:
            params["tools"] = tools
        params.update(extra_params)
        requests.append({"custom_id": f"{qid}_run0", "params": params})

    batch = client.beta.messages.batches.create(requests=requests)
    print(f"  Submitted Claude batch {batch.id} ({len(requests)} requests)")
    return batch.id, out_dir


def _submit_openai_batch(model_cfg: dict, queries: list, out_dir: Path) -> tuple[str, Path]:
    """Submit an OpenAI batch. Returns (batch_id, out_dir)."""
    import openai, os
    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)

    model_name   = model_cfg["model"]
    extra_params = {k: v for k, v in model_cfg.items() if k != "model"}
    web_search   = extra_params.pop("web_search", False)
    max_tokens   = extra_params.pop("max_tokens", 8096)

    lines = []
    for qid, qtext in queries:
        body = {"model": model_name, "messages": [{"role": "user", "content": qtext}],
                "max_tokens": max_tokens}
        if web_search:
            body["tools"] = [{"type": "web_search_preview"}]
        body.update(extra_params)
        lines.append(json.dumps({
            "custom_id": f"{qid}_run0",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }))

    file_obj = client.files.create(
        file=("batch.jsonl", "\n".join(lines).encode()),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Submitted OpenAI batch {batch.id} ({len(lines)} requests)")
    return batch.id, out_dir


def _collect_claude_batch(batch_id: str, model_name: str, out_dir: Path) -> None:
    import anthropic, os
    api_key = os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    for result in client.beta.messages.batches.results(batch_id):
        run_qid = result.custom_id
        if result.result.type == "succeeded":
            text = "".join(b.text for b in result.result.message.content
                           if getattr(b, "type", None) == "text")
            record = {"query_id": run_qid, "model": model_name, "response_text": text}
        else:
            record = {"query_id": run_qid, "model": model_name,
                      "error": str(result.result)}
        save_response(out_dir, run_qid, record)
    print(f"  Saved Claude batch results to {out_dir.name}/")


def _collect_openai_batch(batch_id: str, model_name: str, out_dir: Path) -> None:
    import openai, os
    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed" or not batch.output_file_id:
        errors = getattr(batch.errors, "data", []) if batch.errors else []
        msg = errors[0].message if errors else batch.status
        raise RuntimeError(f"OpenAI batch {batch_id} did not complete: {msg}")
    content = client.files.content(batch.output_file_id).text
    for line in content.splitlines():
        if not line.strip():
            continue
        result  = json.loads(line)
        run_qid = result["custom_id"]
        try:
            text = result["response"]["body"]["choices"][0]["message"]["content"] or ""
            record = {"query_id": run_qid, "model": model_name, "response_text": text}
        except Exception as exc:
            record = {"query_id": run_qid, "model": model_name, "error": str(exc)}
        save_response(out_dir, run_qid, record)
    print(f"  Saved OpenAI batch results to {out_dir.name}/")


def run_models_batch(cfg: dict, queries: list, api_dir: Path, skip_existing: bool) -> None:
    sys.path.insert(0, str(SCRAPER_DIR))
    from api_runner import run_api_query
    from dotenv import load_dotenv
    load_dotenv(SCRAPER_DIR / ".env")

    # Anthropic Batch API: all claude- models.
    # OpenAI Batch API: only gpt-4 era models (gpt-5+ use Responses API, not supported).
    # Everything else: sequential.
    OPENAI_BATCH_MODELS = ("gpt-4", "gpt-3.5")

    claude_jobs, openai_jobs, sequential_jobs = [], [], []
    for model_cfg in cfg["models"]:
        name = model_cfg["model"].lower()
        out_dir = api_dir / model_dir_name(model_cfg)
        if name.startswith("claude-"):
            claude_jobs.append((model_cfg, out_dir))
        elif name.startswith("gemini-"):
            sequential_jobs.append((model_cfg, out_dir))
        elif any(name.startswith(m) for m in OPENAI_BATCH_MODELS):
            openai_jobs.append((model_cfg, out_dir))
        else:
            sequential_jobs.append((model_cfg, out_dir))

    # Submit batches
    pending: list[tuple[str, str, str, Path]] = []  # (provider, batch_id, model_name, out_dir)

    for model_cfg, out_dir in claude_jobs:
        batch_id, out_dir = _submit_claude_batch(model_cfg, queries, out_dir)
        pending.append(("claude", batch_id, model_cfg["model"], out_dir))

    for model_cfg, out_dir in openai_jobs:
        batch_id, out_dir = _submit_openai_batch(model_cfg, queries, out_dir)
        pending.append(("openai", batch_id, model_cfg["model"], out_dir))

    # Sequential fallback for Gemini and unsupported batch models
    for model_cfg, out_dir in sequential_jobs:
        model_name   = model_cfg["model"]
        extra_params = {k: v for k, v in model_cfg.items() if k != "model"}
        name_lower   = model_name.lower()
        if not name_lower.startswith("claude-") and not name_lower.startswith("gemini-"):
            if "max_tokens" in extra_params:
                extra_params["max_completion_tokens"] = extra_params.pop("max_tokens")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {model_name} (sequential fallback) ===")
        for i, (qid, qtext) in enumerate(queries, 1):
            run_qid  = f"{qid}_run0"
            out_file = out_dir / f"{run_qid}.api.json"
            if skip_existing and out_file.exists():
                continue
            print(f"  [{i}/{len(queries)}] {run_qid}")
            run_api_query(run_qid, qtext, str(out_dir), model_name, **extra_params)

    # Poll until all batches complete
    if pending:
        import anthropic, openai, os
        claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY"))

        print(f"\nPolling {len(pending)} batch(es)...")
        remaining = list(pending)
        while remaining:
            still_waiting = []
            for provider, batch_id, model_name, out_dir in remaining:
                if provider == "claude":
                    status = claude_client.beta.messages.batches.retrieve(batch_id).processing_status
                    done   = status == "ended"
                else:
                    status = openai_client.batches.retrieve(batch_id).status
                    done   = status in ("completed", "failed", "expired", "cancelled")
                print(f"  {model_name}: {status}")
                if done:
                    if provider == "claude":
                        _collect_claude_batch(batch_id, model_name, out_dir)
                    else:
                        _collect_openai_batch(batch_id, model_name, out_dir)
                else:
                    still_waiting.append((provider, batch_id, model_name, out_dir))
            remaining = still_waiting
            if remaining:
                print(f"  {len(remaining)} batch(es) still running — sleeping 60s...")
                time.sleep(60)
        print("All batches complete.")


# ── judge + select ────────────────────────────────────────────────────────────

def run_judge(cfg: dict, run_id: str) -> None:
    experiment_dir = Path(cfg["experiment_dir"])
    judge_script   = BENCHMARK_CREATION / "judge.py"
    scored_dir     = experiment_dir / "outputs" / run_id

    cmd = [
        sys.executable, str(judge_script),
        "--api-dir",    str(experiment_dir / "api"),
        "--queries",    cfg["queries"],
        "--output-dir", str(scored_dir),
        "--run-id",     run_id,
    ]
    if cfg.get("judge_mode") == "regex":
        cmd.append("--regex-judge")
    print(f"\n=== Judge ===")
    subprocess.run(cmd, check=True)


def run_select_subset(cfg: dict, run_id: str) -> None:
    experiment_dir = Path(cfg["experiment_dir"])
    scored_dir     = experiment_dir / "outputs" / run_id
    output_path    = Path(cfg["output"])
    select_script  = BENCHMARK_CREATION / "select_subset.py"

    cmd = [sys.executable, str(select_script),
           "--scored-dir", str(scored_dir),
           "--queries",    cfg["queries"],
           "--output",     str(output_path)]
    for flag in ("min_rpbis", "min_std", "max_acc", "rank_sim_threshold"):
        if flag in cfg:
            cmd += [f"--{flag.replace('_', '-')}", str(cfg[flag])]

    print(f"\n=== Selecting subset ===")
    subprocess.run(cmd, check=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",        required=True)
    parser.add_argument("--run-id",        default=None)
    parser.add_argument("--batch",         action="store_true",
                        help="Use Batch APIs for Claude/OpenAI (Gemini stays sequential)")
    parser.add_argument("--sample",        type=int, default=None,
                        help="Only run the first N queries (for testing)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-run",      action="store_true")
    parser.add_argument("--skip-judge",    action="store_true")
    parser.add_argument("--skip-subset",   action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BENCHMARK_CREATION / config_path
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    for key in ("experiment_dir", "queries", "output"):
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str(REPO_ROOT / p)

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Run ID: {run_id}  |  batch={'yes' if args.batch else 'no'}")

    if not args.skip_run:
        sys.path.insert(0, str(SCRAPER_DIR))
        from api_runner import load_queries_from_file
        from dotenv import load_dotenv
        load_dotenv(SCRAPER_DIR / ".env")

        queries     = load_queries_from_file(cfg["queries"])
        if args.sample:
            queries = queries[:args.sample]
        api_dir     = Path(cfg["experiment_dir"]) / "api" / run_id
        print(f"Loaded {len(queries)} queries")

        if args.batch:
            run_models_batch(cfg, queries, api_dir, args.skip_existing)
        else:
            run_models_sequential(cfg, queries, api_dir, args.skip_existing)

    if not args.skip_judge:
        run_judge(cfg, run_id)

    if not args.skip_subset:
        run_select_subset(cfg, run_id)


if __name__ == "__main__":
    main()
