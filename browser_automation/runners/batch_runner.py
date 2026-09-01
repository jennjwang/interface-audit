"""Batch API runner for OpenAI / Anthropic (and sequential fallback for Gemini).

Same yaml format as api_runner.py. Submits one batch per (api_model, experiment),
polls until all complete, then writes per-query *.api.json files into the same
output layout api_runner uses, so existing scoring scripts work unchanged.

Run:
  python browser_automation/batch_runner.py yamls/<file>.yaml \
      --config-api-models --run-id <run_id>

Resume (after submission, before completion):
  python browser_automation/batch_runner.py yamls/<file>.yaml \
      --config-api-models --run-id <run_id> --resume
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_runner import (
    _api_model_dir_name,
    _api_model_and_params,
    _pop_system_prompt,
    _safe_id,
    load_queries_from_file,
    run_api_query,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def _is_claude(name: str) -> bool: return name.lower().startswith("claude-")
def _is_gemini(name: str) -> bool: return name.lower().startswith("gemini-")


# ─── submit ──────────────────────────────────────────────────────────────────

def submit_openai_batch(model_name: str, params: dict, queries: list, system_prompt: str | None) -> str:
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY"))
    lines = []
    for qid, qtext in queries:
        body: dict[str, Any] = {"model": model_name}
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": qtext})
        body["messages"] = msgs
        p = dict(params)
        # gpt-5.x rejects max_tokens (chat-latest and reasoning); rename.
        needs_rename = (
            "reasoning_effort" in p
            or (model_name or "").lower().startswith("gpt-5.")
        )
        if needs_rename and "max_tokens" in p:
            p["max_completion_tokens"] = p.pop("max_tokens")
        body.update(p)
        lines.append(json.dumps({
            "custom_id": qid,
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
    print(f"  Submitted OpenAI batch {batch.id} ({len(lines)} requests, model={model_name})")
    return batch.id


def submit_claude_batch(model_name: str, params: dict, queries: list, system_prompt: str | None) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    p = dict(params)
    p.pop("web_search", None)  # batch API doesn't support web_search tool here
    max_tokens = int(p.pop("max_tokens", 8096))
    thinking_budget = p.pop("budget_tokens", None)
    effort = p.pop("effort", None)  # Opus 4.7 uses adaptive thinking + output_config.effort
    requests = []
    for qid, qtext in queries:
        body: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": qtext}],
        }
        if system_prompt:
            body["system"] = system_prompt
        if thinking_budget is not None:
            body["thinking"] = {"type": "enabled", "budget_tokens": int(thinking_budget)}
            p.pop("temperature", None)
        elif effort is not None:
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": str(effort)}
            p.pop("temperature", None)
        body.update(p)
        requests.append({"custom_id": qid, "params": body})
    batch = client.messages.batches.create(requests=requests)
    print(f"  Submitted Anthropic batch {batch.id} ({len(requests)} requests, model={model_name})")
    return batch.id


def submit_gemini_batch(model_name: str, params: dict, queries: list, system_prompt: str | None) -> tuple[str, list[str]]:
    """Submit a Gemini batch. Returns (batch_name, custom_ids_in_order)."""
    from google import genai
    from google.genai import types as gtypes
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    p = dict(params)
    thinking_level = p.pop("thinking_level", None) or p.pop("thinkingLevel", None)
    temperature = p.pop("temperature", None)
    max_tokens = p.pop("max_tokens", None)
    top_p = p.pop("top_p", None)

    def make_config():
        cfg_kwargs: dict[str, Any] = {}
        if system_prompt: cfg_kwargs["system_instruction"] = system_prompt
        if temperature is not None: cfg_kwargs["temperature"] = float(temperature)
        if max_tokens is not None: cfg_kwargs["max_output_tokens"] = int(max_tokens)
        if top_p is not None: cfg_kwargs["top_p"] = float(top_p)
        if thinking_level is not None:
            cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_level=str(thinking_level))
        return gtypes.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None

    cfg = make_config()
    # Embed custom_id in metadata: Gemini batch does NOT preserve request order
    # for large batches, so we tag each request and match in collect_gemini_batch.
    requests = [
        gtypes.InlinedRequest(contents=qtext, config=cfg, metadata={"custom_id": qid})
        for (qid, qtext) in queries
    ]
    custom_ids = [qid for (qid, _) in queries]
    job = client.batches.create(model=model_name, src=requests)
    print(f"  Submitted Gemini batch {job.name} ({len(requests)} requests, model={model_name})")
    return job.name, custom_ids


def collect_gemini_batch(batch_id: str, model_name: str, system_prompt_chars: int | None,
                         out_dir: Path, prompts_by_qid: dict[str, str],
                         custom_ids: list[str]) -> tuple[int, int]:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    job = client.batches.get(name=batch_id)
    state = str(job.state).split(".")[-1]
    if not state.endswith("SUCCEEDED"):
        raise RuntimeError(f"Gemini batch {batch_id} state={state} (error={job.error})")
    ok = err = 0
    completed_at = datetime.now().isoformat()
    responses = (job.dest.inlined_responses if job.dest else None) or []
    if len(responses) != len(custom_ids):
        print(f"  WARN: response count ({len(responses)}) != submitted ({len(custom_ids)})")
    # Match by metadata.custom_id (Gemini batch doesn't preserve request order)
    by_id = {}
    unmatched = 0
    for ir in responses:
        md = getattr(ir, "metadata", None) or {}
        cid = md.get("custom_id") if isinstance(md, dict) else None
        if cid:
            by_id[cid] = ir
        else:
            unmatched += 1
    if unmatched:
        print(f"  WARN: {unmatched} responses had no custom_id metadata; falling back to order")
        # Fallback for old batches submitted without metadata: positional
        ordered = list(zip(custom_ids, responses))
    else:
        ordered = [(cid, by_id.get(cid)) for cid in custom_ids]
    for qid, ir in ordered:
        rec: dict[str, Any] = {
            "query_id": qid,
            "model": model_name,
            "prompt": prompts_by_qid.get(__import__('re').sub(r"_run\d+$", "", qid), ""),
            "completed_at": completed_at,
        }
        if system_prompt_chars is not None:
            rec["system_prompt_chars"] = system_prompt_chars
        if ir is None:
            rec["error"] = "missing response (no matching metadata)"
            err += 1
            _save_record(out_dir, qid, rec)
            continue
        if getattr(ir, "error", None):
            rec["error"] = str(ir.error)
            err += 1
        else:
            try:
                resp = ir.response
                text = ""
                if resp.candidates and resp.candidates[0].content:
                    parts = resp.candidates[0].content.parts or []
                    text = "".join(getattr(pt, "text", "") or "" for pt in parts)
                rec["response_text"] = text
                rec["response_id"] = getattr(resp, "response_id", None)
                if resp.candidates:
                    rec["finish_reason"] = str(resp.candidates[0].finish_reason)
                if resp.usage_metadata:
                    rec["usage"] = {
                        "input_tokens": resp.usage_metadata.prompt_token_count,
                        "output_tokens": resp.usage_metadata.candidates_token_count,
                    }
                ok += 1
            except Exception as exc:
                rec["error"] = str(exc)
                err += 1
        _save_record(out_dir, qid, rec)
    return ok, err


# ─── collect ─────────────────────────────────────────────────────────────────

def _save_record(out_dir: Path, qid: str, record: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_id(qid) or "unknown"
    (out_dir / f"{safe}.api.json").write_text(
        json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8"
    )


def collect_openai_batch(batch_id: str, model_name: str, system_prompt_chars: int | None,
                         out_dir: Path, prompts_by_qid: dict[str, str]) -> tuple[int, int]:
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY"))
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed" or not batch.output_file_id:
        errs = getattr(batch.errors, "data", []) if batch.errors else []
        msg = errs[0].message if errs else batch.status
        raise RuntimeError(f"OpenAI batch {batch_id} status={batch.status}: {msg}")
    content = client.files.content(batch.output_file_id).text
    ok = err = 0
    completed_at = datetime.now().isoformat()
    for line in content.splitlines():
        if not line.strip(): continue
        result = json.loads(line)
        qid = result["custom_id"]
        rec: dict[str, Any] = {
            "query_id": qid,
            "model": model_name,
            "prompt": prompts_by_qid.get(__import__('re').sub(r"_run\d+$", "", qid), ""),
            "completed_at": completed_at,
        }
        if system_prompt_chars is not None:
            rec["system_prompt_chars"] = system_prompt_chars
        try:
            body = result["response"]["body"]
            choice = body["choices"][0]
            rec["response_text"] = choice["message"]["content"] or ""
            rec["response_id"]  = body.get("id")
            rec["finish_reason"] = choice.get("finish_reason")
            if body.get("usage"):
                rec["usage"] = body["usage"]
            ok += 1
        except Exception as exc:
            rec["error"] = str(exc)
            rec["raw"] = result.get("response") or result.get("error")
            err += 1
        _save_record(out_dir, qid, rec)
    return ok, err


def collect_claude_batch(batch_id: str, model_name: str, system_prompt_chars: int | None,
                          out_dir: Path, prompts_by_qid: dict[str, str]) -> tuple[int, int]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    ok = err = 0
    completed_at = datetime.now().isoformat()
    for result in client.messages.batches.results(batch_id):
        qid = result.custom_id
        rec: dict[str, Any] = {
            "query_id": qid,
            "model": model_name,
            "prompt": prompts_by_qid.get(__import__('re').sub(r"_run\d+$", "", qid), ""),
            "completed_at": completed_at,
        }
        if system_prompt_chars is not None:
            rec["system_prompt_chars"] = system_prompt_chars
        if result.result.type == "succeeded":
            msg = result.result.message
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            rec["response_text"] = text
            rec["response_id"] = msg.id
            rec["finish_reason"] = msg.stop_reason
            if msg.usage:
                rec["usage"] = {
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                }
            ok += 1
        else:
            rec["error"] = str(result.result)
            err += 1
        _save_record(out_dir, qid, rec)
    return ok, err


# ─── orchestration ───────────────────────────────────────────────────────────

def build_jobs(config_path: Path, run_id: str, out_root: Path) -> list[dict]:
    full = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    api_models = full.get("api_models", []) or []
    experiments = full.get("experiments", []) or []
    defaults = full.get("defaults", {}) or {}
    if not api_models or not experiments:
        raise SystemExit("config must have api_models and experiments")

    default_runs = int(defaults.get("runs", 1) or 1)
    default_run_offset = int(defaults.get("run_offset", 0) or 0)
    jobs: list[dict] = []
    for api_cfg in api_models:
        model_name, raw_params = _api_model_and_params(api_cfg)
        model_dir = _api_model_dir_name(api_cfg)
        provider = ("claude" if _is_claude(model_name)
                    else "gemini" if _is_gemini(model_name)
                    else "openai")
        # Resolve system prompt once per model_cfg
        params = dict(raw_params)
        system_prompt = _pop_system_prompt(params)
        sys_chars = len(system_prompt) if system_prompt else None
        # web_search isn't supported in batch path here
        if params.get("web_search") is False:
            params.pop("web_search", None)
        for exp in experiments:
            exp_name = exp.get("name", "experiment")
            qfile = exp.get("query_file", defaults.get("query_file"))
            if not qfile:
                continue
            base_queries = load_queries_from_file(str(qfile))
            runs = int(exp.get("runs", default_runs) or 1)
            run_offset = int(exp.get("run_offset", default_run_offset) or 0)
            # Expand by runs so each repetition gets its own custom_id (_runN)
            queries = [(f"{qid}_run{run_offset + r}", qtext)
                       for (qid, qtext) in base_queries
                       for r in range(runs)]
            out_dir = out_root / run_id / model_dir / "session_00" / exp_name
            prompts_by_qid = {qid: qtext for qid, qtext in base_queries}
            jobs.append({
                "model_name": model_name,
                "params": params,
                "system_prompt": system_prompt,
                "sys_chars": sys_chars,
                "provider": provider,
                "exp_name": exp_name,
                "queries": queries,
                "out_dir": str(out_dir),
                "model_dir": model_dir,
                "prompts_by_qid": prompts_by_qid,
            })
    return jobs


def state_path(out_root: Path, run_id: str) -> Path:
    return out_root / run_id / "_batch_state.json"


def submit_all(jobs: list[dict], state_file: Path) -> dict:
    state: dict = {"jobs": []}
    if state_file.exists():
        state = json.loads(state_file.read_text())
    submitted = {(j["model_dir"], j["exp_name"]): j for j in state["jobs"]}

    for job in jobs:
        key = (job["model_dir"], job["exp_name"])
        if key in submitted:
            existing = submitted[key]
            print(f"  Resume: {job['model_dir']} / {job['exp_name']} → batch {existing.get('batch_id')}")
            continue
        provider = job["provider"]
        print(f"\nSubmitting {provider}: {job['model_dir']} / {job['exp_name']} ({len(job['queries'])} q)")
        custom_ids = None
        if provider == "openai":
            bid = submit_openai_batch(job["model_name"], job["params"], job["queries"], job["system_prompt"])
        elif provider == "claude":
            bid = submit_claude_batch(job["model_name"], job["params"], job["queries"], job["system_prompt"])
        else:
            bid, custom_ids = submit_gemini_batch(job["model_name"], job["params"], job["queries"], job["system_prompt"])
        entry = {
            "model_name": job["model_name"],
            "model_dir": job["model_dir"],
            "exp_name": job["exp_name"],
            "provider": provider,
            "batch_id": bid,
            "out_dir": job["out_dir"],
            "sys_chars": job["sys_chars"],
            "submitted_at": datetime.now().isoformat(),
            "status": "submitted" if bid != "sequential" else "completed",
        }
        if custom_ids is not None:
            entry["custom_ids"] = custom_ids
        state["jobs"].append(entry)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2))
    return state


def poll_and_collect(state: dict, state_file: Path, jobs_by_key: dict, poll_sec: int = 60) -> None:
    pending = [j for j in state["jobs"] if j["status"] not in ("collected", "failed", "completed")]
    if not pending:
        print("\nNo pending batches.")
        return

    import anthropic, openai
    oa = openai.OpenAI(api_key=os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY"))
    an = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    gem_client = None
    if any(e["provider"] == "gemini" for e in pending):
        from google import genai
        gem_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    print(f"\nPolling {len(pending)} pending batch(es)...")
    while pending:
        still = []
        for entry in pending:
            provider = entry["provider"]
            bid = entry["batch_id"]
            if provider == "openai":
                st = oa.batches.retrieve(bid).status
                done = st in ("completed", "failed", "expired", "cancelled")
            elif provider == "claude":
                st = an.messages.batches.retrieve(bid).processing_status
                done = st == "ended"
            else:  # gemini
                gst = gem_client.batches.get(name=bid).state
                st = str(gst).split(".")[-1]
                done = st.endswith("SUCCEEDED") or st.endswith("FAILED") or st.endswith("CANCELLED") or st.endswith("EXPIRED")
            print(f"  {entry['model_dir']}/{entry['exp_name']} [{provider}/{bid}]: {st}")
            if done:
                key = (entry["model_dir"], entry["exp_name"])
                job = jobs_by_key[key]
                try:
                    if provider == "openai":
                        ok, err = collect_openai_batch(bid, entry["model_name"], entry["sys_chars"],
                                                       Path(entry["out_dir"]), job["prompts_by_qid"])
                    elif provider == "claude":
                        ok, err = collect_claude_batch(bid, entry["model_name"], entry["sys_chars"],
                                                       Path(entry["out_dir"]), job["prompts_by_qid"])
                    else:
                        ok, err = collect_gemini_batch(bid, entry["model_name"], entry["sys_chars"],
                                                       Path(entry["out_dir"]), job["prompts_by_qid"],
                                                       entry.get("custom_ids") or [])
                    entry["status"] = "collected"
                    entry["ok"] = ok
                    entry["err"] = err
                    print(f"    ✓ collected: {ok} ok, {err} err → {entry['out_dir']}")
                except Exception as e:
                    entry["status"] = "failed"
                    entry["error"] = str(e)
                    print(f"    ✗ collect failed: {e}")
                state_file.write_text(json.dumps(state, indent=2))
            else:
                still.append(entry)
        pending = still
        if pending:
            time.sleep(poll_sec)
    print("\nAll batches collected.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--config-api-models", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-root", default=str(DATA_DIR / "api"))
    ap.add_argument("--poll-sec", type=int, default=60)
    ap.add_argument("--submit-only", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    load_dotenv(BASE_DIR / ".env")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = BASE_DIR / out_root

    print(f">> run_id: {run_id}")
    print(f">> out_root: {out_root}")

    jobs = build_jobs(config_path, run_id, out_root)
    jobs_by_key = {(j["model_dir"], j["exp_name"]): j for j in jobs}

    sf = state_path(out_root, run_id)
    if not args.collect_only:
        state = submit_all(jobs, sf)
    else:
        state = json.loads(sf.read_text())

    if not args.submit_only:
        poll_and_collect(state, sf, jobs_by_key, poll_sec=args.poll_sec)


if __name__ == "__main__":
    main()
