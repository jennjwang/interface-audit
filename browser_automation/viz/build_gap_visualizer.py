"""Build an HTML visualizer showing one run for the largest-gap model per benchmark.

For each of the 6 metabench benchmarks, finds the interface model with the
largest same-model API-Interface accuracy gap, picks the first run, and renders
a side-by-side table of API vs Interface responses.

Usage:
  python automated-scraper/build_gap_visualizer.py
"""
from __future__ import annotations
import csv, json, re, html as _html
from collections import defaultdict
from pathlib import Path

BASE   = Path(__file__).resolve().parents[1]
REPO   = BASE.parent
BENCH  = REPO / "experiments" / "metabench"
PLOTS  = REPO / "experiments" / "metabench" / "openllm_leaderboard" / "plots"
OUT    = BASE / "outputs" / "gap_visualizer.html"

# ── Coverage CSV paths ──────────────────────────────────────────────────────
COVERAGE = {
    "chatgpt": PLOTS / "coverage_by_run.csv",
    "claude":  PLOTS / "coverage_by_run_claude.csv",
    "gemini":  PLOTS / "coverage_by_run_gemini.csv",
}

# ── Same-model pairs (interface_cond → api_cond) ────────────────────────────
PAIRS = {
    "claude": [
        ("Interface: Haiku",   "API: Claude Haiku 4.5",   "haiku",    "claude-haiku-4-5-20251001",   "session_00"),
        ("Interface: Sonnet",  "API: Claude Sonnet 4.6",  "sonnet",   "claude-sonnet-4-6",           "session_02"),
        ("Interface: Opus",    "API: Claude Opus 4.6",    "opus",     "claude-opus-4-6",             "session_01"),
    ],
    "chatgpt": [
        ("Interface: Instant",  "API: GPT 5.3 Chat (Instant)",  "instant",  "gpt-5.3-chat-latest",                         "session_00"),
        ("Interface: Thinking", "API: GPT 5.4 Reasoning High",  "thinking", "gpt-5.4-2026-03-05_reasoning_effort-high",    "session_01"),
    ],
    "gemini": [
        ("Interface: Fast",     "API: Gemini 3 Flash (Fast)",     "fast",     "gemini-3-flash-preview_thinking_level-low",  "session_00"),
        ("Interface: Thinking", "API: Gemini 3 Flash (Thinking)", "thinking", "gemini-3-flash-preview_thinking_level-high", "session_01"),
    ],
}

BENCHMARKS = ["arc", "gsm8k", "hellaswag", "mmlu", "truthfulqa", "winogrande"]
BENCH_DIR  = {"arc": "metabench-arc", "gsm8k": "metabench-gsm8k",
              "hellaswag": "metabench-hellaswag", "mmlu": "metabench-mmlu",
              "truthfulqa": "metabench-truthfulQA", "winogrande": "metabench-winogrande"}
QUERIES    = {b: BENCH / BENCH_DIR[b] / "queries" / "queries.csv" for b in BENCHMARKS}

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_queries(bench: str) -> dict[str, dict]:
    out = {}
    qf = QUERIES[bench]
    if not qf.exists():
        return out
    with open(qf) as f:
        for row in csv.DictReader(f):
            out[str(row["id"])] = row
    return out


def load_coverage(provider: str) -> list[dict]:
    p = COVERAGE[provider]
    if not p.exists():
        return []
    rows = []
    for r in csv.DictReader(open(p)):
        try:
            float(r["accuracy"])
            rows.append(r)
        except (ValueError, KeyError):
            pass
    return rows


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def find_iface_csv(bench: str, provider: str, timestamp: str, stem: str, session: str) -> Path | None:
    data_dir = BENCH / BENCH_DIR[bench] / f"data-{provider}"
    ts_dir   = data_dir / "interface" / timestamp
    if not ts_dir.exists():
        # Try glob with suffix
        matches = list((data_dir / "interface").glob(f"{timestamp}*"))
        if not matches:
            return None
        ts_dir = matches[0]
    sess_dir = ts_dir / session
    if not sess_dir.exists():
        # Try any session
        sessions = sorted(ts_dir.iterdir()) if ts_dir.exists() else []
        if not sessions:
            return None
        sess_dir = sessions[0]
    # Exact match
    exact = sess_dir / f"{stem}.csv"
    if exact.exists():
        return exact
    # Glob fallback
    matches = sorted(sess_dir.glob(f"{stem}*.csv"))
    if matches:
        return matches[0]
    # Any CSV in session
    all_csvs = sorted(sess_dir.glob("*.csv"))
    return all_csvs[0] if all_csvs else None


def find_api_csv(bench: str, provider: str, timestamp: str, api_stem: str) -> Path | None:
    data_dir = BENCH / BENCH_DIR[bench] / f"data-{provider}"
    ts_dir   = data_dir / "api" / timestamp
    if not ts_dir.exists():
        matches = list((data_dir / "api").glob(f"{timestamp}*"))
        if not matches:
            return None
        ts_dir = matches[0]
    # Recurse for CSV matching api_stem
    for csv_path in sorted(ts_dir.rglob("*.csv")):
        if api_stem in csv_path.stem or csv_path.stem.startswith(api_stem.split("_")[0]):
            return csv_path
    # Fallback: any CSV
    all_csvs = sorted(ts_dir.rglob("*.csv"))
    return all_csvs[0] if all_csvs else None


def load_csv_rows(path: Path) -> dict[str, dict]:
    """Return {id: row} from a scored CSV."""
    out = {}
    if not path or not path.exists():
        return out
    for row in csv.DictReader(open(path)):
        qid = str(row.get("id", "")).strip()
        if qid:
            out[qid] = row
    return out


def get_raw_json(row: dict) -> dict:
    """Return the raw JSON dict for a row (empty dict if unavailable)."""
    path_str = row.get("path", "")
    if not path_str:
        return {}
    p = Path(path_str)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_response_text(row: dict) -> str:
    """Read ai_generated_output_text from path in row."""
    d = get_raw_json(row)
    if d:
        return (d.get("ai_generated_output_text") or d.get("response_text") or "").strip()
    return row.get("answer_llm", "")


# ── Find largest-gap winner per benchmark ───────────────────────────────────

def find_winners() -> dict[str, dict]:
    """For each benchmark, find (provider, iface_cond, api_cond, iface_stem, api_stem, session, gap, timestamp)."""
    # Load coverage data
    cov: dict[str, list[dict]] = {p: load_coverage(p) for p in ["chatgpt", "claude", "gemini"]}

    by_dc: dict[str, dict[tuple, list[float]]] = {}
    ts_map: dict[str, dict[tuple, list[str]]] = {}  # (provider, ds, cond) → [timestamps]
    for provider, rows in cov.items():
        by_dc[provider] = defaultdict(list)
        ts_map[provider] = defaultdict(list)
        for r in rows:
            key = (r["dataset"], r["condition"])
            by_dc[provider][key].append(float(r["accuracy"]))
            ts_map[provider][key].append(r["timestamp"])

    winners: dict[str, dict] = {}
    for bench in BENCHMARKS:
        ds = f"metabench-{bench}" if bench != "truthfulqa" else "metabench-truthfulQA"
        best_gap = None
        best = None
        for provider, pairs in PAIRS.items():
            for ic, ac, stem, api_stem, session in pairs:
                iv = by_dc[provider].get((ds, ic), [])
                av = by_dc[provider].get((ds, ac), [])
                if not iv or not av:
                    continue
                gap = mean(iv) - mean(av)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    timestamps = ts_map[provider].get((ds, ic), [])
                    best = dict(
                        provider=provider, iface_cond=ic, api_cond=ac,
                        stem=stem, api_stem=api_stem, session=session,
                        gap=gap, iface_mean=mean(iv), api_mean=mean(av),
                        iface_n=len(iv), timestamps=timestamps,
                    )
        if best:
            winners[bench] = best
    return winners


# ── HTML ─────────────────────────────────────────────────────────────────────

def truncate(text: str, n: int = 300) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def build_tab(bench: str, winner: dict, gold: dict[str, dict]) -> str:
    provider   = winner["provider"]
    stem       = winner["stem"]
    api_stem   = winner["api_stem"]
    session    = winner["session"]
    timestamps = winner["timestamps"]

    # Pick first available run
    iface_csv = api_csv = None
    chosen_ts = None
    for ts in timestamps:
        ic = find_iface_csv(bench, provider, ts, stem, session)
        ac = find_api_csv(bench, provider, ts, api_stem)
        if ic and ac:
            iface_csv, api_csv, chosen_ts = ic, ac, ts
            break
    if iface_csv is None:
        return f'<div class="no-data">No data found for {bench} / {winner["iface_cond"]}</div>'

    iface_rows = load_csv_rows(iface_csv)
    api_rows   = load_csv_rows(api_csv)

    all_ids = sorted(set(iface_rows) | set(api_rows), key=lambda x: int(x) if x.isdigit() else x)

    rows_html = []
    for qid in all_ids:
        qdata   = gold.get(qid, {})
        q_text  = qdata.get("query", "")
        correct = qdata.get("answer", "").strip()

        ir = iface_rows.get(qid, {})
        ar = api_rows.get(qid, {})

        iface_pred  = ir.get("answer", "").strip()
        api_pred    = ar.get("answer", ar.get("answer_llm", "")).strip()
        iface_ok    = ir.get("correct", "").strip()
        api_ok      = ar.get("correct", "").strip()

        ir_json     = get_raw_json(ir) if ir else {}
        ar_json     = get_raw_json(ar) if ar else {}
        iface_resp  = (ir_json.get("ai_generated_output_text") or ir_json.get("response_text") or "").strip() if ir_json else (ir.get("answer_llm", "") if ir else "")
        api_resp    = (ar_json.get("ai_generated_output_text") or ar_json.get("response_text") or "").strip() if ar_json else (ar.get("answer_llm", "") if ar else "")

        def mark(pred, ok):
            if not pred:
                return '<span class="na">—</span>'
            cls = "correct" if ok == "True" else "wrong"
            return f'<span class="{cls}">{_html.escape(pred)}</span>'

        def csv_fields_html(row: dict, raw_json: dict) -> str:
            """Render CSV extraction fields + JSON path as a mini-table."""
            fields = [
                ("extracted",  row.get("answer", "") or row.get("answer_llm", "")),
                ("regex",      row.get("answer_regex", "")),
                ("gold",       row.get("gold_answer", "")),
                ("correct",    row.get("correct", "")),
            ]
            path_str = row.get("path", "")
            if path_str:
                fields.append(("json path", path_str))
            if not raw_json and path_str:
                fields.append(("⚠ json", "FILE NOT FOUND"))
            rows_f = "".join(
                f'<tr><td class="fk">{_html.escape(k)}</td>'
                f'<td class="fv">{_html.escape(str(v))}</td></tr>'
                for k, v in fields if v
            )
            return f'<table class="fields-tbl">{rows_f}</table>' if rows_f else ""

        q_short  = _html.escape(truncate(q_text, 200))
        q_full   = _html.escape(q_text)
        api_full   = _html.escape(api_resp)
        iface_full = _html.escape(iface_resp)
        api_fields   = csv_fields_html(ar, ar_json) if ar else ""
        iface_fields = csv_fields_html(ir, ir_json) if ir else ""

        row_id = f"{bench}_{qid}"
        rows_html.append(f"""
  <tr class="summary-row" onclick="toggle('{row_id}')">
    <td class="qid">{_html.escape(qid)}</td>
    <td class="question">{q_short}</td>
    <td class="ans">{_html.escape(correct)}</td>
    <td class="pred">{mark(api_pred, api_ok)}</td>
    <td class="pred">{mark(iface_pred, iface_ok)}</td>
    <td class="vote-cell" onclick="event.stopPropagation()">
      <button class="vote-btn" data-id="{_html.escape(qid)}" data-bench="{_html.escape(bench)}" data-val="api_bad"   onclick="castVote(this)">API?</button>
      <button class="vote-btn" data-id="{_html.escape(qid)}" data-bench="{_html.escape(bench)}" data-val="iface_bad" onclick="castVote(this)">Iface?</button>
    </td>
  </tr>
  <tr id="{row_id}" class="detail-row" style="display:none">
    <td colspan="6">
      <div class="detail-grid">
        <div class="detail-block">
          <div class="detail-label">Question</div>
          <pre class="detail-text">{q_full}</pre>
        </div>
        <div class="detail-block">
          <div class="detail-label">API raw response <span class="pred-inline">{mark(api_pred, api_ok)}</span></div>
          {api_fields}
          <pre class="detail-text">{api_full}</pre>
        </div>
        <div class="detail-block">
          <div class="detail-label">Interface raw response <span class="pred-inline">{mark(iface_pred, iface_ok)}</span></div>
          {iface_fields}
          <pre class="detail-text">{iface_full}</pre>
        </div>
      </div>
    </td>
  </tr>""")

    n_iface_correct = sum(1 for r in iface_rows.values() if r.get("correct") == "True")
    n_api_correct   = sum(1 for r in api_rows.values()   if r.get("correct") == "True")
    n_iface_total   = len([r for r in iface_rows.values() if r.get("answer")])
    n_api_total     = len([r for r in api_rows.values()   if r.get("answer") or r.get("answer_llm")])

    iface_acc = n_iface_correct / n_iface_total if n_iface_total else 0
    api_acc   = n_api_correct   / n_api_total   if n_api_total   else 0
    gap_pp    = (iface_acc - api_acc) * 100

    header = f"""
    <div class="run-header">
      <span class="model-tag">{_html.escape(winner["iface_cond"])}</span>
      <span class="stat">API: <strong>{api_acc:.1%}</strong> ({n_api_correct}/{n_api_total})</span>
      <span class="stat">Interface: <strong>{iface_acc:.1%}</strong> ({n_iface_correct}/{n_iface_total})</span>
      <span class="gap {'neg' if gap_pp < 0 else 'pos'}">Gap: {gap_pp:+.1f}pp</span>
      <span class="ts">Run: {_html.escape(chosen_ts)}</span>
    </div>
    <p class="sub">Overall means across {winner['iface_n']} runs — Interface: {winner['iface_mean']:.1%} | API: {winner['api_mean']:.1%} | Gap: {winner['gap']*100:+.1f}pp. Click row to expand.</p>"""

    table = f"""
    <table>
      <thead><tr>
        <th>ID</th><th>Question</th><th>Gold</th>
        <th>API pred</th><th>Interface pred</th><th>Flag</th>
      </tr></thead>
      <tbody>{''.join(rows_html)}
      </tbody>
    </table>"""

    return header + table


def build_html(tabs: dict[str, str]) -> str:
    tab_buttons = "".join(
        f'<button class="tab-btn" onclick="showTab(\'{b}\')" id="btn-{b}">{b.upper()}</button>'
        for b in tabs
    )
    tab_panes = "".join(
        f'<div class="tab-pane" id="tab-{b}">{content}</div>'
        for b, content in tabs.items()
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Largest-Gap Model per Benchmark</title>
<style>
  body {{ font-family: system-ui, sans-serif; font-size: 13px; margin: 0; background: #f8fafc; color: #1e293b; }}
  h1 {{ margin: 16px 20px 4px; font-size: 18px; }}
  .tabs {{ display: flex; gap: 4px; padding: 10px 20px 0; border-bottom: 2px solid #e2e8f0; background: #fff; position: sticky; top: 0; z-index: 10; }}
  .tab-btn {{ padding: 6px 16px; border: 1px solid #cbd5e1; border-bottom: none; border-radius: 6px 6px 0 0; background: #f1f5f9; cursor: pointer; font-size: 13px; font-weight: 600; }}
  .tab-btn.active {{ background: #fff; border-bottom: 2px solid #fff; margin-bottom: -2px; color: #2563eb; }}
  .tab-pane {{ display: none; padding: 16px 20px; }}
  .tab-pane.active {{ display: block; }}
  .run-header {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 6px; }}
  .model-tag {{ background: #dbeafe; color: #1d4ed8; padding: 3px 10px; border-radius: 12px; font-weight: 600; font-size: 12px; }}
  .stat {{ color: #475569; font-size: 12px; }}
  .gap {{ font-weight: 700; font-size: 13px; padding: 2px 8px; border-radius: 8px; }}
  .gap.neg {{ background: #fee2e2; color: #dc2626; }}
  .gap.pos {{ background: #dcfce7; color: #16a34a; }}
  .ts {{ color: #94a3b8; font-size: 11px; }}
  .sub {{ color: #64748b; font-size: 12px; margin: 0 0 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  thead tr {{ background: #f1f5f9; }}
  th {{ padding: 7px 10px; text-align: left; font-weight: 600; border-bottom: 2px solid #e2e8f0; white-space: nowrap; }}
  .summary-row {{ cursor: pointer; border-bottom: 1px solid #e2e8f0; }}
  .summary-row:hover {{ background: #f8fafc; }}
  td {{ padding: 6px 10px; vertical-align: top; }}
  .qid {{ width: 40px; color: #94a3b8; font-size: 11px; }}
  .question {{ max-width: 360px; color: #475569; }}
  .ans {{ width: 40px; font-weight: 700; text-align: center; }}
  .pred {{ width: 80px; text-align: center; font-weight: 700; font-size: 13px; }}
  .correct {{ color: #16a34a; }}
  .wrong {{ color: #dc2626; }}
  .na {{ color: #94a3b8; }}
  .detail-row td {{ background: #f8fafc; padding: 12px 16px; }}
  .detail-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
  .detail-block {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; }}
  .detail-label {{ font-weight: 600; font-size: 11px; color: #64748b; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .pred-inline {{ font-size: 13px; }}
  pre.detail-text {{ margin: 0; white-space: pre-wrap; font-family: inherit; font-size: 12px; color: #334155; max-height: 300px; overflow-y: auto; }}
  .no-data {{ padding: 40px; color: #94a3b8; text-align: center; }}
  .vote-cell {{ width: 100px; white-space: nowrap; }}
  .vote-btn {{ padding: 2px 7px; border: 1px solid #cbd5e1; border-radius: 5px; background: #f8fafc; cursor: pointer; font-size: 11px; font-weight: 600; color: #64748b; }}
  .vote-btn:hover {{ background: #e2e8f0; }}
  .vote-btn.flagged-api_bad   {{ background: #fee2e2; border-color: #fca5a5; color: #dc2626; }}
  .vote-btn.flagged-iface_bad {{ background: #fef3c7; border-color: #fcd34d; color: #b45309; }}
  .summary-row.has-flag {{ background: #fffbeb; }}
  #export-bar {{ position: sticky; bottom: 0; background: #0f172a; color: #e2e8f0; padding: 8px 20px; display: flex; align-items: center; gap: 12px; font-size: 12px; z-index: 20; }}
  #flag-count {{ font-weight: 700; }}
  #export-btn {{ padding: 5px 14px; background: #2563eb; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; }}
  .fields-tbl {{ border-collapse: collapse; margin-bottom: 6px; font-size: 11px; width: 100%; }}
  .fields-tbl td {{ padding: 2px 6px; border: 1px solid #e2e8f0; vertical-align: top; }}
  .fk {{ font-weight: 600; color: #64748b; white-space: nowrap; width: 80px; background: #f8fafc; }}
  .fv {{ font-family: monospace; color: #1e293b; word-break: break-all; }}
</style>
</head>
<body>
<h1>Largest API–Interface gap per benchmark (one representative run)</h1>
<div class="tabs">{tab_buttons}</div>
{tab_panes}
<div id="export-bar">
  <span>Flagged: <span id="flag-count">0</span> items</span>
  <button id="export-btn" onclick="exportFlags()">Export flags JSON</button>
  <span style="color:#64748b;font-size:11px">API? = bad API extraction · Iface? = bad interface extraction</span>
</div>
<script>
const flags = {{}};  // key: "bench/qid" → "api_bad" | "iface_bad"

function castVote(btn) {{
  const bench = btn.dataset.bench;
  const id    = btn.dataset.id;
  const val   = btn.dataset.val;
  const key   = bench + '/' + id;
  const row   = btn.closest('tr');

  // Toggle: clicking same button again clears the flag
  if (flags[key] === val) {{
    delete flags[key];
    row.classList.remove('has-flag');
    row.querySelectorAll('.vote-btn').forEach(b => b.classList.remove('flagged-api_bad','flagged-iface_bad'));
  }} else {{
    flags[key] = val;
    row.classList.add('has-flag');
    row.querySelectorAll('.vote-btn').forEach(b => {{
      b.classList.remove('flagged-api_bad','flagged-iface_bad');
      if (b.dataset.val === val) b.classList.add('flagged-' + val);
    }});
  }}
  document.getElementById('flag-count').textContent = Object.keys(flags).length;
}}

function exportFlags() {{
  const out = {{}};
  for (const [k, v] of Object.entries(flags)) {{
    const [bench, id] = k.split('/');
    if (!out[bench]) out[bench] = {{}};
    out[bench][id] = v;
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'extraction_flags.json';
  a.click();
}}

function showTab(name) {{
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('btn-' + name).classList.add('active');
}}
function toggle(id) {{
  const el = document.getElementById(id);
  el.style.display = el.style.display === 'none' ? 'table-row' : 'none';
}}
// Show first tab
const first = document.querySelector('.tab-btn');
if (first) showTab(first.id.replace('btn-',''));
</script>
</body>
</html>"""


def main():
    print("Finding largest-gap model per benchmark...")
    winners = find_winners()
    for bench, w in winners.items():
        print(f"  {bench:12s}: {w['provider']:8s} {w['iface_cond']:24s} gap={w['gap']*100:+.1f}%  (n={w['iface_n']})")

    print("\nBuilding tabs...")
    tabs = {}
    for bench in BENCHMARKS:
        if bench not in winners:
            print(f"  {bench}: no winner found, skipping")
            continue
        gold = load_queries(bench)
        tabs[bench] = build_tab(bench, winners[bench], gold)
        print(f"  {bench}: done ({len(gold)} questions)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(tabs), encoding="utf-8")
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
