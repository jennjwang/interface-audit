"""Generate an interactive HTML review of Haiku HellaSwag extraction.

For each of the 5 reporting runs, compares the original extractor (no-context
LLM judge) against the context-aware judge, and renders every item with:
  - response snippet (first 500 chars)
  - old extracted letter, new extracted letter, gold answer
  - status tag: fixed / lucky / missed / same-wrong / same-right

Writes: automated-scraper/outputs/hellaswag_extraction_review.html

Usage:
  cd /path/to/personalization
  python automated-scraper/outputs/hellaswag_extraction_review.py [--fast]

--fast: skip re-running the LLM judge; use only the cached answer_llm column from
        the CSV and the context-aware verdict will be set to ??? for review.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import html as html_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parent

load_dotenv(BASE / ".env")

HELLASWAG_DIR = REPO / "experiments" / "metabench" / "metabench-hellaswag"
QUERIES_CSV   = HELLASWAG_DIR / "queries" / "queries.csv"
IFACE_DIR     = HELLASWAG_DIR / "data-claude" / "interface"
PARSED_DIR    = HELLASWAG_DIR / "data-claude" / "parsed_json"
OUT_HTML      = Path(__file__).resolve().parent / "hellaswag_extraction_review.html"

HAIKU_RUNS = [
    ("Run 6",  "2026-05-14_01-20-44-hellaswag"),
    ("Run 7",  "2026-05-14_10-57-37-hellaswag"),
    ("Run 8",  "2026-05-14_12-26-15-hellaswag"),
    ("Run 9",  "2026-05-14_13-32-50-hellaswag"),
    ("Run 10", "2026-05-14_14-42-08-hellaswag"),
]

JUDGE_SYSTEM = (
    "You are extracting the answer letter from a model's response to a "
    "multiple-choice question. The response may contain meta-commentary, "
    "document summaries, or clarifying menus — focus ONLY on which answer "
    "letter the model ultimately selected for the actual question. "
    "Output a single uppercase letter (A, B, C, D, ...) or NONE."
)


def load_queries() -> dict[str, dict]:
    return {r["id"]: r for r in csv.DictReader(open(QUERIES_CSV))}


def question_tail(query: str, chars: int = 700) -> str:
    return query[-chars:].strip()


def judge_one(client: OpenAI, qid: str, query: str, response: str) -> str:
    tail = question_tail(query)
    user = f"Question (end of prompt):\n{tail}\n\nModel's response:\n{response}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user},
            ],
            max_completion_tokens=4,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
        if raw in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and len(raw) == 1:
            return raw
        if raw == "NONE":
            return "NONE"
        for c in raw:
            if c in "ABCD":
                return c
        return "NONE"
    except Exception as exc:
        return f"ERR:{exc}"


def status_tag(old: str, new: str, gold: str) -> tuple[str, str]:
    """Return (tag, css_class)."""
    old_ok = old == gold
    new_ok = new == gold and new != "NONE"
    if old_ok and new_ok:
        return "same-right", "same-right"
    if not old_ok and not new_ok and new == "NONE" and old != gold:
        return "same-wrong (no-answer)", "same-wrong"
    if not old_ok and not new_ok:
        return "same-wrong", "same-wrong"
    if old_ok and new == "NONE":
        return "lucky→no-answer", "lucky"
    if old_ok and not new_ok:
        return "degraded", "degraded"
    if not old_ok and new_ok:
        return "fixed", "fixed"
    return "changed", "changed"


def process_run(run_label: str, ts: str, gold: dict, client: OpenAI | None) -> list[dict]:
    csv_path = IFACE_DIR / ts / "session_00" / "haiku.csv"
    if not csv_path.exists():
        return []
    rows = list(csv.DictReader(open(csv_path)))

    # Build judge inputs
    inputs: list[tuple[str, str, str]] = []
    response_map: dict[str, str] = {}
    for row in rows:
        qid = row["id"]
        path_str = row.get("path", "")
        if not path_str:
            continue
        jp = Path(path_str)
        if not jp.exists():
            continue
        d = json.loads(jp.read_text(encoding="utf-8"))
        resp = d.get("ai_generated_output_text") or d.get("response_text") or ""
        response_map[qid] = resp
        if resp and client:
            inputs.append((qid, gold[qid]["query"], resp))

    # Run new judge in parallel
    new_verdicts: dict[str, str] = {}
    if client and inputs:
        print(f"  Judging {len(inputs)} items for {run_label}...", flush=True)
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {ex.submit(judge_one, client, qid, q, r): qid
                    for qid, q, r in inputs}
            done = 0
            for fut in as_completed(futs):
                qid = futs[fut]
                new_verdicts[qid] = fut.result()
                done += 1
                if done % 30 == 0 or done == len(futs):
                    print(f"    {done}/{len(futs)}", flush=True)
    else:
        for row in rows:
            new_verdicts[row["id"]] = "???"

    items = []
    for row in rows:
        qid = row["id"]
        old_letter = (row.get("answer_llm") or "").strip().upper() or "—"
        new_letter = new_verdicts.get(qid, "???")
        gold_ans   = (gold.get(qid, {}).get("answer") or "").strip().upper()
        resp_text  = response_map.get(qid, "")
        tag, css   = status_tag(old_letter, new_letter, gold_ans)
        items.append({
            "qid": qid,
            "old": old_letter,
            "new": new_letter,
            "gold": gold_ans,
            "status": tag,
            "css": css,
            "response": resp_text,
        })

    # Sort: bad cases first
    order = {"fixed": 0, "lucky": 1, "degraded": 2, "same-wrong": 3, "changed": 4, "same-right": 5}
    items.sort(key=lambda x: (order.get(x["css"], 9), int(x["qid"])))

    old_correct = sum(1 for r in rows if r.get("correct") == "True")
    new_correct = sum(1 for x in items
                      if x["new"] not in ("NONE", "???", "—") and x["new"] == x["gold"])
    n_none = sum(1 for x in items if x["new"] == "NONE")
    print(f"  {run_label}: old={old_correct}/93, new={new_correct}/93, NONE={n_none}/93")
    return items, old_correct, new_correct, n_none


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HellaSwag Haiku Extraction Audit</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:13px; margin:0; background:#f4f4f4; }}
h1 {{ margin:16px; color:#1a1a2e; }}
.tabs {{ display:flex; gap:4px; padding:8px 16px; background:#fff; border-bottom:1px solid #ddd; position:sticky; top:0; z-index:10; }}
.tab {{ cursor:pointer; padding:6px 14px; border-radius:6px; border:1px solid #ccc; background:#f8f8f8; font-size:12px; }}
.tab.active {{ background:#1a1a2e; color:#fff; border-color:#1a1a2e; }}
.panel {{ display:none; padding:16px; }}
.panel.active {{ display:block; }}
.summary {{ background:#fff; border-radius:8px; padding:12px 16px; margin-bottom:12px; font-size:12px; }}
.summary span {{ margin-right:20px; }}
.filters {{ margin-bottom:10px; display:flex; gap:6px; flex-wrap:wrap; }}
.filter-btn {{ cursor:pointer; padding:4px 10px; border-radius:12px; border:1px solid #ccc; font-size:11px; background:#f0f0f0; }}
.filter-btn.active {{ border-color:#333; background:#333; color:#fff; }}
table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px; overflow:hidden; }}
th {{ background:#1a1a2e; color:#fff; padding:8px 10px; text-align:left; font-size:11px; font-weight:600; position:sticky; top:42px; }}
td {{ padding:8px 10px; border-bottom:1px solid #eee; vertical-align:top; font-size:12px; }}
tr:hover td {{ background:#f9f9f9; }}
.resp {{ max-height:120px; overflow-y:auto; font-size:11px; white-space:pre-wrap; word-break:break-word; color:#444; background:#fafafa; padding:4px; border-radius:4px; cursor:pointer; }}
.resp.expanded {{ max-height:none; }}
.badge {{ display:inline-block; padding:2px 7px; border-radius:10px; font-size:10px; font-weight:600; }}
.fixed     {{ background:#d4edda; color:#155724; }}
.lucky     {{ background:#fff3cd; color:#856404; }}
.degraded  {{ background:#f8d7da; color:#721c24; }}
.same-wrong{{ background:#f0f0f0; color:#555; }}
.same-right{{ background:#cce5ff; color:#004085; }}
.changed   {{ background:#fde8d8; color:#843400; }}
.ltr {{ display:inline-block; width:20px; height:20px; border-radius:50%; text-align:center; line-height:20px; font-weight:700; font-size:11px; }}
.correct   {{ background:#28a745; color:#fff; }}
.incorrect {{ background:#dc3545; color:#fff; }}
.none-v    {{ background:#aaa; color:#fff; }}
</style>
</head>
<body>
<h1>HellaSwag Haiku Extraction Audit — Context-Aware vs Original Judge</h1>
<div class="tabs">
{tab_buttons}
</div>
{panels}
<script>
function switchTab(idx) {{
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', i===idx));
  document.querySelectorAll('.panel').forEach((p,i) => p.classList.toggle('active', i===idx));
}}
function toggleResp(el) {{ el.classList.toggle('expanded'); }}
function filterRows(panelIdx, status) {{
  const btns = document.querySelectorAll('#panel-'+panelIdx+' .filter-btn');
  btns.forEach(b => b.classList.toggle('active', b.dataset.status===status||status==='all'));
  document.querySelectorAll('#panel-'+panelIdx+' tr[data-status]').forEach(r => {{
    r.style.display = (status==='all'||r.dataset.status===status) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

PANEL_TEMPLATE = """<div class="panel{active}" id="panel-{idx}">
<div class="summary">
  <span><b>Old judge:</b> {old_correct}/93 = {old_pct:.1f}%</span>
  <span><b>New judge:</b> {new_correct}/93 = {new_pct:.1f}%</span>
  <span><b>NONE:</b> {n_none}/93</span>
  <span><b>Fixed:</b> {n_fixed}</span>
  <span><b>Lucky→NONE:</b> {n_lucky}</span>
  <span><b>Degraded:</b> {n_degraded}</span>
</div>
<div class="filters">
  <span style="font-size:11px;margin-right:4px;line-height:26px">Filter:</span>
  <button class="filter-btn active" data-status="all" onclick="filterRows({idx},'all')">All ({total})</button>
  <button class="filter-btn" data-status="fixed" onclick="filterRows({idx},'fixed')">Fixed ({n_fixed})</button>
  <button class="filter-btn" data-status="lucky" onclick="filterRows({idx},'lucky')">Lucky→NONE ({n_lucky})</button>
  <button class="filter-btn" data-status="degraded" onclick="filterRows({idx},'degraded')">Degraded ({n_degraded})</button>
  <button class="filter-btn" data-status="same-wrong" onclick="filterRows({idx},'same-wrong')">Same-Wrong ({n_same_wrong})</button>
  <button class="filter-btn" data-status="same-right" onclick="filterRows({idx},'same-right')">Same-Right ({n_same_right})</button>
</div>
<table>
<thead><tr>
  <th>QID</th><th>Old</th><th>New</th><th>Gold</th><th>Status</th><th>Response</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>"""

ROW_TEMPLATE = """<tr data-status="{css}">
  <td>{qid}</td>
  <td><span class="ltr {old_cls}">{old}</span></td>
  <td><span class="ltr {new_cls}">{new_disp}</span></td>
  <td><span class="ltr correct">{gold}</span></td>
  <td><span class="badge {css}">{status}</span></td>
  <td><div class="resp" onclick="toggleResp(this)">{resp}</div></td>
</tr>"""


def letter_cls(letter: str, gold: str) -> str:
    if letter in ("NONE", "???", "—", "ERR"):
        return "none-v"
    return "correct" if letter == gold else "incorrect"


def build_html(all_runs_data: list) -> str:
    tab_buttons = ""
    panels = ""
    for idx, (run_label, ts, items, old_correct, new_correct, n_none) in enumerate(all_runs_data):
        active_cls = " active" if idx == 0 else ""
        tab_buttons += f'<button class="tab{active_cls}" onclick="switchTab({idx})">{run_label}</button>'

        rows_html = ""
        for item in items:
            new_disp = item["new"] if item["new"] != "NONE" else "—"
            resp_escaped = html_mod.escape(item["response"][:800])
            rows_html += ROW_TEMPLATE.format(
                qid=item["qid"],
                old=item["old"],
                new_disp=new_disp if new_disp != "—" else "∅",
                gold=item["gold"],
                old_cls=letter_cls(item["old"], item["gold"]),
                new_cls=letter_cls(item["new"], item["gold"]),
                css=item["css"],
                status=item["status"],
                resp=resp_escaped,
            )

        n_fixed      = sum(1 for x in items if x["css"] == "fixed")
        n_lucky      = sum(1 for x in items if x["css"] == "lucky")
        n_degraded   = sum(1 for x in items if x["css"] == "degraded")
        n_same_wrong = sum(1 for x in items if x["css"] == "same-wrong")
        n_same_right = sum(1 for x in items if x["css"] == "same-right")

        panels += PANEL_TEMPLATE.format(
            active=active_cls,
            idx=idx,
            old_correct=old_correct,
            old_pct=100 * old_correct / 93,
            new_correct=new_correct,
            new_pct=100 * new_correct / 93,
            n_none=n_none,
            n_fixed=n_fixed,
            n_lucky=n_lucky,
            n_degraded=n_degraded,
            n_same_wrong=n_same_wrong,
            n_same_right=n_same_right,
            total=len(items),
            rows=rows_html,
        )

    return HTML_TEMPLATE.format(tab_buttons=tab_buttons, panels=panels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="Skip LLM judge; show ??? for new verdict")
    args = ap.parse_args()

    gold = load_queries()
    client = None
    if not args.fast:
        api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_KEY not set — falling back to --fast mode")
            args.fast = True
        else:
            client = OpenAI(api_key=api_key, timeout=60)

    all_runs_data = []
    for run_label, ts in HAIKU_RUNS:
        print(f"\nProcessing {run_label} ({ts})...")
        result = process_run(run_label, ts, gold, client)
        if result:
            items, old_correct, new_correct, n_none = result
            all_runs_data.append((run_label, ts, items, old_correct, new_correct, n_none))

    html = build_html(all_runs_data)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nWrote {OUT_HTML}")


if __name__ == "__main__":
    main()
