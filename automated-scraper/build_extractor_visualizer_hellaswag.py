"""Hellaswag-specific extractor visualizer for the May-24 re-runs.

Covers all May-24 hellaswag runs across providers/conditions:
  - Claude Haiku (Interface + API)
  - ChatGPT Instant / Thinking (Interface + API)

Usage:
  python automated-scraper/build_extractor_visualizer_hellaswag.py
  python automated-scraper/build_extractor_visualizer_hellaswag.py --worst-only
"""
from __future__ import annotations
import argparse, csv, json, re, html as _html
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
HELLA = REPO / "experiments" / "metabench" / "metabench-hellaswag"
OUT = BASE / "outputs" / "extractor_visualizer_hellaswag.html"

DATE_PREFIX = "2026-05-24"

# (condition label, relative glob from HELLA, regex group capturing timestamp)
CONDITIONS = [
    ("Claude Haiku — Interface",
     f"data-claude/interface/{DATE_PREFIX}*-hellaswag/session_*/haiku.csv"),
    ("Claude Haiku — API",
     f"data-claude/api/{DATE_PREFIX}*-hellaswag/haiku/claude-haiku-4-5-20251001.csv"),
    ("ChatGPT Instant — Interface",
     f"data-chatgpt/interface/{DATE_PREFIX}*-hellaswag/session_*/instant.csv"),
    ("ChatGPT Thinking — Interface",
     f"data-chatgpt/interface/{DATE_PREFIX}*-hellaswag/session_*/thinking.csv"),
    ("ChatGPT Instant — API",
     f"data-chatgpt/api/{DATE_PREFIX}*-hellaswag/session_*/gpt-5.3-chat-latest.csv"),
    ("ChatGPT Thinking — API",
     f"data-chatgpt/api/{DATE_PREFIX}*-hellaswag/session_*/gpt-5.4-2026-03-05_reasoning_effort-high.csv"),
]


def load_response(path_str: str) -> str:
    if not path_str:
        return ""
    fp = Path(path_str)
    if not fp.exists():
        return ""
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
        return d.get("ai_generated_output_text") or d.get("response_text") or ""
    except Exception:
        return ""


def run_label(csv_path: Path) -> str:
    """Build a unique label for this run (timestamp + session if present)."""
    parts = csv_path.parts
    ts = next((p for p in parts if p.startswith(DATE_PREFIX)), csv_path.parent.name)
    session = next((p for p in parts if p.startswith("session_")), "")
    return f"{ts}/{session}" if session else ts


EXPECTED_ROWS = 93  # hellaswag subset size


def _is_correct(row: dict) -> bool:
    c = row.get("correct", "").lower()
    return c == "true" or c == "1"


def load_files(worst_only: bool = False) -> dict[str, dict[str, list[dict]]]:
    """Return {condition_label: {run_label: [rows]}}.

    A run is "complete" iff it has EXPECTED_ROWS rows AND every row has a
    non-empty response loaded from its JSON path. Incomplete runs are skipped.

    If worst_only is True, only the worst run per condition is kept, ranked by
    problem_rate = (no_extract + wrong_only) / EXPECTED_ROWS.
    """
    out: dict[str, dict[str, list[dict]]] = {}
    for label, pattern in CONDITIONS:
        matches = sorted(HELLA.glob(pattern))
        if not matches:
            print(f"  [empty]    {label}")
            continue
        runs: dict[str, list[dict]] = {}
        skipped: list[tuple[str, str]] = []
        for csv_path in matches:
            with csv_path.open(encoding="utf-8") as fh:
                raw = list(csv.DictReader(fh))
            rows = [
                {
                    "id":       (r.get("id") or "").strip(),
                    "gold":     (r.get("gold_answer") or "").strip(),
                    "answer":   (r.get("answer") or "").strip(),
                    "correct":  (r.get("correct") or "").strip(),
                    "response": load_response(r.get("path", "")),
                }
                for r in raw
            ]
            rlabel = run_label(csv_path)
            if len(rows) != EXPECTED_ROWS:
                skipped.append((rlabel, f"{len(rows)} rows"))
                continue
            missing_resp = sum(1 for r in rows if not r["response"])
            if missing_resp > 0:
                skipped.append((rlabel, f"{missing_resp} missing responses"))
                continue
            runs[rlabel] = rows

        if worst_only and runs:
            def problem_count(rows: list[dict]) -> int:
                return sum(1 for r in rows if not r["answer"] or not _is_correct(r))
            worst_rlabel = max(runs, key=lambda r: problem_count(runs[r]))
            runs = {worst_rlabel: runs[worst_rlabel]}

        out[label] = dict(sorted(runs.items()))
        print(f"  {label}: {len(runs)} complete runs, "
              f"{sum(len(v) for v in runs.values())} rows"
              + (f"; skipped {len(skipped)}" if skipped else ""))
        for rlabel, why in skipped:
            print(f"      skipped {rlabel} ({why})")
    return out


def build_html(files: dict[str, dict[str, list[dict]]]) -> str:
    # Flatten to {condition__run: rows} keyed for JS, plus group map
    flat: dict[str, list[dict]] = {}
    groups: dict[str, list[str]] = {}
    for cond, runs in files.items():
        names: list[str] = []
        for run_name, rows in runs.items():
            key = f"{cond} :: {run_name}"
            flat[key] = rows
            names.append(key)
        groups[cond] = names

    data_json = json.dumps(flat, ensure_ascii=False)
    groups_json = json.dumps(groups, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Extractor Visualizer — Hellaswag May-24</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: system-ui, sans-serif; font-size: 13px; background: #f8fafc; color: #1e293b; }}
.header {{ background: #fff; border-bottom: 1px solid #e2e8f0;
           padding: 10px 20px; position: sticky; top: 0; z-index: 20; }}
.header h1 {{ font-size: 14px; font-weight: 700; margin-bottom: 8px; }}
.bench-tabs {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
.bench-tab {{ padding: 4px 12px; border: 1px solid #cbd5e1; border-radius: 6px;
              background: #f1f5f9; cursor: pointer; font-size: 12px; font-weight: 500; }}
.bench-tab.active {{ background: #1e40af; color: #fff; border-color: #1e40af; }}
.run-buttons {{ display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 8px; }}
.run-btn {{ padding: 3px 10px; border: 1px solid #cbd5e1; border-radius: 6px;
            background: #f8fafc; cursor: pointer; font-size: 11px; font-weight: 500;
            white-space: nowrap; }}
.run-btn.active  {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
.run-btn.warn-extraction {{ background: #fff7ed; border-color: #fdba74; color: #9a3412; }}
.run-btn.warn-accuracy   {{ background: #fef2f2; border-color: #fca5a5; color: #991b1b; }}
.run-btn.warn-extraction.active,
.run-btn.warn-accuracy.active {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
.controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.filter-btns {{ display: flex; gap: 4px; }}
.filter-btn {{ padding: 4px 11px; border: 1px solid #cbd5e1; border-radius: 6px;
               background: #f1f5f9; cursor: pointer; font-size: 12px; font-weight: 500; }}
.filter-btn.active {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
#search {{ padding: 4px 10px; border: 1px solid #cbd5e1; border-radius: 6px;
           width: 200px; font-size: 12px; }}
.stats {{ font-size: 12px; color: #64748b; white-space: nowrap; }}
#item-list {{ padding: 10px 20px; display: flex; flex-direction: column; gap: 5px; }}
.item {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
.item.correct-item {{ border-left: 3px solid #16a34a; }}
.item.wrong-item   {{ border-left: 3px solid #dc2626; }}
.item.voted-agree    {{ border-left: 3px solid #2563eb; }}
.item.voted-disagree {{ border-left: 3px solid #f59e0b; }}
.item-header {{ display: flex; align-items: center; gap: 10px;
                padding: 8px 14px; cursor: pointer; user-select: none; }}
.item-header:hover {{ background: #f8fafc; }}
.item-id {{ font-size: 11px; color: #94a3b8; min-width: 60px; }}
.item-answer {{ font-size: 13px; font-weight: 700; }}
.item-gold {{ font-size: 12px; color: #64748b; }}
.badge {{ padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700; white-space: nowrap; }}
.badge.c {{ background: #dcfce7; color: #16a34a; }}
.badge.w {{ background: #fee2e2; color: #dc2626; }}
.vote-btns {{ display: flex; gap: 4px; margin-left: 4px; }}
.vote-btn {{ padding: 2px 9px; border-radius: 6px; border: 1px solid #cbd5e1;
             background: #f8fafc; cursor: pointer; font-size: 11px; font-weight: 600; }}
.vote-btn:hover {{ background: #e2e8f0; }}
.vote-btn.agree-active    {{ background: #dbeafe; color: #1d4ed8; border-color: #93c5fd; }}
.vote-btn.disagree-active {{ background: #fef3c7; color: #b45309; border-color: #fcd34d; }}
.item-detail {{ border-top: 1px solid #e2e8f0; padding: 12px 14px; background: #fafbfc; }}
.item-detail.collapsed {{ display: none; }}
.response-label {{ font-size: 10px; font-weight: 700; text-transform: uppercase;
                   letter-spacing: 0.06em; color: #64748b; margin-bottom: 6px; }}
.response-text {{ font-size: 12px; color: #334155; white-space: pre-wrap;
                  font-family: inherit; line-height: 1.6;
                  max-height: 420px; overflow-y: auto;
                  background: #fff; border: 1px solid #e2e8f0;
                  border-radius: 6px; padding: 10px 12px; }}
#empty-msg {{ text-align: center; padding: 60px; color: #94a3b8; font-size: 14px; }}
</style>
</head>
<body>

<div class="header">
  <h1>Extractor Visualizer — Hellaswag (May-24 runs)</h1>
  <div class="bench-tabs" id="bench-tabs"></div>
  <div class="run-buttons" id="run-buttons"></div>
  <div class="controls">
    <div class="filter-btns">
      <button class="filter-btn active" data-filter="all">All</button>
      <button class="filter-btn" data-filter="wrong">Wrong</button>
      <button class="filter-btn" data-filter="correct">Correct</button>
      <button class="filter-btn" data-filter="noextract">No extract</button>
      <button class="filter-btn" data-filter="disagreed">Disagreed</button>
    </div>
    <input id="search" type="text" placeholder="Search response…">
    <span class="stats" id="stats-label"></span>
    <button id="export-btn" style="margin-left:auto;padding:4px 12px;background:#0f172a;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;">Export votes</button>
  </div>
</div>

<div id="item-list"></div>
<div id="empty-msg" style="display:none">No items match the current filter.</div>

<script>
const ALL_DATA = {data_json};
const GROUPS   = {groups_json};

let currentFile   = null;
let currentBench  = null;
let currentFilter = 'all';
let currentSearch = '';
const votes = {{}};

function voteKey(id) {{ return currentFile + '__' + id; }}
function getVote(id) {{ return votes[voteKey(id)]; }}

function isCorrect(row) {{
  return row.correct.toLowerCase() === 'true' || row.correct === '1';
}}

function renderBenchTabs() {{
  const container = document.getElementById('bench-tabs');
  container.innerHTML = Object.keys(GROUPS).map(bench =>
    `<button class="bench-tab${{bench === currentBench ? ' active' : ''}}"
             onclick="selectBench('${{escAttr(bench)}}')">${{escHtml(bench)}}</button>`
  ).join('');
}}

function shortRunLabel(name) {{
  const sep = ' :: ';
  const i = name.indexOf(sep);
  return i >= 0 ? name.slice(i + sep.length) : name;
}}

function renderRunButtons() {{
  const container = document.getElementById('run-buttons');
  const runs = GROUPS[currentBench] || [];
  container.innerHTML = runs.map(name => {{
    const rows  = ALL_DATA[name] || [];
    const n     = rows.length;
    const nc    = rows.filter(r => isCorrect(r)).length;
    const noext = rows.filter(r => !r.answer).length;
    const extractRate = n ? (n - noext) / n : 1;
    const accRate     = n ? nc / n : 1;
    const warnCls = accRate < 0.5 ? 'warn-accuracy'
                  : extractRate < 0.8 ? 'warn-extraction'
                  : '';
    const activeCls = name === currentFile ? ' active' : '';
    const short = shortRunLabel(name);
    const label = noext > 0
      ? `${{escHtml(short)}} (${{nc}}/${{n}}, ${{noext}} no-extract)`
      : `${{escHtml(short)}} (${{nc}}/${{n}})`;
    return `<button class="run-btn ${{warnCls}}${{activeCls}}"
                    onclick="selectRun('${{escAttr(name)}}')" title="${{warnCls}}">${{label}}</button>`;
  }}).join('');
}}

function selectBench(bench) {{
  currentBench = bench;
  const runs = GROUPS[bench] || [];
  currentFile = runs[0] || null;
  renderBenchTabs();
  renderRunButtons();
  renderList();
}}

function selectRun(name) {{
  currentFile = name;
  renderRunButtons();
  renderList();
}}

function castVote(rowId, val, event) {{
  event.stopPropagation();
  const k = currentFile + '__' + rowId;
  if (votes[k] === val) {{
    delete votes[k];
  }} else {{
    votes[k] = val;
  }}
  const itemEl = document.getElementById('item-' + cssId(rowId));
  if (!itemEl) return;
  const v = votes[k];
  itemEl.className = 'item ' + itemClass(itemEl.dataset.ok === 'true', v);
  itemEl.querySelectorAll('.vote-btn').forEach(btn => {{
    btn.classList.remove('agree-active', 'disagree-active');
    if (v === 'agree')    btn.dataset.val === 'agree'    && btn.classList.add('agree-active');
    if (v === 'disagree') btn.dataset.val === 'disagree' && btn.classList.add('disagree-active');
  }});
  updateStats();
}}

function itemClass(ok, vote) {{
  if (vote === 'agree')    return 'voted-agree';
  if (vote === 'disagree') return 'voted-disagree';
  return ok ? 'correct-item' : 'wrong-item';
}}

function updateStats() {{
  const rows      = ALL_DATA[currentFile] || [];
  const shown     = document.querySelectorAll('#item-list .item').length;
  const correct   = rows.filter(r => isCorrect(r)).length;
  const noext     = rows.filter(r => !r.answer).length;
  const disagreed = Object.entries(votes)
    .filter(([k, v]) => k.startsWith(currentFile + '__') && v === 'disagree').length;
  document.getElementById('stats-label').textContent =
    `${{shown}} shown  ·  ${{correct}}/${{rows.length}} correct  ·  ${{noext}} no-extract  ·  ${{disagreed}} disagreed`;
}}

function renderList() {{
  const rows = ALL_DATA[currentFile] || [];
  const list  = document.getElementById('item-list');
  const empty = document.getElementById('empty-msg');
  const q = currentSearch.toLowerCase();

  const filtered = rows.filter(row => {{
    const ok   = isCorrect(row);
    const has  = !!row.answer;
    const vote = getVote(row.id);
    if (currentFilter === 'correct'   && !ok)                 return false;
    if (currentFilter === 'wrong'     && ok)                  return false;
    if (currentFilter === 'noextract' && has)                 return false;
    if (currentFilter === 'disagreed' && vote !== 'disagree') return false;
    if (q && !row.response.toLowerCase().includes(q) &&
             !row.id.toLowerCase().includes(q) &&
             !row.gold.toLowerCase().includes(q) &&
             !row.answer.toLowerCase().includes(q)) return false;
    return true;
  }});

  if (filtered.length === 0) {{
    list.innerHTML = '';
    empty.style.display = 'block';
    updateStats();
    return;
  }}
  empty.style.display = 'none';

  list.innerHTML = filtered.map(row => {{
    const ok         = isCorrect(row);
    const vote       = getVote(row.id);
    const cls        = itemClass(ok, vote);
    const badgeClass = ok ? 'c' : 'w';
    const badgeText  = ok ? '✓ correct' : '✗ wrong';
    const detailId   = 'detail-' + cssId(row.id);
    const ansDisplay = row.answer || '—';
    const ansColor   = !row.answer ? '#94a3b8' : (ok ? '#15803d' : '#dc2626');

    return `
<div class="item ${{cls}}" id="item-${{cssId(row.id)}}" data-ok="${{ok}}">
  <div class="item-header" onclick="toggleDetail('${{detailId}}')">
    <span class="item-id">#${{escHtml(row.id)}}</span>
    <span class="item-answer" style="color:${{ansColor}}">${{escHtml(ansDisplay)}}</span>
    <span class="item-gold">/ ${{escHtml(row.gold)}}</span>
    <span style="flex:1"></span>
    <span class="badge ${{badgeClass}}">${{badgeText}}</span>
    <div class="vote-btns">
      <button class="vote-btn${{vote === 'agree' ? ' agree-active' : ''}}" data-val="agree"
              onclick="castVote('${{escAttr(row.id)}}', 'agree', event)">👍 agree</button>
      <button class="vote-btn${{vote === 'disagree' ? ' disagree-active' : ''}}" data-val="disagree"
              onclick="castVote('${{escAttr(row.id)}}', 'disagree', event)">👎 disagree</button>
    </div>
  </div>
  <div class="item-detail" id="${{detailId}}">
    <div class="response-label">Model response</div>
    <div class="response-text">${{row.response ? escHtml(row.response) : '<em style="color:#94a3b8">not available</em>'}}</div>
  </div>
</div>`;
  }}).join('');

  updateStats();
}}

function toggleDetail(id) {{
  const el = document.getElementById(id);
  if (el) el.classList.toggle('collapsed');
}}

function escHtml(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}
function escAttr(s) {{
  return String(s).replace(/'/g, "\\\\'");
}}
function cssId(s) {{
  return String(s).replace(/[^a-zA-Z0-9_-]/g, '_');
}}

document.querySelectorAll('.filter-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter = btn.dataset.filter;
    renderList();
  }});
}});

let searchTimer;
document.getElementById('search').addEventListener('input', e => {{
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {{
    currentSearch = e.target.value;
    renderList();
  }}, 200);
}});

document.getElementById('export-btn').addEventListener('click', () => {{
  const out = {{}};
  for (const [k, v] of Object.entries(votes)) {{
    const sep = k.indexOf('__');
    const file = k.slice(0, sep);
    const id   = k.slice(sep + 2);
    if (!out[file]) out[file] = {{}};
    out[file][id] = v;
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'extractor_votes_hellaswag.json';
  a.click();
}});

currentBench = Object.keys(GROUPS)[0];
currentFile  = (GROUPS[currentBench] || [])[0] || null;
renderBenchTabs();
renderRunButtons();
renderList();
</script>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worst-only", action="store_true",
                    help="Keep only the worst run per condition (by no_extract + wrong)")
    args = ap.parse_args()

    files = load_files(worst_only=args.worst_only)
    total_runs = sum(len(v) for v in files.values())
    total_rows = sum(len(rows) for runs in files.values() for rows in runs.values())
    print(f"Conditions: {len(files)}, runs: {total_runs}, rows: {total_rows}")

    out_path = OUT.with_name(OUT.stem + ("_worst" if args.worst_only else "") + OUT.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(files), encoding="utf-8")
    print(f"Written: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
