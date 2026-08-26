"""Build a self-contained HTML visualizer for extraction pipeline results.

Shows, for each model response: full original response text, final extracted
answer, gold answer, and verdict.  Only includes complete runs used in the paper
(complete_run=True in coverage_by_run_claude.csv).

Usage:
  python automated-scraper/build_extractor_visualizer.py
"""
from __future__ import annotations
import csv, json, re, html as _html
from pathlib import Path

BASE      = Path(__file__).resolve().parents[1]
REPO      = BASE.parent
META_DIR  = REPO / "experiments" / "metabench"
COVERAGE  = META_DIR / "openllm_leaderboard" / "plots" / "coverage_by_run_claude.csv"
OUT       = BASE / "outputs" / "extractor_visualizer.html"




def load_paper_runs() -> dict[str, list[str]]:
    """Return {dataset: [timestamp, ...]} for complete Interface: Haiku paper runs."""
    paper: dict[str, list[str]] = {}
    with open(COVERAGE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("condition") == "Interface: Haiku"
                    and r.get("complete_run") == "True"):
                paper.setdefault(r["dataset"], []).append(r["timestamp"])
    return paper


def fix_path(p: str) -> Path:
    p = re.sub(r"/experiments/metabench-([^/]+)/",
               r"/experiments/metabench/metabench-\1/", p)
    return Path(p)


def load_response(path_str: str) -> str:
    if not path_str:
        return ""
    fp = fix_path(path_str)
    if not fp.exists():
        return ""
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
        return d.get("ai_generated_output_text") or d.get("response_text") or ""
    except Exception:
        return ""


def load_files(paper_runs: dict[str, list[str]]) -> tuple[dict[str, list[dict]], list[str]]:
    """Return ({run_label: [rows]}, [missing_label, ...]) for paper runs."""
    result: dict[str, list[dict]] = {}
    missing: list[str] = []

    # Index what's on disk
    on_disk: dict[tuple[str, str], Path] = {}
    for csv_path in META_DIR.rglob("*/interface/*/session_*/haiku.csv"):
        parts = csv_path.parts
        try:
            bench = parts[parts.index("metabench") + 1]
            ts    = parts[parts.index("interface") + 1]
        except ValueError:
            continue
        on_disk[(bench, ts)] = csv_path

    for bench, timestamps in sorted(paper_runs.items()):
        for ts in timestamps:
            label = f"{bench} / {ts}"
            csv_path = on_disk.get((bench, ts))
            if csv_path is None:
                missing.append(label)
                continue

            with open(csv_path, encoding="utf-8") as f:
                raw_rows = list(csv.DictReader(f))

            rows = []
            for r in raw_rows:
                response = load_response(r.get("path", ""))
                rows.append({
                    "id":       r.get("id", "").strip(),
                    "gold":     r.get("gold_answer", "").strip(),
                    "answer":   r.get("answer", "").strip(),
                    "correct":  r.get("correct", "").strip(),
                    "response": response,
                })

            result[label] = rows

    return result, missing


def build_html(files: dict[str, list[dict]], missing: list[str]) -> str:
    data_json  = json.dumps(files, ensure_ascii=False)
    if missing:
        items = "".join(f"<li>{_html.escape(m)}</li>" for m in sorted(missing))
        missing_banner = (
            f'<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;'
            f'padding:6px 12px;font-size:12px;color:#991b1b;margin-bottom:6px;">'
            f'<strong>⚠ Missing from disk ({len(missing)}):</strong> '
            f'<ul style="margin:2px 0 0 16px;padding:0">{items}</ul></div>'
        )
    else:
        missing_banner = ""

    # Build bench → sorted list of run keys
    groups: dict[str, list[str]] = {}
    for name in sorted(files):
        bench = name.split(" / ")[0]
        groups.setdefault(bench, []).append(name)

    # Render benchmark tabs + run buttons as JSON for JS
    groups_json = json.dumps(
        {b: names for b, names in sorted(groups.items())},
        ensure_ascii=False,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Extractor Visualizer</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: system-ui, sans-serif; font-size: 13px; background: #f8fafc; color: #1e293b; }}

/* ── Sticky header ── */
.header {{ background: #fff; border-bottom: 1px solid #e2e8f0;
           padding: 10px 20px; position: sticky; top: 0; z-index: 20; }}
.header h1 {{ font-size: 14px; font-weight: 700; margin-bottom: 8px; }}

/* bench tabs */
.bench-tabs {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
.bench-tab {{ padding: 4px 12px; border: 1px solid #cbd5e1; border-radius: 6px;
              background: #f1f5f9; cursor: pointer; font-size: 12px; font-weight: 500; }}
.bench-tab.active {{ background: #1e40af; color: #fff; border-color: #1e40af; }}

/* run buttons */
.run-buttons {{ display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 8px; }}
.run-btn {{ padding: 3px 10px; border: 1px solid #cbd5e1; border-radius: 6px;
            background: #f8fafc; cursor: pointer; font-size: 11px; font-weight: 500;
            white-space: nowrap; }}
.run-btn.active  {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
.run-btn.warn-extraction {{ background: #fff7ed; border-color: #fdba74; color: #9a3412; }}
.run-btn.warn-accuracy   {{ background: #fef2f2; border-color: #fca5a5; color: #991b1b; }}
.run-btn.warn-extraction.active,
.run-btn.warn-accuracy.active {{ background: #2563eb; color: #fff; border-color: #2563eb; }}

/* filter + search row */
.controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.filter-btns {{ display: flex; gap: 4px; }}
.filter-btn {{ padding: 4px 11px; border: 1px solid #cbd5e1; border-radius: 6px;
               background: #f1f5f9; cursor: pointer; font-size: 12px; font-weight: 500; }}
.filter-btn.active {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
#search {{ padding: 4px 10px; border: 1px solid #cbd5e1; border-radius: 6px;
           width: 200px; font-size: 12px; }}
.stats {{ font-size: 12px; color: #64748b; white-space: nowrap; }}

/* ── Item list ── */
#item-list {{ padding: 10px 20px; display: flex; flex-direction: column; gap: 5px; }}
.item {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
.item.correct-item {{ border-left: 3px solid #16a34a; }}
.item.wrong-item   {{ border-left: 3px solid #dc2626; }}
.item.voted-agree    {{ border-left: 3px solid #2563eb; }}
.item.voted-disagree {{ border-left: 3px solid #f59e0b; }}

.item-header {{ display: flex; align-items: center; gap: 10px;
                padding: 8px 14px; cursor: pointer; user-select: none; }}
.item-header:hover {{ background: #f8fafc; }}
.item-id {{ font-size: 11px; color: #94a3b8; min-width: 28px; }}
.item-answer {{ font-size: 13px; font-weight: 700; }}
.item-gold {{ font-size: 12px; color: #64748b; }}
.badge {{ padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700; white-space: nowrap; }}
.badge.c {{ background: #dcfce7; color: #16a34a; }}
.badge.w {{ background: #fee2e2; color: #dc2626; }}

/* vote buttons */
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
  <h1>Extractor Visualizer — paper runs</h1>
  {missing_banner}
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
const votes = {{}};  // key: file+'__'+id  →  'agree' | 'disagree'

function voteKey(id) {{ return currentFile + '__' + id; }}
function getVote(id) {{ return votes[voteKey(id)]; }}

function isCorrect(row) {{
  return row.correct.toLowerCase() === 'true' || row.correct === '1';
}}

function renderBenchTabs() {{
  const container = document.getElementById('bench-tabs');
  container.innerHTML = Object.keys(GROUPS).map(bench =>
    `<button class="bench-tab${{bench === currentBench ? ' active' : ''}}"
             onclick="selectBench('${{bench}}')">${{escHtml(bench)}}</button>`
  ).join('');
}}

function renderRunButtons() {{
  const container = document.getElementById('run-buttons');
  const runs = GROUPS[currentBench] || [];
  container.innerHTML = runs.map(name => {{
    const rows  = ALL_DATA[name] || [];
    const n     = rows.length;
    const nc    = rows.filter(r => isCorrect(r)).length;
    const noext = rows.filter(r => !r.answer).length;
    const ts    = name.split(' / ')[1] || name;
    const extractRate = n ? (n - noext) / n : 1;
    const accRate     = n ? nc / n : 1;
    // warn-accuracy takes priority (worse signal); warn-extraction if extraction < 80%
    const warnCls = accRate < 0.5 ? 'warn-accuracy'
                  : extractRate < 0.8 ? 'warn-extraction'
                  : '';
    const activeCls = name === currentFile ? ' active' : '';
    const label = noext > 0
      ? `${{escHtml(ts)}} (${{nc}}/${{n}}, ${{noext}} no-extract)`
      : `${{escHtml(ts)}} (${{nc}}/${{n}})`;
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
    delete votes[k];   // toggle off
  }} else {{
    votes[k] = val;
  }}
  // patch just this item's classes and buttons in-place
  const itemEl = document.getElementById('item-' + rowId);
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
    if (currentFilter === 'correct'   && !ok)              return false;
    if (currentFilter === 'wrong'     && ok)               return false;
    if (currentFilter === 'noextract' && has)              return false;
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
    const detailId   = 'detail-' + row.id;
    const ansDisplay = row.answer || '—';
    const ansColor   = !row.answer ? '#94a3b8' : (ok ? '#15803d' : '#dc2626');

    return `
<div class="item ${{cls}}" id="item-${{escHtml(row.id)}}" data-ok="${{ok}}">
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
  return String(s).replace(/'/g, "\\'");
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
  a.download = 'extractor_votes.json';
  a.click();
}});

// Init
currentBench = Object.keys(GROUPS)[0];
currentFile  = (GROUPS[currentBench] || [])[0] || null;
renderBenchTabs();
renderRunButtons();
renderList();
</script>
</body>
</html>"""


def main():
    paper_runs = load_paper_runs()
    files, missing = load_files(paper_runs)
    total_expected = sum(len(v) for v in paper_runs.values())
    print(f"Paper runs: {total_expected} expected, {len(files)} on disk, {len(missing)} missing")
    if missing:
        for m in sorted(missing):
            print(f"  MISSING: {m}")
    print(f"Rows: {sum(len(v) for v in files.values())} total")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(files, missing)
    OUT.write_text(html, encoding="utf-8")
    print(f"Written: {OUT}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
