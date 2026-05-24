"""Build a self-contained HTML tool to review and validate extractor judgments.

Embeds all aa-omniscience outputs-200 CSVs as JSON, with:
  - File picker dropdown
  - Filter: All / Wrong / Correct / Flagged
  - Per-item: question, gold answer, response, extractor verdict
  - Click to expand full response
  - Override button to flip correct/wrong for manual validation
  - Export overrides as JSON

Usage:
  python automated-scraper/build_annotation_visualizer.py
"""
from __future__ import annotations
import csv, json, html as _html
from pathlib import Path

BASE    = Path(__file__).resolve().parent
REPO    = BASE.parent
DATA_DIR = REPO / "experiments" / "aa-omniscience" / "outputs-200"
OUT     = BASE / "outputs" / "annotation_visualizer.html"


def load_files() -> dict[str, list[dict]]:
    """Return {filename: [rows]} sorted by filename."""
    result = {}
    for p in sorted(DATA_DIR.glob("*.csv")):
        rows = []
        for r in csv.DictReader(open(p, encoding="utf-8")):
            rows.append({
                "id":          r.get("id", "").strip(),
                "query":       r.get("query", "").strip(),
                "gold":        r.get("gold_answer", "").strip(),
                "response":    r.get("response", "").strip(),
                "correct":     r.get("correct", "").strip(),
            })
        result[p.name] = rows
    return result


def build_html(files: dict[str, list[dict]]) -> str:
    data_json = json.dumps(files, ensure_ascii=False)

    # Build option groups by provider
    providers: dict[str, list[str]] = {}
    for name in files:
        provider = name.split("__")[0]
        providers.setdefault(provider, []).append(name)

    options_html = ""
    for prov, names in sorted(providers.items()):
        options_html += f'<optgroup label="{_html.escape(prov)}">'
        for name in names:
            rows = files[name]
            n = len(rows)
            nc = sum(1 for r in rows if r["correct"] in ("1", "True"))
            label = name.replace(".csv", "")
            options_html += f'<option value="{_html.escape(name)}">{_html.escape(label)} ({nc}/{n})</option>'
        options_html += '</optgroup>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Extractor Validation</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: system-ui, sans-serif; font-size: 13px; background: #f8fafc; color: #1e293b; }}

/* ── Header ── */
.header {{ background: #fff; border-bottom: 1px solid #e2e8f0; padding: 12px 20px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; position: sticky; top: 0; z-index: 20; }}
.header h1 {{ font-size: 15px; font-weight: 700; white-space: nowrap; }}
select#file-picker {{ max-width: 480px; font-size: 12px; padding: 5px 8px; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; }}
.filter-btns {{ display: flex; gap: 4px; }}
.filter-btn {{ padding: 5px 12px; border: 1px solid #cbd5e1; border-radius: 6px; background: #f1f5f9; cursor: pointer; font-size: 12px; font-weight: 500; }}
.filter-btn.active {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
#search {{ padding: 5px 10px; border: 1px solid #cbd5e1; border-radius: 6px; width: 200px; font-size: 12px; }}
.stats {{ font-size: 12px; color: #64748b; white-space: nowrap; }}
#export-btn {{ margin-left: auto; padding: 5px 14px; background: #0f172a; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; }}

/* ── List ── */
#item-list {{ padding: 12px 20px; display: flex; flex-direction: column; gap: 6px; }}
.item {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
.item.correct-item {{ border-left: 3px solid #16a34a; }}
.item.wrong-item   {{ border-left: 3px solid #dc2626; }}
.item.overridden   {{ border-left: 3px solid #f59e0b; }}

.item-header {{ display: flex; align-items: flex-start; gap: 10px; padding: 10px 14px; cursor: pointer; }}
.item-header:hover {{ background: #f8fafc; }}
.item-id   {{ font-size: 11px; color: #94a3b8; min-width: 28px; padding-top: 2px; }}
.item-query {{ flex: 1; color: #1e293b; font-size: 12px; line-height: 1.5; }}
.item-gold  {{ font-size: 12px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; padding: 2px 8px; white-space: nowrap; color: #475569; }}
.gold-label {{ font-size: 10px; color: #94a3b8; margin-right: 4px; }}
.badge {{ padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; white-space: nowrap; }}
.badge.c  {{ background: #dcfce7; color: #16a34a; }}
.badge.w  {{ background: #fee2e2; color: #dc2626; }}
.badge.ov {{ background: #fef3c7; color: #d97706; }}
.override-btn {{ padding: 3px 10px; border-radius: 6px; border: 1px solid #cbd5e1; background: #f8fafc; cursor: pointer; font-size: 11px; font-weight: 600; white-space: nowrap; color: #475569; }}
.override-btn:hover {{ background: #e2e8f0; }}

.item-detail {{ display: none; border-top: 1px solid #e2e8f0; padding: 12px 14px; background: #fafbfc; }}
.item-detail.open {{ display: block; }}
.detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.detail-block {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 12px; }}
.detail-label {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; margin-bottom: 6px; }}
.detail-text {{ font-size: 12px; color: #334155; white-space: pre-wrap; font-family: inherit; line-height: 1.6; max-height: 300px; overflow-y: auto; }}
.gold-full {{ font-size: 13px; font-weight: 700; color: #0f172a; }}

#empty-msg {{ text-align: center; padding: 60px; color: #94a3b8; font-size: 14px; }}
</style>
</head>
<body>

<div class="header">
  <h1>Extractor Validation</h1>
  <select id="file-picker">{options_html}</select>
  <div class="filter-btns">
    <button class="filter-btn active" data-filter="all">All</button>
    <button class="filter-btn" data-filter="wrong">Wrong</button>
    <button class="filter-btn" data-filter="correct">Correct</button>
    <button class="filter-btn" data-filter="flagged">Flagged</button>
  </div>
  <input id="search" type="text" placeholder="Search query / response…">
  <span class="stats" id="stats-label"></span>
  <button id="export-btn">Export overrides</button>
</div>

<div id="item-list"></div>
<div id="empty-msg" style="display:none">No items match the current filter.</div>

<script>
const ALL_DATA = {data_json};
const overrides = {{}};  // id → 'correct' | 'wrong' (overrides extractor)

let currentFile = null;
let currentFilter = 'all';
let currentSearch = '';

function isCorrect(row) {{
  if (overrides[currentFile + '__' + row.id] !== undefined) {{
    return overrides[currentFile + '__' + row.id] === 'correct';
  }}
  return row.correct === '1' || row.correct === 'True';
}}

function isOverridden(row) {{
  return overrides[currentFile + '__' + row.id] !== undefined;
}}

function renderList() {{
  const rows = ALL_DATA[currentFile] || [];
  const list = document.getElementById('item-list');
  const empty = document.getElementById('empty-msg');
  const q = currentSearch.toLowerCase();

  const filtered = rows.filter(row => {{
    const ok = isCorrect(row);
    const ov = isOverridden(row);
    if (currentFilter === 'correct' && !ok) return false;
    if (currentFilter === 'wrong'   && ok)  return false;
    if (currentFilter === 'flagged' && !ov) return false;
    if (q && !row.query.toLowerCase().includes(q) &&
             !row.response.toLowerCase().includes(q) &&
             !row.gold.toLowerCase().includes(q)) return false;
    return true;
  }});

  const correct = rows.filter(r => isCorrect(r)).length;
  const flagged = Object.keys(overrides).filter(k => k.startsWith(currentFile + '__')).length;
  document.getElementById('stats-label').textContent =
    `${{filtered.length}} shown  ·  ${{correct}}/${{rows.length}} correct  ·  ${{flagged}} overridden`;

  if (filtered.length === 0) {{
    list.innerHTML = '';
    empty.style.display = 'block';
    return;
  }}
  empty.style.display = 'none';

  list.innerHTML = filtered.map(row => {{
    const ok = isCorrect(row);
    const ov = isOverridden(row);
    const badgeClass = ov ? 'ov' : ok ? 'c' : 'w';
    const badgeText  = ov ? '★ override' : ok ? '✓ correct' : '✗ wrong';
    const itemClass  = ov ? 'overridden' : ok ? 'correct-item' : 'wrong-item';
    const btnText    = ok ? 'Mark wrong' : 'Mark correct';
    const qShort = row.query.length > 180 ? row.query.slice(0,180) + '…' : row.query;
    const detailId = 'detail-' + CSS.escape(row.id);
    return `
<div class="item ${{itemClass}}" id="item-${{CSS.escape(row.id)}}">
  <div class="item-header" onclick="toggleDetail('${{detailId}}')">
    <span class="item-id">#${{row.id}}</span>
    <span class="item-query">${{escHtml(qShort)}}</span>
    <span class="item-gold"><span class="gold-label">gold</span>${{escHtml(row.gold)}}</span>
    <span class="badge ${{badgeClass}}">${{badgeText}}</span>
    <button class="override-btn" onclick="event.stopPropagation(); toggleOverride('${{row.id}}')">${{btnText}}</button>
  </div>
  <div class="item-detail" id="${{detailId}}">
    <div class="detail-grid">
      <div class="detail-block">
        <div class="detail-label">Question</div>
        <div class="detail-text">${{escHtml(row.query)}}</div>
      </div>
      <div class="detail-block">
        <div class="detail-label">Model response</div>
        <div class="detail-text">${{escHtml(row.response)}}</div>
      </div>
    </div>
    <div style="margin-top:10px" class="detail-block">
      <div class="detail-label">Gold answer</div>
      <div class="gold-full">${{escHtml(row.gold)}}</div>
    </div>
  </div>
</div>`;
  }}).join('');
}}

function toggleDetail(id) {{
  const el = document.getElementById(id);
  if (el) el.classList.toggle('open');
}}

function toggleOverride(rowId) {{
  const key = currentFile + '__' + rowId;
  const row = (ALL_DATA[currentFile] || []).find(r => r.id === rowId);
  if (!row) return;
  if (overrides[key] !== undefined) {{
    delete overrides[key];
  }} else {{
    overrides[key] = isCorrect(row) ? 'wrong' : 'correct';
  }}
  renderList();
}}

function escHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// File picker
document.getElementById('file-picker').addEventListener('change', e => {{
  currentFile = e.target.value;
  renderList();
}});

// Filter buttons
document.querySelectorAll('.filter-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter = btn.dataset.filter;
    renderList();
  }});
}});

// Search
let searchTimer;
document.getElementById('search').addEventListener('input', e => {{
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {{
    currentSearch = e.target.value;
    renderList();
  }}, 200);
}});

// Export
document.getElementById('export-btn').addEventListener('click', () => {{
  const out = {{}};
  for (const [key, val] of Object.entries(overrides)) {{
    const [file, ...rest] = key.split('__');
    const id = rest.join('__');
    if (!out[file]) out[file] = {{}};
    out[file][id] = val;
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'extractor_overrides.json';
  a.click();
}});

// Init
currentFile = document.getElementById('file-picker').value;
renderList();
</script>
</body>
</html>"""


def main():
    print(f"Loading {DATA_DIR}...")
    files = load_files()
    print(f"Loaded {len(files)} files, {sum(len(v) for v in files.values())} total rows")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(files)
    OUT.write_text(html, encoding="utf-8")
    print(f"Written: {OUT}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
