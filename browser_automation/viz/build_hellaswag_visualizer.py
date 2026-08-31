#!/usr/bin/env python3
"""Rebuild hellaswag_rescore_visualizer.html from coverage CSVs + rescored interface CSVs."""

import csv, json, html as _html
from pathlib import Path

REPO     = Path(__file__).resolve().parents[2]
BENCH    = REPO / "experiments/metabench/metabench-hellaswag"
PLOTS    = REPO / "experiments/metabench/openllm_leaderboard/plots"
OUT      = Path(__file__).resolve().parents[1] / "outputs/hellaswag_rescore_visualizer.html"

COVERAGE_FILES = [
    (PLOTS / "coverage_by_run_gemini.csv", "data-gemini"),
    (PLOTS / "coverage_by_run_claude.csv", "data-claude"),
    (PLOTS / "coverage_by_run.csv",        "data-chatgpt"),
]

# (provider, coverage_condition) → display label, stable tab ID, csv stem to look for
COND_INFO = {
    ("data-gemini",  "Interface: Fast"):     ("Interface: Gemini 3 Flash",           "Gemini_Flash",    "fast"),
    ("data-gemini",  "Interface: Thinking"): ("Interface: Gemini 3 Flash (Thinking)","Gemini_Thinking", "thinking"),
    ("data-chatgpt", "Interface: Auto"):     ("Interface: GPT-5 Auto",               "GPT_Auto",        "instant"),
    ("data-chatgpt", "Interface: Instant"):  ("Interface: GPT-5.3 Instant",          "GPT_Instant",     "instant"),
    ("data-chatgpt", "Interface: Thinking"): ("Interface: GPT-5.4 Thinking",         "GPT_Thinking",    "thinking"),
    ("data-claude",  "Interface: Opus"):     ("Interface: Claude Opus 4.6",          "Claude_Opus",     "opus"),
    ("data-claude",  "Interface: Sonnet"):   ("Interface: Claude Sonnet 4.6",        "Claude_Sonnet",   "sonnet"),
    ("data-claude",  "Interface: Haiku"):    ("Interface: Claude Haiku 4.5",         "Claude_Haiku",    "haiku"),
}

TAB_ORDER = [
    "Gemini_Flash", "Gemini_Thinking",
    "GPT_Auto", "GPT_Instant", "GPT_Thinking",
    "Claude_Opus", "Claude_Sonnet", "Claude_Haiku",
]


def find_csv(timestamp: str, provider: str, stem: str) -> Path | None:
    iface_root = BENCH / provider / "interface"
    # Timestamp dir may have a suffix (e.g., "-gemini-hellaswag")
    candidates = list(iface_root.glob(f"{timestamp}*/"))
    if not candidates:
        return None
    ts_dir = candidates[0]
    for sess in sorted(ts_dir.iterdir()):
        if not sess.is_dir():
            continue
        # Exact match first
        p = sess / f"{stem}.csv"
        if p.exists():
            return p
        # Older runs use "{stem}-metabench-hellaswag.csv" etc.
        matches = list(sess.glob(f"{stem}*.csv"))
        if matches:
            return matches[0]
    return None


def read_response(json_path: str) -> str:
    try:
        p = Path(json_path)
        if not p.exists():
            return ""
        d = json.loads(p.read_text(encoding="utf-8"))
        return (d.get("ai_generated_output_text") or "").strip()
    except Exception:
        return ""


def read_csv_stats(csv_path: Path, n_samples: int = 4) -> dict:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    n_total = len(rows)
    if n_total == 0:
        return {}
    new_correct  = sum(1 for r in rows if r.get("correct", "").lower() == "true")
    extracted    = [r for r in rows if r.get("answer_llm", "").strip()]
    unextracted  = [r for r in rows if not r.get("answer_llm", "").strip()]
    n_extracted  = len(extracted)

    def make_sample(r: dict) -> dict:
        return {
            "id":        r.get("id", r.get("query_id", "?")),
            "gold":      r.get("gold_answer", "?"),
            "verdict":   r.get("answer_llm", "").strip() or "—",
            "correct":   r.get("correct", "").lower() == "true",
            "response":  read_response(r.get("path", "")),
        }

    return {
        "n_total":     n_total,
        "new_correct": new_correct,
        "n_extracted": n_extracted,
        "extracted_samples":   [make_sample(r) for r in extracted[:n_samples]],
        "unextracted_samples": [make_sample(r) for r in unextracted[:n_samples]],
    }


def load_tracked_runs() -> dict[str, list[dict]]:
    """Returns tab_id → list of run dicts (sorted by run_index)."""
    by_tab: dict[str, list[dict]] = {t: [] for t in TAB_ORDER}

    for cov_path, provider in COVERAGE_FILES:
        rows = list(csv.DictReader(cov_path.open(encoding="utf-8")))
        # Filter to hellaswag interface conditions
        rows = [r for r in rows
                if "hellaswag" in r.get("dataset", "").lower()
                and r.get("condition", "").startswith("Interface:")]
        for r in rows:
            key = (provider, r["condition"])
            if key not in COND_INFO:
                continue
            label, tab_id, stem = COND_INFO[key]
            timestamp = r["timestamp"]
            csv_path  = find_csv(timestamp, provider, stem)
            if csv_path is None:
                print(f"  WARN: no CSV for {timestamp} / {provider} / {stem}")
                continue
            stats = read_csv_stats(csv_path)
            if not stats:
                continue
            by_tab[tab_id].append({
                "run_index":  int(r["run_index"]),
                "timestamp":  timestamp,
                "old_acc":    float(r["accuracy"]),
                "csv_path":   csv_path,
                "label":      label,
                **stats,
            })

    # Sort each tab by run_index
    for tab_id in by_tab:
        by_tab[tab_id].sort(key=lambda x: x["run_index"])
    return by_tab


# ── HTML generation ────────────────────────────────────────────────────────

def pct(v: float) -> str:
    return f"{v*100:.1f}%"

def delta_class(d: float) -> str:
    if d > 0.005:  return "dpos"
    if d < -0.005: return "dneg"
    return "dneu"

def delta_str(d: float) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{d*100:.1f}pp"

def extr_color(rate: float) -> str:
    if rate >= 0.90: return "#34d399"   # green
    if rate >= 0.75: return "#fbbf24"   # amber
    return "#f87171"                    # red

def h(s: str) -> str:
    return _html.escape(str(s))

def render_item_row(item: dict, ext: bool) -> str:
    resp  = item["response"]
    summary = h(resp[:120].replace("\n", " ")) + ("…" if len(resp) > 120 else "")
    full    = h(resp[:2000])
    check   = "✓" if item["correct"] else "✗"
    css     = "ok" if item["correct"] else "fail"
    row_cls = "row-ext" if ext else "row-unext"
    return (
        f'<tr class="{row_cls}"><td class="c-id">{h(item["id"])}</td>'
        f'<td class="c-gold">{h(item["gold"])}</td>'
        f'<td class="c-verdict">{h(item["verdict"])}</td>'
        f'<td class="c-ok {css}">{check}</td>'
        f'<td class="c-resp"><details><summary>{summary}</summary>'
        f'<pre class="rpre">{full}</pre></details></td></tr>\n'
    )

def render_run(run: dict, tab_id: str) -> str:
    ri      = run["run_index"]
    detail_id = f"{tab_id}_r{ri}"
    new_acc = run["new_correct"] / run["n_total"] if run["n_total"] else 0
    old_acc = run["old_acc"]
    d       = new_acc - old_acc
    rate    = run["n_extracted"] / run["n_total"] if run["n_total"] else 0
    color   = extr_color(rate)
    bar_w   = int(rate * 120)
    ts_short = run["timestamp"][:16].replace("_", " ")

    # Find relative path for title tooltip
    try:
        rel = run["csv_path"].relative_to(REPO / "experiments/metabench")
    except ValueError:
        rel = run["csv_path"]

    row = (
        f'<tr class="run-row" onclick="toggleDetail(\'{detail_id}\')">\n'
        f'  <td class="c-run">Run {ri}</td>\n'
        f'  <td class="c-old">{pct(old_acc)}</td>\n'
        f'  <td class="c-new">{pct(new_acc)}</td>\n'
        f'  <td class="{delta_class(d)}">{delta_str(d)}</td>\n'
        f'  <td class="c-ext">\n'
        f'    <div class="bar-w"><div class="bar-i" style="width:{bar_w}px;background:{color}"></div></div>\n'
        f'    <span style="color:{color};font-weight:700">{pct(rate)}</span>\n'
        f'    ({run["n_extracted"]}/{run["n_total"]})\n'
        f'  </td>\n'
        f'  <td class="c-ts" title="{h(str(rel))}">{h(ts_short)}</td>\n'
        f'</tr>\n'
    )

    # Detail section
    items_html = '<table class="item-tbl"><thead><tr><th>ID</th><th>Gold</th><th>Extracted</th><th></th><th>Response</th></tr></thead>\n<tbody>'
    for item in run["extracted_samples"]:
        items_html += render_item_row(item, ext=True)
    if run["unextracted_samples"]:
        items_html += "<tr class='sep-row'><td colspan='5'>── unextracted ──</td></tr>"
        for item in run["unextracted_samples"]:
            items_html += render_item_row(item, ext=False)
    items_html += "</tbody></table>"

    detail = (
        f'<tr class="detail-row" id="{detail_id}" style="display:none">\n'
        f'  <td colspan="6" class="detail-cell">\n'
        f'    {items_html}\n'
        f'  </td>\n'
        f'</tr>\n'
    )
    return row + detail


def render_panel(tab_id: str, label: str, runs: list[dict]) -> str:
    if not runs:
        return f'<div class="panel" id="panel-{tab_id}"><p style="color:#888">No tracked runs found.</p></div>\n'

    avg_old = sum(r["old_acc"] for r in runs) / len(runs)
    avg_new = sum(r["new_correct"] / r["n_total"] for r in runs if r["n_total"]) / len(runs)
    avg_ext = sum(r["n_extracted"] / r["n_total"] for r in runs if r["n_total"]) / len(runs)
    d = avg_new - avg_old
    d_cls = delta_class(d)

    rows_html = "".join(render_run(run, tab_id) for run in runs)

    return (
        f'<div class="panel" id="panel-{tab_id}">\n'
        f'  <div class="summary-bar">\n'
        f'    <span class="s-cond">{h(label)}</span>\n'
        f'    <span class="s-stat">avg old: <b>{pct(avg_old)}</b></span>\n'
        f'    <span class="s-stat">avg new: <b>{pct(avg_new)}</b></span>\n'
        f'    <span class="s-stat {d_cls}">Δ {delta_str(d)}</span>\n'
        f'    <span class="s-stat">extraction: <b>{pct(avg_ext)}</b></span>\n'
        f'    <span class="s-hint">Click a run to expand samples</span>\n'
        f'  </div>\n'
        f'  <table class="run-tbl">\n'
        f'    <thead><tr><th>Run</th><th>Old acc</th><th>New acc</th><th>Δ</th><th>Extraction rate</th><th>Timestamp</th></tr></thead>\n'
        f'    <tbody>\n{rows_html}    </tbody>\n'
        f'  </table>\n'
        f'</div>\n'
    )


CSS = """
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:16px}
h1{font-size:1.3rem;margin:0 0 4px}p.sub{color:#94a3b8;font-size:.85rem;margin:0 0 12px}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.tb{background:#1e293b;border:1px solid #334155;color:#cbd5e1;border-radius:6px;
    padding:6px 14px;cursor:pointer;font-size:.8rem}
.tb.act{background:#3b82f6;border-color:#3b82f6;color:#fff}
.panel{display:none}.panel.act{display:block}
.summary-bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;
    background:#1e293b;border-radius:8px;padding:10px 16px;margin-bottom:10px}
.s-cond{font-weight:700;font-size:1rem;color:#f1f5f9}
.s-stat{font-size:.85rem;color:#94a3b8}.s-stat b{color:#e2e8f0}
.s-stat.dpos b,.dpos{color:#34d399}.s-stat.dneg b,.dneg{color:#f87171}.dneu{color:#94a3b8}
.s-hint{font-size:.78rem;color:#475569;margin-left:auto}
.run-tbl{width:100%;border-collapse:collapse;font-size:.82rem;margin-bottom:16px}
.run-tbl th{background:#1e293b;color:#94a3b8;padding:6px 8px;text-align:left;
    border-bottom:1px solid #334155}
.run-row{cursor:pointer;border-bottom:1px solid #1e293b}
.run-row:hover{background:#1e293b}
.run-row td{padding:6px 8px}
.c-old,.c-new{font-variant-numeric:tabular-nums}
.c-ext{display:flex;align-items:center;gap:6px;white-space:nowrap}
.bar-w{width:120px;height:8px;background:#1e293b;border-radius:4px;flex-shrink:0}
.bar-i{height:8px;border-radius:4px}
.c-ts{color:#64748b;font-size:.78rem;max-width:160px;overflow:hidden;text-overflow:ellipsis}
.detail-row td{padding:0 8px 12px 8px}
.detail-cell{background:#0f172a}
.item-tbl{width:100%;border-collapse:collapse;font-size:.78rem}
.item-tbl th{color:#64748b;padding:4px 6px;text-align:left;border-bottom:1px solid #1e293b}
.item-tbl td{padding:3px 6px;vertical-align:top;border-bottom:1px solid #0f172a}
.c-id{color:#64748b;width:40px}.c-gold,.c-verdict{font-weight:700;width:50px}
.c-ok{width:20px}.ok{color:#34d399}.fail{color:#f87171}
.c-resp{max-width:600px}
details summary{cursor:pointer;color:#93c5fd;font-size:.75rem;white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis;max-width:580px;display:block}
.rpre{white-space:pre-wrap;word-break:break-word;font-size:.72rem;color:#cbd5e1;
    background:#1e293b;padding:8px;border-radius:4px;margin:4px 0;max-height:300px;overflow-y:auto}
.sep-row td{color:#475569;font-size:.75rem;padding:4px 6px}
.row-unext .c-verdict{color:#f87171}.row-unext .c-ok{color:#f87171}
"""

JS = """
function show(id){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('act'));
  document.querySelectorAll('.tb').forEach(b=>b.classList.remove('act'));
  document.getElementById('panel-'+id).classList.add('act');
  document.getElementById('btn-'+id).classList.add('act');
}
function toggleDetail(id){
  var r=document.getElementById(id);
  r.style.display=r.style.display==='none'?'table-row':'none';
}
"""

def build_html(by_tab: dict) -> str:
    # Tab buttons
    first = True
    btn_html = ""
    for tab_id in TAB_ORDER:
        runs = by_tab.get(tab_id, [])
        if not runs:
            continue
        label = runs[0]["label"]
        act = " act" if first else ""
        btn_html += f'<button class="tb{act}" onclick="show(\'{tab_id}\')" id="btn-{tab_id}">{h(label)}</button>'
        first = False

    # Panels
    panels_html = ""
    first = True
    for tab_id in TAB_ORDER:
        runs = by_tab.get(tab_id, [])
        if not runs:
            continue
        label = runs[0]["label"]
        panel = render_panel(tab_id, label, runs)
        if first:
            panel = panel.replace('<div class="panel"', '<div class="panel act"', 1)
            first = False
        panels_html += panel

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>HellaSwag Rescore Visualizer</title>
<style>{CSS}</style>
</head>
<body>
<h1>HellaSwag Interface Rescore</h1>
<p class="sub">Context-aware gpt-4o-mini judge. Old acc = from coverage CSV (pre-rescore). New acc = after batch. Click any run row to see extracted + unextracted sample items.</p>
<div class="tabs">{btn_html}</div>
{panels_html}
<script>{JS}</script>
</body></html>
"""


def main():
    print("Loading coverage CSVs and rescored CSVs...")
    by_tab = load_tracked_runs()
    total = sum(len(v) for v in by_tab.values())
    print(f"Found {total} tracked run entries across {sum(1 for v in by_tab.values() if v)} conditions")
    for tab_id in TAB_ORDER:
        runs = by_tab.get(tab_id, [])
        if runs:
            print(f"  {runs[0]['label']}: {len(runs)} runs")

    print("Building HTML...")
    html_out = build_html(by_tab)
    OUT.write_text(html_out, encoding="utf-8")
    print(f"Written: {OUT}")


if __name__ == "__main__":
    main()
