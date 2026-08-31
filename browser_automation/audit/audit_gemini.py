"""Browser scraper for gemini.google.com — mirrors audit_claude.py's structure.

Supports:
  - Sending queries from CSV/JSON files via the Gemini web UI
  - Multi-session parallelism (one Chrome profile per session)
  - Saving raw HTML + metadata (same schema as audit.py / audit_claude.py)
  - Clearing conversation history between runs
  - Model selection (Gemini 2.0 Flash, Gemini 2.5 Pro, etc.)
  - YAML experiment configs (same format as exp.yaml)
  - API-only mode via api_runner.py (Google Generative AI SDK)

Usage:
    python audit_gemini.py exp_gemini.yaml
    python audit_gemini.py exp_gemini.yaml --sessions 2
    python audit_gemini.py exp_gemini.yaml --clear-memory
    python audit_gemini.py exp_gemini.yaml --interface-model "2.5 Pro"
    python audit_gemini.py exp_gemini.yaml --api-models gemini-2.5-pro-preview-05-06
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import re
import sys
import threading
import time
import yaml
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from DrissionPage import ChromiumPage, ChromiumOptions
from DrissionPage.common import Keys

# This script lives in audit/; api_runner.py sits at the automated-scraper
# root, so put that root on sys.path before importing from it.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from api_runner import run_api_query

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from parse_raw_run import parse_gemini_run as _parse_gemini_run

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "gemini_data"
# Where auto-parse writes Gemini JSON by default.
# Keep this scoped to the scraper's own data directory so we don't
# directly write into any experiments/* trees (e.g. metabench-mmlu).
_GEMINI_OUTPUT_ROOT = DATA_DIR / "parsed_json"

GEMINI_URL = "https://gemini.google.com"
GEMINI_APP_URL = "https://gemini.google.com/app"
GEMINI_ERROR_13_TEXT = "Something went wrong (13)"
GEMINI_ERROR_13_PAUSE_SECONDS = 15 * 60
THINKING_FAST_RETRY_SECONDS = 20 * 60
INTERVAL_BREAK_EVERY_QUERIES = 0
INTERVAL_BREAK_SECONDS = 10 * 60

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


# ── Timestamped logging ───────────────────────────────────────────────────────

class _TimestampedTee:
    def __init__(self, stream, log_handles):
        self._stream = stream
        self._logs = list(log_handles)
        self._at_line_start = True

    def write(self, data):
        if not data:
            return
        self._stream.write(data)
        self._stream.flush()
        for chunk in data.splitlines(True):
            if self._at_line_start:
                ts = datetime.now().isoformat()
                for log in self._logs:
                    log.write(f"[{ts}] ")
            for log in self._logs:
                log.write(chunk)
            self._at_line_start = chunk.endswith("\n")
        for log in self._logs:
            log.flush()

    def flush(self):
        self._stream.flush()
        for log in self._logs:
            log.flush()


def _setup_timestamped_logging(run_id, session_id):
    log_dir = DATA_DIR / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    combined_log_path = log_dir / f"gemini_session_{session_id:02d}.log"
    combined_handle = open(combined_log_path, "a", encoding="utf-8")
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = _TimestampedTee(stdout_orig, [combined_handle])
    sys.stderr = _TimestampedTee(stderr_orig, [combined_handle])
    return combined_log_path, stdout_orig, stderr_orig, [combined_handle]


def _resolve_user_data_dir(profile_base, session_id, total_sessions):
    base = Path(profile_base) if profile_base else (DATA_DIR / "chrome_profiles_gemini")
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"session_{session_id:02d}")


class FileBasedBarrier:
    """Simple file-based barrier for multiprocessing synchronization."""
    def __init__(self, parties, sync_dir, session_id, barrier_id=0):
        self.parties = parties
        self.sync_dir = Path(sync_dir)
        self.session_id = session_id
        self.barrier_id = barrier_id
        self.sync_dir.mkdir(parents=True, exist_ok=True)

    def wait(self):
        """Wait indefinitely for all parties to reach the barrier."""
        checkpoint_file = self.sync_dir / f"barrier_{self.barrier_id}_session_{self.session_id}.ready"
        checkpoint_file.write_text(str(time.time()))

        while True:
            ready_count = len(list(self.sync_dir.glob(f"barrier_{self.barrier_id}_session_*.ready")))
            if ready_count >= self.parties:
                time.sleep(0.1)
                break
            time.sleep(0.2)

        self.barrier_id += 1


# ── Query loading ─────────────────────────────────────────────────────────────

def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value) if value is not None else None


def _detect_is_mcq(item):
    id_val = str(item.get("id", ""))
    if "_mcq" in id_val.lower():
        return True
    for key in ("is_mcq", "mcq"):
        if key in item:
            return _coerce_bool(item.get(key))
    if item.get("correct_answer") or item.get("choices") or item.get("options"):
        return True
    return None


def _api_model_dir_name(api_cfg):
    """Reproduce the output directory name for an API model config dict."""
    m_name = api_cfg["model"]
    m_params = {k: v for k, v in api_cfg.items() if k != "model"}
    return m_name + ("_" + "_".join(f"{k}-{v}" for k, v in sorted(m_params.items())) if m_params else "")


def find_resume_point(data_dir, query_file, config=None):
    """Return a filtered query list with already-completed queries removed.

    Scans the UNION of completions across ALL prior run dirs (not just the latest),
    so parallel/restarted jobs don't lose track of work done in earlier sessions.

    A query counts as completed when both a browser HTML and (if api_models is
    configured) an API JSON exist for it anywhere in history.
    """
    data_dir = Path(data_dir)
    raw_html_dir = data_dir / "raw_html"
    api_dir = data_dir / "api"

    html_runs = sorted(raw_html_dir.glob("*")) if raw_html_dir.exists() else []
    api_runs = sorted(api_dir.glob("*")) if api_dir.exists() else []
    if not html_runs and not api_runs:
        print(">> [resume] No previous runs found — starting from the beginning.")
        return None

    exp_names = None
    api_model_dirs = None
    if config and isinstance(config, dict):
        exps = config.get("experiments", [])
        if exps:
            exp_names = {e["name"] for e in exps if "name" in e}
        api_models = config.get("api_models", [])
        if api_models:
            api_model_dirs = {_api_model_dir_name(m) for m in api_models}

    benchmark_name = _extract_benchmark_name(query_file) if query_file else None

    def _browser_dir_matches(name):
        if not exp_names:
            return True
        if name in exp_names:
            return True
        if benchmark_name:
            for en in exp_names:
                if name == f"{en}-{benchmark_name}":
                    return True
                if name == f"{en}-resume-cache-{benchmark_name}":
                    return True
        return False

    def _api_top_matches(name):
        if benchmark_name and name in (benchmark_name, f"resume-cache-{benchmark_name}"):
            return True
        return False

    browser_ids = set()
    for run_dir in html_runs:
        if not run_dir.is_dir():
            continue
        for session_dir in sorted(run_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for exp_dir in sorted(session_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                if not _browser_dir_matches(exp_dir.name):
                    continue
                for f in exp_dir.glob("*.html"):
                    browser_ids.add(f.stem)

    api_ids = set()
    for run_dir in api_runs:
        if not run_dir.is_dir():
            continue
        for top_dir in sorted(run_dir.iterdir()):
            if not top_dir.is_dir():
                continue
            if _api_top_matches(top_dir.name):
                for model_dir in sorted(top_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    if api_model_dirs and model_dir.name not in api_model_dirs:
                        continue
                    for f in model_dir.glob("*.api.json"):
                        api_ids.add(f.stem.replace(".api", ""))
            else:
                if api_model_dirs and top_dir.name not in api_model_dirs:
                    continue
                for f in top_dir.glob("*.api.json"):
                    api_ids.add(f.stem.replace(".api", ""))

    has_browser = bool(exp_names)
    has_api = bool(api_model_dirs)
    if has_browser and has_api:
        completed = browser_ids & api_ids
        print(f">> [resume] browser_union={len(browser_ids)} api_union={len(api_ids)} both={len(completed)}")
    elif has_browser:
        completed = browser_ids
        print(f">> [resume] browser_union={len(browser_ids)} (no api_models configured)")
    elif has_api:
        completed = api_ids
        print(f">> [resume] api_union={len(api_ids)} (no experiments configured)")
    else:
        completed = browser_ids | api_ids
        print(f">> [resume] union={len(completed)} (no config filter)")

    if not completed:
        print(">> [resume] No completed queries found — starting from the beginning.")
        return None

    all_queries = load_queries_from_file(query_file)
    if not all_queries:
        return None

    remaining = []
    skipped = 0
    for q in all_queries:
        q_id = str(q.get("id") if isinstance(q, dict) else q[0])
        if f"{q_id}_run0" in completed or q_id in completed:
            skipped += 1
            continue
        remaining.append(q)

    print(f">> [resume] Skipped {skipped} already-completed queries; {len(remaining)} remaining.")
    return remaining


def load_queries_from_file(query_file):
    queries = []
    if not query_file:
        print("Error: No query file specified.")
        return []
    if not os.path.isabs(query_file):
        query_file = BASE_DIR / query_file
    if not os.path.exists(query_file):
        print(f"Error: Query file not found: {query_file}")
        return []
    try:
        if str(query_file).lower().endswith(".json"):
            data = json.loads(Path(query_file).read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    lower_item = {str(k).lower(): v for k, v in item.items()}
                    if "id" in lower_item and "query" in lower_item:
                        queries.append({
                            "id": lower_item["id"],
                            "query": lower_item["query"],
                            "reuse_chat": _coerce_bool(lower_item.get("reuse_chat")),
                            "is_mcq": _detect_is_mcq(lower_item),
                        })
        else:
            with open(query_file, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    lower_row = {str(k).lower(): v for k, v in row.items()}
                    if "id" in lower_row and "query" in lower_row:
                        queries.append({
                            "id": lower_row["id"],
                            "query": lower_row["query"],
                            "reuse_chat": _coerce_bool(lower_row.get("reuse_chat")),
                            "is_mcq": _detect_is_mcq(lower_row),
                        })
    except Exception as e:
        print(f"Error loading queries from {query_file}: {e}")
        return []
    return queries


def shuffle_no_consecutive(queries, seed=None):
    if len(queries) <= 1:
        return queries
    if seed is None:
        random.shuffle(queries)
    else:
        rng = random.Random(seed)
        rng.shuffle(queries)
    return queries


def _slugify_label(value, fallback):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or fallback


def _resolve_query_path(query_file):
    if not query_file:
        return None
    path = Path(query_file)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _extract_benchmark_name(query_file):
    resolved = _resolve_query_path(query_file)
    if not resolved:
        return "benchmark"

    for part in reversed(resolved.parts):
        if part.lower().startswith("metabench-"):
            return _slugify_label(part, "benchmark")

    if resolved.parent.name.lower() == "queries" and resolved.parent.parent.name:
        return _slugify_label(resolved.parent.parent.name, "benchmark")

    stem = resolved.stem
    if stem.lower() in {"query", "queries"} and resolved.parent.name:
        return _slugify_label(resolved.parent.name, "benchmark")
    return _slugify_label(stem, "benchmark")


def _query_file_sequence(experiment, defaults, prefer_defaults_rotate=False):
    """Return query files in sequence for this experiment."""
    if prefer_defaults_rotate:
        rotate_list = defaults.get("rotate_query_files")
        if not rotate_list:
            rotate_list = experiment.get("rotate_query_files")
    else:
        rotate_list = experiment.get("rotate_query_files") or defaults.get("rotate_query_files")

    if rotate_list and isinstance(rotate_list, list):
        return [str(qf) for qf in rotate_list]

    single = experiment.get("query_file", defaults.get("query_file"))
    return [single] if single else []


# ── Human-like typing ─────────────────────────────────────────────────────────

def _js_shift_enter(ele):
    ele.run_js("""
        ['keydown', 'keypress', 'keyup'].forEach(evType => {
            this.dispatchEvent(new KeyboardEvent(evType, {
                key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
                shiftKey: true, bubbles: true, cancelable: true,
            }));
        });
    """)


def human_type(page, selector_or_ele, text, allow_typos=False):
    try:
        if isinstance(selector_or_ele, str):
            ele = page.ele(selector_or_ele, timeout=5)
        else:
            ele = selector_or_ele
        if not ele:
            return False
        try:
            ele.run_js("this.click(); this.focus();")
        except Exception:
            try:
                ele.click(by_js=True)
            except Exception:
                pass
        if text is None:
            text = ""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        for i, line in enumerate(lines):
            burst_remaining = random.randint(6, 18)
            for char in line:
                if burst_remaining <= 0:
                    time.sleep(random.uniform(0.15, 0.5))
                    burst_remaining = random.randint(6, 18)
                escaped_char = json.dumps(char)
                ele.run_js(f"document.execCommand('insertText', false, {escaped_char});")
                base_delay = random.uniform(0.01, 0.03)
                if char in ".!?":
                    base_delay += random.uniform(0.12, 0.35)
                elif char in ",;:":
                    base_delay += random.uniform(0.06, 0.18)
                elif char == " " and random.random() < 0.08:
                    base_delay += random.uniform(0.05, 0.2)
                time.sleep(base_delay)
                burst_remaining -= 1
            if i < len(lines) - 1:
                time.sleep(random.uniform(0.05, 0.12))
                _js_shift_enter(ele)
                time.sleep(random.uniform(0.08, 0.25))
            if random.random() < 0.2:
                time.sleep(random.uniform(0.15, 0.5))
        time.sleep(random.uniform(0.4, 1.0))
        return True
    except Exception as e:
        print(f"Error typing: {e}")
        return False


def paste_prompt(page, selector_or_ele, text):
    """Insert prompt into Gemini's Quill editor via execCommand insertText."""
    try:
        if isinstance(selector_or_ele, str):
            ele = page.ele(selector_or_ele, timeout=5)
        else:
            ele = selector_or_ele
        if not ele:
            return False
        if text is None:
            text = ""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
        # Use JS click/focus to avoid "no location or size" errors on Angular elements
        try:
            ele.run_js("this.click(); this.focus();")
        except Exception:
            try:
                ele.click(by_js=True)
            except Exception:
                pass
        escaped = json.dumps(normalized)
        result = ele.run_js(f"""
            this.focus();
            document.execCommand('selectAll', false, null);
            const ok = document.execCommand('insertText', false, {escaped});
            return ok;
        """)
        print(f">> execCommand insertText result: {result}")
        content = ele.run_js("return this.innerText || this.textContent || '';") or ""
        if len(content.strip()) < min(10, len(normalized) // 2):
            print(">> execCommand failed; trying synthetic paste event")
            ele.run_js(f"""
                this.focus();
                const dt = new DataTransfer();
                dt.setData('text/plain', {escaped});
                this.dispatchEvent(new ClipboardEvent('paste', {{
                    clipboardData: dt, bubbles: true, cancelable: true
                }}));
            """)
            time.sleep(random.uniform(0.3, 0.5))
            content = ele.run_js("return this.innerText || this.textContent || '';") or ""
        if len(content.strip()) < min(10, len(normalized) // 2):
            print(">> Paste still empty after fallback.")
            return False
        return True
    except Exception as e:
        print(f"Error inserting prompt: {e}")
        return False


def _think_delay(text):
    if not text:
        return
    delay = 0.2 + len(text) / 900
    delay += random.uniform(0.0, 0.5)
    delay = min(delay, 2.8)
    if random.random() < 0.05:
        delay += random.uniform(1.0, 3.0)
    time.sleep(delay)


def _maybe_scroll(page):
    if random.random() < 0.15:
        delta = random.randint(-250, 500)
        try:
            page.run_js(f"window.scrollBy(0, {delta});")
            time.sleep(random.uniform(0.2, 0.6))
        except Exception:
            pass


# ── Gemini login ──────────────────────────────────────────────────────────────

def verify_strict_login(page):
    """Return True only if we are genuinely logged into Gemini."""
    try:
        html = page.html or ""
        return (
            'data-test-id="mavatar-footer-settings-button"' in html
            or 'data-test-id="pillbox"' in html
        )
    except Exception:
        return False


def handle_google_login(page, allow_manual=True):
    """Navigate to Gemini and ensure the user is logged in via Google."""
    if verify_strict_login(page):
        return True
    if not allow_manual:
        print(">> Not logged in and manual login is disabled.")
        return False

    print(">> Initiating Gemini login sequence...")

    # Click Sign in
    sign_in_btn = (
        page.ele('text:Sign in', timeout=4) or
        page.ele('a[href*="accounts.google.com"]', timeout=3)
    )
    if sign_in_btn:
        print(">> Clicking 'Sign in'...")
        try:
            sign_in_btn.click()
            time.sleep(2)
        except Exception:
            pass
    else:
        print(">> Sign-in button not found — please log in manually if needed.")

    print("\n" + "=" * 50)
    print(">> PLEASE LOG IN WITH YOUR GOOGLE ACCOUNT NOW")
    print("=" * 50 + "\n")

    print(">> Waiting for successful login...")
    for i in range(240):  # up to 120 s
        if verify_strict_login(page):
            print(">> Login verified!")
            return True
        if i % 20 == 0:
            print(f">> Waiting... ({120 - i * 0.5:.0f}s remaining)")
        time.sleep(0.5)

    print(">> Login timed out.")
    return False


# ── Model selection ───────────────────────────────────────────────────────────

def _get_model_picker(page, timeout=2):
    """Return the model picker button, or None."""
    picker = (
        page.ele('button[data-test-id="bard-mode-menu-button"]', timeout=timeout) or
        page.ele('button[aria-label="Open mode picker"]', timeout=1)
    )
    if picker:
        return picker
    # Fallback: find button containing the logo-pill-label-container
    for btn in (page.eles('tag:button', timeout=2) or []):
        try:
            if btn.ele('css:[data-test-id="logo-pill-label-container"]', timeout=0):
                return btn
        except Exception:
            continue
    return None


def _current_model_label(page):
    """Return the visible model name from the picker button span, or ''."""
    try:
        span = page.ele('css:[data-test-id="logo-pill-label-container"] span.ng-star-inserted', timeout=0.5)
        if span:
            return (span.text or "").strip()
    except Exception:
        pass
    return ""


def select_interface_model(page, model_name, timeout=8):
    """Select a Gemini model from the model picker dropdown."""
    try:
        # Check if already selected without opening the picker
        current = _current_model_label(page)
        cur_l, req_l = current.lower(), model_name.lower()
        if current and (req_l in cur_l or cur_l in req_l):
            print(f">> Model '{model_name}' already selected ('{current}').")
            return True

        picker = _get_model_picker(page)
        if not picker:
            print(">> Model picker not found; skipping model selection.")
            return False

        print(f">> Opening model picker for '{model_name}' (current: '{current}')...")
        try:
            picker.click()
        except Exception:
            picker.click(by_js=True)

        # Wait for menu items to appear (up to 3s).
        # Note: Gemini uses <gem-menu-item data-test-id="bard-mode-option-{hash}">
        # with a hash ID (not a name slug), so we enumerate all options and
        # match on the <span class="label"> text inside each item.
        target = None
        available = []
        for attempt in range(6):
            menu_items = (
                page.eles('css:gem-menu-item[data-test-id^="bard-mode-option-"]', timeout=1) or
                page.eles('css:[data-test-id^="bard-mode-option-"]', timeout=1) or
                page.eles('css:[role="menuitem"]', timeout=1)
            )
            if menu_items:
                break
            time.sleep(0.5)

        print(f">> Menu items found: {len(menu_items or [])}")
        for item in (menu_items or []):
            # Prefer the inner label span; fall back to full element text
            label_el = item.ele('css:span.label', timeout=0)
            text = ((label_el.text if label_el else None) or item.text or "").strip()
            available.append(text)
            if re.search(r'\b' + re.escape(model_name.lower()) + r'\b', text.lower()):
                target = item
                print(f">> Matched '{model_name}' against menu item '{text}'")
                break

        if not target:
            print(f">> '{model_name}' not found in picker. Available: {available}")
            try:
                page.actions.key_down("Escape").key_up("Escape")
            except Exception:
                pass
            return False

        # If the item is already checked AND the current label confirms it, skip clicking
        try:
            already = target.attr("aria-checked") == "true"
        except Exception:
            already = False
        if already:
            current_after = _current_model_label(page)
            ca_l = current_after.lower()
            if req_l in ca_l or ca_l in req_l:
                print(f">> Model '{model_name}' already checked in picker; closing menu.")
                try:
                    page.actions.key_down("Escape").key_up("Escape")
                except Exception:
                    pass
                return True
            # aria-checked is stale/misleading — click anyway to actually switch

        print(f">> Clicking '{(target.text or '').strip()}'...")
        try:
            target.click()
        except Exception:
            target.click(by_js=True)
        time.sleep(0.5)
        print(f">> Selected model '{model_name}'.")
        return True
    except Exception as e:
        print(f">> Error selecting model: {e}")
        return False


# ── Detect model from page HTML ───────────────────────────────────────────────

def detect_model_from_html(html):
    """Return the model identifier shown on a Gemini response page, or None."""
    if not html:
        return None
    # Structured model ID fields
    for pattern in [
        r'data-model-name="([^"]+)"',
        r'"modelVersion"\s*:\s*"([^"]+)"',
        r'"model_version"\s*:\s*"([^"]+)"',
        r'"modelId"\s*:\s*"([^"]+)"',
    ]:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            return m.group(1)

    # Mode picker button text — the display name of the currently selected model
    # e.g. <span ...>Thinking</span> / <span ...>Fast</span> / <span ...>Pro</span>
    # The span may have extra _ngcontent-* attrs before class="ng-star-inserted",
    # and Angular inserts <!---->  comment nodes before the span.
    m = re.search(
        r'data-test-id="logo-pill-label-container"[^>]*>'
        r'(?:<!--.*?-->)*\s*'
        r'<span[^>]*class="ng-star-inserted">([^<]+)</span>',
        html,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()

    return None


# ── New conversation ──────────────────────────────────────────────────────────

def ensure_new_chat(page, attempts=3):
    """Start a fresh Gemini conversation. Returns the prompt input element or None."""
    for _ in range(attempts):
        try:
            page.get(GEMINI_APP_URL)
            el = _find_prompt_input(page, timeout=5)
            if el:
                return el
        except Exception as e:
            print(f">> ensure_new_chat error: {e}")
            time.sleep(1)

    # Fallback: click the New chat button in the sidebar
    for _ in range(attempts):
        btn = (
            page.ele('a[aria-label="New chat"]', timeout=3) or
            page.ele('button[aria-label="New chat"]', timeout=2) or
            page.ele('text:New chat', timeout=2)
        )
        if btn:
            try:
                btn.click()
                el = _find_prompt_input(page, timeout=5)
                if el:
                    return el
            except Exception:
                pass
        page.refresh()
        time.sleep(2)
    return None


# ── Prompt input / send helpers ───────────────────────────────────────────────

def _find_prompt_input(page, timeout=3):
    """Return Gemini's prompt textarea element (the inner contenteditable div)."""
    import time as _time

    deadline = _time.time() + timeout

    while _time.time() < deadline:
        # Try all selector forms — DrissionPage 4.x interprets strings differently
        # depending on whether they start with @, css:, tag:, text:, etc.
        el = (
            # @attr=val syntax (DrissionPage native)
            page.ele('@aria-label=Enter a prompt for Gemini', timeout=0.5) or
            # CSS selector with explicit prefix
            page.ele('css:div[aria-label="Enter a prompt for Gemini"]', timeout=0.5) or
            # role=textbox excludes .ql-clipboard (no role attr)
            page.ele('css:.ql-editor[role="textbox"]', timeout=0.5)
        )
        if el:
            print(f">> Found prompt input")
            return el

        # Try via rich-textarea parent
        rt = page.ele('tag:rich-textarea', timeout=0.5)
        if rt:
            inner = (
                rt.ele('@aria-label=Enter a prompt for Gemini', timeout=0.5) or
                rt.ele('css:.ql-editor[role="textbox"]', timeout=0.5)
            )
            if inner:
                print(f">> Found prompt input via rich-textarea child")
                return inner

        _time.sleep(0.5)

    print(f">> _find_prompt_input: all selectors exhausted after {timeout}s")
    return None


def _find_send_button(page):
    return (
        page.ele('button[aria-label="Send message"]', timeout=3) or
        page.ele('button.send-button.submit', timeout=2) or
        page.ele('button[mattooltip="Send message"]', timeout=2) or
        page.ele('@@aria-label=Send message', timeout=2) or
        page.ele('button[data-test-id="send-button"]', timeout=2)
    )


def _click_send(page, prompt_input, max_attempts=3):
    """Click send and verify generation started. Returns True if sent successfully."""
    for attempt in range(max_attempts):
        send_btn = _find_send_button(page)

        if send_btn:
            # Try progressively more forceful click strategies
            try:
                send_btn.click()
            except Exception:
                pass
            time.sleep(0.2)
            if 'aria-label="Stop response"' not in (page.html or ""):
                try:
                    send_btn.run_js("""
                        this.dispatchEvent(new MouseEvent('click', {
                            bubbles: true, cancelable: true, view: window
                        }));
                    """)
                except Exception:
                    pass
            time.sleep(0.2)
            if 'aria-label="Stop response"' not in (page.html or ""):
                try:
                    send_btn.run_js("this.click();")
                except Exception:
                    pass

        # Also try Enter key on the editor as parallel fallback
        if prompt_input and 'aria-label="Stop response"' not in (page.html or ""):
            try:
                prompt_input.run_js("""
                    this.dispatchEvent(new KeyboardEvent('keydown', {
                        key:'Enter',code:'Enter',keyCode:13,which:13,
                        bubbles:true,cancelable:true
                    }));
                """)
            except Exception:
                pass

        # Wait up to 5s for the stop button to appear (generation started)
        for _ in range(10):
            if 'aria-label="Stop response"' in (page.html or ""):
                print(f">> Send confirmed (attempt {attempt + 1}).")
                return True
            time.sleep(0.5)

        print(f">> Send attempt {attempt + 1} did not trigger generation; retrying...")
        time.sleep(random.uniform(0.5, 1.0))

    print(">> Send failed after all attempts.")
    return False


# ── Wait for response ─────────────────────────────────────────────────────────

def _response_is_complete(page):
    """Return True when Gemini has finished generating."""
    html = page.html or ""

    # Still generating if stop button is present
    if 'aria-label="Stop response"' in html:
        return False

    # Response footer with thumb up/down only appears after generation completes
    if 'response-container-footer' in html:
        return True

    return False


def _pause_on_error_13(html, pause_seconds=GEMINI_ERROR_13_PAUSE_SECONDS):
    """Pause when Gemini shows the transient error banner."""
    if GEMINI_ERROR_13_TEXT.lower() not in (html or "").lower():
        return False
    print(
        f">> Detected '{GEMINI_ERROR_13_TEXT}'. "
        f"Pausing for {pause_seconds // 60} minutes..."
    )
    time.sleep(pause_seconds)
    return True


# ── Clear history ─────────────────────────────────────────────────────────────

def clear_memory(page):
    """Delete all conversations on gemini.google.com."""
    print(">> Clearing Gemini conversation history...")
    try:
        page.get(f"{GEMINI_URL}/settings")
        time.sleep(2)

        clear_btn = (
            page.ele('text:Delete all conversations', timeout=4) or
            page.ele('text:Clear all conversations', timeout=3) or
            page.ele('button[data-test-id="delete-all-conversations"]', timeout=3)
        )
        if not clear_btn:
            print(">> Clear conversations button not found in settings.")
            return False

        clear_btn.click()
        time.sleep(1.5)

        # Confirm dialog
        confirm_btn = (
            page.ele('text:Delete all', timeout=3) or
            page.ele('text:Confirm', timeout=2) or
            page.ele('css:button.confirm-button', timeout=2)
        )
        if confirm_btn:
            confirm_btn.click()
            time.sleep(1.5)
            print(">> Conversation history cleared.")
            return True
        else:
            print(">> Confirm button not found.")
            return False
    except Exception as e:
        print(f">> Error clearing memory: {e}")
        return False


# ── Save response ─────────────────────────────────────────────────────────────

def save_response(output_dir, query_id, content, model_info, sent_at=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join([c for c in query_id if c.isalnum() or c in ("-", "_")]).strip()
    filename = output_dir / f"{safe_id}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f">> Saved response to {filename}")

    meta = {
        "query_id": query_id,
        "saved_at": datetime.now().isoformat(),
        "platform": "gemini.google.com",
    }
    if sent_at:
        meta["sent_at"] = sent_at
    if model_info.get("model_slug"):
        meta["model_slug"] = model_info["model_slug"]

    meta_filename = output_dir / f"{safe_id}.meta.json"
    with open(meta_filename, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f">> Saved metadata to {meta_filename}")


def _api_barrier_worker(
    query_files, runs, shuffle, seed,
    api_cfg, api_key,
    output_base,
    sync_barrier_info, stop_event,
):
    """Standalone API worker that participates in the per-query sync barrier."""
    barrier = FileBasedBarrier(
        parties=sync_barrier_info["parties"],
        sync_dir=sync_barrier_info["sync_dir"],
        session_id=sync_barrier_info["session_id"],
    )

    m_name = api_cfg["model"]
    m_params = {k: v for k, v in api_cfg.items() if k != "model"}
    m_dir = m_name + ("_" + "_".join(f"{k}-{v}" for k, v in sorted(m_params.items())) if m_params else "")

    # Build flat list of (query_dict, output_dir) across all query files.
    # Multiple files get separate subdirectories (matching browser rotate behavior).
    items = []
    use_subdirs = len(query_files) > 1
    for qf in query_files:
        queries_base = load_queries_from_file(qf)
        if not queries_base:
            continue
        if use_subdirs:
            out_dir = Path(output_base) / _extract_benchmark_name(qf) / m_dir
        else:
            out_dir = Path(output_base) / m_dir
        file_queries = []
        for item in queries_base:
            q_id = item.get("id") if isinstance(item, dict) else item[0]
            q_text = item.get("query") if isinstance(item, dict) else item[1]
            for r in range(runs):
                file_queries.append({"id": f"{q_id}_run{r}", "query": q_text, "_out": out_dir})
        if shuffle:
            file_queries = shuffle_no_consecutive(file_queries, seed=seed)
        items.extend(file_queries)

    if not items:
        print(f">> [API {m_name}] No queries loaded. Exiting worker.")
        return

    print(f">> [API {m_name}] Starting barrier worker with {len(items)} queries.")

    for i, item in enumerate(items):
        if stop_event and stop_event.is_set():
            print(f">> [API {m_name}] Stop signal received. Halting.")
            return

        print(f">> [API {m_name}] [{i+1}/{len(items)}] Waiting at barrier...")
        barrier.wait()

        attempt = 0
        while True:
            if stop_event and stop_event.is_set():
                return
            sent_at = datetime.now().isoformat()
            print(f">> [API {m_name}] Sending {item['id']} (attempt {attempt+1})...")
            record = run_api_query(item["id"], item["query"], str(item["_out"]), m_name,
                                   sent_at=sent_at, api_key=api_key, **m_params)
            if not record.get("error"):
                break
            attempt += 1
            wait = min(60 * (2 ** (attempt - 1)), 3600)
            print(f">> [API {m_name}] Query {item['id']} failed: {record['error']} — retrying in {wait}s...")
            time.sleep(wait)

    print(f">> [API {m_name}] All queries complete.")


# ── Core experiment runner ────────────────────────────────────────────────────

def run_experiment(
    page,
    experiment,
    defaults,
    run_id,
    session_id,
    sync_barrier=None,
    seed_override=None,
    api_models=None,
    interface_model=None,
    save_mcq_only=False,
    stop_event=None,
    query_file_override=None,
    output_group_override=None,
):
    api_models = api_models or []
    exp_name = experiment.get("name", "experiment")
    query_file = query_file_override if query_file_override is not None else \
        experiment.get("query_file", defaults.get("query_file"))
    runs = int(experiment.get("runs", defaults.get("runs", 1)))
    shuffle = bool(experiment.get("shuffle", defaults.get("shuffle", False)))
    exp_seed = experiment.get("seed", defaults.get("seed"))
    reuse_chat = bool(experiment.get("reuse_chat", defaults.get("reuse_chat", False)))
    fast_mode = bool(experiment.get("fast", defaults.get("fast", False)))
    wait_after_saves = experiment.get("wait_after_saves", defaults.get("wait_after_saves", [3, 7]))
    interface_model = experiment.get("interface_model", interface_model)
    save_group_name = output_group_override or exp_name

    if seed_override is not None and "seed" not in experiment:
        exp_seed = seed_override

    output_dir = DATA_DIR / "raw_html" / run_id / f"session_{session_id:02d}" / save_group_name
    api_output_base = DATA_DIR / "api" / run_id / f"session_{session_id:02d}" / save_group_name

    print(f"\n>>> STARTING EXPERIMENT: {exp_name}")
    print(f"    Query File: {query_file}")
    if save_group_name != exp_name:
        print(f"    Save Group: {save_group_name}")
    if interface_model:
        print(f"    Interface Model: {interface_model}")
    print(f"    Output Directory: {output_dir}")
    if api_models:
        print(f"    API Models: {', '.join(m['model'] for m in api_models)}")

    queries_base = load_queries_from_file(query_file)
    if not queries_base:
        print("    No queries found. Skipping.")
        return

    queries = []
    for item in queries_base:
        q_id = item.get("id") if isinstance(item, dict) else item[0]
        q_text = item.get("query") if isinstance(item, dict) else item[1]
        q_reuse = item.get("reuse_chat") if isinstance(item, dict) else None
        q_is_mcq = item.get("is_mcq") if isinstance(item, dict) else None
        for r in range(runs):
            queries.append({
                "id": f"{q_id}_run{r}",
                "query": q_text,
                "reuse_chat": q_reuse,
                "is_mcq": q_is_mcq,
            })

    if shuffle:
        queries = shuffle_no_consecutive(queries, seed=exp_seed)
        print(f"    Shuffled {len(queries)} queries.")

    prompt_selector = '.ql-editor'

    for i, item in enumerate(queries):
        if stop_event and stop_event.is_set():
            print(">> Stop signal received. Halting.")
            return

        q_id = item["id"]
        q_text = "Please do not use web search for this question.\n\n" + item["query"]
        item_reuse = item.get("reuse_chat")
        effective_reuse = reuse_chat if item_reuse is None else bool(item_reuse)
        is_mcq = item.get("is_mcq")
        if is_mcq is None:
            is_mcq = "_mcq" in str(q_id or "").lower()

        print(f"\n[{i+1}/{len(queries)}] Processing {q_id}...")
        attempt = 0
        barrier_reached = False
        query_complete = False

        while not query_complete:
            if stop_event and stop_event.is_set():
                print(">> Stop signal received. Halting.")
                return

            attempt += 1
            if attempt > 1:
                print(f">> Re-running {q_id} (attempt {attempt})...")

            skip_query = False
            prompt_input = None
            open_fresh_chat = (not effective_reuse) or attempt > 1
            if open_fresh_chat:
                prompt_input = ensure_new_chat(page)
                if interface_model:
                    select_interface_model(page, interface_model)

            if not prompt_input:
                prompt_input = _find_prompt_input(page, timeout=8)
            if not prompt_input:
                print(">> Prompt input not found. Refreshing...")
                page.refresh()
                time.sleep(5)
                prompt_input = _find_prompt_input(page, timeout=10)
                if not prompt_input:
                    print(">> Still no prompt input. Skipping query.")
                    skip_query = True

            if not skip_query:
                print(f">> Sending: {q_text[:60].replace(chr(10), ' ')}...")

                typed = paste_prompt(page, prompt_input, q_text) or \
                        human_type(page, prompt_input, q_text, allow_typos=not (fast_mode or is_mcq))

                if not typed:
                    print(">> Failed to enter prompt. Skipping.")
                    skip_query = True

            # Cross-session barrier: ALL participants must hit this for every
            # query (even on skip/retry) so the barrier count stays in lockstep.
            if sync_barrier and not barrier_reached:
                print(f">> [session {session_id}] Waiting at query barrier {i}...")
                try:
                    sync_barrier.wait()
                except Exception as e:
                    print(f">> Barrier wait failed ({e}). Proceeding.")
                barrier_reached = True

            # Skip if already saved — allows resuming an interrupted run
            _sid = "".join(c for c in str(q_id or "") if c.isalnum() or c in ("-", "_")).strip()
            if (output_dir / f"{_sid}.html").exists():
                print(f">> [{i+1}/{len(queries)}] Skipping {q_id} (already saved).")
                query_complete = True
                break

            try:
                sent_at = None
                if not skip_query:
                    # Confirm text is in the editor before sending
                    try:
                        content_check = prompt_input.run_js("return this.innerText || this.textContent || '';") or ""
                        print(f">> Editor content check: {repr(content_check[:80])}")
                    except Exception:
                        content_check = "unknown"

                    sent_at = datetime.now().isoformat()
                    _click_send(page, prompt_input)

                if not skip_query:
                    print(">> Waiting for response...")
                    wait_start = time.time()
                    wait_timeout = 600
                    soft_refresh_timeout = 90
                    soft_refreshed = False
                    poll_interval = 0.5

                    # Wait for stop button to appear (Gemini started generating)
                    print(">> Waiting for generation to start...")
                    for _ in range(60):  # up to 30s
                        html = page.html or ""
                        if _pause_on_error_13(html):
                            wait_start = time.time()
                            soft_refreshed = False
                            try:
                                page.refresh()
                                time.sleep(5)
                            except Exception:
                                pass
                            continue
                        if 'aria-label="Stop response"' in html:
                            print(">> Generation started.")
                            break
                        time.sleep(0.5)
                    else:
                        print(">> Stop button never appeared — may have completed instantly.")

                    while True:
                        html = page.html or ""
                        if _pause_on_error_13(html):
                            wait_start = time.time()
                            soft_refreshed = False
                            try:
                                page.refresh()
                                time.sleep(5)
                            except Exception:
                                pass
                            continue
                        # Still generating if stop button present
                        if 'aria-label="Stop response"' in html:
                            pass  # still going
                        elif _response_is_complete(page):
                            print(">> Response complete.")
                            break
                        elapsed = time.time() - wait_start
                        if not soft_refreshed and elapsed > soft_refresh_timeout:
                            print(f">> No response after {soft_refresh_timeout}s — soft-refreshing...")
                            try:
                                page.refresh()
                            except Exception as _ref_err:
                                print(f">> Soft refresh raised: {_ref_err}")
                            time.sleep(5)
                            soft_refreshed = True
                            prompt_input = _find_prompt_input(page, timeout=8)
                            if prompt_input:
                                try:
                                    refreshed_content = prompt_input.run_js(
                                        "return this.innerText || this.textContent || '';"
                                    ) or ""
                                except Exception:
                                    refreshed_content = ""
                                normalized_prompt = " ".join((q_text or "").split())
                                normalized_content = " ".join(refreshed_content.split())
                                if normalized_prompt and normalized_prompt not in normalized_content:
                                    print(">> Soft refresh cleared prompt; re-entering and re-sending...")
                                    retyped = paste_prompt(page, prompt_input, q_text) or \
                                        human_type(
                                            page,
                                            prompt_input,
                                            q_text,
                                            allow_typos=not (fast_mode or is_mcq),
                                        )
                                    if retyped and _click_send(page, prompt_input):
                                        sent_at = datetime.now().isoformat()
                                        wait_start = time.time()
                                    elif not retyped:
                                        print(">> Failed to re-enter prompt after soft refresh.")
                                else:
                                    print(">> Prompt still present after soft refresh.")
                            else:
                                print(">> Prompt input not found after soft refresh; waiting for timeout retry.")
                        if elapsed > wait_timeout:
                            print(f">> Timed out after {wait_timeout}s. Re-sending...")
                            ensure_new_chat(page)
                            if interface_model:
                                select_interface_model(page, interface_model)
                            prompt_input = _find_prompt_input(page)
                            if prompt_input:
                                paste_prompt(page, prompt_input, q_text) or \
                                    human_type(page, prompt_input, q_text, allow_typos=not (fast_mode or is_mcq))
                                _click_send(page, prompt_input)
                            wait_start = time.time()
                        time.sleep(poll_interval)

                    # Use JS to grab HTML synchronously — avoids hanging when
                    # Gemini Pro keeps open network connections after generation.
                    try:
                        final_html = page.run_js("return document.documentElement.outerHTML;") or ""
                    except Exception:
                        final_html = page.html or ""
                    model_slug = detect_model_from_html(final_html)
                    print(f">> Model detected: {model_slug or 'unknown'}")

                    if (
                        interface_model
                        and "thinking" in str(interface_model).lower()
                        and "fast" in str(model_slug or "").lower()
                    ):
                        print(
                            f">> Expected Thinking model but got '{model_slug}'. "
                            f"Waiting {THINKING_FAST_RETRY_SECONDS // 60} minutes, then retrying same query."
                        )
                        pause_end = time.time() + THINKING_FAST_RETRY_SECONDS
                        while time.time() < pause_end:
                            if stop_event and stop_event.is_set():
                                print(">> Stop signal received during retry wait. Halting.")
                                return
                            time.sleep(5)
                        continue

                    skip_save = False
                    if save_mcq_only:
                        is_mcq_item = item.get("is_mcq")
                        if is_mcq_item is False:
                            skip_save = True
                        elif is_mcq_item is None and "_mcq" not in str(q_id or "").lower():
                            skip_save = True

                    if skip_save:
                        print(">> Skipping save (non-MCQ).")
                    else:
                        save_response(
                            output_dir,
                            q_id,
                            final_html,
                            model_info={"model_slug": model_slug} if model_slug else {},
                            sent_at=sent_at,
                        )
                    _maybe_scroll(page)
                else:
                    print(">> Skipping send/wait/save due to earlier failure.")

                query_complete = True

            except Exception as _attempt_err:
                import traceback
                print(f"\n>> !! Query {q_id} attempt {attempt} failed: {_attempt_err}")
                traceback.print_exc()
                print(f">> Will retry (attempt {attempt}/5).")

        # Inter-query wait
        if isinstance(wait_after_saves, list) and len(wait_after_saves) == 2:
            sleep_time = random.uniform(wait_after_saves[0], wait_after_saves[1])
        else:
            sleep_time = 5 if wait_after_saves is None else float(wait_after_saves)
        if sleep_time > 0:
            print(f">> Waiting {sleep_time:.2f}s before next query...")
            time.sleep(sleep_time)

        processed_queries = i + 1
        if (
            INTERVAL_BREAK_EVERY_QUERIES > 0
            and processed_queries % INTERVAL_BREAK_EVERY_QUERIES == 0
            and processed_queries < len(queries)
        ):
            print(
                f">> Processed {processed_queries} queries. "
                f"Taking a {INTERVAL_BREAK_SECONDS // 60}-minute break..."
            )
            time.sleep(INTERVAL_BREAK_SECONDS)


# ── Main audit runner ─────────────────────────────────────────────────────────

def run_audit(
    config_path,
    delay_seconds=0,
    user_data_dir=None,
    run_id=None,
    session_id=0,
    sync_barrier=None,
    sync_barrier_info=None,
    seed_override=None,
    api_models=None,
    interface_model=None,
    allow_manual_login=True,
    page=None,
    log_to_file=True,
    clear_memory_first=False,
    clear_only=False,
    api_only=False,
    save_mcq_only=False,
    experiment_index=None,
    stop_event=None,
    api_ready_event=None,
    parsed_config=None,
):
    combined_log_path = None
    stdout_orig = stderr_orig = log_handles = None
    if run_id and log_to_file:
        combined_log_path, stdout_orig, stderr_orig, log_handles = \
            _setup_timestamped_logging(run_id, session_id)
        print(f">> Logging to {combined_log_path}")

    # Create barrier from info if needed (for multiprocessing)
    if sync_barrier_info and not sync_barrier:
        if sync_barrier_info.get("use_threads"):
            sync_barrier = threading.Barrier(sync_barrier_info["parties"])
        else:
            sync_barrier = FileBasedBarrier(
                parties=sync_barrier_info["parties"],
                sync_dir=sync_barrier_info["sync_dir"],
                session_id=session_id,
            )

    try:
        if parsed_config is not None:
            full_config = parsed_config
        else:
            try:
                with open(config_path, "r") as f:
                    full_config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config {config_path}: {e}")
                return

        if isinstance(full_config, list):
            defaults = {
                "query_file": str(config_path),
                "runs": 1,
                "shuffle": False,
                "wait_after_prompts": 0,
                "wait_after_saves": 0,
            }
            experiments = [{"name": "json_queries", "type": "default"}]
            config_api_models = []
        else:
            defaults = full_config.get("defaults", {})
            experiments = full_config.get("experiments", [])
            config_api_models = full_config.get("api_models", [])

        if api_models is None:
            api_models = config_api_models or []

        if experiment_index is not None and experiment_index < len(experiments):
            experiments = [experiments[experiment_index]]
        elif experiment_index is not None and experiments:
            experiments = [experiments[0]]

        # API-only mode
        if api_only:
            model_names = [m["model"] for m in api_models]
            print(f">> [API] Session {session_id} | Models: {', '.join(model_names)}")
            if api_ready_event:
                print(">> [API] Waiting for interface session to be ready...")
                while not api_ready_event.is_set():
                    if stop_event and stop_event.is_set():
                        print(">> [API] Stop signal received. Halting.")
                        return
                    time.sleep(5)
                print(">> [API] Interface session ready.")

            if delay_seconds > 0:
                time.sleep(delay_seconds)

            run_id_str = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            api_output_base = DATA_DIR / "api" / run_id_str / f"session_{session_id:02d}"

            for experiment in experiments:
                query_files = _query_file_sequence(
                    experiment,
                    defaults,
                    prefer_defaults_rotate=experiment_index is not None,
                )
                if not query_files:
                    print(f">> No query file configured for experiment '{experiment.get('name', 'experiment')}'. Skipping.")
                    continue
                for file_idx, qf in enumerate(query_files):
                    if len(query_files) > 1:
                        print(
                            f">> [rotate-seq] Experiment '{experiment.get('name', 'experiment')}': "
                            f"file [{file_idx + 1}/{len(query_files)}] {qf}"
                        )
                    save_group_name = experiment.get("name", "experiment")
                    if len(query_files) > 1:
                        save_group_name = f"{save_group_name}-{_extract_benchmark_name(qf)}"
                    runs = int(experiment.get("runs", defaults.get("runs", 1)))
                    shuffle = bool(experiment.get("shuffle", defaults.get("shuffle", False)))
                    exp_seed = experiment.get("seed", defaults.get("seed"))
                    if seed_override is not None:
                        exp_seed = seed_override
                    queries_base = load_queries_from_file(qf)
                    if not queries_base:
                        continue
                    queries = []
                    for item in queries_base:
                        q_id = item.get("id") if isinstance(item, dict) else item[0]
                        q_text = item.get("query") if isinstance(item, dict) else item[1]
                        for r in range(runs):
                            queries.append({"id": f"{q_id}_run{r}", "query": q_text})
                    if shuffle:
                        queries = shuffle_no_consecutive(queries, seed=exp_seed)
                    for j, q in enumerate(queries):
                        if stop_event and stop_event.is_set():
                            return
                        for api_cfg in api_models:
                            m_name = api_cfg["model"]
                            m_params = {k: v for k, v in api_cfg.items() if k != "model"}
                            m_dir = m_name + ("_" + "_".join(f"{k}-{v}" for k, v in sorted(m_params.items())) if m_params else "")
                            m_output = api_output_base / save_group_name / m_dir
                            print(f"\n[{j+1}/{len(queries)}] {q['id']} -> {m_name}")
                            run_api_query(q["id"], q["query"], m_output, m_name,
                                          api_key=GEMINI_API_KEY, **m_params)
            print(f"\n>> [API] Session {session_id} complete.")
            return

        # Browser mode
        if page is None:
            co = ChromiumOptions()

            # Give each worker process its own DrissionPage tmp root so that
            # PortFinder's auto-port cleanup doesn't race across processes and
            # trigger FileNotFoundError deep inside shutil.rmtree.
            per_worker_tmp = DATA_DIR / "drission_tmp" / f"session_{session_id:02d}_pid_{os.getpid()}"
            co.set_tmp_path(per_worker_tmp)

            co.auto_port()
            if user_data_dir:
                co.set_argument(f"--user-data-dir={user_data_dir}")
            co.set_argument("--no-first-run")
            co.set_argument("--mute-audio")
            page = ChromiumPage(co)

        if "gemini.google.com" not in page.url:
            print(">> Navigating to Gemini...")
            page.get(GEMINI_APP_URL)
            time.sleep(random.uniform(1.0, 2.0))

        # Login
        login_ok = False
        for attempt in range(5):
            try:
                login_ok = handle_google_login(page, allow_manual=allow_manual_login)
            except Exception as exc:
                print(f">> Login attempt {attempt + 1} crashed: {exc}")
                login_ok = False
            if login_ok:
                break
            wait_secs = min(30 * (attempt + 1), 120)
            print(f">> Login attempt {attempt + 1} failed. Retrying in {wait_secs}s...")
            time.sleep(wait_secs)
            try:
                page.get(GEMINI_APP_URL)
                time.sleep(3)
            except Exception:
                pass

        if not login_ok:
            print(">> Login failed after all retries. Aborting.")
            if stop_event:
                stop_event.set()
            return

        if api_ready_event:
            api_ready_event.set()

        if clear_memory_first:
            clear_memory(page)
            time.sleep(random.uniform(0.6, 1.4))

        if clear_only:
            print(">> Clear-only mode: skipping experiments.")
            return

        if delay_seconds > 0:
            print(f">> Delaying start by {delay_seconds:.1f}s...")
            time.sleep(delay_seconds)

        print(f">> Found {len(experiments)} experiment(s).")
        try:
            for exp in experiments:
                query_files = _query_file_sequence(
                    exp,
                    defaults,
                    prefer_defaults_rotate=experiment_index is not None,
                )
                if not query_files:
                    print(f">> No query file configured for experiment '{exp.get('name', 'experiment')}'. Skipping.")
                    continue

                for file_idx, query_file in enumerate(query_files):
                    if len(query_files) > 1:
                        print(
                            f">> [rotate-seq] Experiment '{exp.get('name', 'experiment')}': "
                            f"file [{file_idx + 1}/{len(query_files)}] {query_file}"
                        )
                    save_group_name = exp.get("name", "experiment")
                    if len(query_files) > 1:
                        save_group_name = f"{save_group_name}-{_extract_benchmark_name(query_file)}"
                    run_experiment(
                        page,
                        exp,
                        defaults,
                        run_id=run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        session_id=session_id,
                        sync_barrier=sync_barrier,
                        seed_override=seed_override,
                        api_models=api_models,
                        interface_model=interface_model,
                        save_mcq_only=save_mcq_only,
                        stop_event=stop_event,
                        query_file_override=query_file,
                        output_group_override=save_group_name,
                    )
        except Exception as exc:
            print(f"\n>> SESSION {session_id} ERROR: {exc}")
            if stop_event:
                stop_event.set()
            raise

        print("\nAll Experiments Completed.")
        try:
            page.get(GEMINI_APP_URL)
            time.sleep(random.uniform(1.0, 2.0))
        except Exception:
            pass

    finally:
        if log_handles:
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig
            for handle in log_handles:
                handle.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Browser scraper for gemini.google.com (mirrors audit_claude.py)")
    parser.add_argument("config", nargs="?", default="yamls/exp_gemini.yaml",
                        help="Path to experiments YAML")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated list of configs (one per session)")
    parser.add_argument("--start-in", type=float, default=0,
                        help="Delay start by N minutes")
    parser.add_argument("--sessions", type=int, default=None,
                        help="Override number of concurrent browser sessions")
    parser.add_argument("--profile-base", type=str, default=None,
                        help="Base directory for per-session Chrome profiles")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for deterministic shuffle")
    parser.add_argument("--no-parse", action="store_true",
                        help="Skip parsing saved HTML after experiments")
    parser.add_argument("--allow-manual-login", action="store_true",
                        help="Allow manual login prompts")
    parser.add_argument("--api-models", type=str, default=None,
                        help='Comma-separated Gemini API models, e.g. '
                             '"gemini-2.5-pro-preview-05-06,gemini-2.0-flash:0.7". '
                             'Param after colon is temperature (float).')
    parser.add_argument("--clear-memory", action="store_true",
                        help="Clear conversation history and exit.")
    parser.add_argument("--interface-model", type=str, default=None,
                        help="Model to select in Gemini UI (e.g. '2.5 Pro', 'Flash').")
    parser.add_argument("--output-tag", type=str, default=None,
                        help="Tag appended to run ID for output directories.")
    parser.add_argument("--num-runs", type=int, default=None,
                        help="Run the full experiment N times using a persistent browser session.")
    parser.add_argument("--repeat-every-hours", type=float, default=None,
                        help="Re-run the experiment every N hours indefinitely.")
    parser.add_argument("--resume", action="store_true",
                        help="Pick up where the last run left off (skip already-completed queries).")
    args = parser.parse_args()

    load_dotenv()

    def _resolve_config(raw):
        p = Path(raw)
        if p.exists():
            return p
        p2 = BASE_DIR / raw
        if p2.exists():
            return p2
        print(f"Config file not found: {raw}")
        sys.exit(1)

    if args.configs:
        raw_configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        config_paths = [_resolve_config(c) for c in raw_configs]
    else:
        config_paths = [_resolve_config(args.config)]

    delay_seconds = (args.start_in or 0) * 60

    # Build CLI api_models list
    cli_api_models = None
    if args.api_models:
        cli_api_models = []
        for raw in args.api_models.split(","):
            raw = raw.strip()
            if not raw or raw.lower() in {"none", "off", "false", "0"}:
                continue
            if ":" in raw:
                model_name, param_value = raw.split(":", 1)
                try:
                    cli_api_models.append({"model": model_name,
                                           "temperature": float(param_value)})
                except ValueError:
                    print(f"Warning: Cannot parse param '{param_value}' for '{model_name}'. Skipping.")
            else:
                cli_api_models.append({"model": raw})

    # Pre-parse configs
    _parsed_configs = {}
    for cp in config_paths:
        try:
            with open(cp, "r") as f:
                _parsed_configs[cp] = yaml.safe_load(f)
        except Exception:
            _parsed_configs[cp] = {}

    _pre_config = _parsed_configs[config_paths[0]]
    config_mode = _pre_config.get("mode", "sequential") if isinstance(_pre_config, dict) else "sequential"
    config_experiments = _pre_config.get("experiments", []) if isinstance(_pre_config, dict) else []

    if args.sessions is not None:
        sessions = max(1, args.sessions)
    elif config_mode == "sync":
        sessions = max(1, len(config_experiments))
    else:
        sessions = 1

    if len(config_paths) not in (1, sessions):
        print(f"--configs count must be 1 or match sessions ({sessions}).")
        sys.exit(1)

    def _auto_parse(run_id):
        run_dir = DATA_DIR / "raw_html" / run_id
        if not run_dir.exists():
            print(f"[parse] run dir not found: {run_dir}")
            return
        print(f"\n[parse] Auto-parsing HTML → JSON for run {run_id} ...")
        _parse_gemini_run(run_dir, _GEMINI_OUTPUT_ROOT)
        print("[parse] Done.\n")

    def run_once(run_delay, run_profile_base, allow_manual, persistent_pages=None):
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.output_tag:
            run_id = f"{run_id}-{args.output_tag}"
        clear_only = bool(args.clear_memory)

        if args.resume:
            import csv as _csv
            cfg = _parsed_configs[config_paths[0]]
            cfg_defaults = cfg.get("defaults", {}) if isinstance(cfg, dict) else {}
            rotate_list = cfg_defaults.get("rotate_query_files")
            single_qf = cfg_defaults.get("query_file") or \
                        (cfg.get("experiments", [{}])[0].get("query_file") if cfg.get("experiments") else None)

            if rotate_list and isinstance(rotate_list, list):
                # Resume each file in the rotation list independently
                new_rotate = []
                any_resumed = False
                for qf in rotate_list:
                    resolved_qf = str((BASE_DIR / qf).resolve()) if not Path(qf).is_absolute() else qf
                    remaining = find_resume_point(DATA_DIR, resolved_qf, config=cfg)
                    all_queries = load_queries_from_file(resolved_qf)
                    if remaining is not None and len(remaining) < len(all_queries):
                        stem = Path(resolved_qf).stem
                        resume_path = DATA_DIR / f"resume_cache_{stem}.csv"
                        keys = list(remaining[0].keys()) if remaining and isinstance(remaining[0], dict) else ["id", "query"]
                        with open(resume_path, "w", newline="", encoding="utf-8") as f:
                            w = _csv.DictWriter(f, fieldnames=keys)
                            w.writeheader()
                            w.writerows(remaining)
                        print(f">> [resume] Written {len(remaining)} queries to {resume_path}")
                        new_rotate.append(str(resume_path))
                        any_resumed = True
                    else:
                        new_rotate.append(qf)
                if any_resumed:
                    for cp in config_paths:
                        _parsed_configs[cp].setdefault("defaults", {})["rotate_query_files"] = new_rotate
            elif single_qf:
                resolved_qf = str(BASE_DIR / single_qf) if not Path(single_qf).is_absolute() else single_qf
                remaining = find_resume_point(DATA_DIR, resolved_qf, config=cfg)
                if remaining is not None and len(remaining) < len(load_queries_from_file(resolved_qf)):
                    resume_path = DATA_DIR / "resume_cache.csv"
                    keys = list(remaining[0].keys()) if remaining and isinstance(remaining[0], dict) else ["id", "query"]
                    with open(resume_path, "w", newline="", encoding="utf-8") as f:
                        w = _csv.DictWriter(f, fieldnames=keys)
                        w.writeheader()
                        w.writerows(remaining)
                    print(f">> [resume] Written {len(remaining)} queries to {resume_path}")
                    for cp in config_paths:
                        _parsed_configs[cp]["defaults"]["query_file"] = str(resume_path)

        if run_profile_base is None:
            if sessions > 1:
                run_profile_base = DATA_DIR / "chrome_profiles_gemini" / "default_multi_session"
            else:
                run_profile_base = DATA_DIR / "chrome_profiles_gemini" / f"run_{run_id}"

        effective_api_models = cli_api_models
        if effective_api_models is None:
            cfg = _parsed_configs[config_paths[0]]
            effective_api_models = cfg.get("api_models", []) if isinstance(cfg, dict) else []

        seed_override = args.seed
        if sessions > 1 and seed_override is None:
            seed_override = int(time.time())
            print(f">> Multi-session mode: seed={seed_override}")

        # Per-query cross-session barrier (sync mode only)
        # API models are included as barrier participants so all requests
        # (browser + API) fire simultaneously at each query.
        n_api_workers = len(effective_api_models) if (config_mode == "sync" and effective_api_models) else 0
        if config_mode == "sync" and (sessions > 1 or n_api_workers > 0):
            sync_dir = DATA_DIR / "sync" / run_id
            sync_dir.mkdir(parents=True, exist_ok=True)
            sync_barrier_info = {
                "parties": sessions + n_api_workers,
                "sync_dir": str(sync_dir),
            }
        else:
            sync_barrier_info = None
        browser_api_models = [] if n_api_workers > 0 else effective_api_models

        stop_event = mp.Event()
        api_ready_event = mp.Event()
        procs = []
        _thread_log_state = None

        use_threads = persistent_pages is not None and sessions > 1

        if use_threads:
            log_dir = DATA_DIR / "logs" / run_id
            log_dir.mkdir(parents=True, exist_ok=True)
            _tl_handle = open(log_dir / "combined.log", "a", encoding="utf-8")
            _tl_stdout = sys.stdout
            _tl_stderr = sys.stderr
            sys.stdout = _TimestampedTee(_tl_stdout, [_tl_handle])
            sys.stderr = _TimestampedTee(_tl_stderr, [_tl_handle])
            _thread_log_state = (_tl_stdout, _tl_stderr, _tl_handle)

        if persistent_pages is not None and sessions > 1:
            for sid in range(sessions):
                session_config = config_paths[sid] if len(config_paths) > 1 else config_paths[0]
                kwargs = {
                    "config_path": session_config,
                    "parsed_config": _parsed_configs[session_config],
                    "delay_seconds": run_delay,
                    "run_id": run_id,
                    "session_id": sid,
                    "sync_barrier_info": sync_barrier_info,
                    "seed_override": seed_override,
                    "api_models": browser_api_models,
                    "interface_model": args.interface_model,
                    "allow_manual_login": allow_manual,
                    "page": persistent_pages[sid],
                    "log_to_file": False,
                    "clear_memory_first": bool(args.clear_memory),
                    "clear_only": clear_only,
                    "stop_event": stop_event,
                    "api_ready_event": api_ready_event,
                }
                if config_mode == "sync":
                    kwargs["experiment_index"] = sid
                t = threading.Thread(target=run_audit, kwargs=kwargs)
                t.start()
                procs.append(t)
        elif persistent_pages is not None and sessions == 1:
            session_config = config_paths[0]
            kwargs = {
                "config_path": session_config,
                "parsed_config": _parsed_configs[session_config],
                "delay_seconds": run_delay,
                "run_id": run_id,
                "session_id": 0,
                "sync_barrier_info": sync_barrier_info,
                "seed_override": seed_override,
                "api_models": browser_api_models,
                "interface_model": args.interface_model,
                "allow_manual_login": allow_manual,
                "page": persistent_pages[0],
                "log_to_file": True,
                "clear_memory_first": bool(args.clear_memory),
                "clear_only": clear_only,
                "stop_event": stop_event,
                "api_ready_event": api_ready_event,
            }
            if config_mode == "sync":
                kwargs["experiment_index"] = 0
            try:
                run_audit(**kwargs)
            except Exception as exc:
                print(f">> Session 0 run error: {exc}")
        elif config_mode == "sync" and sessions > 1:
            for exp_idx in range(sessions):
                user_data_dir = _resolve_user_data_dir(run_profile_base, exp_idx, sessions)
                session_config = config_paths[exp_idx] if len(config_paths) > 1 else config_paths[0]
                p = mp.Process(target=run_audit, kwargs={
                    "config_path": session_config,
                    "parsed_config": _parsed_configs[session_config],
                    "delay_seconds": run_delay,
                    "user_data_dir": user_data_dir,
                    "run_id": run_id,
                    "session_id": exp_idx,
                    "sync_barrier_info": sync_barrier_info,
                    "seed_override": seed_override,
                    "api_models": browser_api_models,
                    "interface_model": args.interface_model,
                    "allow_manual_login": allow_manual,
                    "clear_memory_first": bool(args.clear_memory),
                    "clear_only": clear_only,
                    "experiment_index": exp_idx,
                    "stop_event": stop_event,
                    "api_ready_event": api_ready_event,
                })
                p.start()
                procs.append(p)
        else:
            for session_id in range(sessions):
                user_data_dir = _resolve_user_data_dir(run_profile_base, session_id, sessions)
                session_config = config_paths[session_id] if len(config_paths) > 1 else config_paths[0]
                p = mp.Process(target=run_audit, kwargs={
                    "config_path": session_config,
                    "parsed_config": _parsed_configs[session_config],
                    "delay_seconds": run_delay,
                    "user_data_dir": user_data_dir,
                    "run_id": run_id,
                    "session_id": session_id,
                    "sync_barrier_info": sync_barrier_info,
                    "seed_override": seed_override,
                    "api_models": browser_api_models,
                    "interface_model": args.interface_model,
                    "allow_manual_login": allow_manual,
                    "clear_memory_first": bool(args.clear_memory),
                    "clear_only": clear_only,
                    "stop_event": stop_event,
                    "api_ready_event": api_ready_event,
                })
                p.start()
                procs.append(p)

        # Launch API barrier workers (sync mode: each API model is its own
        # barrier participant, so browser + API requests fire simultaneously).
        if n_api_workers > 0 and sync_barrier_info:
            defaults_cfg = _pre_config.get("defaults", {}) if isinstance(_pre_config, dict) else {}
            _single = defaults_cfg.get("query_file")
            _rotate = defaults_cfg.get("rotate_query_files")
            if _single:
                api_query_files = [str(_single)]
            elif _rotate and isinstance(_rotate, list):
                api_query_files = [str(f) for f in _rotate]
            else:
                api_query_files = []
            api_runs = int(defaults_cfg.get("runs", 1))
            api_shuffle = bool(defaults_cfg.get("shuffle", False))
            api_output_base = DATA_DIR / "api" / run_id

            for api_idx, api_cfg in enumerate(effective_api_models):
                api_sid = sessions + api_idx
                api_bi = {
                    "parties": sync_barrier_info["parties"],
                    "sync_dir": sync_barrier_info["sync_dir"],
                    "session_id": api_sid,
                }
                p = mp.Process(target=_api_barrier_worker, kwargs={
                    "query_files": api_query_files,
                    "runs": api_runs,
                    "shuffle": api_shuffle,
                    "seed": seed_override,
                    "api_cfg": api_cfg,
                    "api_key": GEMINI_API_KEY,
                    "output_base": str(api_output_base),
                    "sync_barrier_info": api_bi,
                    "stop_event": stop_event,
                })
                p.start()
                procs.append(p)

        # Wait for all sessions (browser + API) to finish
        for p in procs:
            p.join()

        if _thread_log_state:
            sys.stdout = _thread_log_state[0]
            sys.stderr = _thread_log_state[1]
            _thread_log_state[2].close()

        if not args.no_parse and not clear_only:
            _auto_parse(run_id)

    _top_defaults = _pre_config.get("defaults", {}) if isinstance(_pre_config, dict) else {}
    num_runs = int(args.num_runs or _top_defaults.get("num_runs", 1))
    repeat_hours = float(args.repeat_every_hours or 0)
    repeat_seconds = max(1.0, repeat_hours * 3600.0) if repeat_hours > 0 else 0
    use_persistent = repeat_hours > 0 or num_runs > 1

    profile_base_path = Path(args.profile_base) if args.profile_base else None
    persistent_pages = None
    if use_persistent:
        repeat_profile_base = profile_base_path or DATA_DIR / "chrome_profiles_gemini" / "default_multi_session"
        persistent_pages = []
        for sid in range(sessions):
            user_data_dir = _resolve_user_data_dir(repeat_profile_base, sid, sessions)
            co = ChromiumOptions()
            co.auto_port()
            co.set_tmp_path(str(DATA_DIR / "drission_tmp" / f"session_{sid:02d}"))
            co.set_argument(f"--user-data-dir={user_data_dir}")
            co.set_argument("--no-first-run")
            co.set_argument("--mute-audio")
            persistent_pages.append(ChromiumPage(co))
        print(f">> Created {sessions} persistent ChromiumPage instances.")
    else:
        repeat_profile_base = profile_base_path

    run_count = 0
    first_run = True
    while True:
        allow_manual = bool(args.allow_manual_login) or first_run
        run_once(delay_seconds if first_run else 0, repeat_profile_base, allow_manual,
                 persistent_pages=persistent_pages)
        run_count += 1
        first_run = False

        if num_runs > 1 and run_count >= num_runs:
            print(f">> Completed {run_count}/{num_runs} runs. Done.")
            break

        if repeat_hours <= 0 and num_runs <= 1:
            break

        print(f"\n>> Run {run_count} complete. Sleeping {repeat_seconds:.0f}s before next run...")
        time.sleep(repeat_seconds)
