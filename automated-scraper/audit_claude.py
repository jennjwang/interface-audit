"""Browser scraper for claude.ai — mirrors audit.py's structure and output format.

Supports:
  - Sending queries from CSV/JSON files via the claude.ai web UI
  - Multi-session parallelism (one Chrome profile per session)
  - Saving raw HTML + metadata (same schema as audit.py)
  - Clearing memory / conversations between runs
  - Custom system prompts (via claude.ai's custom instructions / "What would you like...")
  - YAML experiment configs (same format as audit.py's exp.yaml)
  - API-only mode via api_runner.py (Claude SDK)

Usage:
    python audit_claude.py exp_claude.yaml
    python audit_claude.py exp_claude.yaml --sessions 2
    python audit_claude.py exp_claude.yaml --clear-memory
    python audit_claude.py exp_claude.yaml --interface-model claude-opus-4-7
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

try:
    import fcntl
except Exception:
    fcntl = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

from api_runner import run_api_query

sys.path.append(str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "claude_data"


class FileBasedBarrier:
    """Simple file-based barrier for multiprocessing synchronization."""
    def __init__(self, parties, sync_dir, session_id, barrier_id=0):
        self.parties = parties
        self.sync_dir = Path(sync_dir)
        self.session_id = session_id
        self.barrier_id = barrier_id
        self.sync_dir.mkdir(parents=True, exist_ok=True)

    def wait(self, timeout=120):
        """Wait for all parties to reach the barrier, or proceed after timeout seconds."""
        checkpoint_file = self.sync_dir / f"barrier_{self.barrier_id}_session_{self.session_id}.ready"
        checkpoint_file.write_text(str(time.time()))

        deadline = time.time() + timeout
        while True:
            ready_count = len(list(self.sync_dir.glob(f"barrier_{self.barrier_id}_session_*.ready")))
            if ready_count >= self.parties:
                time.sleep(0.05)
                break
            if time.time() > deadline:
                print(f">> [barrier {self.barrier_id}] Timeout after {timeout}s ({ready_count}/{self.parties} ready) — proceeding anyway.")
                break
            time.sleep(0.1)

        self.barrier_id += 1


CLAUDE_URL = "https://claude.ai/login"
CLAUDE_NEW_CHAT_URL = "https://claude.ai/new"

# Env
load_dotenv()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY")

# Models that map a UI label → API slug substring for detection
CLAUDE_MODEL_PATTERNS = [
    "claude-opus",
    "claude-sonnet",
    "claude-haiku",
    "claude-3",
    "claude-4",
]


# ── Timestamped logging (identical to audit.py) ───────────────────────────────

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
    combined_log_path = log_dir / f"claude_session_{session_id:02d}.log"
    combined_handle = open(combined_log_path, "a", encoding="utf-8")
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = _TimestampedTee(stdout_orig, [combined_handle])
    sys.stderr = _TimestampedTee(stderr_orig, [combined_handle])
    return combined_log_path, stdout_orig, stderr_orig, [combined_handle]


def _resolve_user_data_dir(profile_base, session_id, total_sessions):
    base = Path(profile_base) if profile_base else (DATA_DIR / "chrome_profiles_claude")
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"session_{session_id:02d}")


# ── Query loading (identical to audit.py) ────────────────────────────────────

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


def _build_save_group_name(exp_name, interface_model, query_file):
    model_name = _slugify_label(interface_model or exp_name, "model")
    benchmark_name = _extract_benchmark_name(query_file)
    return f"{model_name}-{benchmark_name}"


# ── Human-like typing helpers ─────────────────────────────────────────────────

def _js_shift_enter(ele):
    ele.run_js("""
        ['keydown', 'keypress', 'keyup'].forEach(evType => {
            this.dispatchEvent(new KeyboardEvent(evType, {
                key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
                shiftKey: true, bubbles: true, cancelable: true,
            }));
        });
    """)


def human_type(page, selector, text, allow_typos=False):
    try:
        ele = selector if not isinstance(selector, str) else page.ele(selector, timeout=5)
        if not ele:
            return False
        ele.click()
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
                ele.input(char)
                base_delay = random.uniform(0.01, 0.03)
                if char in ".!?":
                    base_delay += random.uniform(0.12, 0.35)
                elif char in ",;:":
                    base_delay += random.uniform(0.06, 0.18)
                elif char == " " and random.random() < 0.08:
                    base_delay += random.uniform(0.05, 0.2)
                time.sleep(base_delay)
                if allow_typos and char.isalpha() and random.random() < 0.006:
                    ele.input(Keys.BACKSPACE)
                    time.sleep(random.uniform(0.02, 0.08))
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


def paste_prompt(page, selector, text):
    """Insert prompt via synthetic clipboard paste."""
    try:
        ele = selector if not isinstance(selector, str) else page.ele(selector, timeout=5)
        if not ele:
            return False
        if text is None:
            text = ""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
        ele.click()
        time.sleep(0.1)
        escaped = json.dumps(normalized)
        js = f"""
            const el = this;
            el.focus();
            const dt = new DataTransfer();
            dt.setData('text/plain', {escaped});
            const pasteEvt = new ClipboardEvent('paste', {{
                clipboardData: dt,
                bubbles: true,
                cancelable: true,
            }});
            el.dispatchEvent(pasteEvt);
        """
        ele.run_js(js)
        time.sleep(random.uniform(0.3, 0.6))
        # Verify; fall back to direct value set
        content = ele.run_js("return this.innerText || this.value || '';") or ""
        if len(content.strip()) < min(20, len(normalized) // 2):
            print(">> Paste not handled; falling back to direct input")
            ele.run_js(f"this.value = {escaped}; this.dispatchEvent(new Event('input', {{bubbles:true}}));")
            time.sleep(random.uniform(0.2, 0.4))
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


# ── Claude.ai login ───────────────────────────────────────────────────────────

def verify_strict_login(page):
    """Return True only if we are genuinely logged into claude.ai with the chat UI ready."""
    chk = 0.5

    def _visible(ele):
        try:
            return ele and ele.states.is_displayed
        except Exception:
            return False

    # Fail fast on explicit logged-out indicators
    login_btn = page.ele('text:Log in', timeout=chk) or page.ele('text:Sign in', timeout=chk)
    if _visible(login_btn):
        return False
    signup_btn = page.ele('text:Sign up', timeout=chk)
    if _visible(signup_btn):
        return False

    # Require the prompt input to be present and visible — the strongest
    # signal that the full chat UI has loaded for this account
    prompt = (
        page.ele('@@data-testid=chat-input', timeout=chk) or
        page.ele('div[contenteditable="true"]', timeout=chk)
    )
    if _visible(prompt):
        return True

    return False


def handle_google_login(page, allow_manual=True):
    """Navigate to claude.ai and ensure the user is logged in."""
    if verify_strict_login(page):
        return True
    if not allow_manual:
        print(">> Not logged in and manual login is disabled.")
        return False

    print(">> Initiating Claude login sequence...")

    # Click "Log in" / "Sign in" on the landing page if present
    login_btn = (
        page.ele('text:Log in', timeout=3) or
        page.ele('text:Sign in', timeout=2) or
        page.ele('a[href*="login"]', timeout=2)
    )
    if login_btn:
        print(">> Clicking 'Log in'...")
        try:
            login_btn.click()
            time.sleep(1.5)
        except Exception:
            pass

    # Try "Continue with Google"
    google_btn = (
        page.wait.ele_displayed('@@data-testid=login-with-google', timeout=4) or
        page.wait.ele_displayed('text:Continue with Google', timeout=4) or
        page.wait.ele_displayed('@@data-provider=google', timeout=2)
    )
    if google_btn:
        print(">> Clicking 'Continue with Google'...")
        google_btn.click()
        print("\n" + "=" * 60)
        print(">> PLEASE LOG IN MANUALLY NOW (Google sign-in)")
        print(">> IMPORTANT: Select the account that HAS the chats to delete.")
        print("=" * 60 + "\n")
    else:
        print(">> Google button not found — please log in manually if needed.")

    print(">> Waiting for successful login...")
    wait_limit = 600  # 10 minutes
    for i in range(wait_limit * 2):  # 0.5s steps
        if verify_strict_login(page):
            print(">> Login verified!")
            return True
        remaining = wait_limit - i * 0.5
        if i % 40 == 0:
            print(f">> Waiting... ({remaining:.0f}s remaining)")
        time.sleep(0.5)

    print(">> Login timed out.")
    return False


# ── Model selection ───────────────────────────────────────────────────────────

def _extract_model_family(name):
    """Extract core model family (opus/sonnet/haiku) from any model name format.

    Handles: 'opus', 'sonnet', 'haiku', 'claude-opus-4-7', 'claude-sonnet-4-6',
             'Opus 4.7', 'Sonnet 4.6', etc.
    """
    lower = name.lower().replace("-", " ").replace("_", " ")
    for family in ("opus", "sonnet", "haiku"):
        if family in lower:
            return family
    return lower.strip()


def _model_matches(model_name, text):
    """Check if model_name matches a dropdown menu item text."""
    # Direct substring match
    if model_name.lower() in text.lower():
        return True
    # Compare extracted family names (e.g. 'claude-opus-4-7' matches 'Opus 4.7')
    return _extract_model_family(model_name) == _extract_model_family(text)


def select_interface_model(page, model_name, timeout=8):
    """Select a Claude model from the model picker (if available)."""
    try:
        time.sleep(2)
        # Find the model picker button via text content, then walk up to the button
        picker = None
        for label in ('Sonnet', 'Opus', 'Haiku'):
            text_el = page.ele(f'text:{label}', timeout=timeout)
            if text_el:
                # Walk up to the nearest <button> ancestor
                btn = text_el.parent('tag:button')
                if btn:
                    print(f">> Model picker found via '{label}' text -> parent button")
                    picker = btn
                    break
        if not picker:
            print(f">> Model picker not found; skipping model selection.")
            return False

        current_text = (picker.text or "").strip()
        if _model_matches(model_name, current_text):
            print(f">> Model '{model_name}' already selected ({current_text}).")
            return True

        print(f">> Opening model picker for '{model_name}'...")
        picker.click()
        time.sleep(random.uniform(0.8, 1.5))

        menu_items = (
            page.eles('@@role=menuitemradio', timeout=timeout) or
            page.eles('@@role=menuitem', timeout=timeout) or
            page.eles('css:div[role="menuitem"]', timeout=3) or
            page.eles('css:li[role="option"]', timeout=3)
        )
        if not menu_items:
            print(f">> No model menu items found.")
            try:
                page.actions.key_down("Escape").key_up("Escape")
            except Exception:
                pass
            return False

        target = None
        available = []
        for item in menu_items:
            text = (item.text or "").strip()
            available.append(text)
            if _model_matches(model_name, text):
                target = item
                break

        if not target:
            print(f">> '{model_name}' not found in picker. Available: {available}")
            try:
                page.actions.key_down("Escape").key_up("Escape")
            except Exception:
                pass
            return False

        target_label = target.text.split('\n')[0].strip()
        print(f">> Clicking '{target_label}'...")
        try:
            target.click(by_js=True)
        except Exception:
            target.click()
        time.sleep(random.uniform(0.8, 1.5))
        print(f">> Selected model '{model_name}'.")
        return True
    except Exception as e:
        print(f">> Error selecting model: {e}")
        return False


# ── Detect model from response HTML ──────────────────────────────────────────

def detect_model_from_html(html):
    """Return the model slug shown in a claude.ai response page, or None."""
    if not html:
        return None
    # claude.ai embeds model info in data attributes or script tags
    patterns = [
        r'data-model-slug="([^"]+)"',
        r'"modelId"\s*:\s*"([^"]+)"',
        r'"model_id"\s*:\s*"([^"]+)"',
        r'"model"\s*:\s*"(claude-[^"]+)"',
        r'(claude-(?:opus|sonnet|haiku)-[\d\-\.]+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


# ── New chat ──────────────────────────────────────────────────────────────────

def ensure_new_chat(page, attempts=3):
    """Navigate to a fresh claude.ai conversation."""
    for _ in range(attempts):
        try:
            page.get(CLAUDE_NEW_CHAT_URL)
            time.sleep(random.uniform(1.5, 2.5))
            # Wait for the prompt textarea to appear
            if page.ele('div[contenteditable="true"]', timeout=8) or \
               page.ele('textarea', timeout=3):
                return True
        except Exception as e:
            print(f">> ensure_new_chat error: {e}")
            time.sleep(1)
    # Try clicking the New Chat button as fallback
    for _ in range(attempts):
        btn = (
            page.ele('a[href="/new"]', timeout=3) or
            page.ele('text:New chat', timeout=2) or
            page.ele('text:New Chat', timeout=2)
        )
        if btn:
            try:
                btn.click()
                time.sleep(random.uniform(1.0, 2.0))
                return True
            except Exception:
                pass
        page.refresh()
        time.sleep(2)
    return False


# ── Prompt input selector ─────────────────────────────────────────────────────

def _find_prompt_input(page, timeout=8):
    """Return the prompt input element on claude.ai, waiting until it is visible."""
    for selector in (
        '@@data-testid=chat-input',
        'div[contenteditable="true"]',
        '@@contenteditable=true',
        'textarea[placeholder*="message"]',
        'textarea',
    ):
        try:
            ele = page.wait.ele_displayed(selector, timeout=timeout)
            if ele:
                return ele
        except Exception:
            pass
        timeout = 3  # shorter for subsequent fallbacks
    return None


def _find_send_button(page):
    return (
        page.ele('button[aria-label="Send message"]', timeout=3) or
        page.ele('button[data-testid="send-button"]', timeout=2) or
        page.ele('@@aria-label=Send message', timeout=2) or
        page.ele('button[type="submit"]', timeout=2)
    )


# ── Wait for response to complete ────────────────────────────────────────────

def _response_is_complete(page):
    """Return True when Claude has stopped generating.

    The audio/read-aloud button (equalizer icon, viewBox="0 0 21 21") only
    appears after the response is fully rendered.
    """
    return bool(page.ele('css:button svg[viewBox="0 0 21 21"]', timeout=0.2))


# ── Memory management ─────────────────────────────────────────────────────────

def clear_memory(page):
    """Delete all conversation history on claude.ai via the /recents page."""
    print(">> Clearing Claude conversation history...")
    try:
        page.get("https://claude.ai/recents")
        time.sleep(2)

        deleted_any = False
        consecutive_failures = 0

        for iteration in range(500):  # safety cap

            # Check for empty state — no chat list means all deleted
            if not page.ele('css:[data-dd-action-name="conversation-cell"]', timeout=2):
                print(">> No chat list found — all chats deleted.")
                break

            # Click "Select" button to enter selection mode
            select_btn = page.ele('text:Select', timeout=3)
            if select_btn:
                print(">> Clicking 'Select' to enter selection mode...")
                try:
                    select_btn.click()
                except Exception:
                    select_btn.click(by_js=True)
                time.sleep(1.0)
            else:
                print(">> 'Select' button not found — done.")
                break

            # Click "Select all" checkbox (sr-only span with text "Select all")
            select_all_cb = page.ele('css:input[type="checkbox"].sr-only.peer', timeout=3)
            if not select_all_cb:
                print(">> No 'Select all' checkbox found after clicking Select — done.")
                break

            print(">> Clicking 'Select all'...")
            selected = 0
            try:
                select_all_cb.click(by_js=True)
                selected = 1
            except Exception:
                try:
                    select_all_cb.click()
                    selected = 1
                except Exception:
                    pass

            if selected == 0:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print(">> Could not click 'Select all' after 3 attempts — done.")
                    break
                time.sleep(1)
                continue

            time.sleep(5)  # wait for UI to register all selections

            # Click the trash / delete-selected button
            delete_btn = (
                page.ele('@@tag()=button@@aria-label:Delete @@aria-label:selected item', timeout=5) or
                page.ele('css:button[aria-label^="Delete "]', timeout=3) or
                page.ele('css:button[aria-label*="Delete"]', timeout=3) or
                page.ele('css:button[aria-label*="delete"]', timeout=2) or
                page.ele('text:Delete selected', timeout=2) or
                page.ele('css:button[data-testid*="delete"]', timeout=2)
            )
            if delete_btn:
                print(f">> Delete button found: aria-label='{delete_btn.attr('aria-label')}'")
            if not delete_btn:
                print(">> Delete-selected button not found.")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
                # Reload page and retry
                page.get("https://claude.ai/recents")
                time.sleep(2)
                continue

            consecutive_failures = 0

            try:
                delete_btn.click()
            except Exception:
                delete_btn.click(by_js=True)
            print(f">> Delete button clicked — waiting for confirm dialog...")

            # Wait for confirm dialog to appear, try all selectors
            confirm_btn = None
            for _ in range(10):
                time.sleep(0.5)
                confirm_btn = (
                    page.ele('button[data-testid="delete-modal-confirm"]', timeout=0.5) or
                    page.ele('css:div[role="dialog"][data-state="open"] button.Button_danger__cHFE4', timeout=0.5) or
                    page.ele('css:div[role="dialog"][data-state="open"] button[type="button"]:last-child', timeout=0.5)
                )
                if confirm_btn:
                    break

            if not confirm_btn:
                print(">> Confirm dialog not found — reloading and retrying...")
                page.get("https://claude.ai/recents")
                time.sleep(2)
                continue

            print(">> Confirm dialog found — clicking Delete...")
            try:
                confirm_btn.click()
            except Exception:
                confirm_btn.click(by_js=True)
            print(f">> Clicked confirm — waiting for dialog to close...")
            # Wait until no open dialog is visible (up to 15s)
            dialog_closed = False
            for _ in range(30):
                time.sleep(0.5)
                try:
                    dialog = page.ele('css:div[role="dialog"][data-state="open"]', timeout=0.3)
                    if not dialog or not dialog.states.is_displayed:
                        dialog_closed = True
                        break
                except Exception:
                    dialog_closed = True
                    break
            if not dialog_closed:
                print(">> Dialog may still be open — proceeding anyway...")
            print(f">> Deleted batch of {selected} chat(s) — refreshing page...")
            page.get("https://claude.ai/recents")
            time.sleep(2)
            deleted_any = True

        if deleted_any:
            print(">> Conversation history cleared.")
        else:
            print(">> No conversations deleted.")
        return deleted_any
    except Exception as e:
        print(f">> Error clearing memory: {e}")
        return False


# ── Parse saved HTML ──────────────────────────────────────────────────────────

def parse_html_file(html_path, meta_path=None):
    """Extract user query and assistant response from a saved claude.ai or chatgpt.com HTML file."""
    if BeautifulSoup is None:
        raise RuntimeError("BeautifulSoup (bs4) is required — pip install beautifulsoup4")
    content = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(content, "html.parser")

    user_text = ""
    asst_text = ""
    model_slug = None

    if "chatgpt.com" in content or "data-message-author-role" in content:
        # ChatGPT: data-message-author-role="user" / "assistant"
        user_nodes = soup.find_all(attrs={"data-message-author-role": "user"})
        if user_nodes:
            node = user_nodes[-1]
            target = node.find("div", class_=lambda c: c and "whitespace-pre-wrap" in c) or node
            user_text = target.get_text("\n", strip=True)
        asst_nodes = soup.find_all(attrs={"data-message-author-role": "assistant"})
        if asst_nodes:
            node = asst_nodes[-1]
            target = node.find("div", class_=lambda c: c and "markdown" in c) or node
            asst_text = target.get_text("\n", strip=True)
            model_slug = node.get("data-message-model-slug")
    else:
        # Claude: data-testid="user-message" + data-is-streaming div
        user_node = soup.find(attrs={"data-testid": "user-message"})
        user_text = user_node.get_text("\n", strip=True) if user_node else ""
        asst_nodes = soup.find_all(attrs={"data-is-streaming": True})
        if asst_nodes:
            asst_text = asst_nodes[-1].get_text("\n", strip=True)

    if not user_text and not asst_text:
        return None

    meta = {}
    if meta_path and Path(meta_path).exists():
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        except Exception:
            pass

    if not model_slug:
        model_slug = meta.get("model_slug")

    parsed = {
        "query_parsed": user_text,
        "ai_generated_output_text": asst_text,
    }
    if model_slug:
        parsed["model_slug"] = model_slug
    if meta:
        parsed["meta"] = meta
    return parsed


def parse_saved_htmls(run_id, raw_root=None, parsed_root=None):
    """Parse all saved HTML files for a run into JSON."""
    if BeautifulSoup is None:
        print(">> Parse skipped: BeautifulSoup not installed (pip install beautifulsoup4)")
        return
    raw_root = Path(raw_root) if raw_root else (DATA_DIR / "raw_html" / run_id)
    parsed_root = Path(parsed_root) if parsed_root else (DATA_DIR / "parsed_json" / run_id)

    if not raw_root.exists():
        print(f">> Parse skipped: raw_html not found at {raw_root}")
        return

    html_files = sorted(raw_root.rglob("*.html"))
    if not html_files:
        print(f">> Parse skipped: no HTML files found under {raw_root}")
        return

    parsed_count = 0
    skipped = 0
    for html_path in html_files:
        rel_path = html_path.relative_to(raw_root)
        output_path = parsed_root / rel_path.with_suffix(".json")
        if output_path.exists():
            skipped += 1
            continue
        meta_path = html_path.with_suffix(".meta.json")
        try:
            parsed = parse_html_file(html_path, meta_path=meta_path)
        except Exception as e:
            print(f">> Parse failed for {html_path}: {e}")
            continue
        if not parsed:
            print(f">> Parse skipped (empty) for {html_path}")
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        parsed_count += 1

    print(f">> Parsed {parsed_count} HTML files to {parsed_root} (skipped {skipped} already parsed)")


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
        "platform": "claude.ai",
    }
    if sent_at:
        meta["sent_at"] = sent_at
    if model_info.get("model_slug"):
        meta["model_slug"] = model_info["model_slug"]

    meta_filename = output_dir / f"{safe_id}.meta.json"
    with open(meta_filename, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f">> Saved metadata to {meta_filename}")


# ── Core experiment runner ────────────────────────────────────────────────────

def _rotate_query_file(experiment, defaults, rotate_slot=None):
    """Return the query file to use, rotating through a list if configured.

    If ``rotate_query_files`` (list) is present in the experiment or defaults,
    the next file in the list is selected based on a small JSON state file
    stored alongside the data directory. By default rotation is global across
    experiments so parallel model sessions advance through the list instead of
    all choosing index 0. Set ``rotate_scope: experiment`` to keep independent
    per-experiment rotation state.

    Falls back to ``query_file`` (single value) when no rotation list is found.
    """
    rotate_list = experiment.get("rotate_query_files") or defaults.get("rotate_query_files")
    if not rotate_list or not isinstance(rotate_list, list):
        return experiment.get("query_file", defaults.get("query_file")), None

    exp_name = experiment.get("name", "default")
    if rotate_slot is not None:
        idx = int(rotate_slot) % len(rotate_list)
        chosen = rotate_list[idx]
        print(
            f">> [rotate] Experiment '{exp_name}': using file [{idx+1}/{len(rotate_list)}] "
            f"{chosen} (scope=session-slot, slot={rotate_slot})"
        )
        return chosen, idx

    rotate_scope = str(experiment.get("rotate_scope", defaults.get("rotate_scope", "global"))).strip().lower()
    if rotate_scope in {"experiment", "per_experiment", "per-experiment"}:
        key = f"exp::{exp_name}"
    else:
        # Key by list contents so unrelated configs don't share a counter.
        sig = "||".join(str(x) for x in rotate_list)
        key = f"global::{hashlib.sha1(sig.encode('utf-8')).hexdigest()[:16]}"

    state_path = DATA_DIR / "rotate_state.json"
    lock_path = DATA_DIR / "rotate_state.lock"
    lock_handle = None
    state = {}
    idx = 0

    try:
        lock_handle = open(lock_path, "a+", encoding="utf-8")
        if fcntl:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        raw_idx = state.get(key)
        if raw_idx is None:
            # Backward compatibility: pre-scope state used plain experiment names.
            for legacy_key, legacy_val in state.items():
                if isinstance(legacy_key, str) and (
                    legacy_key.startswith("global::") or legacy_key.startswith("exp::")
                ):
                    continue
                try:
                    raw_idx = int(legacy_val)
                    break
                except Exception:
                    continue
        idx = int(raw_idx or 0) % len(rotate_list)
        state[key] = (idx + 1) % len(rotate_list)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as e:
        print(f">> Warning: could not update rotate state: {e}")
    finally:
        if lock_handle:
            if fcntl:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            lock_handle.close()

    chosen = rotate_list[idx]
    print(
        f">> [rotate] Experiment '{exp_name}': using file [{idx+1}/{len(rotate_list)}] "
        f"{chosen} (scope={rotate_scope})"
    )
    return chosen, idx


def _query_file_sequence(experiment, defaults, prefer_defaults_rotate=False):
    """Return query files in sequence for this experiment.

    In sync multi-session mode we prefer defaults-level rotation so all
    experiments advance through the same file list together.
    """
    if prefer_defaults_rotate:
        rotate_list = defaults.get("rotate_query_files")
    else:
        rotate_list = experiment.get("rotate_query_files") or defaults.get("rotate_query_files")

    if rotate_list and isinstance(rotate_list, list):
        return [str(qf) for qf in rotate_list]

    single = experiment.get("query_file", defaults.get("query_file"))
    return [single] if single else []


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
    rotate_slot=None,
    query_file_override=None,
):
    api_models = api_models or []
    exp_name = experiment.get("name", "experiment")
    if query_file_override is not None:
        query_file = query_file_override
        _rotate_idx = None
    else:
        query_file, _rotate_idx = _rotate_query_file(experiment, defaults, rotate_slot=rotate_slot)
    runs = int(experiment.get("runs", defaults.get("runs", 1)))
    shuffle = bool(experiment.get("shuffle", defaults.get("shuffle", False)))
    exp_seed = experiment.get("seed", defaults.get("seed"))
    reuse_chat = bool(experiment.get("reuse_chat", defaults.get("reuse_chat", False)))
    fast_mode = bool(experiment.get("fast", defaults.get("fast", False)))
    wait_after_saves = experiment.get("wait_after_saves", defaults.get("wait_after_saves", [3, 7]))
    interface_model = experiment.get("interface_model", interface_model)
    save_group_name = _build_save_group_name(exp_name, interface_model, query_file)

    if seed_override is not None and "seed" not in experiment:
        exp_seed = seed_override

    output_dir = DATA_DIR / "raw_html" / run_id / f"session_{session_id:02d}" / save_group_name
    api_output_base = DATA_DIR / "api" / run_id / save_group_name

    print(f"\n>>> STARTING EXPERIMENT: {exp_name}")
    print(f"    Query File: {query_file}")
    if interface_model:
        print(f"    Interface Model: {interface_model}")
    print(f"    Save Group: {save_group_name}")
    print(f"    Output Directory: {output_dir}")
    if api_models:
        print(f"    API Models: {', '.join(m['model'] for m in api_models)}")

    if _rotate_idx is not None:
        print(f">> [rotate] Clearing memory before experiment '{exp_name}'...")
        clear_memory(page)
        time.sleep(random.uniform(0.6, 1.4))

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

    prev_reuse = None
    for i, item in enumerate(queries):
        if stop_event and stop_event.is_set():
            print(">> Stop signal received. Halting.")
            return

        q_id = item["id"]
        q_text = item["query"]
        item_reuse = item.get("reuse_chat")
        effective_reuse = reuse_chat if item_reuse is None else bool(item_reuse)
        is_mcq = item.get("is_mcq")
        if is_mcq is None:
            is_mcq = "_mcq" in str(q_id or "").lower()

        # Cross-session barrier: all sessions synchronize at the top of each query
        if sync_barrier:
            print(f">> [session {session_id}] Waiting at query barrier {i}...")
            try:
                sync_barrier.wait()
                print(f">> [session {session_id}] All sessions at query {i+1}. Proceeding.")
            except TimeoutError as e:
                print(f">> Barrier timeout: {e} — proceeding anyway")

        # Skip if already saved — allows resuming an interrupted run
        _sid = "".join(c for c in str(q_id or "") if c.isalnum() or c in ("-", "_")).strip()
        if (output_dir / f"{_sid}.html").exists():
            print(f">> [{i+1}/{len(queries)}] Skipping {q_id} (already saved).")
            continue

        api_threads = []
        try:
            print(f"\n[{i+1}/{len(queries)}] Processing {q_id}...")
            skip_query = False

            # Open a new chat when not reusing
            if not effective_reuse:
                ensure_new_chat(page)
                if interface_model:
                    select_interface_model(page, interface_model)
            prev_reuse = effective_reuse

            prompt_input = _find_prompt_input(page)
            if not prompt_input:
                print(">> Prompt input not found. Refreshing...")
                page.refresh()
                time.sleep(5)
                prompt_input = _find_prompt_input(page)
                if not prompt_input:
                    print(">> Still no prompt input. Skipping query.")
                    skip_query = True

            if not skip_query:
                print(f">> Sending: {q_text[:60].replace(chr(10), ' ')}...")

                # Re-fetch to avoid stale element after navigation / model selection
                prompt_input = _find_prompt_input(page)
                if not prompt_input:
                    print(">> Prompt input gone before typing. Skipping.")
                    skip_query = True

            if not skip_query:
                if fast_mode or is_mcq:
                    typed = paste_prompt(page, prompt_input, q_text)
                else:
                    _think_delay(q_text)
                    typed = human_type(page, prompt_input, q_text) or \
                            paste_prompt(page, prompt_input, q_text)

                if not typed:
                    print(">> Failed to enter prompt. Skipping.")
                    skip_query = True

            sent_at = None
            if not skip_query:
                send_btn = _find_send_button(page)
                if send_btn:
                    time.sleep(random.uniform(0.15, 0.6))
                    try:
                        send_btn.click()
                    except Exception:
                        send_btn.click(by_js=True)
                else:
                    # Fallback: Enter key
                    prompt_input = _find_prompt_input(page, timeout=3)
                    if prompt_input:
                        time.sleep(random.uniform(0.15, 0.4))
                        prompt_input.run_js("""
                            this.dispatchEvent(new KeyboardEvent('keydown', {
                                key:'Enter',code:'Enter',keyCode:13,which:13,
                                bubbles:true,cancelable:true
                            }));
                        """)
                sent_at = datetime.now().isoformat()
                if session_id == 0:
                    for api_cfg in api_models:
                        m_name = api_cfg["model"]
                        m_params = {k: v for k, v in api_cfg.items() if k != "model"}
                        m_output = api_output_base / m_name
                        t = threading.Thread(
                            target=run_api_query,
                            args=(q_id, q_text, m_output, m_name),
                            kwargs={"sent_at": sent_at, "api_key": ANTHROPIC_KEY, **m_params},
                            daemon=True,
                        )
                        t.start()
                        api_threads.append(t)

            if not skip_query:
                print(">> Waiting for response...")
                wait_start = time.time()
                wait_timeout = 600
                soft_refresh_timeout = 90
                soft_refreshed = False
                poll_interval = 0.5

                while True:
                    if _response_is_complete(page):
                        print(">> Response complete.")
                        break
                    elapsed = time.time() - wait_start
                    if not soft_refreshed and elapsed > soft_refresh_timeout:
                        print(f">> No response after {soft_refresh_timeout}s — soft-refreshing...")
                        page.refresh()
                        time.sleep(5)
                        soft_refreshed = True
                    if elapsed > wait_timeout:
                        print(f">> Timed out after {wait_timeout}s. Re-sending...")
                        ensure_new_chat(page)
                        if interface_model:
                            select_interface_model(page, interface_model)
                        prompt_input = _find_prompt_input(page)
                        if prompt_input:
                            if fast_mode or is_mcq:
                                paste_prompt(page, prompt_input, q_text)
                            else:
                                human_type(page, prompt_input, q_text) or \
                                    paste_prompt(page, prompt_input, q_text)
                            send_btn = _find_send_button(page)
                            if send_btn:
                                time.sleep(random.uniform(0.15, 0.4))
                                try:
                                    send_btn.click()
                                except Exception:
                                    send_btn.click(by_js=True)
                        wait_start = time.time()
                    time.sleep(poll_interval)

                final_html = page.html
                model_slug = detect_model_from_html(final_html)
                print(f">> Model detected: {model_slug or 'unknown'}")

                is_mcq_item = item.get("is_mcq")
                skip_save = False
                if save_mcq_only:
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

        except Exception as _query_err:
            import traceback
            print(f"\n>> !! Query {q_id} failed with unexpected error: {_query_err}")
            traceback.print_exc()
            print(">> Continuing to next query.")
        finally:
            for t in api_threads:
                t.join()

        # Wait between queries (skip if configured to 0)
        if isinstance(wait_after_saves, list) and len(wait_after_saves) == 2:
            sleep_time = random.uniform(wait_after_saves[0], wait_after_saves[1])
        else:
            sleep_time = 0 if wait_after_saves is None else float(wait_after_saves)
        if sleep_time > 0:
            print(f">> Waiting {sleep_time:.2f}s before next query...")
            time.sleep(sleep_time)


# ── Main audit runner ─────────────────────────────────────────────────────────

def run_audit(
    config_path,
    delay_seconds=0,
    user_data_dir=None,
    run_id=None,
    session_id=0,
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
    sync_barrier_info=None,
    login_barrier_obj=None,
):
    # Reconstruct sync barrier from serialisable info (for multiprocessing)
    sync_barrier = None
    if sync_barrier_info:
        sync_barrier = FileBasedBarrier(
            parties=sync_barrier_info["parties"],
            sync_dir=sync_barrier_info["sync_dir"],
            session_id=session_id,
        )

    # login_barrier_obj is a manager.Barrier passed directly
    login_barrier = login_barrier_obj

    combined_log_path = None
    stdout_orig = stderr_orig = log_handles = None
    if run_id and log_to_file:
        combined_log_path, stdout_orig, stderr_orig, log_handles = \
            _setup_timestamped_logging(run_id, session_id)
        print(f">> Logging to {combined_log_path}")

    try:
        # Load config
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
                for file_idx, qf in enumerate(query_files):
                    if len(query_files) > 1:
                        print(
                            f">> [rotate-seq] Experiment '{experiment.get('name', 'experiment')}': "
                            f"file [{file_idx + 1}/{len(query_files)}] {qf}"
                        )
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
                            m_output = api_output_base / m_dir
                            print(f"\n[{j+1}/{len(queries)}] {q['id']} -> {m_name}")
                            run_api_query(q["id"], q["query"], m_output, m_name,
                                          api_key=ANTHROPIC_KEY, **m_params)
            print(f"\n>> [API] Session {session_id} complete.")
            return

        # Browser mode
        if page is None:
            co = ChromiumOptions()
            co.auto_port()
            co.set_tmp_path(str(DATA_DIR / "drission_tmp" / f"session_{session_id:02d}"))
            if user_data_dir:
                co.set_argument(f"--user-data-dir={user_data_dir}")
            co.set_argument("--no-first-run")
            co.set_argument("--mute-audio")
            page = ChromiumPage(co)

        print(">> Navigating to Claude.ai...")
        page.get(CLAUDE_NEW_CHAT_URL)
        time.sleep(random.uniform(1.5, 2.5))

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
                page.get(CLAUDE_URL)
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

        if login_barrier:
            print(f">> [session {session_id}] Logged in — waiting for all sessions to log in...")
            try:
                login_barrier.wait()
                print(f">> [session {session_id}] All sessions logged in. Starting experiments.")
            except Exception as e:
                print(f">> [session {session_id}] Login barrier error ({e}) — proceeding anyway.")

        # Navigate to new chat so the model picker and prompt input are ready
        ensure_new_chat(page)

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
                        try:
                            print(f">> [rotate-seq] Clearing memory before file [{file_idx + 1}]...")
                            clear_memory(page)
                            time.sleep(random.uniform(0.6, 1.4))
                        except Exception as e:
                            print(f">> [rotate-seq] Clear memory failed: {e}")
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
                    )
        except Exception as exc:
            print(f"\n>> SESSION {session_id} ERROR: {exc}")
            if stop_event:
                stop_event.set()
            raise

        print("\nAll Experiments Completed.")

        # Always clear conversation history at end of run
        try:
            clear_memory(page)
            time.sleep(random.uniform(0.6, 1.4))
        except Exception as e:
            print(f">> End-of-run clear failed: {e}")

        # Navigate back to a clean state (use /new so login check works on next repeat)
        try:
            page.get(CLAUDE_NEW_CHAT_URL)
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
        description="Browser scraper for claude.ai (mirrors audit.py)")
    parser.add_argument("config", nargs="?", default="yamls/exp_claude.yaml",
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
                        help='Comma-separated Claude API models, e.g. '
                             '"claude-opus-4-7,claude-sonnet-4-6:0.7". '
                             'Param after colon is temperature (float) or '
                             'budget_tokens (int for extended thinking).')
    parser.add_argument("--clear-memory", action="store_true",
                        help="Clear conversation history before running experiments.")
    parser.add_argument("--email", type=str, default=None,
                        help="Google account email to select during login (e.g. user@gmail.com).")
    parser.add_argument("--interface-model", type=str, default=None,
                        help="Model to select in claude.ai UI dropdown.")
    parser.add_argument("--output-tag", type=str, default=None,
                        help="Tag appended to run ID for output directories.")
    parser.add_argument("--repeat-every-hours", type=float, default=None,
                        help="Re-run the experiment every N hours indefinitely.")
    args = parser.parse_args()

    load_dotenv()

    def _resolve_config(raw, required=True):
        p = Path(raw)
        if p.exists():
            return p
        p2 = BASE_DIR / raw
        if p2.exists():
            return p2
        if not required:
            return None
        print(f"Config file not found: {raw}")
        sys.exit(1)

    clear_only_mode = bool(args.clear_memory) and not args.configs and args.config == "exp_claude.yaml"

    if args.configs:
        raw_configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        config_paths = [_resolve_config(c) for c in raw_configs]
    else:
        resolved = _resolve_config(args.config, required=not clear_only_mode)
        config_paths = [resolved] if resolved else [None]

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
                    val = float(param_value)
                    if val == int(val) and val > 1:
                        cli_api_models.append({"model": model_name,
                                               "budget_tokens": int(val)})
                    else:
                        cli_api_models.append({"model": model_name,
                                               "temperature": val})
                except ValueError:
                    print(f"Warning: Cannot parse param '{param_value}' for '{model_name}'. Skipping.")
            else:
                cli_api_models.append({"model": raw})

    # Pre-parse configs
    _parsed_configs = {}
    for cp in config_paths:
        if cp is None:
            _parsed_configs[cp] = {}
            continue
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

    def run_once(run_delay, run_profile_base, allow_manual, persistent_pages=None):
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.output_tag:
            run_id = f"{run_id}-{args.output_tag}"
        clear_only = bool(args.clear_memory)

        if run_profile_base is None:
            if sessions > 1 or clear_only:
                run_profile_base = DATA_DIR / "chrome_profiles_claude" / "default_multi_session"
            else:
                run_profile_base = DATA_DIR / "chrome_profiles_claude" / f"run_{run_id}"

        effective_api_models = cli_api_models
        if effective_api_models is None:
            cfg = _parsed_configs[config_paths[0]]
            effective_api_models = cfg.get("api_models", []) if isinstance(cfg, dict) else []

        seed_override = args.seed
        if sessions > 1 and seed_override is None:
            seed_override = int(time.time())
            print(f">> Multi-session mode: seed={seed_override}")

        stop_event = mp.Event()
        api_ready_event = mp.Event()
        procs = []
        _thread_log_state = None

        # Create barrier info for per-query cross-session synchronisation
        sync_barrier_info = None
        if config_mode == "sync" and sessions > 1:
            sync_dir = DATA_DIR / "sync" / run_id / "queries"
            sync_dir.mkdir(parents=True, exist_ok=True)
            sync_barrier_info = {
                "parties": sessions,
                "sync_dir": str(sync_dir),
            }

        use_threads = persistent_pages is not None

        if use_threads and sessions > 1:
            log_dir = DATA_DIR / "logs" / run_id
            log_dir.mkdir(parents=True, exist_ok=True)
            _tl_handle = open(log_dir / "combined.log", "a", encoding="utf-8")
            _tl_stdout = sys.stdout
            _tl_stderr = sys.stderr
            sys.stdout = _TimestampedTee(_tl_stdout, [_tl_handle])
            sys.stderr = _TimestampedTee(_tl_stderr, [_tl_handle])
            _thread_log_state = (_tl_stdout, _tl_stderr, _tl_handle)

        if persistent_pages is not None:
            for sid in range(sessions):
                session_config = config_paths[sid] if len(config_paths) > 1 else config_paths[0]
                kwargs = {
                    "config_path": session_config,
                    "parsed_config": _parsed_configs[session_config],
                    "delay_seconds": run_delay,
                    "run_id": run_id,
                    "session_id": sid,
                    "seed_override": seed_override,
                    "api_models": effective_api_models,
                    "interface_model": args.interface_model,
                    "allow_manual_login": allow_manual,
                    "page": persistent_pages[sid],
                    "log_to_file": False,
                    "clear_memory_first": bool(args.clear_memory),
                    "clear_only": clear_only,
                    "stop_event": stop_event,
                    "api_ready_event": api_ready_event,
                    "sync_barrier_info": sync_barrier_info,
                }
                if config_mode == "sync":
                    kwargs["experiment_index"] = sid
                t = threading.Thread(target=run_audit, kwargs=kwargs)
                t.start()
                procs.append(t)
        elif config_mode == "sync" and sessions > 1:
            manager = mp.Manager()
            login_barrier_obj = manager.Barrier(sessions) if sessions > 1 else None
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
                    "seed_override": seed_override,
                    "api_models": effective_api_models,
                    "interface_model": args.interface_model,
                    "allow_manual_login": allow_manual,
                    "clear_memory_first": bool(args.clear_memory),
                    "clear_only": clear_only,
                    "experiment_index": exp_idx,
                    "stop_event": stop_event,
                    "api_ready_event": api_ready_event,
                    "sync_barrier_info": sync_barrier_info,
                    "login_barrier_obj": login_barrier_obj,
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
                    "seed_override": seed_override,
                    "api_models": effective_api_models,
                    "interface_model": args.interface_model,
                    "allow_manual_login": allow_manual,
                    "clear_memory_first": bool(args.clear_memory),
                    "clear_only": clear_only,
                    "stop_event": stop_event,
                    "api_ready_event": api_ready_event,
                })
                p.start()
                procs.append(p)

        for p in procs:
            p.join()

        if _thread_log_state:
            sys.stdout = _thread_log_state[0]
            sys.stderr = _thread_log_state[1]
            _thread_log_state[2].close()

        if not args.no_parse:
            print(f"\n>> Parsing saved HTML files for run {run_id}...")
            parse_saved_htmls(run_id)

    profile_base_path = Path(args.profile_base) if args.profile_base else None
    if args.repeat_every_hours:
        interval = args.repeat_every_hours * 3600
        # Create persistent browser pages once so login state is preserved across rounds
        repeat_profile_base = profile_base_path or DATA_DIR / "chrome_profiles_claude" / "default_multi_session"
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
        run_once(delay_seconds, repeat_profile_base, True, persistent_pages=persistent_pages)
        while True:
            print(f"\n>> Waiting {args.repeat_every_hours}h before next run...")
            time.sleep(interval)
            run_once(0, repeat_profile_base, False, persistent_pages=persistent_pages)
    else:
        run_once(delay_seconds, profile_base_path, True)
