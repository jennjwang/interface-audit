import time
import random
import os
import sys
import yaml
import json
import csv
import argparse
import re
import multiprocessing as mp
import threading
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from DrissionPage import ChromiumPage, ChromiumOptions
from DrissionPage.common import Keys
from api_runner import run_api_query
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# Allow imports from parent directory for utils if needed, though we implementing logic here
sys.path.append(str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

class _TimestampedTee:
    """Write to console as-is, and to log(s) with timestamps per line."""
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
    combined_log_path = log_dir / "combined.log"
    combined_handle = open(combined_log_path, "a", encoding="utf-8")
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = _TimestampedTee(stdout_orig, [combined_handle])
    sys.stderr = _TimestampedTee(stderr_orig, [combined_handle])
    return combined_log_path, stdout_orig, stderr_orig, [combined_handle]

def _resolve_user_data_dir(profile_base, session_id, total_sessions):
    """Return per-session Chrome user-data-dir (avoid profile contention)."""
    base = Path(profile_base) if profile_base else (DATA_DIR / "chrome_profiles")
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
        # Create our checkpoint file
        checkpoint_file = self.sync_dir / f"barrier_{self.barrier_id}_session_{self.session_id}.ready"
        checkpoint_file.write_text(str(time.time()))

        # Wait for all other parties
        while True:
            ready_count = len(list(self.sync_dir.glob(f"barrier_{self.barrier_id}_session_*.ready")))
            if ready_count >= self.parties:
                # All parties ready, clean up and proceed
                time.sleep(0.1)  # Small delay to ensure all see the files
                break

            time.sleep(0.2)

        # Increment barrier_id for next wait
        self.barrier_id += 1

# Load credentials
load_dotenv()
EMAIL = os.getenv("OPENAI_EMAIL")
PASSWORD = os.getenv("OPENAI_PASSWORD")
OPENAI_KEY = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")

# Path to your Chrome Profile
USER_DATA_PATH = r"/Users/jenniferwang/Library/Application Support/Google/Chrome"

# Models that indicate a rate-limit downgrade — abort immediately.
DOWNGRADE_SLUGS = {"gpt-4o-mini", "gpt-5-mini"}

# Page-level error text that signals the session should stop.
UNUSUAL_ACTIVITY_TEXT = "our systems have detected unusual activity coming from your system"


class ModelDowngradeError(RuntimeError):
    """Raised when a response comes from a downgraded model or wrong model variant."""
    pass


def is_expected_model(model_slug: str | None) -> bool:
    """Check if model slug matches expected gpt-5 variants (excluding mini).

    Accepts: gpt-5, gpt-5.2, gpt-5-2, etc.
    Rejects: gpt-5-mini, gpt-4o, gpt-4o-mini, None, etc.
    """
    if not model_slug:
        return False
    # Must start with gpt-5
    if not model_slug.startswith("gpt-5"):
        return False
    # Must not be a mini variant
    if "mini" in model_slug.lower():
        return False
    return True


class UnusualActivityError(RuntimeError):
    """Raised when ChatGPT shows the 'unusual activity' block page."""
    pass


def _check_unusual_activity(page):
    """Raise UnusualActivityError if the page shows the unusual-activity warning."""
    html = (page.html or "").lower()
    if UNUSUAL_ACTIVITY_TEXT in html:
        raise UnusualActivityError(
            "ChatGPT blocked this session: 'Our systems have detected unusual activity "
            "coming from your system. Please try again later.'"
        )

def _js_shift_enter(ele):
    """Send Shift+Enter via JS KeyboardEvent — avoids the modifier-key
    timing bug where a bare Enter accidentally submits the chat."""
    ele.run_js("""
        ['keydown', 'keypress', 'keyup'].forEach(evType => {
            this.dispatchEvent(new KeyboardEvent(evType, {
                key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
                shiftKey: true, bubbles: true, cancelable: true,
            }));
        });
    """)


def human_type(page, selector, text, allow_typos=False):
    """Simulate human typing, preserving newlines via Shift+Enter."""
    try:
        ele = page.ele(selector, timeout=5)
        if not ele: return False
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
                    ele.input(char)
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
    """Insert prompt via a synthetic clipboard paste so React state stays in sync."""
    try:
        ele = page.ele(selector, timeout=5)
        if not ele:
            return False
        if text is None:
            text = ""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if normalized.endswith("\n"):
            normalized = normalized.rstrip("\n")
        ele.click()
        time.sleep(0.1)

        import json
        escaped = json.dumps(normalized)  # JS-safe string

        # Dispatch a synthetic ClipboardEvent('paste') with the text in its
        # DataTransfer.  ChatGPT's ProseMirror / React handler listens for
        # paste events and will insert multi-line text correctly (including
        # wrapping lines in <p> tags and updating internal state).
        js = f"""
            const el = this;
            el.focus();

            // Clear any existing content first
            el.innerHTML = '<p><br></p>';
            el.dispatchEvent(new InputEvent('input', {{bubbles: true}}));

            // Build a synthetic paste event
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

        # Verify content was inserted (fallback to old innerHTML method)
        content = ele.run_js("return this.innerText;") or ""
        if len(content.strip()) < min(20, len(normalized) // 2):
            print(">> Paste event not handled; falling back to innerHTML method")
            js_fallback = f"""
                const el = this;
                el.focus();
                const lines = {escaped}.split('\\n');
                el.innerHTML = lines.map(l => '<p>' + (l || '<br>') + '</p>').join('');
                el.dispatchEvent(new InputEvent('input', {{bubbles: true}}));
            """
            ele.run_js(js_fallback)
            time.sleep(random.uniform(0.2, 0.4))

        return True
    except Exception as e:
        print(f"Error inserting prompt: {e}")
        return False

def _maybe_scroll(page):
    if random.random() < 0.15:
        delta = random.randint(-250, 500)
        try:
            page.run_js(f"window.scrollBy(0, {delta});")
            time.sleep(random.uniform(0.2, 0.6))
        except Exception:
            pass

def _think_delay(text):
    if not text:
        return
    length = len(text)
    delay = 0.2 + (length / 900)
    delay += random.uniform(0.0, 0.5)
    delay = min(delay, 2.8)
    if random.random() < 0.05:
        delay += random.uniform(1.0, 3.0)
    time.sleep(delay)
def verify_strict_login(page):
    """
    STRICT CHECK: Returns True ONLY if we are genuinely logged in.
    Guest mode also has a profile button, so we must check for the ABSENCE
    of 'Log in' / 'Sign up' buttons.
    """
    chk_timeout = 0.1

    # 0. CHECK FOR CHAT HISTORY (Strong Signal)
    if page.ele('nav[aria-label="Chat history"]', timeout=chk_timeout) or \
       page.ele('div.flex-1.overflow-y-auto', timeout=chk_timeout): 
         return True
    
    def _is_displayed(ele):
        try:
            return ele and ele.states.is_displayed
        except Exception:
            return False

    # 1. Check for "Log in" or "Sign up" buttons.
    login_btn = page.ele('text:Log in', timeout=chk_timeout)
    signup_btn = page.ele('text:Sign up', timeout=chk_timeout)
    
    if _is_displayed(login_btn):
        return False
    
    if _is_displayed(signup_btn):
        return False

    # 2. If no visible login buttons, AND we see logged-in elements
    profile_btn = page.ele('@@data-testid=profile-button', timeout=chk_timeout)
    profile_img = page.ele('img[alt="Profile image"]', timeout=chk_timeout)
    user_menu = page.ele('.user-menu', timeout=chk_timeout)
    model_selector = (
        page.ele('button[aria-label^="Model selector"]', timeout=chk_timeout) or
        page.ele('button.__composer-pill[aria-haspopup="menu"]', timeout=1)
    )
    new_chat = page.ele('text:New chat', timeout=chk_timeout)

    if _is_displayed(profile_btn) or \
       _is_displayed(profile_img) or \
       _is_displayed(user_menu) or \
       _is_displayed(model_selector) or \
       _is_displayed(new_chat):
        return True
    
    return False

def handle_google_login(page, allow_manual=True):
    # 1. STRICT CHECK
    if verify_strict_login(page):
        return True
    if not allow_manual:
        print(">> Not logged in and manual login is disabled. Aborting.")
        return False

    print(">> Initiating Login Sequence...")
    
    # 2. FIND THE LOGIN BUTTON
    login_btn = (
        page.ele('@@data-testid=login-button', timeout=2) or 
        page.ele('button:Log in', timeout=1) or 
        page.ele('a:Log in', timeout=1) or
        page.ele('text:Log in', timeout=1)
    )
    
    if login_btn:
        print(">> Clicking 'Log in'...")
        login_btn.click()
    else:
        print(">> !! Could not find 'Log in' button. Are you on a weird page?")

    # 3. CLICK "CONTINUE WITH GOOGLE"
    print(">> Looking for Google button...")
    google_btn = (
        page.wait.ele_displayed('text:Continue with Google', timeout=3) or
        page.wait.ele_displayed('@@data-provider=google', timeout=2) or
        page.wait.ele_displayed('@@aria-label=Continue with Google', timeout=2)
    )

    if google_btn:
        print(">> Clicking Google Button...")
        google_btn.click()
        
        print("\n" + "="*50)
        print(">> PLEASE LOG IN MANUALLY NOW")
        print(">> I have clicked 'Continue with Google'.")
        print(">> Please select your account or type credentials in the browser.")
        print("="*50 + "\n")
        
    else:
        print(">> Google button not found or already in a flow. Please log in manually if needed.")

    # 5. WAIT FOR LOGIN SUCCESS (Or Manual Intervention)
    print(">> Waiting for successful login...")
    
    for i in range(240): # 120s
        if verify_strict_login(page):
            print("\n>> Login Verified! proceeding...")
            return True
        if i % 10 == 0:
            print(f">> Waiting... ({120 - i*0.5:.0f}s). Please log in manually.")
        time.sleep(0.5)
            
    print(">> Login timed out. Proceeding regardless (might be in guest mode or failed).")
    return False

def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    if value is None:
        return None
    return bool(value)

def _detect_is_mcq(item):
    """Best-effort MCQ detector for JSON/CSV query rows."""
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
    """Load queries from a CSV or JSON file with 'id' and 'query' entries.
    JSON items can optionally include 'reuse_chat' to control chat reuse per query.
    """
    queries = []
    if not query_file:
        print("Error: No query file specified.")
        return []
    # Make path absolute if it's relative
    if not os.path.isabs(query_file):
        # assume relative to this script's folder
        query_file = BASE_DIR / query_file

    print(f"Loading queries from: {query_file}")
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
                    if 'id' in lower_item and 'query' in lower_item:
                        queries.append(
                            {
                                "id": lower_item["id"],
                                "query": lower_item["query"],
                                "reuse_chat": _coerce_bool(lower_item.get("reuse_chat")),
                                "is_mcq": _detect_is_mcq(lower_item),
                            }
                        )
            else:
                print(f"Error: JSON query file must be a list of objects: {query_file}")
        else:
            with open(query_file, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    lower_row = {str(k).lower(): v for k, v in row.items()}
                    if 'id' in lower_row and 'query' in lower_row:
                        queries.append(
                            {
                                "id": lower_row["id"],
                                "query": lower_row["query"],
                                "reuse_chat": _coerce_bool(lower_row.get("reuse_chat")),
                                "is_mcq": _detect_is_mcq(lower_row),
                            }
                        )
    except Exception as e:
        print(f"Error loading queries from {query_file}: {e}")
        return []
    return queries

def shuffle_no_consecutive(queries, seed=None):
    """Simple shuffle implementation"""
    if len(queries) <= 1:
        return queries
    if seed is None:
        random.shuffle(queries)
    else:
        rng = random.Random(seed)
        rng.shuffle(queries)
    return queries

def clear_chat(page):
    """Clear the chat."""
    url = "https://chatgpt.com/#settings/DataControls"
    delete_label = "Delete all"
    modal_header = "Clear your chat history"

    def _modal_visible() -> bool:
        # User-provided success signal (prefer live element queries; fall back to HTML).
        try:
            are_you_sure = page.ele("text:are you sure", timeout=0.2) or page.ele("text:Are you sure", timeout=0.2)
            confirm = page.ele("text:Confirm deletion", timeout=0.2) or page.ele("text:Confirm Deletion", timeout=0.2)
            # "Clear your chat history" text can exist on the settings page itself, so require a modal-specific signal.
            if (are_you_sure and getattr(are_you_sure, "states", None) and are_you_sure.states.is_displayed) or \
               (confirm and getattr(confirm, "states", None) and confirm.states.is_displayed):
                return True
        except Exception:
            pass
        html = (page.html or "").lower()
        return (modal_header.lower() in html) and ("are you sure" in html)

    def _find_delete_button():
        # Prefer the actual <button> element (text selectors may return a nested <div>).
        # Anchor on the "Delete all chats" row to avoid matching other "Delete all" buttons.
        btn = page.ele(
            'xpath://div[.//div[normalize-space()="Delete all chats"]]//button[contains(@class,"btn-danger-outline") and .//div[normalize-space()="Delete all"]]',
            timeout=1,
        )
        if btn:
            return (
                btn,
                'xpath://div[.//div[normalize-space()="Delete all chats"]]//button[contains(@class,"btn-danger-outline") and .//div[normalize-space()="Delete all"]]',
            )

        btn = page.ele(
            'xpath://button[contains(@class,"btn-danger-outline") and .//div[normalize-space()="Delete all"]]',
            timeout=1,
        )
        if btn:
            return btn, 'xpath://button[contains(@class,"btn-danger-outline") and .//div[normalize-space()="Delete all"]]'

        btn = page.ele('xpath://button[.//div[normalize-space()="Delete all"]]', timeout=1)
        if btn:
            return btn, 'xpath://button[.//div[normalize-space()="Delete all"]]'

        btn = page.ele(f"text:{delete_label}", timeout=1)
        if btn:
            return btn, f"text:{delete_label} (may be nested)"
        try:
            for i, b in enumerate(page.eles("tag:button", timeout=1) or []):
                try:
                    if delete_label.lower() in (b.text or "").strip().lower():
                        return b, f"tag:button[{i}]"
                except Exception:
                    continue
        except Exception:
            pass
        return None, None

    def _debug_btn_state(btn, prefix: str) -> None:
        try:
            if hasattr(btn, "states"):
                print(f">> {prefix} displayed={getattr(btn.states, 'is_displayed', None)} enabled={getattr(btn.states, 'is_enabled', None)}")
        except Exception:
            pass

        for attr in ("class", "disabled", "aria-disabled"):
            try:
                if hasattr(btn, "attr"):
                    print(f">> {prefix} attr[{attr}]={btn.attr(attr)}")
            except Exception:
                pass
        try:
            print(f">> {prefix} text={repr((btn.text or '').strip())}")
        except Exception:
            pass

    def _click_delete(btn) -> None:
        # Attempt to ensure the element is in view even inside nested scroll containers.
        try:
            if hasattr(btn, "run_js"):
                btn.run_js("this.scrollIntoView({block: 'center', inline: 'center'});")
                rect = btn.run_js(
                    "const r=this.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height};"
                )
                print(f">> Delete all rect: {rect}")
                try:
                    outer = btn.run_js("return this.outerHTML;")
                    if outer:
                        outer_s = str(outer).replace("\n", " ")
                        print(f">> Delete all outerHTML[:200]={outer_s[:200]!r}")
                except Exception as exc:
                    print(f">> outerHTML failed: {exc}")
        except Exception as exc:
            print(f">> scrollIntoView/rect failed: {exc}")
        try:
            if hasattr(btn, "scroll"):
                try:
                    btn.scroll.to_center()
                except Exception:
                    pass
            btn.click()
            print(">> Clicked Delete all (normal)")
            return
        except Exception as exc:
            print(f">> Normal click failed: {exc}")
        try:
            btn.click(by_js=True)
            print(">> Clicked Delete all (by_js)")
            return
        except Exception as exc:
            print(f">> by_js click failed: {exc}")

        # Last resort: dispatch a full mouse event sequence.
        try:
            if hasattr(btn, "run_js"):
                btn.run_js(
                    """
                    const el = this;
                    const events = ['mouseover','mouseenter','mousemove','mousedown','mouseup','click'];
                    for (const type of events) {
                      el.dispatchEvent(new MouseEvent(type, {bubbles:true, cancelable:true, view:window}));
                    }
                    """
                )
                print(">> Dispatched mouse events to Delete all")
                return
        except Exception as exc:
            print(f">> Dispatch mouse events failed: {exc}")

    def _dump_button_texts(limit: int = 40) -> None:
        try:
            buttons = page.eles("tag:button", timeout=1) or []
        except Exception:
            buttons = []
        print(f">> Found {len(buttons)} <button> elements on page.")
        shown = 0
        for b in buttons:
            if shown >= limit:
                break
            try:
                t = (b.text or "").strip()
            except Exception:
                continue
            if not t:
                continue
            shown += 1
            print(f">> button[{shown}]: {t[:120]}")

    def _try_confirm_deletion() -> bool:
        """Click the confirmation button after the modal appears."""
        dialog = None
        try:
            dialog = page.ele('[role="dialog"]', timeout=0.5) or page.ele('[aria-modal="true"]', timeout=0.5)
            if dialog:
                print(">> Found dialog container.")
        except Exception:
            dialog = None

        candidates = ["text:Confirm deletion", "text:Confirm Deletion", "text:Confirm", "text:Delete"]
        for sel in candidates:
            try:
                btn = (dialog.ele(sel, timeout=0.8) if dialog else page.ele(sel, timeout=0.8))
                if btn and getattr(btn, "states", None) and btn.states.is_displayed:
                    try:
                        btn.click()
                        print(f">> Clicked confirm via {sel} (normal)")
                    except Exception as exc:
                        print(f">> Confirm normal click failed via {sel}: {exc}")
                        btn.click(by_js=True)
                        print(f">> Clicked confirm via {sel} (by_js)")
                    return True
            except Exception:
                continue
        print(">> Confirm button not found in modal.")
        return False

    max_page_attempts = 3
    max_click_attempts = 15
    modal_wait_s = 2.5

    for attempt in range(1, max_page_attempts + 1):
        print(f">> clear_chat page attempt {attempt}/{max_page_attempts}: {url}")
        try:
            page.get(url)
        except Exception as exc:
            print(f">> Navigation failed: {exc}")
            time.sleep(1)
            continue
        time.sleep(2.5)
        try:
            print(f">> URL now: {getattr(page, 'url', None)}")
        except Exception:
            pass

        # Quick sanity check that the settings view actually loaded.
        try:
            settings_hint = page.ele("text:Data controls", timeout=1) or page.ele("text:Data Controls", timeout=1)
            print(f">> Data controls header visible: {bool(settings_hint)}")
        except Exception as exc:
            print(f">> Data controls header check failed: {exc}")

        if _modal_visible():
            print(">> Modal already visible; delete already triggered?")
            if _try_confirm_deletion():
                time.sleep(1.5)
                return not _modal_visible()
            return False

        delete_btn, found_by = _find_delete_button()
        if not delete_btn:
            print(f">> '{delete_label}' button not found on page.")
            _dump_button_texts()
            time.sleep(1)
            continue

        print(f">> Found '{delete_label}' via {found_by}")
        _debug_btn_state(delete_btn, "Delete all button state:")

        for click_attempt in range(1, max_click_attempts + 1):
            print(f">> Clicking '{delete_label}' ({click_attempt}/{max_click_attempts})...")
            try:
                _click_delete(delete_btn)
            except Exception as exc:
                print(f">> Click threw: {exc}")

            deadline = time.time() + modal_wait_s
            while time.time() < deadline:
                if _modal_visible():
                    print('>> Saw modal text: "Clear your chat history - are you sure?"')
                    if not _try_confirm_deletion():
                        return False
                    time.sleep(1.5)
                    if _modal_visible():
                        print(">> Modal still visible after confirming (deletion may have failed).")
                        return False
                    print(">> Modal dismissed; chat deletion likely triggered.")
                    return True
                time.sleep(0.2)
            # Debug: detect any dialog even if the expected text doesn't match.
            try:
                dialog = page.ele('[role="dialog"]', timeout=0.2) or page.ele('[aria-modal="true"]', timeout=0.2)
                if dialog and getattr(dialog, "states", None) and dialog.states.is_displayed:
                    dt = (dialog.text or "").strip().replace("\n", " ")
                    print(f">> Dialog detected but expected text not found. dialog.text[:200]={dt[:200]!r}")
            except Exception:
                pass

            # Re-find to avoid stale element handles.
            delete_btn, found_by = _find_delete_button()
            if delete_btn:
                print(f">> Re-found '{delete_label}' via {found_by}")
            else:
                print(f">> Re-find '{delete_label}' failed (DOM may be changing).")

        print(f">> Modal never appeared after {max_click_attempts} clicks.")
        time.sleep(1)

    print(">> Clear chat failed after retries.")
    return False

def _save_memories_snapshot(
    page,
    run_id=None,
    session_id=None,
    label=None,
    persona_name=None,
    persona_id=None,
):
    """Extract memory rows from the open 'Saved memories' modal and persist them.

    Memories are saved to ``data/memory_snapshots/<run_id>/memories.jsonl``
    (one JSON object per snapshot).  Each snapshot contains a list of
    memory strings, a timestamp, and optional session/label metadata.
    The raw memory modal HTML is always saved alongside as a fallback.

    NOTE: This function expects the memory management modal to already be open
    (i.e., the "Manage" button has been clicked in settings).
    """
    try:
        out_dir = DATA_DIR / "memory_snapshots"
        if run_id:
            out_dir = out_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now()
        ts_str = ts.strftime("%Y%m%d_%H%M%S")

        # ── Wait for and capture the memory modal (not settings dialog) ──
        # The memory modal appears after clicking "Manage" and contains the table
        print(">> Waiting for memory modal to load...")
        time.sleep(1.5)  # Give modal time to fully render

        # Try multiple selectors to find the memory modal
        # It should contain either the table or "Saved memories" heading
        dialog = None
        for attempt in range(3):
            # Look for dialog containing the memories table
            dialog = page.ele('css:div[role="dialog"]:has(table)', timeout=2)
            if dialog:
                print(">> Found memory modal with table")
                break
            # Fallback: any dialog with "Saved memories" or "Memory" heading
            dialog = page.ele('css:div[role="dialog"]', timeout=2)
            if dialog and dialog.text and ('saved memor' in dialog.text.lower() or 'memory' in dialog.text.lower()):
                print(">> Found memory modal via text content")
                break
            if attempt < 2:
                print(f">> Memory modal not found yet, retrying... ({attempt + 1}/3)")
                time.sleep(1)

        # ── Always save the raw memory modal HTML as a fallback ──────────────
        if dialog:
            dialog_html = dialog.html or ""
            if dialog_html:
                safe_label = (label or "snapshot").replace("/", "_")
                safe_persona = None
                if persona_id:
                    safe_persona = f"persona{persona_id}"
                elif persona_name:
                    safe_persona = re.sub(r"[^A-Za-z0-9_-]+", "_", str(persona_name)).strip("_") or None
                if safe_persona:
                    html_name = f"memories_s{session_id or 0}_{safe_label}_{safe_persona}_{ts_str}.html"
                else:
                    html_name = f"memories_s{session_id or 0}_{safe_label}_{ts_str}.html"
                html_file = out_dir / html_name
                html_file.write_text(dialog_html, encoding="utf-8")
                print(f">> Saved raw memory modal HTML to {html_file}")
        else:
            print(">> No memory modal element found — cannot save HTML fallback.")
            return []

        # ── Extract structured memory text ─────────────────────────────
        # Memory items are in divs with whitespace-pre-wrap class inside tbody
        # Structure: <tbody><div class="group ..."><div><div class="whitespace-pre-wrap">MEMORY</div></div></div></tbody>
        memories = []

        # Try to find memory items via whitespace-pre-wrap divs
        memory_divs = page.eles('css:div[role="dialog"] table tbody div.whitespace-pre-wrap', timeout=3)
        if memory_divs:
            print(f">> Found {len(memory_divs)} memory div elements")
            for div in memory_divs:
                txt = (div.text or "").strip()
                if txt:
                    memories.append(txt)
        else:
            print(">> No memory divs found in modal (table might be empty).")

        if not memories:
            print(">> Memory rows present but no text extracted (HTML fallback saved).")
            return []

        print(f">> Found {len(memories)} saved memories.")

        # ── Persist structured snapshot ────────────────────────────────
        snapshot = {
            "saved_at": ts.isoformat(),
            "run_id": run_id,
            "session_id": session_id,
            "label": label,
            "persona_name": persona_name,
            "persona_id": persona_id,
            "count": len(memories),
            "memories": memories,
        }
        jsonl_file = out_dir / "memories.jsonl"
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        print(f">> Saved memory snapshot ({len(memories)} items) to {jsonl_file}")
        return memories
    except Exception as exc:
        print(f">> Failed to save memory snapshot: {exc}")
        return []


def clear_memory(page, run_id=None, session_id=None, label=None, persona_name=None, persona_id=None):
    """Clear the memory, saving any existing memories first.

    Returns:
        True if memory was cleared successfully, False otherwise.
    """
    try:
        # Navigate to personalization settings
        page.get('https://chatgpt.com/#settings/Personalization')
        time.sleep(2)

        # Click "Manage" button to open memory modal
        print(">> Opening memory manager...")
        manage_btn = page.ele('text:Manage', timeout=3)
        if not manage_btn:
            print(">> 'Manage' button not found in settings")
            return False
        manage_btn.click()
        time.sleep(2)

        # ── Save memories before deleting ──────────────────────────────────
        memories = _save_memories_snapshot(
            page,
            run_id=run_id,
            session_id=session_id,
            label=label,
            persona_name=persona_name,
            persona_id=persona_id,
        )

        # If no memories exist, nothing to delete
        if not memories:
            print(">> No memories found to delete")
            # Close the modal
            close_btn = page.ele('css:button[aria-label="Close"]', timeout=2)
            if close_btn:
                close_btn.click()
                time.sleep(0.5)
            return True

        # ── Try to find direct "Delete all" button first ──────────────────
        print(">> Looking for direct 'Delete all' button...")
        delete_btn = None

        # Sometimes there's a direct button: <button class="btn-danger-outline" data-testid="reset-memories-button">Delete all</button>
        direct_delete_selectors = [
            'css:button[data-testid="reset-memories-button"]',
            'xpath://button[@data-testid="reset-memories-button" and contains(., "Delete all")]',
            'xpath://button[contains(@class, "btn-danger-outline") and contains(., "Delete all")]',
        ]

        for selector in direct_delete_selectors:
            try:
                delete_btn = page.ele(selector, timeout=1)
                if delete_btn and hasattr(delete_btn, 'states') and delete_btn.states.is_displayed:
                    print(f">> Found direct 'Delete all' button via {selector}")
                    # Skip to checkbox/confirmation handling
                    delete_btn.click()
                    time.sleep(1.5)
                    # Jump to checkbox handling
                    break
            except Exception:
                continue

        # If direct button was found and clicked, skip the More options flow
        if delete_btn:
            print(">> Using direct delete button (skipping More options menu)")
        else:
            # ── Click "More options" menu to reveal delete option ──────────────
            print(">> No direct button found, opening 'More options' menu...")
            more_options_btn = None

            # Try to find the More options button
            more_selectors = [
                'css:button[aria-label="More options"]',
                'xpath://button[@aria-label="More options"]',
                'xpath://div[@role="dialog"]//button[@aria-label="More options"]',
            ]

            for selector in more_selectors:
                try:
                    more_options_btn = page.ele(selector, timeout=2)
                    if more_options_btn and hasattr(more_options_btn, 'states') and more_options_btn.states.is_displayed:
                        print(f">> Found 'More options' button via {selector}")
                        break
                except Exception:
                    continue

            if not more_options_btn:
                print(">> 'More options' button not found in memory modal")
                # Try to dump available buttons for debugging
                try:
                    dialog = page.ele('css:div[role="dialog"]', timeout=1)
                    if dialog:
                        buttons = dialog.eles('tag:button', timeout=1) or []
                        for i, btn in enumerate(buttons):
                            aria_label = btn.attr('aria-label') if hasattr(btn, 'attr') else None
                            btn_text = (btn.text or "").strip()
                            print(f">> Button {i}: aria-label={aria_label}, text={btn_text[:50]}")
                except Exception:
                    pass
                return False

            # Click More options
            print(">> Clicking 'More options'...")
            try:
                more_options_btn.click()
            except Exception:
                more_options_btn.click(by_js=True)
            time.sleep(1)

            # ── Find and click "Delete all" in the dropdown menu ──────────────
            print(">> Looking for 'Delete all' option in menu...")
            delete_menuitem = None

            # Try different selectors for the Delete all menu item
            # It's a <div role="menuitem">Delete all memories</div>
            delete_selectors = [
                'xpath://div[@role="menuitem" and contains(., "Delete all")]',
                'xpath://div[@role="menuitem" and contains(., "Delete all memories")]',
                'css:div[role="menuitem"][data-color="danger"]',
                'text:Delete all memories',
                'text:Delete all',
                'xpath://button[@role="menuitem" and contains(., "Delete all")]',
                'xpath://*[@role="menu"]//*[contains(., "Delete all")]',
            ]

            for selector in delete_selectors:
                try:
                    delete_menuitem = page.ele(selector, timeout=2)
                    if delete_menuitem and hasattr(delete_menuitem, 'states') and delete_menuitem.states.is_displayed:
                        print(f">> Found 'Delete all' option via {selector}")
                        break
                except Exception:
                    continue

            if not delete_menuitem:
                print(">> 'Delete all' option not found in menu")
                # Try to dump available menu items for debugging
                try:
                    menu_items = page.eles('css:div[role="menuitem"]', timeout=1) or []
                    if not menu_items:
                        menu_items = page.eles('css:div[role="menu"] > *', timeout=1) or []
                    menu_texts = [item.text.strip() for item in menu_items if item.text]
                    print(f">> Available menu items: {menu_texts}")
                except Exception:
                    pass
                return False

            # Click Delete all menuitem
            print(">> Clicking 'Delete all'...")
            try:
                delete_menuitem.click()
            except Exception:
                delete_menuitem.click(by_js=True)
            time.sleep(1.5)

        # ── Check the "delete past versions" checkbox if present ──────────
        print(">> Looking for 'delete past versions' checkbox...")
        checkbox = None

        checkbox_selectors = [
            'css:input[id="delete-past-versions"]',
            'css:input[data-testid="delete-past-versions"]',
            'xpath://input[@type="checkbox" and (@id="delete-past-versions" or @data-testid="delete-past-versions")]',
        ]

        for selector in checkbox_selectors:
            try:
                checkbox = page.ele(selector, timeout=2)
                if checkbox:
                    print(f">> Found checkbox via {selector}")
                    break
            except Exception:
                continue

        if checkbox:
            # Check if it's already checked (check both attribute and property)
            try:
                # For DrissionPage, check using JavaScript
                is_checked = checkbox.run_js("return this.checked;") if hasattr(checkbox, 'run_js') else False
            except Exception:
                is_checked = False

            if not is_checked:
                print(">> Checking 'delete past versions' checkbox...")
                try:
                    checkbox.click()
                except Exception:
                    try:
                        checkbox.click(by_js=True)
                    except Exception:
                        # Fallback: use JS to set checked property
                        if hasattr(checkbox, 'run_js'):
                            checkbox.run_js("this.checked = true; this.dispatchEvent(new Event('change', {bubbles: true}));")
                time.sleep(0.5)
            else:
                print(">> Checkbox already checked")
        else:
            print(">> No checkbox found (might not be required)")

        # ── Handle confirmation dialog ──────────────────────────────────────
        print(">> Looking for confirmation dialog...")
        confirm_btn = None

        # Try different selectors for the confirmation button
        # Two possible buttons:
        # 1. <button data-testid="confirm-reset-memories-button" class="btn-danger">Clear memory</button> (direct delete flow)
        # 2. <button class="btn-danger">Delete all</button> (menu flow)
        confirm_selectors = [
            'css:button[data-testid="confirm-reset-memories-button"]',
            'xpath://button[@data-testid="confirm-reset-memories-button" and contains(., "Clear memory")]',
            'xpath://button[contains(@class, "btn-danger") and contains(., "Clear memory")]',
            'xpath://button[contains(@class, "btn-danger") and contains(., "Delete all")]',
            'css:button.btn-danger',
            'xpath://div[@role="dialog"]//button[contains(@class, "btn-danger")]',
            'text:Clear memory',
            'text:Delete all',
            'text:Confirm',
        ]

        for selector in confirm_selectors:
            try:
                confirm_btn = page.ele(selector, timeout=2)
                if confirm_btn and hasattr(confirm_btn, 'states') and confirm_btn.states.is_displayed:
                    print(f">> Found confirmation button via {selector}")
                    break
            except Exception:
                continue

        if not confirm_btn:
            print(">> Confirmation button not found")
            # Try to dump available buttons in confirmation dialog
            try:
                dialog = page.ele('css:div[role="dialog"]', timeout=1)
                if dialog:
                    buttons = dialog.eles('tag:button', timeout=1) or []
                    button_info = []
                    for btn in buttons:
                        aria_label = btn.attr('aria-label') if hasattr(btn, 'attr') else None
                        btn_text = (btn.text or "").strip()
                        btn_classes = btn.attr('class') if hasattr(btn, 'attr') else None
                        button_info.append(f"text='{btn_text[:30]}', aria-label={aria_label}, classes={btn_classes[:50] if btn_classes else None}")
                    print(f">> Available buttons in confirmation dialog: {button_info}")
            except Exception as e:
                print(f">> Error dumping confirmation buttons: {e}")
            return False

        # Click confirmation
        print(">> Confirming memory deletion...")
        try:
            confirm_btn.click()
        except Exception:
            confirm_btn.click(by_js=True)
        time.sleep(2)

        print(">> Memory cleared successfully")
        return True

    except Exception as e:
        print(f">> Error clearing memory: {e}")
        import traceback
        traceback.print_exc()
        return False


def set_custom_instructions(page, instructions_text):
    """Set (or clear) the custom instructions on the Personalization settings page.

    Args:
        page: DrissionPage browser page object.
        instructions_text: The text to set. Pass '' or None to clear.

    Returns:
        True if instructions were set successfully, False otherwise.
    """
    instructions_text = instructions_text or ""
    action = "Clearing" if not instructions_text else "Setting"
    print(f">> {action} custom instructions...")

    try:
        page.get('https://chatgpt.com/#settings/Personalization')
        time.sleep(2)

        # Find the textarea by name attribute
        textarea = None
        textarea_selectors = [
            'css:textarea[name="traits_model_message"]',
            'xpath://textarea[@name="traits_model_message"]',
            'css:textarea[placeholder*="behavior, style, and tone"]',
        ]

        for selector in textarea_selectors:
            try:
                textarea = page.ele(selector, timeout=3)
                if textarea:
                    print(f">> Found custom instructions textarea via {selector}")
                    break
            except Exception:
                continue

        if not textarea:
            print(">> Custom instructions textarea not found on Personalization page")
            try:
                textareas = page.eles('tag:textarea', timeout=2) or []
                for i, ta in enumerate(textareas):
                    ta_name = ta.attr('name') if hasattr(ta, 'attr') else None
                    ta_placeholder = ta.attr('placeholder') if hasattr(ta, 'attr') else None
                    print(f">> Textarea {i}: name={ta_name}, placeholder={ta_placeholder}")
            except Exception:
                pass
            return False

        # The textarea is inside a button-like trigger (aria-haspopup="dialog").
        # Clicking it may open a dialog/popover for editing. Try clicking first.
        try:
            textarea.click()
            time.sleep(0.8)
        except Exception:
            try:
                textarea.click(by_js=True)
                time.sleep(0.8)
            except Exception:
                pass

        # After clicking, the textarea might expand or a dialog might open.
        # Re-locate the textarea in case the DOM changed.
        active_textarea = None
        active_selectors = [
            'css:textarea[name="traits_model_message"]',
            'xpath://div[@role="dialog"]//textarea[@name="traits_model_message"]',
            'css:textarea[placeholder*="behavior, style, and tone"]',
            'xpath://div[@role="dialog"]//textarea[contains(@placeholder, "behavior")]',
        ]

        for selector in active_selectors:
            try:
                active_textarea = page.ele(selector, timeout=2)
                if active_textarea:
                    break
            except Exception:
                continue

        if not active_textarea:
            active_textarea = textarea

        # Clear existing content and type new instructions
        print(f">> Writing custom instructions ({len(instructions_text)} chars)...")
        try:
            # Select all + delete to clear, then type new text
            active_textarea.click()
            time.sleep(0.3)
            # Use Ctrl+A to select all, then type replacement
            active_textarea.input('')  # Clear
            time.sleep(0.3)
            if instructions_text:
                active_textarea.input(instructions_text)
                time.sleep(0.5)
        except Exception:
            # Fallback: use JavaScript to set the value
            print(">> Using JS fallback to set textarea value...")
            escaped = instructions_text.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
            js_code = (
                f"const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;"
                f"nativeInputValueSetter.call(this, '{escaped}');"
                f"this.dispatchEvent(new Event('input', {{bubbles: true}}));"
                f"this.dispatchEvent(new Event('change', {{bubbles: true}}));"
            )
            active_textarea.run_js(js_code)
            time.sleep(0.5)

        # Look for a Save button to confirm changes
        save_btn = None
        save_selectors = [
            'css:button[data-testid="save-button"]',
            'xpath://button[contains(., "Save")]',
            'xpath://div[@role="dialog"]//button[contains(., "Save")]',
            'text:Save',
        ]

        for selector in save_selectors:
            try:
                save_btn = page.ele(selector, timeout=2)
                if save_btn and hasattr(save_btn, 'states') and save_btn.states.is_displayed:
                    print(f">> Found Save button via {selector}")
                    break
            except Exception:
                continue

        if save_btn:
            print(">> Clicking Save...")
            try:
                save_btn.click()
            except Exception:
                save_btn.click(by_js=True)
            time.sleep(1.5)
        else:
            print(">> No Save button found (changes may auto-save)")

        # Close settings if still open
        close_btn = page.ele('css:button[aria-label="Close"]', timeout=2)
        if close_btn:
            try:
                close_btn.click()
                time.sleep(0.5)
            except Exception:
                pass

        print(f">> Custom instructions {'cleared' if not instructions_text else 'set'} successfully")
        return True

    except Exception as e:
        print(f">> Error setting custom instructions: {e}")
        import traceback
        traceback.print_exc()
        return False


def ensure_new_chat(page, attempts=3, temporary_chat=False):
    """Start a fresh chat. When temporary_chat=True, navigate via URL param."""
    if temporary_chat:
        print(">> Starting new temporary chat...")
        page.get('https://chatgpt.com/?temporary-chat=true')
        time.sleep(random.uniform(1.2, 2.4))
        return True

    for i in range(attempts):
        print(">> Clicking 'New chat'...")
        new_chat_btn = (
            page.ele('@@data-testid=create-new-chat-button', timeout=2) or
            page.ele('a@@data-testid=create-new-chat-button', timeout=2) or
            page.ele('a[href="/"]@@data-testid=create-new-chat-button', timeout=2) or
            page.ele('button[data-testid="new-chat-button"]', timeout=2) or
            page.ele('a[href="/"]', timeout=2) or 
            page.ele('text:New chat', timeout=2)
        )
        if new_chat_btn:
            try:
                if hasattr(new_chat_btn, "states") and not new_chat_btn.states.is_displayed:
                    print(">> New chat button not visible yet. Waiting...")
                    time.sleep(1)
                if hasattr(new_chat_btn, "scroll"):
                    try:
                        new_chat_btn.scroll.to_center()
                    except Exception:
                        pass
                time.sleep(random.uniform(0.1, 0.4))
                new_chat_btn.click()
                time.sleep(random.uniform(0.8, 1.6))
                return True
            except Exception as e:
                print(f">> New chat click failed: {e}. Retrying with JS click...")
                try:
                    time.sleep(random.uniform(0.1, 0.4))
                    new_chat_btn.click(by_js=True)
                    time.sleep(random.uniform(0.8, 1.6))
                    return True
                except Exception as e2:
                    print(f">> JS click failed: {e2}. Refreshing...")
        print(">> New chat button not found. Refreshing...")
        page.refresh()
        time.sleep(3)
    return False


def _infer_persona_from_value(value):
    if not value:
        return None, None
    text = str(value)
    base = Path(text).stem if ("/" in text or "\\" in text or text.endswith(".json")) else text
    m = re.search(r"persona(\d+)", base, flags=re.I)
    if not m:
        return None, None
    return base, m.group(1)


def select_interface_model(page, model_name, timeout=8):
    """Select a model from the ChatGPT model-switcher dropdown.

    Args:
        page: DrissionPage ChromiumPage instance.
        model_name: Substring to match in the menu item text (e.g. 'instant', 'thinking', '5.2 Instant').
        timeout: Max seconds to wait for elements.

    Returns:
        True if model was selected (or already active), False on failure.
    """
    try:
        # Wait for the page to fully render after New chat navigation
        time.sleep(3)

        # Find the model switcher button — try composer-pill first, then legacy selectors
        s1 = page.ele('css:button.__composer-pill[aria-haspopup="menu"]', timeout=timeout)
        print(f">> [switcher] css:button.__composer-pill: {bool(s1)}")
        s2 = s1 or page.ele('button@@data-testid=model-switcher-dropdown-button', timeout=3)
        print(f">> [switcher] data-testid=model-switcher-dropdown-button: {bool(s2)}")
        s3 = s2 or page.ele('@@data-testid=model-switcher-dropdown-button', timeout=3)
        print(f">> [switcher] @@data-testid fallback: {bool(s3)}")
        s4 = s3 or page.ele('button@@aria-label^=Model selector', timeout=3)
        print(f">> [switcher] aria-label^=Model selector: {bool(s4)}")
        switcher = s4
        if not switcher:
            print(f">> Model switcher button not found. Page may not support model selection.")
            return False

        # Check current model from aria-label and button text
        current_label = switcher.attr('aria-label') or ''
        current_text = (switcher.text or '').strip()
        print(f">> Model switcher found - aria-label: '{current_label}', text: '{current_text}'")

        # Check if desired model is already selected
        # For "instant" or "thinking", check if it's in the aria-label or text
        if model_name.lower() in current_label.lower() or model_name.lower() in current_text.lower():
            print(f">> Model '{model_name}' already selected.")
            return True

        # Click the dropdown to open menu
        print(f">> Opening model selector to pick '{model_name}'...")
        switcher.click()
        time.sleep(random.uniform(0.8, 1.5))

        # gpt-5.4 is not a direct menu item — open Configure... to set it
        if re.search(r'5[.\-]4', model_name):
            configure = page.ele('@@data-testid=model-configure-modal', timeout=timeout)
            if not configure:
                print(f">> 'Configure...' item not found in model dropdown.")
                try:
                    page.actions.key_down('Escape').key_up('Escape')
                except Exception:
                    pass
                return False
            print(f">> Clicking 'Configure...' for model '{model_name}'...")
            try:
                configure.click()
            except Exception as e:
                print(f">> Normal click failed: {e}, trying JS click...")
                configure.click(by_js=True)
            time.sleep(random.uniform(1.5, 2.5))

            # Click the model combobox inside the configure modal
            c1 = page.ele('button@@role=combobox@@aria-labelledby=model-selection-label', timeout=timeout)
            print(f">> [combobox] role=combobox@@aria-labelledby: {bool(c1)}")
            c2 = c1 or page.ele('css:button[aria-labelledby="model-selection-label"]', timeout=3)
            print(f">> [combobox] css aria-labelledby: {bool(c2)}")
            c3 = c2 or page.ele('button@@aria-labelledby=model-selection-label', timeout=3)
            print(f">> [combobox] @@aria-labelledby: {bool(c3)}")
            model_combobox = c3
            if not model_combobox:
                print(f">> Model combobox not found in configure modal.")
                return False
            print(f">> Clicking model combobox (text='{(model_combobox.text or '').strip()}')...")
            try:
                model_combobox.click()
            except Exception as e:
                print(f">> Normal click failed: {e}, trying JS click...")
                model_combobox.click(by_js=True)
            time.sleep(random.uniform(0.8, 1.5))

            # Extract version number (e.g. "5.4") from strings like "gpt 5.4 thinking" or "gpt-5.4"
            # Normalize version: "5-4" or "5.4" → "5.4" for combobox matching
            _ver_match = re.search(r'(\d+)[.\-](\d+)', model_name)
            version = f"{_ver_match.group(1)}.{_ver_match.group(2)}" if _ver_match else model_name
            option = (
                page.ele(f'@@role=option@@text()={version}', timeout=timeout) or
                page.ele(f'css:[role="option"]', timeout=3)
            )
            # If exact match failed, scan all options for the version string
            if not option or version not in (option.text or ''):
                all_options = page.eles('@@role=option', timeout=3) or []
                option = next((o for o in all_options if version in (o.text or '')), None)

            if not option:
                print(f">> Version '{version}' not found in combobox dropdown.")
                return False
            print(f">> Selecting version '{option.text.strip()}'...")
            try:
                option.click()
            except Exception as e:
                print(f">> Normal click failed: {e}, trying JS click...")
                option.click(by_js=True)
            time.sleep(random.uniform(0.5, 1.0))

            # If "thinking" in model name, select the Thinking radio button in the modal
            if 'thinking' in model_name.lower():
                thinking_radio = None
                for radio in (page.eles('@@role=radio', timeout=5) or []):
                    if 'thinking' in (radio.text or '').lower():
                        thinking_radio = radio
                        break
                if thinking_radio:
                    print(f">> Selecting Thinking mode...")
                    try:
                        thinking_radio.click()
                    except Exception as e:
                        thinking_radio.click(by_js=True)
                    time.sleep(random.uniform(0.3, 0.6))
                else:
                    print(f">> Thinking radio button not found in modal.")

            # Close the configure modal
            close_btn = page.ele('@@data-testid=close-button', timeout=5)
            if close_btn:
                print(f">> Closing configure modal...")
                try:
                    close_btn.click()
                except Exception as e:
                    close_btn.click(by_js=True)
                time.sleep(random.uniform(0.3, 0.6))
            else:
                print(f">> Close button not found, pressing Escape...")
                try:
                    page.actions.key_down('Escape').key_up('Escape')
                except Exception:
                    pass
            return True

        # Find the menu items — include both menuitem and menuitemradio roles
        menu_items = (
            page.eles('@@role=menuitemradio', timeout=timeout) or
            page.eles('css:[role="menuitemradio"]', timeout=3)
        )
        if not menu_items:
            menu_items = (
                page.eles('@@role=menuitem', timeout=3) or
                page.eles('@@data-testid^model-switcher', timeout=3)
            )

        if not menu_items:
            print(f">> No menu items found in model dropdown.")
            try:
                page.actions.key_down('Escape').key_up('Escape')
            except Exception:
                pass
            return False

        print(f">> Found {len(menu_items)} menu items")

        # Find the matching item by text (case-insensitive)
        target = None
        available_items = []
        for item in menu_items:
            item_text = (item.text or '').strip()
            available_items.append(item_text)
            if model_name.lower() in item_text.lower():
                target = item
                print(f">> Found matching menu item: '{item_text}'")
                break

        if not target:
            print(f">> Model '{model_name}' not found in menu.")
            print(f">> Available options: {available_items}")
            try:
                page.actions.key_down('Escape').key_up('Escape')
            except Exception:
                pass
            return False

        # Click the target model
        print(f">> Clicking on '{target.text.strip()}'...")
        try:
            target.click()
        except Exception as e:
            print(f">> Normal click failed: {e}, trying JS click...")
            target.click(by_js=True)

        time.sleep(random.uniform(0.8, 1.5))
        print(f">> Selected model '{model_name}'.")
        return True

    except Exception as e:
        print(f">> Error selecting model '{model_name}': {e}")
        import traceback
        traceback.print_exc()
        return False

def detect_model_info_from_html(html, debug=False):
    """Detect model slug from ChatGPT page HTML. Returns slug string or None."""
    if not html:
        return None

    slug_patterns = [
        r'data-message-model-slug="([^"]+)"',
        r'data-model-slug="([^"]+)"[^>]*(?:aria-checked="true"|data-checked="true"|data-selected="true"|aria-selected="true")',
        r'data-model-slug="([^"]+)"',
        r'"model_slug"\s*:\s*"([^"]+)"',
        r'"modelSlug"\s*:\s*"([^"]+)"',
    ]
    for i, pattern in enumerate(slug_patterns, 1):
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(1)
        elif debug:
            print(f">> Pattern {i} no match: {pattern[:50]}...")

    if debug:
        # Show sample of HTML for debugging
        print(f">> HTML sample (first 500 chars): {html[:500]}")
        print(f">> HTML sample (last 500 chars): {html[-500:]}")

    return None

def _class_has_token(value, token):
    if not value:
        return False
    if isinstance(value, str):
        return token in value.split()
    return token in value

def _extract_role_text(soup, role):
    messages = soup.find_all(attrs={"data-message-author-role": role})
    if not messages:
        return "", None
    message = messages[-1]
    if role == "assistant":
        target = message.find("div", class_=lambda c: _class_has_token(c, "markdown")) or message
    else:
        target = message.find("div", class_=lambda c: _class_has_token(c, "whitespace-pre-wrap")) or message
    text = target.get_text("\n", strip=True)
    # Fallback: if the assistant markdown div is empty (e.g. thinking model response
    # not yet rendered into the DOM), use the sr-only accessibility div which always
    # contains the full response text.
    if role == "assistant" and not text:
        sr_div = soup.find("div", class_="sr-only")
        if sr_div:
            sr_text = sr_div.get_text("\n", strip=True)
            # Strip the "ChatGPT says: " prefix added for screen readers
            sr_text = re.sub(r"^ChatGPT says:\s*", "", sr_text)
            if sr_text:
                text = sr_text
    return text, message

def parse_html_file(html_path, meta_path=None):
    if BeautifulSoup is None:
        raise RuntimeError("BeautifulSoup (bs4) is required to parse HTML files.")
    content = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(content, "html.parser")

    user_text = ""
    assistant_text = ""
    model_slug = None

    if "chatgpt.com" in content or "data-message-author-role" in content:
        # ChatGPT: data-message-author-role="user" / "assistant"
        user_text, _ = _extract_role_text(soup, "user")
        assistant_text, assistant_node = _extract_role_text(soup, "assistant")
        if assistant_node is not None:
            model_slug = assistant_node.get("data-message-model-slug")
    else:
        # Claude: data-testid="user-message" + data-is-streaming div
        user_node = soup.find(attrs={"data-testid": "user-message"})
        user_text = user_node.get_text("\n", strip=True) if user_node else ""
        asst_nodes = soup.find_all(attrs={"data-is-streaming": True})
        if asst_nodes:
            assistant_text = asst_nodes[-1].get_text("\n", strip=True)

    if not user_text and not assistant_text:
        return None

    meta = {}
    if meta_path and Path(meta_path).exists():
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    if not model_slug:
        model_slug = meta.get("model_slug")

    parsed = {
        "query_parsed": user_text,
        "ai_generated_output_text": assistant_text,
    }
    if model_slug:
        parsed["model_slug"] = model_slug
    if meta:
        parsed["meta"] = meta
    return parsed

def parse_saved_htmls(run_id, raw_root=None, parsed_root=None):
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

    print(f">> Parsed {parsed_count} HTML files to {parsed_root} (skipped {skipped})")

def save_response(output_dir, query_id, content, scraping_round, model_info, sent_at=None):
    """Save the response content to a file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize query_id for filename
    safe_id = "".join([c for c in query_id if c.isalnum() or c in ('-', '_')]).strip()
    filename = output_dir / f"{safe_id}.html"
    
    # Save the HTML content directly
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f">> Saved response to {filename}")

    meta = {
        "query_id": query_id,
        "scraping_round": scraping_round,
        "saved_at": datetime.now().isoformat(),
    }
    if sent_at:
        meta["sent_at"] = sent_at
    model_slug = model_info.get("model_slug")
    if model_slug:
        meta["model_slug"] = model_slug
    meta_filename = output_dir / f"{safe_id}.meta.json"
    with open(meta_filename, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f">> Saved metadata to {meta_filename}")


def _api_barrier_worker(
    query_file, runs, shuffle, seed,
    api_cfg, api_key,
    output_base,
    sync_barrier_info, stop_event,
    batch_size=0, pause_hours=3,
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
    output_dir = Path(output_base) / m_dir

    queries_base = load_queries_from_file(query_file)
    if not queries_base:
        print(f">> [API {m_name}] No queries loaded. Exiting worker.")
        return

    queries = []
    for item in queries_base:
        q_id = item.get("id") if isinstance(item, dict) else item[0]
        q_text = item.get("query") if isinstance(item, dict) else item[1]
        for r in range(runs):
            queries.append({"id": f"{q_id}_run{r}", "query": q_text})

    if shuffle:
        queries = shuffle_no_consecutive(queries, seed=seed)

    print(f">> [API {m_name}] Starting barrier worker with {len(queries)} queries.")
    queries_done = 0

    for i, item in enumerate(queries):
        if stop_event and stop_event.is_set():
            print(f">> [API {m_name}] Stop signal received. Halting.")
            return

        print(f">> [API {m_name}] [{i+1}/{len(queries)}] Waiting at send barrier...")
        barrier.wait()

        sent_at = datetime.now().isoformat()
        print(f">> [API {m_name}] Sending {item['id']}...")
        run_api_query(item["id"], item["query"], str(output_dir), m_name,
                      sent_at=sent_at, api_key=api_key, **m_params)

        queries_done += 1
        if batch_size > 0 and queries_done % batch_size == 0:
            pause_secs = pause_hours * 3600
            print(f">> [API {m_name}] Batch of {batch_size} done. Pausing {pause_hours:.1f}h...")
            pause_end = time.time() + pause_secs
            while time.time() < pause_end:
                if stop_event and stop_event.is_set():
                    return
                time.sleep(30)
            print(f">> [API {m_name}] Resuming after batch pause.")

    print(f">> [API {m_name}] All queries complete.")


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
):
    """Run a single experiment.

    Args:
        api_models: list of dicts, each with at least {"model": "..."}.
            Additional keys (temperature, top_p, …) are forwarded as
            hyperparameters to the OpenAI API.
    """
    api_models = api_models or []
    exp_name = experiment.get("name", "experiment")
    exp_type = experiment.get("type", "")
    query_file = experiment.get("query_file", defaults.get("query_file"))
    runs = int(experiment.get("runs", defaults.get("runs", 1)))
    shuffle = bool(experiment.get("shuffle", defaults.get("shuffle", False)))
    exp_seed = experiment.get("seed", defaults.get("seed"))
    reuse_chat = bool(experiment.get("reuse_chat", defaults.get("reuse_chat", False)))
    batch_size = int(experiment.get("batch_size", defaults.get("batch_size", 0)))
    pause_hours = float(experiment.get("pause_hours", defaults.get("pause_hours", 3)))
    if seed_override is not None and "seed" not in experiment:
        exp_seed = seed_override

    # Per-experiment interface_model (falls back to CLI / caller arg)
    interface_model = experiment.get("interface_model", interface_model)
    fast_mode = bool(experiment.get("fast", defaults.get("fast", False)))
    
    # Wait configuration
    wait_after_prompts = experiment.get("wait_after_prompts", defaults.get("wait_after_prompts", [3, 7]))
    wait_after_saves = experiment.get("wait_after_saves", defaults.get("wait_after_saves", [3, 7]))

    # Output Directory
    output_dir = DATA_DIR / "raw_html" / run_id / f"session_{session_id:02d}" / exp_name
    api_output_base = DATA_DIR / "api" / run_id / f"session_{session_id:02d}" / exp_name
    print(f"\n>>> STARTING EXPERIMENT: {exp_name}")
    print(f"    Query File: {query_file}")
    print(f"    Type: {exp_type or '(default)'}")
    if interface_model:
        print(f"    Interface Model: {interface_model}")
    print(f"    Output Directory: {output_dir}")
    if api_models:
        model_names = [m['model'] for m in api_models]
        print(f"    API Models: {', '.join(model_names)}")
    else:
        print("    API Models: (none configured) — skipping API queries")
    
    queries_base = load_queries_from_file(query_file)
    if not queries_base:
        print("    No queries found. Skipping.")
        return

    # Prepare detailed query list
    queries = []
    for item in queries_base:
        if isinstance(item, dict):
            q_id = item.get("id")
            q_text = item.get("query")
            q_reuse = item.get("reuse_chat")
            q_is_mcq = item.get("is_mcq")
        else:
            q_id, q_text = item
            q_reuse = None
            q_is_mcq = None
        for r in range(runs):
            # Format: {id}_run{r}
            full_id = f"{q_id}_run{r}"
            queries.append(
                {
                    "id": full_id,
                    "query": q_text,
                    "reuse_chat": q_reuse,
                    "is_mcq": q_is_mcq,
                }
            )
            
    if shuffle:
        queries = shuffle_no_consecutive(queries, seed=exp_seed)
        print(f"    Shuffled {len(queries)} queries.")
        if exp_seed is not None:
            print(f"    Shuffle seed: {exp_seed}")

    # Process Queries
    prev_reuse = None
    queries_done = 0
    for i, item in enumerate(queries):
        if stop_event and stop_event.is_set():
            print(">> Stop signal received. Halting.")
            return

        q_id = item.get("id")
        q_text = item.get("query")
        item_reuse = item.get("reuse_chat")
        effective_reuse = reuse_chat if item_reuse is None else bool(item_reuse)
        q_preview = (q_text or "").replace("\n", " ")

        # Skip if already saved — allows resuming an interrupted run
        _sid = "".join(c for c in str(q_id or "") if c.isalnum() or c in ("-", "_")).strip()
        if (output_dir / f"{_sid}.html").exists():
            print(f">> [{i+1}/{len(queries)}] Skipping {q_id} (already saved).")
            continue

        try:
            print(f"\n[{i+1}/{len(queries)}] Processing {q_id}...")
            skip_query = False

            # Check if this query will update memory — if so, snapshot first
            if item.get("updated"):
                print(">> Query has updated:true — taking memory snapshot before processing...")
                page.get('https://chatgpt.com/#settings/Personalization')
                time.sleep(2)
                try:
                    page.ele('text:Manage', timeout=2).click()
                    time.sleep(1.5)
                    _save_memories_snapshot(
                        page,
                        run_id=run_id,
                        session_id=session_id,
                        label=f"before_{q_id}",
                        persona_name=persona_name,
                        persona_id=persona_id,
                    )
                    # Close the memory dialog
                    close_btn = page.ele('css:button[aria-label="Close"]', timeout=2)
                    if close_btn:
                        close_btn.click()
                    time.sleep(0.5)
                except Exception as exc:
                    print(f">> Failed to take memory snapshot: {exc}")
                # Navigate back to chat
                page.get("https://chatgpt.com/")
                time.sleep(2)

            # 1. New Chat — only when the current query does NOT want to reuse
            if not effective_reuse:
                use_temp = (exp_type == "temporary") and (exp_name != "json_queries")
                ensure_new_chat(page, attempts=3, temporary_chat=use_temp)
                # Select interface model after opening a new chat
                if interface_model:
                    select_interface_model(page, interface_model)
            prev_reuse = effective_reuse

            if not page.ele('#prompt-textarea', timeout=5):
                print(">> Input box still missing. Refreshing...")
                page.refresh()
                time.sleep(5)

            # Barrier: wait for all sessions to finish model selection before sending
            if sync_barrier:
                print(f">> [session {session_id}] Waiting at send barrier {i}...")
                try:
                    sync_barrier.wait()
                    print(f">> [session {session_id}] All sessions ready to send query {i+1}.")
                except Exception as e:
                    print(f">> Send barrier wait failed ({e}) — proceeding anyway")

            # Check for unusual-activity block before attempting each query
            _check_unusual_activity(page)

            is_mcq = item.get("is_mcq")
            if is_mcq is None:
                is_mcq = "_mcq" in str(q_id or "").lower()

            # 2. Type Prompt
            print(f">> Sending: {q_preview[:50]}...")
            if fast_mode or is_mcq:
                if not paste_prompt(page, '#prompt-textarea', q_text):
                    print(">> Failed to paste prompt.")
                    skip_query = True
            else:
                _think_delay(q_text)
                if not human_type(page, '#prompt-textarea', q_text):
                    if not paste_prompt(page, '#prompt-textarea', q_text):
                        print(">> Failed to type or paste prompt.")
                        skip_query = True

            # 3. Click Send
            send_btn = page.ele('button[data-testid="send-button"]', timeout=2)
            sent_at = None
            if not skip_query:
                sent_at = datetime.now().isoformat()
                if send_btn:
                    time.sleep(random.uniform(0.15, 0.6))
                    send_btn.click()
                else:
                    time.sleep(random.uniform(0.15, 0.6))
                    page.ele('#prompt-textarea').input('\n')

            if not skip_query:
                # 4. Wait for Response
                print(">> Waiting for response generation...")
                final_response_html = None

                # We monitor text stability and completion buttons
                print(">> Monitoring generation...")
                poll_interval = 0.2
                wait_timeout = 600  # 10 minutes hard timeout (re-send query)
                soft_refresh_timeout = 90  # seconds before a soft page refresh (no re-send)
                soft_refreshed = False
                wait_start = time.time()

                while True:
                    try:
                        _html = page.html or ""
                    except Exception as _html_err:
                        print(f">> page.html error: {_html_err}; retrying...")
                        time.sleep(2)
                        continue
                    voice_visible = 'start voice' in _html.lower()
                    if voice_visible:
                        final_response_html = _html
                        break
                    elapsed = time.time() - wait_start
                    # Soft refresh: unstick a stalled render without re-sending the query
                    if not soft_refreshed and elapsed > soft_refresh_timeout:
                        print(f">> No response after {soft_refresh_timeout}s — soft-refreshing page to unstick...")
                        page.refresh()
                        time.sleep(5)
                        soft_refreshed = True
                    if elapsed > wait_timeout:
                        print(f">> Response timed out after {wait_timeout}s. Refreshing page and re-sending query...")
                        page.refresh()
                        time.sleep(5)
                        if not page.ele('#prompt-textarea', timeout=10):
                            print(">> Input box not found after refresh. Retrying...")
                            page.refresh()
                            time.sleep(5)
                        _check_unusual_activity(page)
                        if fast_mode or is_mcq:
                            paste_prompt(page, '#prompt-textarea', q_text)
                        else:
                            human_type(page, '#prompt-textarea', q_text) or paste_prompt(page, '#prompt-textarea', q_text)
                        send_btn = page.ele('button[data-testid="send-button"]', timeout=2)
                        if send_btn:
                            time.sleep(random.uniform(0.15, 0.6))
                            send_btn.click()
                        else:
                            time.sleep(random.uniform(0.15, 0.6))
                            page.ele('#prompt-textarea').input('\n')
                        wait_start = time.time()
                    time.sleep(poll_interval)

                # Retry model detection indefinitely until we get a valid result
                model_slug = None
                retry = 0
                while True:
                    # Enable debug mode every 5th attempt
                    debug = (retry > 0 and retry % 5 == 0)
                    model_slug = detect_model_info_from_html(final_response_html, debug=debug)

                    retry_msg = f" (attempt {retry + 1})" if retry > 0 else ""
                    print(f">> Model detected{retry_msg}: {model_slug or 'unknown'}")

                    # If we got a model slug, validate it
                    if model_slug:
                        if is_expected_model(model_slug):
                            break  # Valid model detected, proceed

                        # Wrong model / downgrade — pause and re-send instead of aborting
                        is_downgrade = model_slug in DOWNGRADE_SLUGS
                        tag = "DOWNGRADE" if is_downgrade else "WRONG MODEL"
                        pause_secs = 7200
                        print(f">> {tag}: got '{model_slug}' on query {q_id}. "
                              f"Pausing {pause_secs}s then re-sending...")
                        pause_end = time.time() + pause_secs
                        while time.time() < pause_end:
                            if stop_event and stop_event.is_set():
                                print(">> Stop signal received during downgrade pause. Halting.")
                                return
                            time.sleep(5)
                        page.get("https://chatgpt.com/")
                        time.sleep(5)
                        ensure_new_chat(page, attempts=3,
                                       temporary_chat=(exp_type == "temporary" and exp_name != "json_queries"))
                        if interface_model:
                            select_interface_model(page, interface_model)
                        if not page.ele('#prompt-textarea', timeout=10):
                            page.refresh()
                            time.sleep(5)
                        _check_unusual_activity(page)
                        if fast_mode or is_mcq:
                            paste_prompt(page, '#prompt-textarea', q_text)
                        else:
                            human_type(page, '#prompt-textarea', q_text) or paste_prompt(page, '#prompt-textarea', q_text)
                        send_btn_retry = page.ele('button[data-testid="send-button"]', timeout=2)
                        if send_btn_retry:
                            time.sleep(random.uniform(0.15, 0.6))
                            send_btn_retry.click()
                        else:
                            time.sleep(random.uniform(0.15, 0.6))
                            page.ele('#prompt-textarea').input('\n')
                        # Wait for the new response
                        downgrade_wait_start = time.time()
                        while True:
                            try:
                                _dhtml = page.html or ""
                            except Exception:
                                time.sleep(2)
                                continue
                            if 'start voice' in _dhtml.lower():
                                final_response_html = _dhtml
                                break
                            if time.time() - downgrade_wait_start > wait_timeout:
                                print(f">> Re-send timed out after {wait_timeout}s. Refreshing...")
                                page.refresh()
                                time.sleep(5)
                                downgrade_wait_start = time.time()
                            time.sleep(poll_interval)
                        retry = 0
                        continue

                    # Model detection failed (None/unknown) - keep retrying
                    retry += 1
                    if retry >= 10:
                        print(f">> Model detection failed after {retry} attempts — saving with unknown model.")
                        break
                    wait_time = min(2 + retry * 0.5, 10)  # Gradual backoff, max 10s
                    print(f">> Model detection failed, waiting {wait_time:.1f}s and refreshing page...")
                    time.sleep(wait_time)
                    page.refresh()
                    time.sleep(20)
                    final_response_html = page.html

                # Check if the response page contains the unusual-activity block
                if UNUSUAL_ACTIVITY_TEXT in (final_response_html or "").lower():
                    raise UnusualActivityError(
                        f"Unusual activity detected on query {q_id} — aborting session."
                    )

                is_mcq = item.get("is_mcq")
                q_id_str = str(q_id or "").lower()
                skip_save = False
                if save_mcq_only:
                    # Persona mode: only persist MCQ responses; non-MCQ queries
                    # still run (to build chat context) but HTML is not saved.
                    if is_mcq is False:
                        skip_save = True
                    elif is_mcq is None and "_mcq" not in q_id_str:
                        skip_save = True
                if skip_save:
                    print(">> Skipping save (non-MCQ JSON query).")
                else:
                    save_response(
                        output_dir,
                        q_id,
                        final_response_html,
                        0,
                        model_info={"model_slug": model_slug} if model_slug else {},
                        sent_at=sent_at,
                    )
                _maybe_scroll(page)
            else:
                print(">> Skipping send/wait/save due to earlier failure.")

        except UnusualActivityError:
            raise  # always propagate — this blocks the whole session
        except Exception as _query_err:
            import traceback
            print(f"\n>> !! Query {q_id} failed with unexpected error: {_query_err}")
            traceback.print_exc()
            print(">> Continuing to next query.")

        # Inter-query wait
        if True:
            if isinstance(wait_after_saves, list) and len(wait_after_saves) == 2:
                sleep_time = random.uniform(wait_after_saves[0], wait_after_saves[1])
            else:
                sleep_time = 5 if wait_after_saves is None else float(wait_after_saves)

            if sleep_time <= 0:
                sleep_time = random.uniform(0.2, 0.8)
            else:
                sleep_time *= random.uniform(0.8, 1.4)
                if random.random() < 0.08:
                    sleep_time += random.uniform(4, 12)
            print(f">> Waiting {sleep_time:.2f}s before next query...")
            time.sleep(sleep_time)

        if not skip_query:
            queries_done += 1
            if batch_size > 0 and queries_done % batch_size == 0:
                pause_secs = pause_hours * 3600
                print(f"\n>> Batch of {batch_size} queries complete. Pausing {pause_hours:.1f}h ({pause_secs:.0f}s) before continuing...")
                pause_end = time.time() + pause_secs
                while time.time() < pause_end:
                    if stop_event and stop_event.is_set():
                        return
                    remaining = pause_end - time.time()
                    if int(remaining) % 600 == 0 and remaining > 0:
                        print(f">> Batch pause: {remaining/3600:.1f}h remaining...")
                    time.sleep(30)
                print(f">> Resuming after batch pause.")

def run_audit(
    config_path,
    delay_seconds=0,
    user_data_dir=USER_DATA_PATH,
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
    clear_chat_first=False,
    clear_memory_first=False,
    custom_instructions=None,
    clear_only=False,
    api_only=False,
    skip_end_clear=False,
    skip_end_clear_memory=True,
    save_mcq_only=False,
    experiment_index=None,
    stop_event=None,
    api_ready_event=None,
    parsed_config=None,
):
    combined_log_path = None
    stdout_orig = None
    stderr_orig = None
    log_handles = None
    if run_id and log_to_file:
        combined_log_path, stdout_orig, stderr_orig, log_handles = _setup_timestamped_logging(run_id, session_id)
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
        # Load Config — prefer pre-parsed dict to avoid race conditions
        # when multiple subprocesses read the same YAML file from disk.
        if parsed_config is not None:
            full_config = parsed_config
        else:
            try:
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config {config_path}: {e}")
                return

        if isinstance(full_config, list):
            # Treat a raw list (e.g., JSON queries) as a single simple experiment.
            defaults = {
                "query_file": str(config_path),
                "runs": 1,
                "shuffle": False,
                "wait_after_prompts": 0,
                "wait_after_saves": 0,
            }
            experiments = [
                {
                    "name": "json_queries",
                    "type": "temporary",
                }
            ]
            config_api_models = []
        else:
            defaults = full_config.get('defaults', {})
            experiments = full_config.get('experiments', [])
            config_api_models = full_config.get('api_models', [])

        # CLI api_models override YAML; if neither, empty list.
        if api_models is None:
            api_models = config_api_models or []

        # In sync mode a single experiment may be selected (only if there are multiple experiments).
        if experiment_index is not None and experiment_index < len(experiments):
            experiments = [experiments[experiment_index]]
        elif experiment_index is not None and len(experiments) > 0:
            # If experiment_index is out of range but we have experiments, just use the first one
            # This happens when using separate config files with --configs flag
            experiments = [experiments[0]]

        # Infer persona info (used for memory snapshot metadata)
        persona_name = None
        persona_id = None
        if isinstance(full_config, list):
            persona_name, persona_id = _infer_persona_from_value(config_path)
        else:
            if experiments and len(experiments) == 1:
                persona_name, persona_id = _infer_persona_from_value(experiments[0].get("name"))
            if persona_id is None:
                persona_name, persona_id = _infer_persona_from_value(
                    (experiments[0].get("query_file") if experiments else None) or defaults.get("query_file")
                )

        if api_only:
            # ── API-only mode: skip browser, just run API queries ──
            # Deduplicate by query file — experiments differ only in browser
            # settings (interface_model, type) which are irrelevant for API.
            model_names = [m['model'] for m in api_models]
            print(f">> [API] Session {session_id} | Models: {', '.join(model_names)}")
            if api_ready_event:
                print(">> [API] Waiting for interface session to be ready...")
                while not api_ready_event.is_set():
                    if stop_event and stop_event.is_set():
                        print(">> [API] Stop signal received while waiting. Halting.")
                        return
                    time.sleep(5)
                print(">> [API] Interface session ready. Proceeding.")
            if delay_seconds > 0:
                print(f">> Delaying start by {delay_seconds:.1f} seconds...")
                time.sleep(delay_seconds)

            # Collect unique query files (keyed by resolved path)
            seen_query_files = set()
            unique_query_configs = []
            for experiment in experiments:
                qf = experiment.get("query_file", defaults.get("query_file"))
                resolved = str(Path(qf).resolve()) if qf else qf
                if resolved not in seen_query_files:
                    seen_query_files.add(resolved)
                    unique_query_configs.append({
                        "query_file": qf,
                        "runs": int(experiment.get("runs", defaults.get("runs", 1))),
                        "shuffle": bool(experiment.get("shuffle", defaults.get("shuffle", False))),
                        "seed": experiment.get("seed", defaults.get("seed")),
                    })

            run_id_str = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            api_output_base = DATA_DIR / "api" / run_id_str / f"session_{session_id:02d}"

            for qcfg in unique_query_configs:
                query_file = qcfg["query_file"]
                runs = qcfg["runs"]
                shuffle = qcfg["shuffle"]
                exp_seed = qcfg["seed"]
                if seed_override is not None:
                    exp_seed = seed_override

                print(f"\n>>> [API] Query file: {query_file}  Models: {', '.join(model_names)}")
                print(f"    Output: {api_output_base}")
                queries_base = load_queries_from_file(query_file)
                if not queries_base:
                    print("    No queries found. Skipping.")
                    continue
                queries = []
                for item in queries_base:
                    if isinstance(item, dict):
                        q_id = item.get("id")
                        q_text = item.get("query")
                    else:
                        q_id, q_text = item
                    for r in range(runs):
                        queries.append({"id": f"{q_id}_run{r}", "query": q_text})
                if shuffle:
                    queries = shuffle_no_consecutive(queries, seed=exp_seed)
                api_key = OPENAI_KEY

                # API worker runs independently - no barrier synchronization
                for i, item in enumerate(queries):
                    if stop_event and stop_event.is_set():
                        print(f">> [API] Stop signal received (interface session errored). Halting API queries.")
                        return
                    for api_cfg in api_models:
                        m_name = api_cfg["model"]
                        m_params = {k: v for k, v in api_cfg.items() if k != "model"}
                        m_dir = m_name + ("_" + "_".join(f"{k}-{v}" for k, v in sorted(m_params.items())) if m_params else "")
                        m_output = api_output_base / m_dir
                        print(f"\n[{i+1}/{len(queries)}] {item['id']} -> {m_name}")
                        run_api_query(item["id"], item["query"], m_output, m_name, api_key=api_key, **m_params)

            print(f"\n>> [API] Session {session_id} complete.")
            return

        # ── Browser mode ──
        # Setup Browser (if not provided)
        if page is None:
            co = ChromiumOptions()
            co.auto_port()
            co.set_tmp_path(str(DATA_DIR / "drission_tmp" / f"session_{session_id:02d}"))
            co.set_argument(f'--user-data-dir={user_data_dir}')
            co.set_argument('--no-first-run')
            co.set_argument('--mute-audio')
            page = ChromiumPage(co)
        
        # 1. Listen for console logs or errors if we want
        # page.listen.start('backend-anon/f/conversation') # Optional
        
        # Navigate
        if 'chatgpt.com' not in page.url:
            print(">> Navigating to ChatGPT...")
            page.get('https://chatgpt.com/')
            time.sleep(random.uniform(0.8, 1.8))
        
        # Login with retries
        login_max_retries = 5
        login_ok = False
        for login_attempt in range(login_max_retries):
            try:
                login_ok = handle_google_login(page, allow_manual=allow_manual_login)
            except Exception as exc:
                print(f">> Login attempt {login_attempt + 1}/{login_max_retries} crashed: {exc}")
                login_ok = False
            if login_ok:
                break
            wait_secs = min(30 * (login_attempt + 1), 120)
            print(f">> Login attempt {login_attempt + 1}/{login_max_retries} failed. "
                  f"Refreshing and retrying in {wait_secs}s...")
            time.sleep(wait_secs)
            try:
                page.get('https://chatgpt.com/')
                time.sleep(3)
            except Exception:
                pass
        if not login_ok:
            print(">> Login failed after all retries. Aborting.")
            if stop_event:
                print(">> Signalling API worker to stop...")
                stop_event.set()
            return

        if api_ready_event:
            api_ready_event.set()

        if clear_chat_first:
            clear_chat(page)
            time.sleep(random.uniform(0.6, 1.4))
        if clear_memory_first:
            clear_memory(
                page,
                run_id=run_id,
                session_id=session_id,
                label="pre_run",
                persona_name=persona_name,
                persona_id=persona_id,
            )
            time.sleep(random.uniform(0.6, 1.4))

        if custom_instructions is not None:
            set_custom_instructions(page, custom_instructions)
            time.sleep(random.uniform(0.6, 1.4))

        if clear_only:
            # No barriers - sessions run independently
            print(">> Clear-only mode: skipping experiments.")
            return
        
        print(f">> Delaying start by {delay_seconds:.1f} seconds...")
        time.sleep(delay_seconds)

        # Note: No login barrier needed - sessions can proceed independently
        # They will sync at query execution time via barriers in run_experiment

        print(f">> Found {len(experiments)} experiments.")
        
        # Run Experiments
        try:
            for exp in experiments:
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
                )
        except Exception as exc:
            print(f"\n>> SESSION {session_id} ERROR: {exc}")
            if stop_event:
                print(">> Signalling API worker to stop...")
                stop_event.set()
            raise

        print("\nAll Experiments Completed.")
        if not skip_end_clear:
            # Clear chat history and memory at the end of the run
            try:
                clear_chat(page)
                time.sleep(random.uniform(0.6, 1.4))
            except Exception as e:
                print(f">> End-of-run clear chat failed: {e}")
            if not skip_end_clear_memory:
                try:
                    clear_memory(
                        page,
                        run_id=run_id,
                        session_id=session_id,
                        label="end_of_run",
                        persona_name=persona_name,
                        persona_id=persona_id,
                    )
                    time.sleep(random.uniform(0.6, 1.4))
                except Exception as e:
                    print(f">> End-of-run clear memory failed: {e}")
            if custom_instructions is not None:
                try:
                    set_custom_instructions(page, '')
                    time.sleep(random.uniform(0.6, 1.4))
                except Exception as e:
                    print(f">> End-of-run clear custom instructions failed: {e}")
            # Refresh back to a clean chat page after clearing
            try:
                page.get('https://chatgpt.com/')
                time.sleep(random.uniform(1.5, 2.5))
            except Exception as e:
                print(f">> End-of-run refresh failed: {e}")
    finally:
        if log_handles:
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig
            for handle in log_handles:
                handle.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='yamls/exp_chatgpt.yaml', help='Path to experiments yaml')
    parser.add_argument('--configs', type=str, default=None, help='Comma-separated list of configs (one per session)')
    parser.add_argument('--start-in', type=float, default=0, help='Delay start by N minutes')
    parser.add_argument('--sessions', type=int, default=None, help='Override number of concurrent browser sessions')
    parser.add_argument('--profile-base', type=str, default=None, help='Base directory for per-session Chrome profiles')
    parser.add_argument('--seed', type=int, default=None, help='Seed for deterministic shuffle (applies when shuffle=true)')
    parser.add_argument('--no-parse', action='store_true', help='Skip parsing saved HTML after experiments complete')
    parser.add_argument('--repeat-every-hours', type=float, default=0, help='Repeat full experiment every N hours (0 = run once)')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of full experiment passes to run (default: 1; combine with --repeat-every-hours to add a delay between passes)')
    parser.add_argument('--allow-manual-login', action='store_true', help='Allow manual login prompts when repeating runs')
    parser.add_argument('--api-models', type=str, default=None,
                        help='Comma-separated API models (overrides YAML api_models). '
                             'Format: model_name or model_name:param_value '
                             'e.g. "gpt-5.2-chat-latest:0.7,gpt-5.2-2025-12-11:low,gpt-5.2-2025-12-11:high" '
                             'where param can be temperature (float) or reasoning_effort (low/medium/high)')
    parser.add_argument('--clear-chat', action='store_true', help='Clear chat history and exit (no experiments).')
    parser.add_argument('--clear-memory', action='store_true', help='Clear memory and exit (no experiments).')
    parser.add_argument('--custom-instructions', type=str, default=None,
                        help='Custom instructions to set before running experiments. '
                             'Pass a string or @path/to/file.txt to read from file. '
                             'Cleared automatically at end of run.')
    parser.add_argument('--interface-model', type=str, default=None,
                        help='Model to select in ChatGPT UI dropdown (overrides per-experiment config).')
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Tag appended to the run ID for output directories '
                             '(e.g. --output-tag mmlu produces 2026-03-07_12-49-14-mmlu). '
                             'Applied to raw_html, parsed_json, api, logs, and memory_snapshots.')
    args = parser.parse_args()
    delay_seconds = 0

    # Resolve custom instructions (support @file.txt to read from file)
    custom_instructions = args.custom_instructions
    if custom_instructions and custom_instructions.startswith('@'):
        ci_path = Path(custom_instructions[1:])
        if not ci_path.exists():
            ci_path = BASE_DIR / custom_instructions[1:]
        if ci_path.exists():
            custom_instructions = ci_path.read_text(encoding='utf-8').strip()
            print(f">> Loaded custom instructions from {ci_path} ({len(custom_instructions)} chars)")
        else:
            print(f"Custom instructions file not found: {custom_instructions[1:]}")
            sys.exit(1)

    def _resolve_config_path(raw_path):
        path = Path(raw_path)
        if path.exists():
            return path
        possible_path = BASE_DIR / raw_path
        if possible_path.exists():
            return possible_path
        print(f"Config file not found: {raw_path}")
        sys.exit(1)

    if args.configs:
        raw_configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        if not raw_configs:
            print("No valid configs provided in --configs.")
            sys.exit(1)
        config_paths = [_resolve_config_path(c) for c in raw_configs]
    else:
        config_paths = [_resolve_config_path(args.config)]

    if args.start_in and args.start_in > 0:
        delay_seconds = args.start_in * 60

    # ── Build CLI api_models override (list of dicts) ──
    cli_api_models = None  # None = defer to YAML config
    if args.api_models:
        cli_api_models = []
        for raw in args.api_models.split(","):
            raw = raw.strip()
            if not raw or raw.lower() in {"none", "off", "false", "0"}:
                continue
            # Support "model_name:param" shorthand
            if ":" in raw:
                parts = raw.split(":", 1)
                model_name = parts[0]
                param_value = parts[1]

                # Check if param is a reasoning_effort value (low/medium/high) or temperature (float)
                if param_value.lower() in {"low", "medium", "high"}:
                    cli_api_models.append({"model": model_name, "reasoning_effort": param_value.lower()})
                else:
                    # Try to parse as temperature (float)
                    try:
                        cli_api_models.append({"model": model_name, "temperature": float(param_value)})
                    except ValueError:
                        print(f"Warning: Could not parse parameter '{param_value}' for model '{model_name}'. Skipping.")
                        continue
            else:
                cli_api_models.append({"model": raw})

    # ── Read ALL configs once upfront and reuse the parsed dicts ──
    # This avoids a race condition where subprocesses re-read the YAML
    # from disk independently and can see different file contents if
    # the user edits the file between subprocess launches.
    _parsed_configs = {}
    for cp in config_paths:
        if cp not in _parsed_configs:
            try:
                with open(cp, 'r') as f:
                    _parsed_configs[cp] = yaml.safe_load(f)
            except Exception:
                _parsed_configs[cp] = {}

    _pre_config = _parsed_configs[config_paths[0]]
    if isinstance(_pre_config, dict):
        config_mode = _pre_config.get("mode", "sequential")
        config_experiments = _pre_config.get("experiments", [])
    else:
        config_mode = "sequential"
        config_experiments = []

    # Determine session count: CLI --sessions overrides; otherwise mode decides.
    if args.sessions is not None:
        sessions = max(1, args.sessions)
    elif config_mode == "sync":
        sessions = max(1, len(config_experiments))
    else:
        sessions = 1

    if len(config_paths) not in (1, sessions):
        print(f"--configs count must be 1 or match sessions ({sessions}). Got {len(config_paths)}.")
        sys.exit(1)

    def run_once(run_delay_seconds, run_profile_base, allow_manual_login, page=None, pages=None, use_threads=False):
        # Re-read configs from disk at the start of each cycle so that
        # edits made during --repeat-every-hours sleeps take effect,
        # while still ensuring all subprocesses within a single cycle
        # see the exact same config snapshot.
        for cp in list(_parsed_configs):
            try:
                with open(cp, 'r') as f:
                    _parsed_configs[cp] = yaml.safe_load(f)
            except Exception:
                pass

        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.output_tag:
            run_id = f"{run_id}-{args.output_tag}"
        clear_only = bool(args.clear_chat) or bool(args.clear_memory)
        if run_profile_base is None:
            # Automatically use persistent profile for multi-session mode
            if sessions > 1:
                run_profile_base = DATA_DIR / "chrome_profiles" / "default_multi_session"
                print(f">> Multi-session mode: using persistent profile base: {run_profile_base}")
            else:
                run_profile_base = DATA_DIR / "chrome_profiles" / f"run_{run_id}"
                print(f">> Using ephemeral profile base: {run_profile_base}")
        else:
            print(f">> Using persistent profile base: {run_profile_base}")

        # Resolve the effective api_models list for this run.
        # cli_api_models is None when no CLI override → let run_audit read YAML.
        # To keep browser sessions API-free, we pass an empty list to them and
        # spawn a dedicated API-only worker with the real list.
        current_primary_config = _parsed_configs[config_paths[0]]
        if cli_api_models is not None:
            effective_api_models = cli_api_models
        else:
            effective_api_models = []
            if isinstance(current_primary_config, dict):
                effective_api_models = current_primary_config.get("api_models", [])

        # Per-query cross-session barrier (sync mode only)
        # API models are included as barrier participants so all requests
        # (browser + API) fire simultaneously at each query.
        n_api_workers = len(effective_api_models) if (config_mode == "sync" and sessions > 1 and effective_api_models) else 0
        if config_mode == "sync" and sessions > 1:
            sync_dir = DATA_DIR / "sync" / run_id
            sync_dir.mkdir(parents=True, exist_ok=True)
            sync_barrier_info = {
                "parties": sessions + n_api_workers,
                "sync_dir": str(sync_dir),
            }
        else:
            sync_barrier_info = None
        # Browser sessions get an empty api_models list when API workers
        # participate in the barrier separately.
        browser_api_models = [] if n_api_workers > 0 else effective_api_models

        # Seed for consistent shuffles across sessions
        seed_override = args.seed
        if sessions > 1 and seed_override is None:
            seed_override = int(time.time())
            print(f">> Multi-session mode: using seed {seed_override} for consistent shuffles.")

        print(f">> Mode: {config_mode} | Sessions: {sessions}")

        # Shared stop signal: interface sessions set this on error to halt the API worker
        stop_event = mp.Event()
        # Shared ready signal: set after first login + 2h sleep so API worker waits
        api_ready_event = mp.Event()

        procs = []
        _thread_log_state = None

        use_threads = (page is not None and sessions == 1) or \
                      (pages is not None and sessions > 1)

        if use_threads:
            # Set up combined logging once for all threads (avoids race
            # condition where each thread's run_audit replaces sys.stdout).
            log_dir = DATA_DIR / "logs" / run_id
            log_dir.mkdir(parents=True, exist_ok=True)
            _tl_handle = open(log_dir / "combined.log", "a", encoding="utf-8")
            _tl_stdout = sys.stdout
            _tl_stderr = sys.stderr
            sys.stdout = _TimestampedTee(_tl_stdout, [_tl_handle])
            sys.stderr = _TimestampedTee(_tl_stderr, [_tl_handle])
            _thread_log_state = (_tl_stdout, _tl_stderr, _tl_handle)

        if page is not None and sessions == 1:
            # ── Persistent single browser: reuse session via thread ──
            t = threading.Thread(target=run_audit, kwargs={
                "config_path": config_paths[0],
                "parsed_config": _parsed_configs[config_paths[0]],
                "delay_seconds": run_delay_seconds,
                "run_id": run_id,
                "session_id": 0,
                "seed_override": seed_override,
                "api_models": browser_api_models,
                "interface_model": args.interface_model,
                "allow_manual_login": allow_manual_login,
                "page": page,
                "log_to_file": False,
                "clear_chat_first": bool(args.clear_chat),
                "clear_memory_first": bool(args.clear_memory),
                "custom_instructions": custom_instructions,
                "clear_only": clear_only,
                "stop_event": stop_event,
                "api_ready_event": api_ready_event,
                "sync_barrier_info": sync_barrier_info,
            })
            t.start()
            procs.append(t)
        elif pages is not None and sessions > 1:
            # ── Persistent multi-browser: reuse sessions via threads ──
            for sid in range(sessions):
                session_config = config_paths[sid] if len(config_paths) > 1 else config_paths[0]
                kwargs = {
                    "config_path": session_config,
                    "parsed_config": _parsed_configs[session_config],
                    "delay_seconds": run_delay_seconds,
                    "run_id": run_id,
                    "session_id": sid,
                    "seed_override": seed_override,
                    "api_models": browser_api_models,
                    "interface_model": args.interface_model,
                    "allow_manual_login": allow_manual_login,
                    "page": pages[sid],
                    "log_to_file": False,
                    "clear_chat_first": bool(args.clear_chat),
                    "clear_memory_first": bool(args.clear_memory),
                    "custom_instructions": custom_instructions,
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
            # ── Sync mode: one browser session per experiment (new processes) ──
            for exp_idx in range(sessions):
                user_data_dir = _resolve_user_data_dir(run_profile_base, exp_idx, sessions)
                session_config = config_paths[exp_idx] if len(config_paths) > 1 else config_paths[0]
                p = mp.Process(
                    target=run_audit,
                    kwargs={
                        "config_path": session_config,
                        "parsed_config": _parsed_configs[session_config],
                        "delay_seconds": run_delay_seconds,
                        "user_data_dir": user_data_dir,
                        "run_id": run_id,
                        "session_id": exp_idx,
                        "sync_barrier_info": sync_barrier_info,
                        "seed_override": seed_override,
                        "api_models": browser_api_models,
                        "interface_model": args.interface_model,
                        "allow_manual_login": allow_manual_login,
                        "clear_chat_first": bool(args.clear_chat),
                        "clear_memory_first": bool(args.clear_memory),
                        "custom_instructions": custom_instructions,
                        "clear_only": clear_only,
                        "experiment_index": exp_idx,
                        "stop_event": stop_event,
                        "api_ready_event": api_ready_event,
                    },
                )
                p.start()
                procs.append(p)
        else:
            # ── Sequential mode: all experiments in one browser session (new processes) ──
            for session_id in range(sessions):
                user_data_dir = _resolve_user_data_dir(run_profile_base, session_id, sessions)
                session_config = config_paths[session_id] if len(config_paths) > 1 else config_paths[0]
                p = mp.Process(
                    target=run_audit,
                    kwargs={
                        "config_path": session_config,
                        "parsed_config": _parsed_configs[session_config],
                        "delay_seconds": run_delay_seconds,
                        "user_data_dir": user_data_dir,
                        "run_id": run_id,
                        "session_id": session_id,
                        "sync_barrier_info": sync_barrier_info,
                        "seed_override": seed_override,
                        "api_models": browser_api_models,
                        "interface_model": args.interface_model,
                        "allow_manual_login": allow_manual_login,
                        "clear_chat_first": bool(args.clear_chat),
                        "clear_memory_first": bool(args.clear_memory),
                        "custom_instructions": custom_instructions,
                        "clear_only": clear_only,
                        "stop_event": stop_event,
                    },
                )
                p.start()
                procs.append(p)

        # Launch API barrier workers (sync mode: each API model is its own
        # barrier participant, so browser + API requests fire simultaneously).
        if n_api_workers > 0 and sync_barrier_info:
            defaults_cfg = current_primary_config.get("defaults", {}) if isinstance(current_primary_config, dict) else {}
            api_query_file = defaults_cfg.get("query_file")
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
                    "query_file": api_query_file,
                    "runs": api_runs,
                    "shuffle": api_shuffle,
                    "seed": seed_override,
                    "api_cfg": api_cfg,
                    "api_key": OPENAI_KEY,
                    "output_base": str(api_output_base),
                    "sync_barrier_info": api_bi,
                    "stop_event": stop_event,
                    "batch_size": int(defaults_cfg.get("batch_size", 0)),
                    "pause_hours": float(defaults_cfg.get("pause_hours", 3)),
                })
                p.start()
                procs.append(p)

        # Wait for all sessions (browser + API) to finish
        for p in procs:
            p.join()

        # Restore logging after threads complete
        if _thread_log_state:
            sys.stdout = _thread_log_state[0]
            sys.stderr = _thread_log_state[1]
            _thread_log_state[2].close()
            _thread_log_state = None

        if (not args.no_parse) and (not clear_only):
            parse_saved_htmls(run_id)

    _top_defaults = _pre_config.get("defaults", {}) if isinstance(_pre_config, dict) else {}
    repeat_hours = float(args.repeat_every_hours or _top_defaults.get("repeat_every_hours", 0))
    if args.profile_base:
        repeat_profile_base = Path(args.profile_base)
    elif repeat_hours > 0:
        repeat_profile_base = DATA_DIR / "chrome_profiles" / "repeat_profile"
    else:
        repeat_profile_base = None

    num_runs = int(args.num_runs or _top_defaults.get("num_runs", 1))
    repeat_seconds = max(1.0, repeat_hours * 3600.0) if repeat_hours > 0 else 0
    use_persistent = repeat_hours > 0 or num_runs > 1

    persistent_page = None
    persistent_pages = None
    if use_persistent:
        if sessions == 1:
            user_data_dir = _resolve_user_data_dir(repeat_profile_base, 0, sessions)
            co = ChromiumOptions()
            co.auto_port()
            co.set_tmp_path(str(DATA_DIR / "drission_tmp" / "session_00"))
            co.set_argument(f'--user-data-dir={user_data_dir}')
            co.set_argument('--no-first-run')
            co.set_argument('--mute-audio')
            persistent_page = ChromiumPage(co)
        else:
            persistent_pages = []
            for session_id in range(sessions):
                user_data_dir = _resolve_user_data_dir(repeat_profile_base, session_id, sessions)
                co = ChromiumOptions()
                co.auto_port()
                co.set_tmp_path(str(DATA_DIR / "drission_tmp" / f"session_{session_id:02d}"))
                co.set_argument(f'--user-data-dir={user_data_dir}')
                co.set_argument('--no-first-run')
                co.set_argument('--mute-audio')
                persistent_pages.append(ChromiumPage(co))
            print(f">> Created {sessions} persistent ChromiumPage instances.")

    # infinite loop when repeat_hours set and num_runs==1; finite loop otherwise
    run_count = 0
    first_run = True
    while True:
        allow_manual_login = bool(args.allow_manual_login) or first_run
        run_once(
            delay_seconds if first_run else 0,
            repeat_profile_base,
            allow_manual_login,
            page=persistent_page,
            pages=persistent_pages,
            use_threads=(sessions > 1),
        )
        run_count += 1
        first_run = False

        # Stop if we've hit the requested number of runs
        if num_runs > 1 and run_count >= num_runs:
            print(f">> Completed {run_count}/{num_runs} runs. Done.")
            break

        # Stop if no repeat schedule and num_runs==1 (default single-run behaviour)
        if repeat_hours <= 0 and num_runs <= 1:
            break

        print(f">> Run {run_count} complete. Sleeping {repeat_seconds:.0f}s before next run...")
        time.sleep(repeat_seconds)
