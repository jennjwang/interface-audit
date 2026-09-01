"""Throughput limiter shared by the interface scrapers.

Enforces the collection protocol documented in README.md
(§ Responsible-Use Guidance): a cap on queries per rolling window, and a
cooldown after any rate-limit event. Before this module those limits were
operational conventions the software did not apply; they are now enforced.

One limiter instance is shared by every session of a run, so the cap covers
the run as a whole rather than each browser session independently.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque

# Defaults match the protocol described in the README.
DEFAULT_MAX_QUERIES = 150
DEFAULT_WINDOW_SECONDS = 3 * 60 * 60      # 150 queries per three hours
DEFAULT_COOLDOWN_SECONDS = 2 * 60 * 60    # two hours after a rate-limit event

# Phrases providers use when rate-limiting or capping usage. Matched against
# scraped response text, case-insensitively. Kept deliberately narrow: a false
# positive costs a two-hour pause, so these must not match ordinary refusals
# or a model merely talking about rate limits in the abstract.
RATE_LIMIT_PATTERNS = (
    r"you(?:'ve| have) (?:reached|hit) (?:your|the) .{0,40}\blimit\b",
    r"(?:message|usage|rate) limit reached",
    r"you(?:'re| are) sending messages too (?:quickly|fast)",
    r"\btoo many requests\b",
    r"limit will reset",
    r"try again (?:in|after) \d+ (?:second|minute|hour)",
    r"upgrade to .{0,30}continue",
)

_RATE_LIMIT_RE = re.compile("|".join(RATE_LIMIT_PATTERNS), re.IGNORECASE)

# Granularity of interruptible sleeps: how fast a stop signal is noticed.
_TICK_SECONDS = 1


def looks_rate_limited(text: str | None) -> bool:
    """True if scraped text carries a provider rate-limit or usage-cap notice."""
    if not text:
        return False
    return bool(_RATE_LIMIT_RE.search(text))


class RateLimiter:
    """Sliding-window query cap plus a cooldown after rate-limit events.

    Thread-safe: sessions running concurrently share one instance.
    """

    def __init__(
        self,
        max_queries: int = DEFAULT_MAX_QUERIES,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        name: str = "",
    ):
        self.max_queries = max(0, int(max_queries))
        self.window_seconds = float(window_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.name = name
        self._lock = threading.Lock()
        self._sent: deque[float] = deque()
        self._cooldown_until = 0.0

    @property
    def enabled(self) -> bool:
        return self.max_queries > 0

    def describe(self) -> str:
        if not self.enabled:
            return "rate limiting DISABLED (--max-queries-per-window 0)"
        return (
            f"{self.max_queries} queries / {self.window_seconds / 3600:.3g}h, "
            f"{self.cooldown_seconds / 3600:.3g}h cooldown on rate-limit events"
        )

    def _sleep(self, seconds: float, stop_event) -> bool:
        """Interruptible sleep. False if stop_event fired."""
        end = time.time() + seconds
        while time.time() < end:
            if stop_event is not None and stop_event.is_set():
                return False
            time.sleep(min(_TICK_SECONDS, max(0.0, end - time.time())))
        return True

    def acquire(self, stop_event=None) -> bool:
        """Block until another query may be sent, then record it.

        Returns False if stop_event fired while waiting; callers should halt.
        """
        if not self.enabled:
            return True

        announced = False
        while True:
            if stop_event is not None and stop_event.is_set():
                return False

            with self._lock:
                now = time.time()
                if now < self._cooldown_until:
                    wait = self._cooldown_until - now
                    reason = "cooldown"
                else:
                    cutoff = now - self.window_seconds
                    while self._sent and self._sent[0] <= cutoff:
                        self._sent.popleft()
                    if len(self._sent) < self.max_queries:
                        self._sent.append(now)
                        return True
                    wait = self._sent[0] + self.window_seconds - now
                    reason = "window full"

            if not announced:
                print(
                    f">> Rate limit ({reason}): waiting {wait / 60:.1f} min "
                    f"before the next query. [{self.describe()}]"
                )
                announced = True
            if not self._sleep(min(wait, 60), stop_event):
                return False

    def trigger_cooldown(self, reason: str = "", stop_event=None) -> bool:
        """Start the post-rate-limit cooldown and block for its duration.

        Returns False if stop_event fired during the pause.
        """
        if self.cooldown_seconds <= 0:
            return True
        with self._lock:
            self._cooldown_until = max(
                self._cooldown_until, time.time() + self.cooldown_seconds
            )
            remaining = self._cooldown_until - time.time()
        label = f" ({reason})" if reason else ""
        print(
            f">> RATE LIMIT{label}: pausing {remaining / 60:.0f} min "
            f"before resuming collection."
        )
        return self._sleep(remaining, stop_event)

    def check_response(self, text: str | None, stop_event=None) -> bool:
        """Cooldown if scraped text shows a rate-limit notice.

        Returns False only if stop_event fired during the resulting pause.
        """
        if looks_rate_limited(text):
            return self.trigger_cooldown("provider notice in response", stop_event)
        return True


def add_cli_arguments(parser) -> None:
    """Attach the shared rate-limit flags to an audit script's parser."""
    parser.add_argument(
        "--max-queries-per-window",
        type=int,
        default=DEFAULT_MAX_QUERIES,
        help=(
            f"Cap on queries per rolling window (default: {DEFAULT_MAX_QUERIES}). "
            "0 disables the cap — only for platforms where you have confirmed "
            "higher throughput is permitted."
        ),
    )
    parser.add_argument(
        "--rate-limit-window",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"Window length in seconds (default: {DEFAULT_WINDOW_SECONDS}, i.e. 3h).",
    )
    parser.add_argument(
        "--rate-limit-cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help=f"Pause after a rate-limit event, in seconds (default: {DEFAULT_COOLDOWN_SECONDS}, i.e. 2h).",
    )


def from_args(args, name: str = "") -> RateLimiter:
    """Build a limiter from parsed CLI arguments."""
    return RateLimiter(
        max_queries=getattr(args, "max_queries_per_window", DEFAULT_MAX_QUERIES),
        window_seconds=getattr(args, "rate_limit_window", DEFAULT_WINDOW_SECONDS),
        cooldown_seconds=getattr(args, "rate_limit_cooldown", DEFAULT_COOLDOWN_SECONDS),
        name=name,
    )
