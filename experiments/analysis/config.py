"""Shared configuration for metabench analysis scripts.

Works across any benchmark dataset (MMLU, TruthfulQA, ARC, etc.) and any
provider (ChatGPT, Claude, Gemini).  The provider is auto-detected from the
data directory name (e.g. data-claude -> claude, data-gemini -> gemini,
data / data-chatgpt -> chatgpt).

Usage:
    from config import get_config
    cfg = get_config()
    cfg.data_dir   # Path to data/
    cfg.base_dir   # Parent of data_dir (benchmark root, for plots etc.)
    cfg.provider   # "chatgpt" | "claude" | "gemini"
    cfg.models     # dict  {label: (source, filename)}
    cfg.labels     # list of selected label strings
    cfg.csv_path(source, filename, timestamp)  # -> Path

    # From CLI:
    python fleiss_kappa.py --data-dir metabench-truthfulQA/data-chatgpt
    python plot_accuracy.py --data-dir metabench-mmlu/data-claude
    python plot_accuracy.py --data-dir metabench-mmlu/data-gemini
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Provider-specific model definitions
# ---------------------------------------------------------------------------
CHATGPT_MODELS = {
    "API: GPT 5.3 Chat (Instant)":  ("api",       "gpt-5.3-chat-latest.csv"),
    "API: GPT 5 Chat (Auto)":       ("api",       "gpt-5-chat-latest.csv"),
    "API: GPT 5.4 Reasoning High":  ("api",       "gpt-5.4-2026-03-05_reasoning_effort-high.csv"),
    "Interface: Instant":            ("interface",  "instant.csv"),
    "Interface: Thinking":           ("interface",  "thinking.csv"),
    "Interface: Auto":               ("interface",  "auto.csv"),
}

CLAUDE_MODELS = {
    "API: Claude Opus 4.6":          ("api",       "claude-opus-4-6.csv"),
    "API: Claude Haiku 4.5":         ("api",       "claude-haiku-4-5-20251001.csv"),
    "API: Claude Sonnet 4.6":        ("api",       "claude-sonnet-4-6.csv"),
    "Interface: Haiku":              ("interface",  "haiku.csv"),
    "Interface: Opus":               ("interface",  "opus.csv"),
    "Interface: Sonnet":             ("interface",  "sonnet.csv"),
}

GEMINI_MODELS = {
    "API: Gemini 3 Flash (High)":    ("api",       "gemini-3-flash-preview_thinking_level-high.csv"),
    "API: Gemini 3 Flash (Low)":     ("api",       "gemini-3-flash-preview_thinking_level-low.csv"),
    "Interface: Fast":               ("interface",  "fast.csv"),
    "Interface: Thinking":           ("interface",  "thinking.csv"),
}

PROVIDER_MODELS = {
    "chatgpt": CHATGPT_MODELS,
    "claude":  CLAUDE_MODELS,
    "gemini":  GEMINI_MODELS,
}

# ---------------------------------------------------------------------------
# Session maps (interface filename -> session directory)
# ---------------------------------------------------------------------------
CHATGPT_SESSION_MAP = {
    "instant.csv":  "session_00",
    "thinking.csv": "session_01",
    "auto.csv":     "session_02",
}

CLAUDE_SESSION_MAP = {
    "haiku.csv":  "session_00",
    "opus.csv":   "session_01",
    "sonnet.csv": "session_02",
}

GEMINI_SESSION_MAP = {
    "fast.csv":     "session_00",
    "thinking.csv": "session_01",
}

PROVIDER_SESSION_MAPS = {
    "chatgpt": CHATGPT_SESSION_MAP,
    "claude":  CLAUDE_SESSION_MAP,
    "gemini":  GEMINI_SESSION_MAP,
}

# ---------------------------------------------------------------------------
# CLI aliases  (--only / --exclude shorthand)
# ---------------------------------------------------------------------------
ALIASES = {
    # ChatGPT
    "chat-latest": "API: Chat-Latest",
    "chat":        "API: Chat-Latest",
    "reas-low":    "API: Reasoning Low",
    "low":         "API: Reasoning Low",
    "reas-high":   "API: GPT 5.4 Reasoning High",
    "high":        "API: GPT 5.4 Reasoning High",
    "instant":     "Interface: Instant",
    "thinking":    "Interface: Thinking",
    "auto":        "Interface: Auto",
    # Claude
    "opus":   "API: Claude Opus 4.6",
    "haiku":  "API: Claude Haiku 4.5",
    "sonnet": "API: Claude Sonnet 4.6",
    # Gemini
    "flash-high":  "API: Gemini 3 Flash (High)",
    "flash-low":   "API: Gemini 3 Flash (Low)",
    "flash":       "API: Gemini 3 Flash (High)",
    "fast":        "Interface: Fast",
}

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
CONDITION_COLORS = {
    # ChatGPT — paired by hue (API dark, Interface light)
    #   blue   = Instant pair
    #   purple = Thinking / Reasoning pair
    #   rose   = Auto pair
    "API: GPT 5.3 Chat (Instant)": "#2563EB",
    "Interface: Instant":          "#93C5FD",
    "API: GPT 5.4 Reasoning High": "#7C3AED",
    "Interface: Thinking":         "#C4B5FD",
    "API: GPT 5 Chat (Auto)":      "#DB2777",
    "Interface: Auto":             "#F9A8D4",
    # legacy ChatGPT names (same hue scheme)
    "API: Chat-Latest":            "#2563EB",
    "API: Reasoning Low":          "#10B981",
    "API: Reasoning High":         "#7C3AED",

    # Claude — paired by hue
    #   emerald = Haiku pair
    #   amber   = Opus pair
    #   indigo  = Sonnet pair
    "API: Claude Haiku 4.5":       "#059669",
    "Interface: Haiku":            "#6EE7B7",
    "API: Claude Opus 4.6":        "#D97706",
    "Interface: Opus":             "#FCD34D",
    "API: Claude Sonnet 4.6":      "#4F46E5",
    "Interface: Sonnet":           "#A5B4FC",

    # Gemini — paired by hue
    #   teal = Flash Low ↔ Fast pair
    #   red  = Flash High ↔ Thinking pair
    "API: Gemini 3 Flash (Low)":   "#0D9488",
    "Interface: Fast":             "#5EEAD4",
    "API: Gemini 3 Flash (High)":  "#DC2626",
    "Interface: Thinking":         "#F87171",
}

_DEFAULT_COLOR = "#6B7280"


def _resolve_name(name: str, models: dict) -> str:
    key = name.strip().lower()
    if key in ALIASES:
        resolved = ALIASES[key]
        if resolved in models:
            return resolved
    for label in models:
        if key == label.lower():
            return label
    for label in models:
        if key in label.lower():
            return label
    raise ValueError(
        f"Unknown condition '{name}'. Available: {', '.join(models.keys())}"
    )


def _detect_provider(data_dir: Path) -> str:
    """Infer provider from the data directory name."""
    name = data_dir.name.lower()
    if "claude" in name:
        return "claude"
    if "gemini" in name:
        return "gemini"
    return "chatgpt"


class Config:
    def __init__(
        self,
        data_dir: Path,
        models: dict,
        session_map: dict,
        provider: str = "chatgpt",
    ):
        self.data_dir = data_dir
        self.base_dir = data_dir.parent
        self.models = models
        self.session_map = session_map
        self.labels = list(models.keys())
        self.colors = {l: CONDITION_COLORS.get(l, _DEFAULT_COLOR) for l in self.labels}
        self.provider = provider

    def csv_path(self, source: str, filename: str, ts: str) -> Path:
        if source == "api":
            if self.provider == "chatgpt":
                direct = self.data_dir / "api" / ts / "session_03" / filename
                if direct.exists():
                    return direct
            api_ts = self.data_dir / "api" / ts
            if api_ts.exists():
                matches = sorted(api_ts.rglob(filename))
                if matches:
                    return matches[0]
                stem = Path(filename).stem.split("_")[0]
                prefix_matches = sorted(
                    p for p in api_ts.rglob("*.csv") if p.stem.startswith(stem)
                )
                if prefix_matches:
                    return prefix_matches[0]
            return api_ts / filename
        return self.data_dir / "interface" / ts / self.session_map[filename] / filename

    def get_timestamps(self) -> list[str]:
        for subdir_name in ("api", "interface", "parsed_json"):
            d = self.data_dir / subdir_name
            if d.exists():
                ts = sorted(p.name for p in d.iterdir() if p.is_dir())
                if ts:
                    return ts
        return []

    @property
    def api_labels(self) -> list[str]:
        return [l for l in self.labels if l.startswith("API")]

    @property
    def iface_labels(self) -> list[str]:
        return [l for l in self.labels if l.startswith("Interface")]

    def short_label(self, label: str) -> str:
        return label.split(": ", 1)[1] if ": " in label else label


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--data-dir", "-d",
        required=True,
        help="Path to the benchmark's data directory (e.g. metabench-mmlu/data-claude)",
    )
    p.add_argument(
        "--only",
        nargs="+",
        metavar="COND",
        default=None,
        help="Only include these conditions (e.g. --only reas-low instant)",
    )
    p.add_argument(
        "--exclude", "-x",
        nargs="+",
        metavar="COND",
        default=None,
        help="Exclude these conditions (e.g. --exclude chat-latest auto)",
    )
    return p


def get_config(extra_args: list[str] | None = None) -> Config:
    parser = _build_parser()
    args, _ = parser.parse_known_args(extra_args)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (Path.cwd() / data_dir).resolve()

    provider = _detect_provider(data_dir)
    # Purely provider-based: pick the model set for this provider.
    models = dict(PROVIDER_MODELS.get(provider, CHATGPT_MODELS))

    if args.only:
        selected = {_resolve_name(n, models) for n in args.only}
        models = {k: v for k, v in models.items() if k in selected}
    elif args.exclude:
        excluded = {_resolve_name(n, models) for n in args.exclude}
        models = {k: v for k, v in models.items() if k not in excluded}

    if not models:
        raise ValueError("No conditions selected after filtering.")

    base_session_map = PROVIDER_SESSION_MAPS.get(provider, CHATGPT_SESSION_MAP)
    session_map = {fn: s for fn, s in base_session_map.items()
                   if any(fn == v[1] for v in models.values())}

    return Config(data_dir, models, session_map, provider=provider)
