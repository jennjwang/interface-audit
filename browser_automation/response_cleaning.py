"""Shared normalisation for interface-scraped responses.

Applied at parse time so that every consumer of parsed_json/ sees the same text,
rather than each analysis script re-implementing the fix (previously the label
was stripped independently in four places, and the duplication not at all).

This module is intentionally free of browser dependencies so it can be imported
by the audit scrapers, the offline parsers, and the analysis scripts alike.
"""

from __future__ import annotations

import re

# Provider accessibility headings, e.g. "Claude responded: ".
_LABEL_RE = re.compile(
    r"^\s*(?:Claude|Assistant|ChatGPT|Gemini)\s+(?:responded|said|replied)\s*:\s*",
    re.IGNORECASE,
)


def strip_provider_label(text: str | None) -> str | None:
    """Remove a provider accessibility heading and the duplicate it introduces.

    Claude's response container holds a screen-reader ``<h2 class="sr-only">``
    that restates the response, truncated to 160 characters. Because extraction
    targets the enclosing container, ``get_text()`` captures that heading along
    with the response body: every Claude record is prefixed with
    "Claude responded: ...", and short answers appear twice.

    The heading is a truncation of the body and never carries content the body
    lacks -- verified across all 6,623 Claude interface records -- so it is
    dropped in full. Where a record somehow has no body, the heading is kept as
    the response rather than emptying the record.

    Text without such a heading is returned unchanged, so this is safe to apply
    to every provider.
    """
    if not text:
        return text
    match = _LABEL_RE.match(text)
    if not match:
        return text
    head, _, body = text[match.end():].partition("\n")
    return (body if body.strip() else head).strip()


def normalise_response(text: str | None) -> str | None:
    """Full scraper-level normalisation applied before scoring."""
    return strip_provider_label(text)
