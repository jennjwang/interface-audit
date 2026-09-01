"""Shared loader for experiment YAML configs.

The consolidated configs in ``yamls/`` hold several YAML *documents* separated by
``---`` (e.g. ``interface_scraping.yaml`` carries one document per
provider+model).  The runtime contract, however, is **one config per session /
run**: ``--configs`` takes a comma-separated list where each entry supplies a
single session, and ``batch_runner`` / ``api_runner`` each expect one
``api_models`` + ``experiments`` pair.

A multi-document file therefore needs an explicit selector appended to the path:

    yamls/interface_scraping.yaml#gpt-5-4-thinking   # by experiment name
    yamls/interface_scraping.yaml#1                  # by document index

Single-document files load exactly as before, with or without a selector, so
existing invocations are unaffected.

Selecting is deliberately explicit rather than automatic: the documents in one
file may span several providers, and silently running document 0 (or merging
all of them) would point a vendor's scraper at another vendor's config.
"""

from __future__ import annotations

from pathlib import Path

import yaml

SELECTOR_SEP = "#"


def split_selector(ref) -> tuple[str, str | None]:
    """Split ``path#selector`` into ``(path, selector)``; selector may be None."""
    text = str(ref)
    if SELECTOR_SEP not in text:
        return text, None
    path, _, selector = text.rpartition(SELECTOR_SEP)
    selector = selector.strip()
    return (path, selector or None) if path else (text, None)


def load_documents(path) -> list[dict]:
    """Return every non-empty YAML document in ``path``."""
    text = Path(path).read_text(encoding="utf-8")
    return [doc for doc in yaml.safe_load_all(text) if doc]


def document_labels(doc: dict) -> list[str]:
    """Selector names a document answers to: its experiment and api-model names."""
    labels: list[str] = []
    if not isinstance(doc, dict):
        return labels
    for exp in doc.get("experiments") or []:
        if isinstance(exp, dict) and exp.get("name"):
            labels.append(str(exp["name"]))
    for model in doc.get("api_models") or []:
        if isinstance(model, dict) and model.get("model"):
            labels.append(str(model["model"]))
    return labels


def describe_documents(docs: list[dict]) -> str:
    """Human-readable index of the documents in a file, for error messages."""
    lines = []
    for i, doc in enumerate(docs):
        labels = document_labels(doc)
        lines.append(f"    #{i}" + (f"  ({', '.join(dict.fromkeys(labels))})" if labels else ""))
    return "\n".join(lines)


def load_config(ref) -> dict:
    """Load one config document from ``ref`` (``path`` or ``path#selector``).

    Raises ValueError when a multi-document file is referenced without a
    selector, or when the selector does not identify exactly one document.
    """
    path, selector = split_selector(ref)
    docs = load_documents(path)

    if not docs:
        return {}
    if len(docs) == 1 and selector is None:
        return docs[0]

    if selector is None:
        raise ValueError(
            f"{path} contains {len(docs)} YAML documents; append a selector to "
            f"choose one, e.g. '{path}{SELECTOR_SEP}0'. Available documents:\n"
            f"{describe_documents(docs)}"
        )

    if selector.isdigit():
        index = int(selector)
        if index >= len(docs):
            raise ValueError(
                f"{path}{SELECTOR_SEP}{selector}: document index out of range "
                f"({len(docs)} documents). Available documents:\n{describe_documents(docs)}"
            )
        return docs[index]

    matches = [doc for doc in docs if selector in document_labels(doc)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"{path}{SELECTOR_SEP}{selector}: no document defines that experiment "
            f"or api model. Available documents:\n{describe_documents(docs)}"
        )
    raise ValueError(
        f"{path}{SELECTOR_SEP}{selector}: matches {len(matches)} documents; "
        f"use a document index instead. Available documents:\n{describe_documents(docs)}"
    )
