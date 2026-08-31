"""Consolidate the manual extractor-validation votes into one table.

Sources (raw exports from the extractor/annotation visualizers, copied from
~/Downloads into outputs/extractor_votes_raw/):

  * five ``*extractor_votes*.json`` files exported by the "Export votes" button
    of build_extractor_visualizer*.py.  Format::

        {"<benchmark> / <Model> — <Surface> :: <run>": {"<item_id>": "disagree", ...}}

    (the HellaSwag file omits the ``<benchmark> / `` prefix; its runs end in
    ``-hellaswag``).  Only non-default votes are exported, so every row is a
    "disagree" = the annotator judged that the pipeline's extracted answer did
    NOT match what the model actually said.
  * ``aa_omni_annotations_sample100_*.csv`` — the 100-item random sample of
    AA-Omniscience LLM-grader verdicts (columns: condition, auto_correct,
    my_label, agree).  Blank ``agree`` == disagree.

Writes:
  outputs/extractor_votes_consolidated.csv   (long format, one row per vote)
  outputs/extractor_votes_consolidated.json  (merged, same shape as the exports)
  outputs/extractor_votes_summary.csv        (disagreements per benchmark/run)

Usage:
  python automated-scraper/compile/consolidate_extractor_votes.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "outputs" / "extractor_votes_raw"
OUT_CSV = BASE / "outputs" / "extractor_votes_consolidated.csv"
OUT_JSON = BASE / "outputs" / "extractor_votes_consolidated.json"
OUT_SUMMARY = BASE / "outputs" / "extractor_votes_summary.csv"

KEY_RE = re.compile(r"^(?:(?P<bench>[^/]+?) / )?(?P<model>.+?) — (?P<surface>API|Interface) :: (?P<run>.+)$")

# condition strings used in the AA-Omni sample CSV -> (model, surface)
AA_COND = {
    "Instant API": ("ChatGPT Instant", "API"),
    "Instant Interface": ("ChatGPT Instant", "Interface"),
    "Thinking API": ("ChatGPT Thinking", "API"),
    "Thinking Interface": ("ChatGPT Thinking", "Interface"),
}


def parse_key(key: str) -> tuple[str, str, str, str]:
    m = KEY_RE.match(key)
    if not m:
        raise ValueError(f"unparseable key: {key!r}")
    bench = m.group("bench")
    run = m.group("run")
    if bench is None:
        if "hellaswag" not in run.lower():
            raise ValueError(f"no benchmark in key and run is not hellaswag: {key!r}")
        bench = "hellaswag"
    return bench.strip(), m.group("model").strip(), m.group("surface"), run


def main() -> None:
    rows: list[dict] = []
    merged: dict[str, dict[str, str]] = OrderedDict()
    seen: dict[tuple, str] = {}  # (bench, model, surface, run, item) -> first source file

    for p in sorted(RAW.glob("*.json")):
        data = json.load(open(p, encoding="utf-8"))
        for key, votes in data.items():
            bench, model, surface, run = parse_key(key)
            norm_key = f"{bench} / {model} — {surface} :: {run}"
            for item, vote in votes.items():
                k = (bench, model, surface, run, str(item))
                if k in seen:
                    prev = merged[norm_key][str(item)]
                    if prev != vote:
                        raise RuntimeError(f"conflicting vote for {k}: {prev} ({seen[k]}) vs {vote} ({p.name})")
                    continue  # duplicate across cumulative exports
                seen[k] = p.name
                merged.setdefault(norm_key, {})[str(item)] = vote
                rows.append({
                    "benchmark": bench, "model": model, "surface": surface, "run": run,
                    "item_id": str(item), "vote": vote, "annotation_type": "extractor",
                    "source_file": p.name,
                })

    # AA-Omniscience 100-item grader sample (agree + disagree both recorded)
    for p in sorted(RAW.glob("aa_omni_annotations*.csv")):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            model, surface = AA_COND[r["condition"].strip()]
            vote = "agree" if r["agree"].strip().lower() == "true" else "disagree"
            rows.append({
                "benchmark": "aa-omniscience", "model": model, "surface": surface,
                "run": "sample100", "item_id": r["id"], "vote": vote,
                "annotation_type": "grader_sample", "source_file": p.name,
            })

    fields = ["benchmark", "model", "surface", "run", "item_id", "vote", "annotation_type", "source_file"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    json.dump(merged, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # summary
    per_run = Counter(); per_bench = Counter(); reviewed_bench = Counter()
    for r in rows:
        reviewed_bench[(r["benchmark"], r["annotation_type"])] += 1
        if r["vote"] == "disagree":
            per_run[(r["benchmark"], r["model"], r["surface"], r["run"])] += 1
            per_bench[(r["benchmark"], r["annotation_type"])] += 1
    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "model", "surface", "run", "n_disagree"])
        for (b, m, s, rn), n in sorted(per_run.items()):
            w.writerow([b, m, s, rn, n])

    print(f"{len(rows)} rows -> {OUT_CSV.relative_to(BASE)}")
    print(f"{len(merged)} runs -> {OUT_JSON.relative_to(BASE)}")
    print("\nDisagreements by benchmark:")
    for (b, t), n in sorted(per_bench.items()):
        extra = f" (of {reviewed_bench[(b, t)]} reviewed)" if t == "grader_sample" else ""
        print(f"  {b:15s} {t:14s} {n:4d}{extra}")
    print(f"  {'TOTAL':15s} {'':14s} {sum(per_bench.values()):4d}")


if __name__ == "__main__":
    main()
