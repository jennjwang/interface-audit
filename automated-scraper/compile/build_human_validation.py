"""Save the human validation of the extraction pipeline (paper App. "Human Validation").

Inputs
  <repo>/extractor_visualizer_worst.html         the artifact that was reviewed:
                                                 137 conditions x worst run, embedded as
                                                 `const ALL_DATA = {...}`
  outputs/extractor_votes_consolidated.json      merged "disagree" votes
                                                 (from compile/consolidate_extractor_votes.py)
  outputs/extractor_votes_raw/aa_omni_annotations_sample100_*.csv
                                                 100-item AA-Omniscience grader sample

Outputs (outputs/human_validation/)
  reviewed_items.csv    every item the annotator reviewed: all wrong / non-extractable items
                        in the worst run per condition (+ the 100 AA-Omni sample), with
                        gold, extracted answer, full response, and the annotator's verdict
  disagreements.csv     the subset where the annotator disagreed with the pipeline,
                        including the 54 flags on runs/items no longer in the final visualizer
  summary_by_benchmark.csv
  README.md

Usage:
  python automated-scraper/compile/build_human_validation.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parent
HTML = REPO / "extractor_visualizer_worst.html"
VOTES = BASE / "outputs" / "extractor_votes_consolidated.json"
AA_SAMPLE = sorted((BASE / "outputs" / "extractor_votes_raw").glob("aa_omni_annotations*.csv"))
OUT = BASE / "outputs" / "human_validation"

AA_COND = {
    "Instant API": ("ChatGPT Instant", "API"),
    "Instant Interface": ("ChatGPT Instant", "Interface"),
    "Thinking API": ("ChatGPT Thinking", "API"),
    "Thinking Interface": ("ChatGPT Thinking", "Interface"),
}
PAPER = {  # App. table, for the README comparison
    "arc": (48, 0), "gsm8k": (88, 0), "hellaswag": (111, 2), "mmlu": (106, 1),
    "truthfulqa": (253, 10), "winogrande": (199, 0), "bbq": (265, 1),
    "aa-omniscience": (100, 3), "elephant-og": (70, 0), "elephant-flip": (40, 0),
}


def load_all_data() -> dict[str, list[dict]]:
    h = HTML.read_text(encoding="utf-8")
    marker = "const ALL_DATA = "
    data, _ = json.JSONDecoder().raw_decode(h[h.index(marker) + len(marker):])
    return data


def split_key(key: str) -> tuple[str, str, str, str]:
    bench, rest = key.split(" / ", 1)
    model, rest = rest.split(" — ", 1)
    surface, run = rest.split(" :: ", 1)
    return bench, model, surface, run


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = load_all_data()
    votes = json.load(open(VOTES, encoding="utf-8"))

    fields = ["benchmark", "model", "surface", "run", "item_id", "gold", "extracted",
              "pipeline_correct", "no_extract", "annotator_verdict", "in_final_visualizer",
              "annotation_type", "response"]
    reviewed: list[dict] = []
    seen_votes: set[tuple[str, str]] = set()

    # 1. worst-run review set (extractor validation)
    for key, rows in data.items():
        bench, model, surface, run = split_key(key)
        if bench == "aa-omniscience":
            continue  # reviewed via the 100-item sample instead
        flagged = votes.get(key, {})
        for r in rows:
            if r["correct"] == "True":
                continue
            item = str(r["id"])
            seen_votes.add((key, item))
            reviewed.append({
                "benchmark": bench, "model": model, "surface": surface, "run": run,
                "item_id": item, "gold": r["gold"], "extracted": r["answer"],
                "pipeline_correct": False, "no_extract": not r["answer"],
                "annotator_verdict": "disagree" if item in flagged else "agree",
                "in_final_visualizer": True, "annotation_type": "extractor", "response": r["response"],
            })

    # 2. AA-Omniscience grader sample
    for p in AA_SAMPLE:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            model, surface = AA_COND[r["condition"].strip()]
            reviewed.append({
                "benchmark": "aa-omniscience", "model": model, "surface": surface, "run": "sample100",
                "item_id": r["id"], "gold": r["gold"], "extracted": r["my_label"],
                "pipeline_correct": r["auto_correct"].strip().lower() == "true", "no_extract": False,
                "annotator_verdict": "agree" if r["agree"].strip().lower() == "true" else "disagree",
                "in_final_visualizer": False, "annotation_type": "grader_sample", "response": r["response"],
            })

    # 3. disagree votes not covered above (superseded runs, or items now scored correct)
    orphan = []
    for key, items in votes.items():
        bench, model, surface, run = split_key(key)
        rows_by_id = {str(r["id"]): r for r in data.get(key, [])}
        for item in items:
            if (key, item) in seen_votes:
                continue
            r = rows_by_id.get(item)
            orphan.append({
                "benchmark": bench, "model": model, "surface": surface, "run": run,
                "item_id": item, "gold": r["gold"] if r else "", "extracted": r["answer"] if r else "",
                "pipeline_correct": (r["correct"] == "True") if r else "",
                "no_extract": (not r["answer"]) if r else "",
                "annotator_verdict": "disagree", "in_final_visualizer": key in data,
                "annotation_type": "extractor", "response": r["response"] if r else "",
            })

    def write(path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    write(OUT / "reviewed_items.csv", reviewed)
    disagreements = [r for r in reviewed if r["annotator_verdict"] == "disagree"] + orphan
    write(OUT / "disagreements.csv", disagreements)

    # summary
    rev = Counter(r["benchmark"] for r in reviewed)
    dis = Counter(r["benchmark"] for r in reviewed if r["annotator_verdict"] == "disagree")
    orph = Counter(r["benchmark"] for r in orphan)
    with open(OUT / "summary_by_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "reviewed", "disagree", "agreement_pct",
                    "paper_reviewed", "paper_disagree", "superseded_disagree_flags"])
        for b, (pr, pd) in PAPER.items():
            w.writerow([b, rev[b], dis[b], f"{100*(1-dis[b]/rev[b]):.1f}" if rev[b] else "", pr, pd, orph[b]])
        tr, td = sum(rev.values()), sum(dis.values())
        w.writerow(["TOTAL", tr, td, f"{100*(1-td/tr):.1f}", sum(p[0] for p in PAPER.values()),
                    sum(p[1] for p in PAPER.values()), sum(orph.values())])

    (OUT / "README.md").write_text(f"""# Human validation of the extraction pipeline

Built by `automated-scraper/compile/build_human_validation.py` from
`extractor_visualizer_worst.html` (the artifact reviewed, 137 conditions x worst run),
the exported extractor votes (`outputs/extractor_votes_raw/`, consolidated by
`compile/consolidate_extractor_votes.py`), and the 100-item AA-Omniscience grader sample.

* `reviewed_items.csv` — {len(reviewed)} items the annotator reviewed (every wrong / non-extractable
  item in the worst run per condition, plus the 100-item AA-Omni sample) with the annotator's verdict.
* `disagreements.csv` — {len(disagreements)} rows: the {td} disagreements within the reviewed set, plus
  {len(orphan)} "disagree" flags from earlier review rounds on runs/items that are no longer in the
  final visualizer (run superseded as "worst", or item now scored correct after extractor fixes).
* `summary_by_benchmark.csv` — reviewed / disagree per benchmark, next to the numbers in the paper
  appendix (1,280 reviewed, 17 disagree). Differences likely come from the paper table being computed
  from an earlier build of the visualizer than the one on disk, and from the votes here being the earlier review rounds (the final 17-disagreement pass was not exported).

Verdict semantics: `disagree` = the response contained an extractable answer but the pipeline
returned empty or a different answer than the model stated. A wrong model answer that was
extracted correctly is `agree`.
""", encoding="utf-8")

    print(f"reviewed: {len(reviewed)}  disagree-in-set: {td}  orphan flags: {len(orphan)}  -> {OUT.relative_to(BASE)}/")
    for b, (pr, pd) in PAPER.items():
        print(f"  {b:15s} reviewed {rev[b]:4d} (paper {pr:4d})  disagree {dis[b]:3d} (paper {pd:2d})  superseded flags {orph[b]:3d}")


if __name__ == "__main__":
    main()
