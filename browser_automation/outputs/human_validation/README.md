# Human validation of the extraction pipeline

Built by `automated-scraper/compile/build_human_validation.py` from
`extractor_visualizer_worst.html` (the artifact reviewed, 137 conditions x worst run),
the exported extractor votes (`outputs/extractor_votes_raw/`, consolidated by
`compile/consolidate_extractor_votes.py`), and the 100-item AA-Omniscience grader sample.

* `reviewed_items.csv` — 1327 items the annotator reviewed (every wrong / non-extractable
  item in the worst run per condition, plus the 100-item AA-Omni sample) with the annotator's verdict.
* `disagreements.csv` — 266 rows: the 62 disagreements within the reviewed set, plus
  204 "disagree" flags from earlier review rounds on runs/items that are no longer in the
  final visualizer (run superseded as "worst", or item now scored correct after extractor fixes).
* `summary_by_benchmark.csv` — reviewed / disagree per benchmark, next to the numbers in the paper
  appendix (1,280 reviewed, 17 disagree). Differences likely come from the paper table being computed
  from an earlier build of the visualizer than the one on disk, and from the votes here being the earlier review rounds (the final 17-disagreement pass was not exported).

Verdict semantics: `disagree` = the response contained an extractable answer but the pipeline
returned empty or a different answer than the model stated. A wrong model answer that was
extracted correctly is `agree`.
