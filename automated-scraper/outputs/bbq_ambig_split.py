"""BBQ accuracy split by ambig vs disambig.

Joins the paper's per_query.csv (5 runs/item/side) with the ambig classification
in bbq-subset-200.csv and reports accuracy for each model, side (API/Interface),
and question type. Uses the same "ignore" failure-mode as
paper/tables/compute_test_retest.py: only runs with ext=1 contribute.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
SUBSET = REPO / "benchmark_creation" / "results" / "bbq-subset-200.csv"
PER_Q  = REPO / "experiments" / "plots" / "per_query.csv"
OUT    = THIS.parent / "bbq_ambig_split.csv"

MODELS = [
    ("GPT 5.3 Instant",   "chatgpt", "gpt-5-3-instant"),
    ("GPT 5.4 Thinking",  "chatgpt", "gpt-5-4-thinking"),
    ("Claude Haiku",      "claude",  "haiku"),
    ("Claude Opus",       "claude",  "opus"),
    ("Claude Sonnet",     "claude",  "sonnet"),
    ("Gemini 3 Thinking", "gemini",  "thinking"),
    ("Gemini 3 Fast",     "gemini",  "fast"),
]

UNK = {"can't be determined","can't answer",'cannot answer','cannot be determined',
       'not answerable','not enough info','not enough information','not known',
       'undetermined','unknown'}


def classify(query, gold):
    m = re.findall(r'^([ABC])\.\s*(.+?)\s*$', query, re.MULTILINE)
    opts = dict(m)
    unk = next((k for k,v in opts.items() if v.strip().lower() in UNK), None)
    if unk is None:
        return None
    return 'ambig' if gold == unk else 'disambig'


def main():
    # 1. ambig label per qid
    label = {}
    with SUBSET.open() as f:
        for r in csv.DictReader(f):
            c = classify(r['query'], r['answer'])
            if c is not None:
                label[int(r['id'])] = c

    # 2. per_query rows for BBQ → group by (model, qid) -> list of runs
    runs = defaultdict(list)  # (mlabel, qid) -> [(ext_api, corr_api, ext_ifc, corr_ifc), ...]
    mkey = {(p, m): lbl for lbl, p, m in MODELS}
    with PER_Q.open() as f:
        for r in csv.DictReader(f):
            if r['benchmark'] != 'bbq':
                continue
            key = mkey.get((r['provider'], r['model']))
            if key is None:
                continue
            try:
                qid = int(r['qid'])
            except ValueError:
                continue
            if qid not in label:
                continue
            runs[(key, qid)].append((
                int(r['api_extracted']), int(r['api_correct']),
                int(r['ifc_extracted']), int(r['ifc_correct']),
            ))

    # 3. aggregate. "ignore" failure mode: count only extracted runs.
    out_rows = []
    print(f"{'Model':<22} {'Type':<9} {'N_items':>7} "
          f"{'API_n':>6} {'API%':>6}  {'IFC_n':>6} {'IFC%':>6}")
    print('-' * 70)
    for lbl, prov, mslug in MODELS:
        for kind in ('ambig', 'disambig', 'all'):
            api_ext = api_cor = ifc_ext = ifc_cor = 0
            n_items = 0
            for (mlabel, qid), rs in runs.items():
                if mlabel != lbl:
                    continue
                if kind != 'all' and label.get(qid) != kind:
                    continue
                n_items += 1
                # cap at first 5 runs (paper convention)
                for ea, ca, ei, ci in rs[:5]:
                    if ea:
                        api_ext += 1
                        api_cor += ca
                    if ei:
                        ifc_ext += 1
                        ifc_cor += ci
            api_pct = 100*api_cor/api_ext if api_ext else float('nan')
            ifc_pct = 100*ifc_cor/ifc_ext if ifc_ext else float('nan')
            print(f"{lbl:<22} {kind:<9} {n_items:>7} "
                  f"{api_ext:>6} {api_pct:>6.1f}  {ifc_ext:>6} {ifc_pct:>6.1f}")
            out_rows.append(dict(
                model=lbl, type=kind, n_items=n_items,
                api_n_extracted=api_ext, api_pct=f"{api_pct:.2f}" if api_pct==api_pct else "",
                ifc_n_extracted=ifc_ext, ifc_pct=f"{ifc_pct:.2f}" if ifc_pct==ifc_pct else "",
            ))
        print()

    with OUT.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
