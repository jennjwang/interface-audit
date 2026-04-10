"""Compare discordant (flipped) items between API and Interface conditions.

For matched model pairs (e.g., API Opus vs API Haiku, Interface Opus vs Interface Haiku),
identifies which question IDs flip and measures overlap. This helps disentangle whether
accuracy differences are driven by a few "tricky" benchmark items vs. systematic model behavior.

Usage:
    python compare_flipped_items.py --data-dir metabench-mmlu/data-claude
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import get_config
from plot_mcnemar_rankings import load_question_correctness, build_paired_arrays


def get_discordant_items(
    qids: list[int],
    mat: np.ndarray,
    i: int,
    j: int,
) -> tuple[set[int], set[int]]:
    """Return (items where i correct & j wrong, items where j correct & i wrong)."""
    a = mat[:, i].astype(bool)
    b = mat[:, j].astype(bool)
    i_wins = {qids[k] for k in range(len(qids)) if a[k] and not b[k]}
    j_wins = {qids[k] for k in range(len(qids)) if b[k] and not a[k]}
    return i_wins, j_wins


def jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def match_api_iface_pairs(api_labels: list[str], iface_labels: list[str], provider: str):
    """Match API and Interface labels that correspond to the same underlying model."""
    pairs = []
    if provider == "claude":
        match_map = {
            "API: Claude Haiku 4.5": "Interface: Haiku",
            "API: Claude Opus 4.6": "Interface: Opus",
            "API: Claude Sonnet 4.6": "Interface: Sonnet",
        }
    elif provider == "chatgpt":
        match_map = {
            "API: GPT 5.3 Chat (Instant)": "Interface: Instant",
            "API: GPT 5.4 Reasoning High": "Interface: Thinking",
            "API: GPT 5 Chat (Auto)": "Interface: Auto",
        }
    elif provider == "gemini":
        match_map = {
            "API: Gemini 3 Flash (Low)": "Interface: Fast",
            "API: Gemini 3 Flash (High)": "Interface: Thinking",
        }
    else:
        return []

    for api_l, iface_l in match_map.items():
        if api_l in api_labels and iface_l in iface_labels:
            pairs.append((api_l, iface_l))
    return pairs


def main():
    cfg = get_config()
    data = load_question_correctness(cfg)
    all_labels = [l for l in cfg.labels if data.get(l)]

    # Build paired arrays using ALL models so qids are consistent
    qids, mat = build_paired_arrays(data, all_labels)
    n_q = len(qids)
    print(f"Total shared questions: {n_q}")
    print(f"Models: {all_labels}\n")

    label_idx = {l: i for i, l in enumerate(all_labels)}

    api_labels = [l for l in all_labels if l.startswith("API")]
    iface_labels = [l for l in all_labels if l.startswith("Interface")]

    # ── 1. Same model, API vs Interface: which items flip? ──
    print("=" * 70)
    print("SAME MODEL, API vs INTERFACE — which items differ?")
    print("=" * 70)
    matched = match_api_iface_pairs(api_labels, iface_labels, cfg.provider)
    api_iface_flips = {}
    for api_l, iface_l in matched:
        ai, ii = label_idx[api_l], label_idx[iface_l]
        api_wins, iface_wins = get_discordant_items(qids, mat, ai, ii)
        all_flipped = api_wins | iface_wins
        api_iface_flips[(api_l, iface_l)] = all_flipped

        api_acc = mat[:, ai].mean()
        iface_acc = mat[:, ii].mean()
        print(f"\n{cfg.short_label(api_l)} (API {api_acc:.1%}) vs {cfg.short_label(iface_l)} (Iface {iface_acc:.1%}):")
        print(f"  Total discordant: {len(all_flipped)} / {n_q} ({len(all_flipped)/n_q:.1%})")
        print(f"    API correct, Iface wrong: {len(api_wins)}")
        print(f"    Iface correct, API wrong: {len(iface_wins)}")

    # Cross-pair overlap of API-vs-Interface flips
    if len(api_iface_flips) > 1:
        print(f"\n{'─' * 50}")
        print("Overlap of API↔Interface flipped items across models:")
        keys = list(api_iface_flips.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                s1 = api_iface_flips[keys[i]]
                s2 = api_iface_flips[keys[j]]
                overlap = s1 & s2
                j_sim = jaccard(s1, s2)
                name1 = cfg.short_label(keys[i][0])
                name2 = cfg.short_label(keys[j][0])
                print(f"  {name1} vs {name2}: {len(overlap)} shared flipped items "
                      f"(Jaccard={j_sim:.3f}, |A|={len(s1)}, |B|={len(s2)})")

    # ── 2. Pairwise within API vs pairwise within Interface ──
    print(f"\n{'=' * 70}")
    print("PAIRWISE COMPARISONS: same model pair, API vs Interface")
    print("=" * 70)

    from itertools import combinations

    # Build matching between API pairs and Interface pairs
    api_to_short = {}
    iface_to_short = {}
    for api_l, iface_l in matched:
        short = cfg.short_label(api_l).split()[0]  # e.g., "Claude" or "GPT"
        # Use a canonical short name
        api_to_short[api_l] = cfg.short_label(api_l)
        iface_to_short[iface_l] = cfg.short_label(iface_l)

    # For each pair of models, compare discordant items in API vs Interface
    for (api_a, iface_a), (api_b, iface_b) in combinations(matched, 2):
        ai_a, ai_b = label_idx[api_a], label_idx[api_b]
        ii_a, ii_b = label_idx[iface_a], label_idx[iface_b]

        # API pair discordant items
        api_a_wins, api_b_wins = get_discordant_items(qids, mat, ai_a, ai_b)
        api_disc = api_a_wins | api_b_wins

        # Interface pair discordant items
        iface_a_wins, iface_b_wins = get_discordant_items(qids, mat, ii_a, ii_b)
        iface_disc = iface_a_wins | iface_b_wins

        overlap = api_disc & iface_disc
        j_sim = jaccard(api_disc, iface_disc)

        short_a = cfg.short_label(api_a)
        short_b = cfg.short_label(api_b)
        print(f"\n{short_a} vs {short_b}:")
        print(f"  API discordant:   {len(api_disc):>4d}  (A>{len(api_a_wins)}, B>{len(api_b_wins)})")
        print(f"  Iface discordant: {len(iface_disc):>4d}  (A>{len(iface_a_wins)}, B>{len(iface_b_wins)})")
        print(f"  Overlap:          {len(overlap):>4d}  (Jaccard = {j_sim:.3f})")
        print(f"  API-only flips:   {len(api_disc - iface_disc):>4d}")
        print(f"  Iface-only flips: {len(iface_disc - api_disc):>4d}")

        # Are the "frequent flippers" concentrated?
        all_disc = api_disc | iface_disc
        if all_disc:
            both = api_disc & iface_disc
            pct_both = len(both) / len(all_disc)
            print(f"  Items flipping in BOTH conditions: {len(both)}/{len(all_disc)} ({pct_both:.1%})")

    # ── 3. Concentration analysis: are a few items responsible for most flips? ──
    print(f"\n{'=' * 70}")
    print("FLIP CONCENTRATION: how many items account for discordance?")
    print("=" * 70)

    # Count how many pairwise comparisons each item is discordant in
    from itertools import combinations as combs
    flip_counts = defaultdict(int)
    total_pairs = 0
    for i, j in combs(range(len(all_labels)), 2):
        total_pairs += 1
        a_wins, b_wins = get_discordant_items(qids, mat, i, j)
        for qid in a_wins | b_wins:
            flip_counts[qid] += 1

    if flip_counts:
        counts = np.array(list(flip_counts.values()))
        all_qid_items = list(flip_counts.items())
        all_qid_items.sort(key=lambda x: -x[1])

        print(f"\n  Total model pairs: {total_pairs}")
        print(f"  Items that flip in at least 1 pair: {len(flip_counts)} / {n_q} ({len(flip_counts)/n_q:.1%})")
        print(f"  Items that flip in >50% of pairs:   ", end="")
        high_flip = [q for q, c in flip_counts.items() if c > total_pairs / 2]
        print(f"{len(high_flip)}")

        # Top 20 most frequently flipping items
        print(f"\n  Top 20 most discordant items (qid: # pairs where it flips / {total_pairs}):")
        for qid, cnt in all_qid_items[:20]:
            pct = cnt / total_pairs
            bar = "█" * int(pct * 30)
            print(f"    qid {qid:>5d}: {cnt:>3d} ({pct:>5.1%}) {bar}")

        # What % of total discordance comes from top N items?
        total_disc_events = sum(counts)
        cumsum = np.cumsum(sorted(counts, reverse=True))
        for pct_target in [0.25, 0.50, 0.75]:
            n_items = int(np.searchsorted(cumsum, pct_target * total_disc_events) + 1)
            print(f"  Top {n_items} items ({n_items/len(flip_counts):.0%} of flipping items) "
                  f"account for {pct_target:.0%} of all flip events")


if __name__ == "__main__":
    main()
