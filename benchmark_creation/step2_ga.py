"""
EssenceBench Step 2: faithful port of train_multi_round.py.

Multi-round GA + EBM pipeline:
  Round 0: GA on full Step-1 candidate pool, fitness via TorchGAM surrogate.
           Collect top-200 individuals, fit EBM per individual to get feature
           importances, split candidates into High / Low / Rand groups.
  Round 1+: GA on each group independently, keep the best-performing group
             as the candidate pool for the next round.

Small-N adjustment: stratified_split falls back to a single held-out model
when the population is too small for 10% validation.

Usage:
    python step2_ga.py \
        --scored-dir  experiments/aa-omniscience/outputs/2026-04-23_19-14-53 \
        --subset-csv  benchmark_creation/results/aa-omniscience-subset.csv \
        --queries     benchmark_creation/queries/aa-omniscience.csv \
        --output      benchmark_creation/results/aa-omniscience-subset-100.csv \
        --k           100
"""
import argparse
import json
import logging
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import pearsonr
from interpret.glassbox import ExplainableBoostingRegressor

ESSENCEBENCH = Path(__file__).resolve().parent / "EssenceBench" / "src" / "subset_selection"
sys.path.insert(0, str(ESSENCEBENCH))
from torch_gam import TorchGAM1D  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("ga_ebm")


# ── data ──────────────────────────────────────────────────────────────────────

def load_scored_matrix(scored_dir: Path):
    frames = {}
    for f in sorted(scored_dir.glob("*.csv")):
        df = pd.read_csv(f)
        if "id" not in df.columns or "correct" not in df.columns:
            continue
        s = pd.to_numeric(df.set_index("id")["correct"], errors="coerce")
        frames[f.stem] = s
    if not frames:
        raise FileNotFoundError(f"No scored CSVs in {scored_dir}")
    all_ids = sorted(set.intersection(*[set(s.index) for s in frames.values()]))
    nan_ids = set()
    for s in frames.values():
        nan_ids.update(s.reindex(all_ids)[s.reindex(all_ids).isna()].index)
    question_ids = [q for q in all_ids if q not in nan_ids]
    model_names = list(frames.keys())
    matrix = np.array(
        [[int(frames[m].loc[qid]) for qid in question_ids] for m in model_names],
        dtype=np.int8,
    ).T  # (n_questions, n_models)
    return model_names, question_ids, matrix


def stratified_split(y: np.ndarray, val_ratio: float = 0.1, seed: int = 42):
    """Split model indices into train/val. Falls back to 1 val model for small N."""
    n = len(y)
    n_val = max(1, int(round(n * val_ratio)))
    rng = np.random.default_rng(seed)
    # sort by score and interleave to approximate stratification
    order = np.argsort(y)
    val = order[::max(1, n // n_val)][:n_val]
    tr  = np.array([i for i in range(n) if i not in set(val)])
    return tr, val


# ── GA operators (matching EssenceBench exactly) ──────────────────────────────

def crossover(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    mask = torch.rand_like(p1.float()) < 0.5
    return (p1 & mask) | (p2 & ~mask)


def mutate(child: torch.Tensor, p: float) -> torch.Tensor:
    return child ^ (torch.rand_like(child.float()) < p)


def repair(child: torch.Tensor, k: int, rng: np.random.Generator) -> torch.Tensor:
    """Adjust child to have exactly k True entries."""
    diff = int(child.sum()) - k
    if diff > 0:
        ones = child.nonzero(as_tuple=True)[0].numpy()
        child[rng.choice(ones, diff, replace=False)] = False
    elif diff < 0:
        zeros = (~child).nonzero(as_tuple=True)[0].numpy()
        child[rng.choice(zeros, -diff, replace=False)] = True
    return child


def tournament(pop: list, fit: torch.Tensor, k: int,
               rng: np.random.Generator) -> torch.Tensor:
    idx = rng.choice(len(pop), k, replace=False)
    return pop[int(idx[fit[idx].argmax()])]


# ── fitness ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def fast_fitness(pop: list[torch.Tensor], S: torch.Tensor, k: int,
                 y: torch.Tensor, val_idx: np.ndarray,
                 gam: TorchGAM1D) -> torch.Tensor:
    """Batch GAM fitness evaluated on val_idx only (matches EssenceBench)."""
    M     = torch.stack(pop, 0).float()           # (P, d)
    y_hat = gam.predict_batch((S @ M.T) / k * 100.0)  # (n_models, P)
    rmse  = torch.sqrt(((y_hat[val_idx] - y[val_idx, None]) ** 2).mean(0))
    return -rmse   # higher = better (for tournament/topk)


# ── single-round GA ───────────────────────────────────────────────────────────

def run_ga(S: torch.Tensor, y: torch.Tensor, val_idx: np.ndarray,
           k: int, gam: TorchGAM1D,
           pop: int = 200, elite: int = 20,
           generations: int = 1000, seed: int = 0,
           ) -> tuple[torch.Tensor, float, list[torch.Tensor]]:
    rng  = np.random.default_rng(seed)
    d    = S.size(1)
    mut_rate = 1.0 / k

    pop_masks = []
    for _ in range(pop):
        m = torch.zeros(d, dtype=torch.bool)
        m[torch.from_numpy(rng.choice(d, k, replace=False))] = True
        pop_masks.append(m)

    best_rmse, best_mask = 1e9, None

    for gen in range(1, generations + 1):
        fit = fast_fitness(pop_masks, S, k, y, val_idx, gam)

        cur_best = -fit.max().item()
        if cur_best < best_rmse:
            best_rmse = cur_best
            best_mask = pop_masks[int(fit.argmax())].clone()

        elite_idx = fit.topk(elite).indices
        new_pop   = [pop_masks[int(i)].clone() for i in elite_idx]

        while len(new_pop) < pop:
            p1    = tournament(pop_masks, fit, k=3, rng=rng)
            p2    = tournament(pop_masks, fit, k=3, rng=rng)
            child = crossover(p1, p2)
            child = mutate(child, mut_rate)
            child = repair(child, k, rng)
            new_pop.append(child)

        pop_masks = new_pop

        if gen % 100 == 0:
            logger.info("  gen %4d/%d  best RMSE=%.4f", gen, generations, best_rmse)

    # return top-200 for EBM importance aggregation
    fit   = fast_fitness(pop_masks, S, k, y, val_idx, gam)
    order = fit.argsort(descending=True)[:200]
    top200 = [pop_masks[int(i)].cpu() for i in order]
    return best_mask.cpu(), best_rmse, top200


# ── EBM feature importance aggregation (matches EssenceBench) ─────────────────

def aggregate_ebm(top_inds: list[torch.Tensor], S: torch.Tensor,
                  y: torch.Tensor, tr_idx: np.ndarray) -> np.ndarray:
    d = S.size(1)
    imp_sum   = np.zeros(d, dtype=np.float32)
    imp_count = np.zeros(d, dtype=np.int32)
    S_np = S.cpu().numpy()
    y_np = y.cpu().numpy()

    for idx, m in enumerate(top_inds):
        sel = m.nonzero(as_tuple=True)[0].numpy()
        if sel.size == 0:
            continue
        X = S_np[tr_idx][:, sel]
        if X.shape[0] < 2:
            continue
        try:
            ebm = ExplainableBoostingRegressor(interactions=0, random_state=0)
            ebm.fit(X, y_np[tr_idx])
            imp = ebm.term_importances()
            for j, feat in enumerate(sel):
                imp_sum[feat]   += imp[j]
                imp_count[feat] += 1
        except Exception as e:
            logger.warning("EBM fit failed (idx=%d): %s", idx, e)

    with np.errstate(divide="ignore", invalid="ignore"):
        imp_mean = np.divide(imp_sum, imp_count, where=imp_count > 0)
    return imp_mean


# ── multi-round search ────────────────────────────────────────────────────────

def multi_round_search(S: torch.Tensor, y: torch.Tensor,
                       tr_idx: np.ndarray, val_idx: np.ndarray,
                       rounds: int = 2, k: int = 100,
                       pop: int = 200, elite: int = 20,
                       generations: int = 1000) -> torch.Tensor:
    d_total          = S.size(1)
    feat_ids_current = np.arange(d_total)
    global_best_rmse = 1e9
    global_best_mask = None

    for R in range(rounds):
        logger.info("=== ROUND %d  |candidates|=%d ===", R, len(feat_ids_current))
        S_cur = S[:, feat_ids_current]

        gam = TorchGAM1D(n_splines=25, lam=5.0, device=str(S.device))
        gam.fit_closed((S_cur.mean(1) * 100)[tr_idx], y[tr_idx])

        best_mask, best_rmse, top200 = run_ga(
            S_cur, y, val_idx,
            k=min(k, S_cur.size(1)), gam=gam,
            pop=pop, elite=elite, generations=generations, seed=R)

        if best_rmse < global_best_rmse:
            global_best_rmse = best_rmse
            gmask = torch.zeros(d_total, dtype=torch.bool)
            gmask[feat_ids_current[best_mask.nonzero(as_tuple=True)[0]]] = True
            global_best_mask = gmask
        logger.info("ROUND %d  best RMSE=%.4f", R, best_rmse)

        if R == rounds - 1:
            break   # no pruning needed after last round

        # EBM importance → High / Low / Rand groups (keep top 75%)
        logger.info("Aggregating EBM importances over top-200 individuals …")
        imp = aggregate_ebm(top200, S_cur, y, tr_idx)

        keep     = max(k, len(feat_ids_current) * 3 // 4)
        ranked   = np.argsort(-imp)
        high_set = feat_ids_current[ranked[:keep]]
        low_set  = feat_ids_current[ranked[-keep:]]
        rand_set = np.random.default_rng(R).choice(feat_ids_current, keep, replace=False)

        best_group_rmse  = 1e9
        for name, feats in [("high", high_set), ("low", low_set), ("rand", rand_set)]:
            logger.info("  subgroup %-4s  |feats|=%d", name, len(feats))
            S_sub = S[:, feats]
            gam_sub = TorchGAM1D(n_splines=25, lam=5.0, device=str(S.device))
            gam_sub.fit_closed((S_sub.mean(1) * 100)[tr_idx], y[tr_idx])

            sub_mask, sub_rmse, _ = run_ga(
                S_sub, y, val_idx,
                k=min(k, len(feats)), gam=gam_sub,
                pop=max(60, pop // 3), elite=max(6, elite // 3),
                generations=max(200, generations // 2),
                seed=hash((R, name)) & 0xFFFF)

            if sub_rmse < best_group_rmse:
                best_group_rmse  = sub_rmse
                feat_ids_current = feats

            if sub_rmse < global_best_rmse:
                global_best_rmse = sub_rmse
                gmask = torch.zeros(d_total, dtype=torch.bool)
                gmask[feats[sub_mask.nonzero(as_tuple=True)[0]]] = True
                global_best_mask = gmask
            logger.info("  subgroup %-4s  RMSE=%.4f", name, sub_rmse)

        k = min(k, len(feat_ids_current))

    return global_best_mask


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored-dir",  required=True)
    parser.add_argument("--subset-csv",  required=True)
    parser.add_argument("--queries",     required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--k",           type=int, default=100)
    parser.add_argument("--rounds",      type=int, default=2)
    parser.add_argument("--pop-size",    type=int, default=200)
    parser.add_argument("--generations", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading scored matrix …")
    model_names, question_ids, matrix_np = load_scored_matrix(Path(args.scored_dir))
    n_total     = len(question_ids)
    full_scores = matrix_np.sum(axis=0).astype(float)

    logger.info("%d questions × %d models", n_total, len(model_names))
    for i, m in enumerate(model_names):
        logger.info("  %s: %.3f", m, full_scores[i] / n_total)

    step1_ids         = set(pd.read_csv(args.subset_csv)["id"])
    candidate_indices = np.array([i for i, qid in enumerate(question_ids) if qid in step1_ids])
    logger.info("Step 1 candidates: %d  →  k=%d", len(candidate_indices), args.k)

    # S: (n_models, n_candidates) — EssenceBench layout
    S     = torch.from_numpy(matrix_np[candidate_indices].T.astype(np.float32)).to(device)
    y_pct = torch.tensor(full_scores / n_total * 100, dtype=torch.float32, device=device)

    tr_idx, val_idx = stratified_split(full_scores / n_total, val_ratio=0.15)
    logger.info("Train models (%d): %s", len(tr_idx), [model_names[i] for i in tr_idx])
    logger.info("Val   models (%d): %s", len(val_idx), [model_names[i] for i in val_idx])

    best_mask = multi_round_search(
        S, y_pct, tr_idx=tr_idx, val_idx=val_idx,
        rounds=args.rounds, k=args.k,
        pop=args.pop_size, elite=max(5, args.pop_size // 10),
        generations=args.generations)

    selected_local  = best_mask.nonzero(as_tuple=True)[0].numpy()
    selected_global = candidate_indices[selected_local]
    selected_ids    = [question_ids[i] for i in selected_global]

    # Output CSV
    queries_df  = pd.read_csv(args.queries)
    selected_df = queries_df[queries_df["id"].isin(selected_ids)].copy()
    rpbis_map   = {}
    for i, qid in enumerate(question_ids):
        r, _ = pearsonr(matrix_np[i].astype(float), full_scores)
        rpbis_map[qid] = round(float(r), 4) if not np.isnan(r) else 0.0
    selected_df["rpbis"] = selected_df["id"].map(rpbis_map)
    selected_df = selected_df.sort_values("rpbis", ascending=False)

    # Linear calibration for the output formula
    ga_subset_scores = matrix_np[selected_global].sum(axis=0).astype(float)
    coeffs = np.polyfit(ga_subset_scores, full_scores, 1)
    pred   = np.polyval(coeffs, ga_subset_scores)
    r2     = 1 - float(np.sum((full_scores - pred) ** 2)) / float(np.sum((full_scores - full_scores.mean()) ** 2))

    logger.info("Calibration on %d-question subset:", args.k)
    logger.info("  full_score ≈ %.4f × subset_score + %.4f  (R²=%.4f)",
                coeffs[0], coeffs[1], r2)
    print(f"\n  {'Model':<50} {'Full':>6} {'Subset':>8} {'Pred':>8} {'Err':>7}")
    for i, name in enumerate(model_names):
        p = float(np.polyval(coeffs, ga_subset_scores[i]))
        print(f"  {name:<50} {int(full_scores[i]):>6} {int(ga_subset_scores[i]):>8} {p:>8.1f} {p-full_scores[i]:>+7.1f}")

    selected_df.to_csv(output_path, index=False)
    cal_path = output_path.with_suffix(".calibration.json")
    cal_path.write_text(json.dumps({
        "n_total": n_total,
        "n_step1_candidates": len(candidate_indices),
        "k": args.k,
        "score_reconstruction": {
            "formula": "full_score ≈ scale * subset_score + offset",
            "scale":  round(float(coeffs[0]), 6),
            "offset": round(float(coeffs[1]), 6),
            "r2":     round(r2, 6),
        },
        "training_models": [
            {"model": model_names[i],
             "full_score": int(full_scores[i]),
             "subset_score": int(ga_subset_scores[i]),
             "predicted_full": round(float(np.polyval(coeffs, ga_subset_scores[i])), 2)}
            for i in range(len(model_names))
        ],
    }, indent=2), encoding="utf-8")

    logger.info("Saved %d questions → %s", len(selected_df), output_path)
    logger.info("Calibration → %s", cal_path)


if __name__ == "__main__":
    main()
