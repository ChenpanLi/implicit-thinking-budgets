"""The per-item composition size of fragility-aware batching, under an accuracy budget.

    python -m llm.probe.assign_k --arm measured                  # from the measured drops
    python -m llm.probe.assign_k --arm probe                     # from the heads
    python -m llm.probe.assign_k --arm measured --dataset gsm8k --model qwen3-8b

Every item goes to the size that maximises

    -E[drop at k] + lambda * (1 - cost_k)          (k=1 scores 0)

with ``cost_k`` the dataset's output tokens per item at k over those at k=1
(``token-cost.csv``). ``E[drop]`` is a per-k isotonic map from a score to the measured
drop, fitted on the outer training slice:

* ``measured``: each item's trials at every k (isolated and random, pooled over both
  passes) are split in half at random (hypergeometrically, seed 0). The drop on one
  half is the score; the map is fitted against the drop on the other half, so it
  learns E[drop | noisy drop] rather than the identity.
* ``probe``: the score is the survival probability of the four cross-fitted heads of
  the outer fold. On the training slice each item is read by the one head that held
  it out; on the held-out fold the four heads' expected drops are averaged.

lambda is chosen on the training slice alone: the slice is cut into 5 blocks four
times over, the map is refitted without each block and applied to it, accuracy and
tokens are recorded at 121 rates, the averaged accuracy curve is made non-increasing
in lambda by isotonic regression, and the cheapest rate whose accuracy clears the
mean isolated accuracy minus ``--budget`` by ``--margin`` standard errors wins.

Token prices come from ``$RUN_DIR/probe/token-cost.csv``; ``--token-cost`` names
another table (Claude Opus 5 has one per arm, see ``probe.sh``).

Writes ``$RUN_DIR/probe/assignments/<arm>/<ds>/<model>.csv`` with the outer fold and
``pred_largest_k_lagrangian`` per item (0 = asked alone), the column the
``predicted-calibration`` batching reads.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

from llm.probe import common
from llm.probe import head as head_mod
from llm.probe import tokens as tokens_mod

LAMBDAS = np.concatenate([[0.0], np.exp(np.linspace(np.log(1e-4), np.log(3.0), 120))])
DEFAULT_BUDGET = 0.005
DEFAULT_MARGIN = 1.0
SPLIT_SEED = 0
INNER_REPEATS = (0, 1, 2, 3)
INNER_BLOCKS = 5
K_COLUMN = "pred_largest_k_lagrangian"


# ------------------------------------------------------------------ the rule

def fit_maps(score: np.ndarray, drops: np.ndarray, increasing: bool, rows=None) -> list:
    """Per k, a monotone map from the score to the expected drop."""
    rows = np.arange(len(score)) if rows is None else rows
    maps = []
    for j in range(drops.shape[1]):
        good = rows[np.isfinite(score[rows, j]) & np.isfinite(drops[rows, j])]
        if good.size < 20:
            maps.append(float(np.nanmean(drops[:, j])) if np.isfinite(drops[:, j]).any()
                        else 0.0)
            continue
        maps.append(IsotonicRegression(increasing=increasing, out_of_bounds="clip")
                    .fit(score[good, j], drops[good, j]))
    return maps


def apply_maps(maps: list, score: np.ndarray) -> np.ndarray:
    out = np.zeros_like(score, dtype=float)
    for j, model in enumerate(maps):
        out[:, j] = (model if isinstance(model, float)
                     else model.predict(np.nan_to_num(score[:, j], nan=0.0)))
    return out


def assign(expected: np.ndarray, costs: np.ndarray, lam: float) -> np.ndarray:
    """Index into the sizes (0 = alone) maximising accuracy kept plus lambda * tokens saved."""
    value = -expected + lam * (1.0 - costs[1:])[None, :]
    return np.argmax(np.concatenate([np.zeros((len(expected), 1)), value], axis=1), axis=1)


def realised(assigned, drops, solo, costs) -> tuple[float, float]:
    """Mean accuracy and mean token ratio of an assignment, read off measurements."""
    rows = np.arange(len(assigned))
    taken = np.where(assigned > 0,
                     drops[rows, np.clip(assigned - 1, 0, drops.shape[1] - 1)], 0.0)
    return float((solo - taken).mean()), float(costs[assigned].mean())


def rate_curve(score, drops, solo, costs, increasing) -> np.ndarray:
    """``[accuracy, token ratio, standard error]`` at every rate, out of sample."""
    n = len(score)
    per_block, tokens, weight = [], np.zeros(len(LAMBDAS)), 0
    for seed in INNER_REPEATS:
        rng = np.random.default_rng(seed)
        for block in np.array_split(rng.permutation(n), INNER_BLOCKS):
            keep = np.setdiff1d(np.arange(n), block)
            hat = apply_maps(fit_maps(score, drops, increasing, keep), score[block])
            row = np.zeros(len(LAMBDAS))
            for index, lam in enumerate(LAMBDAS):
                accuracy, ratio = realised(assign(hat, costs, lam), drops[block],
                                           solo[block], costs)
                row[index] = accuracy
                tokens[index] += ratio * len(block)
            per_block.append(row)
            weight += len(block)
    stacked = np.stack(per_block)
    accuracy = -IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
        LAMBDAS, -stacked.mean(axis=0))
    error = stacked.std(axis=0, ddof=1) / np.sqrt(len(stacked))
    return np.stack([accuracy, tokens / weight, error], axis=1)


def choose_rate(curve: np.ndarray, floor: float, margin: float) -> int:
    """The cheapest rate whose estimate clears the floor by ``margin`` standard errors."""
    safe = curve[:, 0] - margin * curve[:, 2]
    feasible = np.flatnonzero(safe >= floor)
    if feasible.size:
        return int(feasible[np.argmin(curve[feasible, 1])])
    return int(np.argmax(safe))


# ------------------------------------------------------------------ the measurements

def trial_counts(dataset: str, model: str, isolated, random) -> pd.DataFrame:
    """Per item: trials and correct answers alone, and at every k, pooled over stages."""
    rungs = common.ladder(dataset)
    solo = common.accuracy(common.occurrences(isolated, dataset, model), ["question_id"])
    per_k = common.accuracy(common.occurrences(random, dataset, model),
                            ["question_id", "nominal_k"])
    table = pd.DataFrame({"n_solo_trials": solo["n_trials"],
                          "n_solo_correct": solo["n_correct"]})
    for k in rungs:
        at_k = per_k.xs(k, level="nominal_k")
        table[f"n_trials_k{k}"] = at_k["n_trials"]
        table[f"accuracy_k{k}"] = at_k["accuracy"]
    table.index = table.index.astype(str)
    spread = {c: sorted(int(v) for v in table[c].dropna().unique())
              for c in table if c.startswith("n_") and "correct" not in c}
    print(f"  {dataset}/{model}: trials per item {spread}", flush=True)
    return table


def split_halves(counts: pd.DataFrame, rungs, seed: int) -> tuple:
    """Two independent halves of every item's trials: ``(drops_a, drops_b, solo_b)``."""
    rng = np.random.default_rng(seed)

    def halves(trials, correct):
        n_a = (trials // 2).astype(float)
        n_b = trials - n_a
        ok = np.isfinite(trials) & np.isfinite(correct) & (trials >= 2)
        got_a = np.zeros(len(trials))
        rows = np.flatnonzero(ok)
        if rows.size:
            got_a[rows] = rng.hypergeometric(
                np.rint(correct[rows]).astype(int),
                np.rint(trials[rows] - correct[rows]).astype(int),
                n_a[rows].astype(int))
        return (np.where(ok, got_a / np.where(n_a > 0, n_a, 1), np.nan),
                np.where(ok, (correct - got_a) / np.where(n_b > 0, n_b, 1), np.nan))

    solo_a, solo_b = halves(counts["n_solo_trials"].to_numpy(float),
                            counts["n_solo_correct"].to_numpy(float))
    cols_a, cols_b = {}, {}
    for k in rungs:
        trials = counts[f"n_trials_k{k}"].to_numpy(float)
        a, b = halves(trials, counts[f"accuracy_k{k}"].to_numpy(float) * trials)
        cols_a[k], cols_b[k] = solo_a - a, solo_b - b
    return (pd.DataFrame(cols_a, index=counts.index), pd.DataFrame(cols_b, index=counts.index),
            pd.Series(solo_b, index=counts.index))


def _take(frame, ids) -> np.ndarray:
    return frame.reindex([str(i) for i in ids]).to_numpy(float)


# ------------------------------------------------------------------ one outer fold

def _fold_items(dataset, model, fold, embeddings, labels, read_heads: bool):
    """Per inner fold: the held-out training items (and the head's scores), plus the test fold."""
    rungs = common.ladder(dataset)
    solo_of = labels["solo_accuracy"]
    slices, test = [], None
    for val_fold, prefix in enumerate(common.PREFIXES):
        data = head_mod.bundle(dataset, model, fold, val_fold, embeddings)
        module = None
        if read_heads:
            path = common.head_dir(prefix, dataset, model, fold) / "checkpoints" / "best.pt"
            if not path.is_file():
                raise SystemExit(f"no trained head at {path}; run python -m llm.probe.train")
            module = head_mod.OrdinalHead(data.train.X.shape[1], len(rungs))
            module.load_state_dict(torch.load(path, map_location="cpu",
                                              weights_only=False)["model_state"])

        def read(split):
            drops = np.asarray(split.drops, dtype=float)
            solo = solo_of.loc[split.item_ids].to_numpy(dtype=float)
            keep = np.isfinite(drops).all(axis=1) & np.isfinite(solo)
            score = (head_mod.probabilities(module, split.X)[keep]
                     if module is not None else None)
            return np.asarray(split.item_ids)[keep], score, drops[keep], solo[keep]

        slices.append(read(data.val))
        current = read(data.test)
        if test is None:
            test = current
        elif not np.array_equal(current[0], test[0]):
            raise SystemExit(f"{dataset}/{model} fold {fold}: the heads disagree on the "
                             "held-out items")
        if read_heads:
            slices[-1] = slices[-1] + (current[1],)
    return slices, test


def one_fold(arm, dataset, model, fold, costs, budget, margin, embeddings, labels,
             halves=None) -> pd.DataFrame:
    sizes = common.sizes(dataset)
    slices, (test_ids, _, test_drops, test_solo) = _fold_items(
        dataset, model, fold, embeddings, labels, read_heads=(arm == "probe"))
    fit_ids = np.concatenate([s[0] for s in slices])
    if len(np.unique(fit_ids)) != len(fit_ids):
        raise SystemExit(f"{dataset}/{model} fold {fold}: the inner folds overlap")

    if arm == "measured":
        drops_a, drops_b, solo_b = halves
        score, drops = _take(drops_a, fit_ids), _take(drops_b, fit_ids)
        solo = solo_b.reindex([str(i) for i in fit_ids]).to_numpy(float)
        good = np.isfinite(score).all(axis=1) & np.isfinite(drops).all(axis=1) & np.isfinite(solo)
        score, drops, solo = score[good], drops[good], solo[good]
        test_score = _take(drops_a, test_ids)
        test_score = np.where(np.isfinite(test_score), test_score, 0.0)
        increasing = True
    else:
        score = np.concatenate([s[1] for s in slices])
        drops = np.concatenate([s[2] for s in slices])
        solo = np.concatenate([s[3] for s in slices])
        increasing = False

    curve = rate_curve(score, drops, solo, costs, increasing)
    index = choose_rate(curve, float(solo.mean()) - budget, margin)
    maps = fit_maps(score, drops, increasing)
    if arm == "measured":
        expected = apply_maps(maps, test_score)
    else:
        expected = np.mean([apply_maps(maps, s[4]) for s in slices], axis=0)
    assigned = assign(expected, costs, float(LAMBDAS[index]))
    n_rungs = len(sizes) - 1
    print(f"  {dataset}/{model} fold {fold}: lambda {LAMBDAS[index]:.4f}, estimated "
          f"{100 * (curve[index, 0] - solo.mean()):+.2f}pp at {1 - curve[index, 1]:.1%} "
          f"tokens saved", flush=True)
    return pd.DataFrame({
        "domain": dataset, "model": model, "fold": fold, "question_id": test_ids,
        K_COLUMN: [sizes[a] if a > 0 else 0 for a in assigned],
        "expected_drop": expected[np.arange(len(assigned)),
                                  np.clip(assigned - 1, 0, n_rungs - 1)] * (assigned > 0),
        "predicted": True,
    })


def run_pair(arm, dataset, model, *, budget, margin, isolated, random, cost_table) -> None:
    costs = np.array(tokens_mod.cost_vector(cost_table, dataset, model), dtype=float)
    costs = costs / costs[0]
    labels = pd.read_csv(common.labels_file(dataset, model),
                         dtype={"question_id": str}).set_index("question_id")
    halves = None
    if arm == "measured":
        isolated, random = common.stages(dataset, model, isolated, random)
        counts = trial_counts(dataset, model, isolated, random)
        halves = split_halves(counts, common.ladder(dataset), SPLIT_SEED)
    embeddings = common.load_embeddings(dataset)
    tables = [one_fold(arm, dataset, model, fold, costs, budget, margin, embeddings,
                       labels, halves) for fold in range(common.N_FOLDS)]
    out = common.assignment_file(arm, dataset, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(out, index=False)
    print(f"  -> {out}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=["measured", "probe"], required=True)
    parser.add_argument("--dataset", nargs="*", default=None)
    parser.add_argument("--model", nargs="*", default=None)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                        help="accuracy allowed below the isolated run, as a fraction")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help="standard errors the chosen rate must clear the floor by")
    parser.add_argument("--token-cost", default=None,
                        help="the token cost table (default: $RUN_DIR/probe/token-cost.csv)")
    common.stage_args(parser)
    args = parser.parse_args(argv)

    cost_table = pd.read_csv(args.token_cost or common.token_cost_file())
    for dataset, model in common.pairs(args.dataset, args.model):
        run_pair(args.arm, dataset, model, budget=args.budget, margin=args.margin,
                 isolated=args.isolated, random=args.random, cost_table=cost_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
