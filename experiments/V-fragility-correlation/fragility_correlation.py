#!/usr/bin/env python3
"""Is an item's fragility shared across models, across k, and between two passes?

    python experiments/V-fragility-correlation/fragility_correlation.py
    python experiments/V-fragility-correlation/fragility_correlation.py --dataset mmlu gsm8k

Fragility is the per-item accuracy drop, drop_k = isolated accuracy - accuracy in
random batches of size k (see `llm.core.analysis.fragility`). Every item is measured
in two independent passes, A (isolated-a + random-a) and B (isolated-b + random-b).
Three kinds of correlation, each over the items of one dataset:

* same (one model, one k): drop from pass A against drop from pass B. This test-retest
  correlation is the reliability the other two are disattenuated by.
* crossk (one model, k against k'): drop_k from pass A against drop_k' from pass B,
  averaged (Fisher z) with the swapped pairing, so the two sides never share the
  isolated accuracy they subtract. The same-pass value is kept as ``r_same_pass``.
* crossmodel (model m against m', one k): drops pooled over both passes, against
  reliabilities stepped up by Spearman-Brown (2r / (1 + r)).

``r_disatt`` = r / sqrt(rel_1 * rel_2), left empty when a reliability is not positive,
and not clipped. A model is used on a dataset only when it has all four stages there.

Every statistic has a 95% percentile bootstrap interval (``_lo`` / ``_hi``) from
``--boot`` resamples of the items. One resample per replicate is shared by every model,
pass and k of a dataset, so reliabilities and correlations are recomputed on the same
items. The pooled ``all`` row and the heatmap cells pair replicate b of every dataset.
Correlations are averaged on the Fisher-z scale, ``r_disatt`` arithmetically.

Writes to $RUN_DIR/results/V-fragility-correlation/:
fragility_correlation_{same,crossk,crossmodel}.csv (one row per pair),
fragility_correlation_summary.csv (means per dataset, plus ``all``),
fragility_correlation_cells.csv (a k pair or a model pair, pooled over datasets) and
fragility_correlation_paper.csv (the three numbers of the paper, with half the width of
their interval as the +-).
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from llm import paths
from llm.core.analysis import fragility
from llm.specs import dataset as dataset_spec

TAU = 0.02
STAGES = ("isolated-a", "random-a", "isolated-b", "random-b")

# The datasets in the order they are processed, each with its bootstrap seed offset
# (seed = --seed + offset).
DATASETS = {"integrals": 0, "mmlu": 1, "onewordqa": 2, "gpqa-diamond": 4, "gsm8k": 5,
            "math500": 6, "aime": 7, "strategyqa": 8}

SAME_STATS = ("r", "rho", "phi", "rel_pooled")
CROSSK_STATS = ("r", "rho", "phi", "r_disatt", "r_same_pass")
CROSSMODEL_STATS = ("r", "rho", "phi", "r_disatt")


def measure(name: str) -> dict:
    """Per model: pass-A, pass-B and pooled drop tables for one dataset."""
    models = fragility.models_with(name, STAGES)
    present = set().union(*(paths.models_with_runs(s, name) for s in STAGES))
    ks = tuple(dataset_spec(name).random_ks)
    out = {"dataset": name, "ks": ks, "models": models,
           "dropped": sorted(present - set(models)), "tables": {}, "trials": {}}
    for model in models:
        sel = [fragility.load_outcomes(stage, name, model) for stage in STAGES]
        out["tables"][model] = {
            "A": fragility.drop_table(sel[0], sel[1], ks),
            "B": fragility.drop_table(sel[2], sel[3], ks),
            "pooled": fragility.drop_table(pd.concat([sel[0], sel[2]]),
                                           pd.concat([sel[1], sel[3]]), ks),
        }
        out["trials"][model] = {
            "solo_A": fragility.trial_counts(sel[0], ["question_id"]),
            "solo_B": fragility.trial_counts(sel[2], ["question_id"]),
            "k_A": fragility.trial_counts(sel[1], ["question_id", "nominal_k"]),
            "k_B": fragility.trial_counts(sel[3], ["question_id", "nominal_k"]),
        }
    return out


def _corr(X: np.ndarray) -> np.ndarray:
    """Column correlation matrix; a constant column gives NaN rather than a warning."""
    Xc = X - X.mean(axis=0)
    sd = np.sqrt((Xc ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        Xn = Xc / np.where(sd > 0, sd, np.nan)
    return Xn.T @ Xn


def _mats(X: np.ndarray, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    """(3, n_pairs): Pearson, Spearman and phi (on drop > TAU) for column pairs (I, J)."""
    return np.stack([_corr(X)[I, J], _corr(rankdata(X, axis=0))[I, J],
                     _corr((X > TAU).astype(float))[I, J]])


def _fzmean(*arrays: np.ndarray) -> np.ndarray:
    """Elementwise Fisher-z mean of several correlations, ignoring NaN."""
    z = np.arctanh(np.clip(np.stack(arrays), -.999, .999))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.tanh(np.nanmean(z, axis=0))


def _average(stat: str, *arrays: np.ndarray) -> np.ndarray:
    """Fisher-z mean for a correlation; plain mean for ``r_disatt``, which can exceed 1."""
    if stat == "r_disatt":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.stack(arrays), axis=0)
    return _fzmean(*arrays)


def _pos(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.nan)


class _Pairs:
    """Registry of the column pairs whose correlations the rows need."""

    def __init__(self, col):
        self.col, self.index = col, {}

    def __call__(self, a, b) -> int:
        key = (self.col[a], self.col[b])
        return self.index.setdefault(key, len(self.index))

    def arrays(self):
        keys = list(self.index)
        return np.array([k[0] for k in keys]), np.array([k[1] for k in keys])


def analyse(result: dict, n_boot: int, seed: int):
    """Every row's statistics at the point estimate and over `n_boot` item resamples."""
    name, ks, models = result["dataset"], result["ks"], result["models"]
    names, blocks = [], []
    for model in models:
        for side in ("A", "B", "pooled"):
            for k in ks:
                names.append((model, side, k))
                blocks.append(result["tables"][model][side][f"drop_k{k}"])
    frame = pd.concat(blocks, axis=1, keys=range(len(blocks)))
    complete = frame.dropna()
    if len(complete) < len(frame):
        print(f"[{name}] {len(frame) - len(complete)} items lack a measurement; dropped")
    X = complete.to_numpy(dtype=float)
    col = {key: i for i, key in enumerate(names)}
    pair = _Pairs(col)

    same_ix = {(m, k): pair((m, "A", k), (m, "B", k)) for m in models for k in ks}
    kpairs = list(itertools.combinations(ks, 2))
    crossk_ix = {(m, k1, k2): (pair((m, "A", k1), (m, "B", k2)),
                               pair((m, "B", k1), (m, "A", k2)),
                               pair((m, "A", k1), (m, "A", k2)),
                               pair((m, "B", k1), (m, "B", k2)))
                 for m in models for k1, k2 in kpairs}
    mpairs = list(itertools.combinations(models, 2))
    model_ix = {(m1, m2, k): pair((m1, "pooled", k), (m2, "pooled", k))
                for m1, m2 in mpairs for k in ks}
    I, J = pair.arrays()

    def rows(C: np.ndarray) -> dict[str, np.ndarray]:
        """C: (..., 3, n_pairs) -> per-kind arrays (..., n_rows, n_stats)."""
        r, rho, phi = C[..., 0, :], C[..., 1, :], C[..., 2, :]
        rel = {key: r[..., i] for key, i in same_ix.items()}
        same = np.stack([np.stack([r[..., same_ix[key]], rho[..., same_ix[key]],
                                   phi[..., same_ix[key]],
                                   2 * _pos(rel[key]) / (1 + _pos(rel[key]))], -1)
                         for key in same_ix], -2)
        crossk = []
        for (m, k1, k2), (ab, ba, aa, bb) in crossk_ix.items():
            rk = _fzmean(r[..., ab], r[..., ba])
            crossk.append(np.stack([
                rk, _fzmean(rho[..., ab], rho[..., ba]), _fzmean(phi[..., ab], phi[..., ba]),
                rk / np.sqrt(_pos(rel[(m, k1)]) * _pos(rel[(m, k2)])),
                _fzmean(r[..., aa], r[..., bb])], -1))
        crossk = np.stack(crossk, -2)
        step = {key: 2 * _pos(v) / (1 + _pos(v)) for key, v in rel.items()}
        crossmodel = np.stack([np.stack([
            r[..., i], rho[..., i], phi[..., i],
            r[..., i] / np.sqrt(step[(m1, k)] * step[(m2, k)])], -1)
            for (m1, m2, k), i in model_ix.items()], -2)
        return {"same": same, "crossk": crossk, "crossmodel": crossmodel}

    point = rows(_mats(X, I, J))
    rng = np.random.default_rng(seed)
    n = len(X)
    boot = np.empty((n_boot, 3, len(I)))
    for b in range(n_boot):
        boot[b] = _mats(X[rng.integers(0, n, n)], I, J)
    draws = rows(boot)

    stats = {"same": SAME_STATS, "crossk": CROSSK_STATS, "crossmodel": CROSSMODEL_STATS}
    keys = {"same": [dict(model=m, k=k) for m, k in same_ix],
            "crossk": [dict(model=m, k1=k1, k2=k2) for m, k1, k2 in crossk_ix],
            "crossmodel": [dict(model1=m1, model2=m2, k=k) for m1, m2, k in model_ix]}
    out = {}
    for kind in ("same", "crossk", "crossmodel"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanpercentile(draws[kind], 2.5, axis=0)
            hi = np.nanpercentile(draws[kind], 97.5, axis=0)
        table = []
        for i, key in enumerate(keys[kind]):
            row = {"dataset": name, **key, "n_items": n}
            for j, stat in enumerate(stats[kind]):
                row[stat] = point[kind][i, j]
                row[f"{stat}_lo"], row[f"{stat}_hi"] = lo[i, j], hi[i, j]
            table.append(row)
        out[kind] = table
    for row in out["same"]:
        t = result["tables"][row["model"]]["pooled"][f"drop_k{row['k']}"]
        row["fragile_rate_pooled"] = float((t > TAU).mean())
        row["mean_drop_pooled"] = float(t.mean())
        row.update(result["trials"][row["model"]])

    summary = {"dataset": name, "n_models": len(models), "n_items": n}
    for kind in ("same", "crossk", "crossmodel"):
        for j, stat in enumerate(stats[kind]):
            est = _average(stat, *point[kind][:, j])
            dist = _average(stat, *np.moveaxis(draws[kind][:, :, j], 1, 0))
            summary[f"{kind}_{stat}"] = float(est)
            summary[f"{kind}_{stat}_lo"] = float(np.nanpercentile(dist, 2.5))
            summary[f"{kind}_{stat}_hi"] = float(np.nanpercentile(dist, 97.5))
    return out["same"], out["crossk"], out["crossmodel"], summary, (point, draws, keys)


def pooled_summary(parts: list) -> dict:
    """The ``all`` row: means over every dataset's rows, replicate by replicate."""
    stats = {"same": SAME_STATS, "crossk": CROSSK_STATS, "crossmodel": CROSSMODEL_STATS}
    row = {"dataset": "all", "n_models": np.nan, "n_items": np.nan}
    for kind, names in stats.items():
        point = np.concatenate([p[kind] for p, _, _ in parts], axis=0)
        draws = np.concatenate([d[kind] for _, d, _ in parts], axis=1)
        for j, stat in enumerate(names):
            dist = _average(stat, *np.moveaxis(draws[:, :, j], 1, 0))
            row[f"{kind}_{stat}"] = float(_average(stat, *point[:, j]))
            row[f"{kind}_{stat}_lo"] = float(np.nanpercentile(dist, 2.5))
            row[f"{kind}_{stat}_hi"] = float(np.nanpercentile(dist, 97.5))
    return row


def pooled_cells(parts: list) -> list[dict]:
    """One row per heatmap cell: a k pair (pooled over datasets and models) or a
    model pair (pooled over datasets and k), averaged as in the summary."""
    stats = {"crossk": (CROSSK_STATS, ("k1", "k2")),
             "crossmodel": (CROSSMODEL_STATS, ("model1", "model2"))}
    out = []
    for kind, (names, by) in stats.items():
        keys = [tuple(key[b] for b in by) for _, _, ks in parts for key in ks[kind]]
        point = np.concatenate([p[kind] for p, _, _ in parts], axis=0)
        draws = np.concatenate([d[kind] for _, d, _ in parts], axis=1)
        for cell in dict.fromkeys(keys):
            idx = [i for i, key in enumerate(keys) if key == cell]
            row = {"kind": kind, by[0]: cell[0], by[1]: cell[1], "n_rows": len(idx)}
            for j, stat in enumerate(names):
                dist = _average(stat, *draws[:, idx, j].T)
                row[stat] = float(_average(stat, *point[idx, j]))
                row[f"{stat}_lo"] = float(np.nanpercentile(dist, 2.5))
                row[f"{stat}_hi"] = float(np.nanpercentile(dist, 97.5))
            out.append(row)
    return out


def paper_numbers(summary: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """The three values of Section 3.2: mean disattenuated r, +- half the interval."""
    pooled = summary.set_index("dataset").loc["all"]
    pair = cells[(cells["kind"] == "crossmodel")
                 & (cells[["model1", "model2"]].apply(set, axis=1) == {"qwen3-8b", "qwen3-14b"})]
    rows = []
    for label, src, prefix in (("across models, fixed k", pooled, "crossmodel_r_disatt"),
                               ("Qwen3 8B x Qwen3 14B", pair.iloc[0], "r_disatt"),
                               ("within a model, across k", pooled, "crossk_r_disatt")):
        lo, hi = src[f"{prefix}_lo"], src[f"{prefix}_hi"]
        rows.append({"quantity": label, "r_disatt": src[prefix], "lo": lo, "hi": hi,
                     "plus_minus": (hi - lo) / 2})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--boot", type=int, default=5000, help="item resamples per dataset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    names = [d for d in DATASETS if d in args.dataset]

    same, crossk, crossmodel, summary, parts = [], [], [], [], []
    for name in names:
        result = measure(name)
        print(f"[{result['dataset']}] {len(result['models'])} models"
              + (f"; dropped (not all four stages): {result['dropped']}"
                 if result["dropped"] else ""), flush=True)
        s, c, m, row, part = analyse(result, args.boot, args.seed + DATASETS[result["dataset"]])
        same += s; crossk += c; crossmodel += m; summary.append(row); parts.append(part)
    cells = []
    if len(parts) > 1:
        summary.append(pooled_summary(parts))
        cells = pooled_cells(parts)

    out = paths.results("V-fragility-correlation")
    for label, rows in (("same", same), ("crossk", crossk), ("crossmodel", crossmodel),
                        ("summary", summary), ("cells", cells)):
        frame = pd.DataFrame(rows)
        frame.insert(1, "n_boot", args.boot)
        frame.to_csv(out / f"fragility_correlation_{label}.csv", index=False)

    frame = pd.DataFrame(summary)
    with pd.option_context("display.width", 250):
        for kind, stats in (("same", SAME_STATS), ("crossk", CROSSK_STATS),
                            ("crossmodel", CROSSMODEL_STATS)):
            table = pd.DataFrame({stat: [f"{a:.2f} [{b:.2f}, {c:.2f}]" for a, b, c in zip(
                frame[f"{kind}_{stat}"], frame[f"{kind}_{stat}_lo"], frame[f"{kind}_{stat}_hi"])]
                for stat in stats}, index=frame["dataset"])
            print(f"\n{kind}\n{table.to_string()}")
    if cells:
        headline = paper_numbers(frame, pd.DataFrame(cells))
        headline.to_csv(out / "fragility_correlation_paper.csv", index=False)
        print()
        for row in headline.itertuples():
            print(f"{row.quantity:28s} {row.r_disatt:.2f} +- {row.plus_minus:.2f}")
    print(f"-> {out}/fragility_correlation_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
