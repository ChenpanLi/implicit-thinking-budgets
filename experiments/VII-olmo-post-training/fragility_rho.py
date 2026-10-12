"""How much the three OLMo 3 stages share their fragile tasks: disattenuated rho per pair.

As for the fragility correlation of the main models: per pair of conditions
r* = r / sqrt(r_AA r_BB), r_AA the test-retest correlation of the per-task accuracy drop
between two independent passes, Delta_A taken from one pass and Delta_B from the other;
r* averaged arithmetically (unclipped) over every condition entering a value; the +- is
half the width of a 95% percentile interval from a bootstrap that resamples each
dataset's tasks and recomputes every correlation and reliability. Unparsed answers
count as wrong.

OLMo has one pass, so its two passes are random halves of each task's trials: isolated
trials 2|3, per-term-trace occurrences 5|5. A condition is (stage, k) on one dataset, k
over that dataset's ladder (GPQA-Diamond 2..10, MATH-500 and MMLU-Redux 2..20); a
stage pair's value averages r* over all (dataset, k). Each correlation is Fisher-z
averaged over --splits random splits; every bootstrap replicate draws --boot-splits
fresh ones.

    python experiments/VII-olmo-post-training/fragility_rho.py             # -> fragility-rho.json
    python experiments/VII-olmo-post-training/fragility_rho.py --boot 20   # quick look
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from llm import paths
from llm.specs import MODEL_NAMES, OLMO, OLMO_DATASETS

from olmo_data import frames

NT, NS = 5, 10   # isolated trials, seeds per k
PAIRS = list(itertools.combinations(range(len(OLMO)), 2))


def load(dataset: str):
    """(ks, C [stage, task, trial], Y [stage, k, task, seed]) over the tasks every stage ran."""
    per = {m: frames(dataset, m) for m in OLMO}
    as_score = lambda f: f.assign(c=f.correct.astype("Float64").fillna(0.0).astype(float))  # noqa: E731
    pos = {m: as_score(per[m]["positions"]) for m in OLMO}
    trials = {m: as_score(per[m]["solo_trials"]) for m in OLMO}
    ks = sorted(set().union(*(set(p.k) for p in pos.values())))
    items = sorted(set.intersection(*[set(t.item_id) for t in trials.values()]) &
                   set.intersection(*[set(p.item_id) for p in pos.values()]))
    ix = {it: i for i, it in enumerate(items)}
    C = np.full((len(OLMO), len(items), NT), np.nan)
    Y = np.full((len(OLMO), len(ks), len(items), NS), np.nan)
    for mi, m in enumerate(OLMO):
        for it, g in trials[m][trials[m].item_id.isin(ix)].groupby("item_id"):
            C[mi, ix[it], :min(len(g), NT)] = g.c.to_numpy()[:NT]
        p = pos[m][pos[m].item_id.isin(ix)]
        for ki, k in enumerate(ks):
            for it, g in p[p.k == k].groupby("item_id"):
                v = g.sort_values("seed").c.to_numpy()[:NS]
                Y[mi, ki, ix[it], :len(v)] = v
    assert not np.isnan(C).any(), f"{dataset}: every task needs {NT} isolated trials"
    assert not np.isnan(Y).any(), f"{dataset}: every task needs {NS} occurrences per k"
    print(f"{dataset}: {len(items)} tasks, k = {ks}", flush=True)
    return ks, C, Y


def halves(gen, C, Y, sel):
    """(drop_A, drop_B), each [stage, k, task], from one random split of every task's trials."""
    c = C[:, sel]
    c = np.take_along_axis(c, np.argsort(gen.random(c.shape), -1), -1)
    nA = np.where(gen.random((len(OLMO), len(sel), 1)) < .5, 2, 3)
    mA = np.arange(NT) < nA
    sA = np.where(mA, c, 0).sum(-1) / mA.sum(-1)
    sB = np.where(~mA, c, 0).sum(-1) / (~mA).sum(-1)
    y = Y[:, :, sel]
    y = np.take_along_axis(y, np.argsort(gen.random(y.shape), -1), -1)
    return sA[:, None] - y[..., :NS // 2].mean(-1), sB[:, None] - y[..., NS // 2:].mean(-1)


def corr(a, b):
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


def z(r):
    return np.arctanh(np.clip(r, -.999, .999))


def star_one(C, Y, sel, gen, n_splits):
    """r* [pair, k] on one dataset; correlations Fisher-z averaged over splits."""
    nk = Y.shape[1]
    z_rel, z_x = np.zeros((len(OLMO), nk)), np.zeros((len(PAIRS), nk))
    for _ in range(n_splits):
        A, B = halves(gen, C, Y, sel)
        z_rel += z(corr(A, B))
        for pi, (i, j) in enumerate(PAIRS):     # Delta_A from one pass, Delta_B from the other
            z_x[pi] += (z(corr(A[i], B[j])) + z(corr(B[i], A[j]))) / 2
    rel, rx = np.tanh(z_rel / n_splits), np.tanh(z_x / n_splits)
    positive = lambda r: np.where(r > 0, r, np.nan)  # noqa: E731
    with np.errstate(invalid="ignore"):
        star = np.array([rx[pi] / np.sqrt(positive(rel[i]) * positive(rel[j]))
                         for pi, (i, j) in enumerate(PAIRS)])
    return star, rel


def rho(data, gen, n_splits, boot=False):
    stars, per = [], {}
    for ds, (ks, C, Y) in data.items():
        n = C.shape[1]
        sel = gen.integers(0, n, n) if boot else np.arange(n)
        star, rel = star_one(C, Y, sel, gen, n_splits)
        stars.append(star)
        per[ds] = {"ks": [int(k) for k in ks], "rho_per_k": star.tolist(),
                   "reliability_per_k": rel.tolist(), "rho": np.nanmean(star, axis=1).tolist()}
    return np.nanmean(np.concatenate(stars, axis=1), axis=1), per


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--boot", type=int, default=5000)
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--boot-splits", type=int, default=10)
    args = parser.parse_args()

    data = {ds: load(ds) for ds in OLMO_DATASETS}
    gen = np.random.default_rng(0)
    point, per = rho(data, gen, args.splits)
    boot = np.array([rho(data, gen, args.boot_splits, boot=True)[0] for _ in range(args.boot)])
    lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    result = {"datasets": OLMO_DATASETS, "boot": args.boot, "splits": args.splits,
              "boot_splits": args.boot_splits, "pairs": [[OLMO[i], OLMO[j]] for i, j in PAIRS],
              "rho": point.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
              "half_width": ((hi - lo) / 2).tolist(), "per_dataset": per}
    name = "fragility-rho.json" if args.boot == 5000 else f"fragility-rho-boot{args.boot}.json"
    out = paths.results("VII-olmo-post-training") / name
    out.write_text(json.dumps(result, indent=1))

    for (i, j), v, l, h in zip(PAIRS, point, lo, hi):
        print(f"{MODEL_NAMES[OLMO[i]]} vs {MODEL_NAMES[OLMO[j]]}: rho* = {v:.2f} +- {(h - l) / 2:.2f} "
              f"[{l:.2f}, {h:.2f}]")
    for ds in OLMO_DATASETS:
        print(ds, "rho per pair:", np.round(per[ds]["rho"], 2).tolist(),
              "| reliability per k:", np.round(per[ds]["reliability_per_k"], 2).tolist())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
