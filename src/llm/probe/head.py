"""The ordinal head, the data it is trained on, and the metrics its checkpoint is chosen by.

The head is a linear probe on the item embedding with one shared score and an
increasing threshold per k (a proportional-odds model):

    logit(survives k_j) = w.e + b - t_j,    t_1 = first,  t_j = t_(j-1) + softplus(gap_j)

so the predicted survival probability is non-increasing in k by construction.

A head is trained on one outer fold's training slice minus one of its four inner
folds, and that inner fold is its validation set: the checkpoint is chosen there, and
``assign_k`` reads it as the head's out-of-sample prediction for those items.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from llm.probe import common
from llm.probe.make_splits import deal

INNER_FOLDS = 4
INNER_SEED = 0
DECISION_PENALTIES = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0)
MAE_WEIGHT = 0.25
OVER_WEIGHT = 0.25


class OrdinalHead(nn.Module):
    def __init__(self, input_dim: int, n_outputs: int):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)
        self.first = nn.Parameter(torch.zeros(1))
        self.gaps = nn.Parameter(torch.full((n_outputs - 1,), 0.5413))  # softplus -> 1

    def thresholds(self) -> torch.Tensor:
        steps = nn.functional.softplus(self.gaps)
        return torch.cat([self.first, self.first + torch.cumsum(steps, dim=0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x) - self.thresholds().unsqueeze(0)


def probabilities(module: nn.Module, X: np.ndarray) -> np.ndarray:
    """Survival probability per k, computed in float32 as during training."""
    module.eval()
    with torch.inference_mode():
        logits = module(torch.tensor(X)).float().cpu().numpy()
    module.train()
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


# ------------------------------------------------------------------ data

@dataclass
class Split:
    item_ids: list[str]
    X: np.ndarray
    y: np.ndarray           # label_k, float32, one column per k
    drops: np.ndarray       # drop_k, float32
    categories: list


@dataclass
class Bundle:
    train: Split
    val: Split
    test: Split


def bundle(dataset: str, model: str, fold: int, val_fold: int,
           embeddings: tuple | None = None) -> Bundle:
    """Outer fold ``fold`` held out; inner fold ``val_fold`` of the rest is validation."""
    labels = pd.read_csv(common.labels_file(dataset, model), dtype={"question_id": str})
    labelled = labels.set_index("question_id")
    rungs = common.ladder(dataset)
    label_cols = [f"label_k{k}" for k in rungs]
    drop_cols = [f"drop_k{k}" for k in rungs]
    matrix, rows = embeddings or common.load_embeddings(dataset)

    has_label = labelled[label_cols].notna().any(axis=1)
    train_ids, test_ids = common.load_splits(dataset, fold)

    def usable(ids):
        return [i for i in ids if i in labelled.index and bool(has_label.get(i, False))
                and i in rows]

    train_ids, test_ids = usable(train_ids), usable(test_ids)
    inner = deal(train_ids, [labelled.loc[i, "category"] for i in train_ids],
                 folds=INNER_FOLDS, seed=INNER_SEED)
    val_ids = [i for i in train_ids if inner[i] == val_fold]
    train_ids = [i for i in train_ids if inner[i] != val_fold]

    def split(ids):
        return Split(ids, matrix[[rows[i] for i in ids]].astype(np.float32),
                     labelled.loc[ids, label_cols].to_numpy(dtype=np.float32),
                     labelled.loc[ids, drop_cols].to_numpy(dtype=np.float32),
                     labelled.loc[ids, "category"].tolist())

    return Bundle(split(train_ids), split(val_ids), split(test_ids))


def positive_weights(y: np.ndarray) -> np.ndarray:
    """Per k, n_negative / n_positive on the training slice (1 where a class is empty)."""
    out = np.ones(y.shape[1], dtype=np.float32)
    for column in range(y.shape[1]):
        values = y[:, column]
        values = values[~np.isnan(values)]
        positives, negatives = float((values >= 0.5).sum()), float((values < 0.5).sum())
        if positives > 0 and negatives > 0:
            out[column] = negatives / positives
    return out


# ------------------------------------------------------------------ selection metrics
#
# The checkpoint is chosen on the validation objective
#     macro AUC - 0.25 * MAE in ladder steps - 0.25 * max(0, mean signed step error)
# where the step error is that of the "decision" readout: per-k logit shifts fitted on
# the training slice (so its predicted positive rate matches its base rate), then the
# rung minimising an asymmetric expected step error whose over-prediction penalty is
# also fitted on the training slice.

_EPS = 1e-15


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def auc_macro(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    aucs = []
    for column in range(y.shape[1]):
        seen = ~np.isnan(y[:, column])
        truth = y[seen, column]
        if truth.size and len(np.unique(truth)) == 2:
            aucs.append(float(roc_auc_score(truth, p[seen, column])))
    return float(np.mean(aucs)) if aucs else float("nan")


def auc_by_k(p: np.ndarray, y: np.ndarray, rungs) -> dict[str, float]:
    return {str(k): auc_macro(p[:, [j]], y[:, [j]]) for j, k in enumerate(rungs)}


def base_rate_shifts(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per k, minus the logit of the cut whose positive rate matches the base rate."""
    cuts = np.full(p.shape[1], 0.5)
    for column in range(p.shape[1]):
        seen = ~np.isnan(y[:, column])
        truth = y[seen, column]
        if truth.size and len(np.unique(truth)) == 2:
            cuts[column] = float(np.quantile(p[seen, column], 1.0 - truth.mean()))
    return -_logit(cuts)


def calibrate(p: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(_logit(p) + shifts.reshape(1, -1))))


def largest_true_k(binary: np.ndarray, rungs) -> np.ndarray:
    values = np.nan_to_num(np.asarray(binary, dtype=float), nan=0.0) >= 0.5
    out = np.zeros(values.shape[0], dtype=int)
    for column, k in enumerate(rungs):
        out = np.where(values[:, column], k, out)
    return out


def _steps(k_values: np.ndarray, rungs) -> np.ndarray:
    index = {int(k): i for i, k in enumerate((0,) + tuple(rungs))}
    return np.array([index[int(k)] for k in k_values], dtype=int)


def decide(p: np.ndarray, penalty: float, rungs) -> np.ndarray:
    """The k minimising the expected step error, with over-prediction weighted by penalty."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    n, m = p.shape
    none_above = np.ones((n, m + 1))
    for column in range(m - 1, -1, -1):
        none_above[:, column] = none_above[:, column + 1] * (1.0 - p[:, column])
    dist = np.empty((n, m + 1))
    dist[:, 0] = none_above[:, 0]
    for column in range(m):
        dist[:, column + 1] = p[:, column] * none_above[:, column + 1]
    steps = np.arange(m + 1, dtype=float)
    diff = steps.reshape(-1, 1) - steps.reshape(1, -1)
    cost = np.maximum(diff, 0.0) * float(penalty) + np.maximum(-diff, 0.0)
    return np.asarray((0,) + tuple(rungs), dtype=int)[np.argmin(dist @ cost.T, axis=1)]


def fit_penalty(p_cal: np.ndarray, y: np.ndarray, rungs) -> float:
    """Lowest training MAE; if it over-predicts on average, the best one that does not."""
    truth = _steps(largest_true_k(y, rungs), rungs)
    scanned = []
    for penalty in DECISION_PENALTIES:
        error = _steps(decide(p_cal, penalty, rungs), rungs) - truth
        scanned.append((float(np.abs(error).mean()), penalty, float(error.mean())))
    by_mae = sorted(scanned, key=lambda row: (row[0], row[1]))
    if by_mae[0][2] <= 0.0:
        return by_mae[0][1]
    lower = [row for row in by_mae if row[2] <= 0.0]
    return lower[0][1] if lower else max(DECISION_PENALTIES)


def val_objective(p_train, y_train, p_val, y_val, rungs) -> float:
    p_train, p_val = np.asarray(p_train, dtype=float), np.asarray(p_val, dtype=float)
    y_train, y_val = np.asarray(y_train, dtype=float), np.asarray(y_val, dtype=float)
    shifts = base_rate_shifts(p_train, y_train)
    penalty = fit_penalty(calibrate(p_train, shifts), y_train, rungs)
    predicted = decide(calibrate(p_val, shifts), penalty, rungs)
    error = _steps(predicted, rungs) - _steps(largest_true_k(y_val, rungs), rungs)
    return (auc_macro(p_val, y_val) - MAE_WEIGHT * float(np.abs(error).mean())
            - OVER_WEIGHT * max(0.0, float(error.mean())))
