"""Train the ordinal heads of one (dataset, model): 5 outer folds x 4 inner folds (CPU).

    python -m llm.probe.train --dataset gsm8k --model qwen3-8b                    # 20 heads
    python -m llm.probe.train --dataset gsm8k --model qwen3-8b --fold 0 --val-fold 0

The recipe of the paper: the item embedding as the only input, a class-balanced
binary cross-entropy over every k, AdamW at lr 3e-3 and weight decay 1.0 with a cosine
schedule, 2000 steps of batch 64, gradient clipping at 1.0, seed 0. Every 100 steps the
head is scored on its inner validation fold and the best checkpoint is kept (see
:func:`llm.probe.head.val_objective`).

Each head goes to ``$RUN_DIR/probe/heads/<prefix>-<ds>-<model>-fold<F>/``:
``checkpoints/best.pt`` (what ``assign_k`` reads), ``predictions.csv`` (the held-out
fold as read by the last step), ``metrics.json`` and ``manifest.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from llm.probe import common
from llm.probe import head as head_mod

RECIPE = {
    "features": "Qwen3-Embedding-8B item embedding, no normalisation",
    "head": "ordinal (linear score, increasing thresholds)",
    "loss": "BCE with per-k pos_weight = n_negative / n_positive on the training slice",
    "optimizer": "adamw", "lr": 3e-3, "weight_decay": 1.0, "betas": [0.9, 0.999],
    "scheduler": "cosine", "steps": 2000, "batch_size": 64, "grad_clip": 1.0,
    "eval_every": 100, "seed": 0,
    "select_on": "val_objective = val macro AUC - 0.25 MAE - 0.25 max(0, signed)",
    "inner_folds": head_mod.INNER_FOLDS, "inner_seed": head_mod.INNER_SEED,
}


def fit(data: head_mod.Bundle, rungs) -> tuple[torch.nn.Module, dict, int, float]:
    """Train one head. Returns the last-step module, the best state, its step and value."""
    torch.manual_seed(RECIPE["seed"])
    np.random.seed(RECIPE["seed"])
    module = head_mod.OrdinalHead(data.train.X.shape[1], data.train.y.shape[1])

    pos_weight = torch.tensor(head_mod.positive_weights(data.train.y))
    k_weights = torch.ones(data.train.y.shape[1])

    def loss_fn(logits, targets):
        mask = ~torch.isnan(targets)
        element = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.nan_to_num(targets, nan=0.0), reduction="none",
            pos_weight=pos_weight)
        weights = mask.float() * k_weights
        return (element * weights).sum() / weights.sum()

    steps = RECIPE["steps"]
    optimizer = torch.optim.AdamW(module.parameters(), lr=RECIPE["lr"],
                                  weight_decay=RECIPE["weight_decay"],
                                  betas=tuple(RECIPE["betas"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1.0 + math.cos(math.pi * min(max(step / steps, 0.0), 1.0))))

    train_x, train_y = torch.tensor(data.train.X), torch.tensor(data.train.y)
    batch_size = min(RECIPE["batch_size"], len(data.train.item_ids))
    generator = np.random.default_rng(RECIPE["seed"])
    best_state, best_step, best_value = None, steps, None
    for step in range(1, steps + 1):
        index = torch.tensor(generator.choice(len(data.train.item_ids), size=batch_size,
                                              replace=False))
        loss = loss_fn(module(train_x[index]), train_y[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), RECIPE["grad_clip"])
        optimizer.step()
        scheduler.step()
        if step % RECIPE["eval_every"] == 0 or step == steps:
            value = head_mod.val_objective(
                head_mod.probabilities(module, data.train.X), data.train.y,
                head_mod.probabilities(module, data.val.X), data.val.y, rungs)
            if np.isfinite(value) and (best_value is None or value > best_value):
                best_state, best_step, best_value = copy.deepcopy(module.state_dict()), step, value
    if best_state is None:
        best_state = copy.deepcopy(module.state_dict())
    return module, best_state, best_step, best_value


def train_one(dataset: str, model: str, fold: int, val_fold: int, embeddings=None) -> None:
    started = time.time()
    rungs = common.ladder(dataset)
    data = head_mod.bundle(dataset, model, fold, val_fold, embeddings)
    module, best_state, best_step, best_value = fit(data, rungs)

    out = common.head_dir(common.PREFIXES[val_fold], dataset, model, fold)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"step": best_step, "model_state": best_state}, out / "checkpoints" / "best.pt")

    last = head_mod.probabilities(module, data.test.X)
    predictions = pd.DataFrame({"question_id": data.test.item_ids,
                                "category": data.test.categories})
    for j, k in enumerate(rungs):
        predictions[f"p_k{k}"] = last[:, j]
        predictions[f"label_k{k}"] = data.test.y[:, j]
        predictions[f"drop_k{k}"] = data.test.drops[:, j]
    predictions["true_largest_k"] = head_mod.largest_true_k(data.test.y, rungs)
    predictions.to_csv(out / "predictions.csv", index=False)

    module.load_state_dict(best_state)
    best_test = head_mod.probabilities(module, data.test.X)
    best_val = head_mod.probabilities(module, data.val.X)
    metrics = {
        "best_step": best_step,
        "best_val_objective": best_value,
        "val_auc_macro": head_mod.auc_macro(best_val.astype(float), data.val.y),
        "test_auc_macro": head_mod.auc_macro(best_test.astype(float), data.test.y),
        "test_auc_by_k": head_mod.auc_by_k(best_test.astype(float), data.test.y, rungs),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {
        "head": out.name, "dataset": dataset, "model": model, "fold": fold,
        "val_fold": val_fold, "k_ladder": list(rungs),
        "n_train": len(data.train.item_ids), "n_val": len(data.val.item_ids),
        "n_test": len(data.test.item_ids),
        "labels": str(common.labels_file(dataset, model)),
        "splits": str(common.splits_file(dataset)),
        "embeddings": str(common.embeddings_file(dataset)),
        "recipe": RECIPE,
        "command": "python -m llm.probe.train " + " ".join(sys.argv[1:]),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  {out.name}: best step {best_step}, val objective {best_value:.4f}, "
          f"held-out AUC {metrics['test_auc_macro']:.3f} ({time.time() - started:.0f}s)",
          flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, nargs="*", default=list(range(common.N_FOLDS)))
    parser.add_argument("--val-fold", type=int, nargs="*",
                        default=list(range(head_mod.INNER_FOLDS)))
    args = parser.parse_args(argv)
    embeddings = common.load_embeddings(args.dataset)
    for fold in args.fold:
        for val_fold in args.val_fold:
            train_one(args.dataset, args.model, fold, val_fold, embeddings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
