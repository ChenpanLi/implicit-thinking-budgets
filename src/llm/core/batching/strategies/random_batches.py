"""The baseline: shuffle the pool and cut it into batches of k.

This is the baseline fragility-aware batching is measured against, and it is
deliberately the simplest thing that satisfies the contract -- one uniform
shuffle of the whole pool, cut into consecutive runs of ``k``. It conditions on
nothing: not on difficulty, not on category, not on the model.

Two properties are worth stating, because both are easy to lose:

* **The short tail is kept.** A pool of 1548 items at k=10 gives 154 batches of
  10 and one of 8, and that last batch is emitted as it is. Dropping it would
  break the partition -- eight items would go ungraded, and the comparison
  between strategies would no longer be over the same corpus. Padding it by
  reusing items would break it the other way. So the batch is simply short, and
  its ``k`` field says 8, which is the true size of the prompt the model saw.
* **Each k is drawn independently.** ``--k 3 6 10 20`` yields four passes per
  seed, each its own full partition, each shuffled from its own derived stream
  (``ctx.derive(k)``). Consuming one shared stream in sequence instead would
  mean that adding k=50 to a later sweep silently re-drew the k=3 partition.
"""
from __future__ import annotations

from collections.abc import Iterator

from ..registry import BatchingContext, Pass, strategy

# What ``--k`` defaults to when none is given (`llm.batches` passes the dataset's
# random_ks). A single value: every k is a full pass over the pool.
DEFAULT_K = [10]


@strategy(
    "random",
    description="Shuffle the whole pool and cut it into batches of exactly k "
                "(the last batch of a pass may be shorter).",
)
def random_batches(ctx: BatchingContext) -> Iterator[Pass]:
    """One pass per requested k: a uniform shuffle cut into runs of k."""
    n = len(ctx.items)
    for k in ctx.int_list_param("k", DEFAULT_K):
        if k < 1:
            raise SystemExit(f"--k must be at least 1, got {k}")
        if k > n:
            raise SystemExit(
                f"cannot batch k={k} from a pool of {n} items; every pass must "
                "cover the pool, so k cannot exceed it"
            )
        order = list(range(n))
        # A stream of its own per k -- see the module docstring.
        ctx.derive(k).shuffle(order)
        groups = [order[start:start + k] for start in range(0, n, k)]
        yield Pass(
            groups=groups,
            label=f"k{k}",
            nominal_k=k,
            info={
                "shuffle": "uniform",
                "n_full_batches": sum(1 for g in groups if len(g) == k),
                "tail_size": len(groups[-1]) if groups[-1] and len(groups[-1]) != k else None,
            },
        )
