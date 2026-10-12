"""The control: no batching at all -- every item asked on its own, k=1.

``random`` is the control for *how* a pool is grouped; this is the control for
whether it is grouped in the first place. It partitions the pool into singleton
groups, so the model sees one item per prompt and the run measures what the
corpus is worth when nothing is composed. That number is the ceiling every
batching strategy is trying to keep: a strategy's score over the whole corpus is
only meaningful next to the score the same corpus gets un-batched, with the same
items, the same template, the same grader and the same seed budget.

Three properties, each of which is why this is a strategy of its own rather than
``random --k 1``:

* **k is 1 by construction, not by parameter.** The pass is the one partition of
  the pool into singletons, so there is nothing to sweep. ``--k`` is refused
  unless it is 1, because a control that quietly batched by ten would be
  indistinguishable, in the run directory, from one that did not.
* **Its own stages.** Its batches and runs are the ``isolated-a`` and
  ``isolated-b`` stages, so the un-batched baseline is never mixed in with a
  batching's runs -- which is exactly what ``random --k 1`` would do.
* **The seeds are repeats, not re-draws.** There is only one way to cut a pool
  into singletons, so every seed yields the same partition and asks the same
  prompts again. That is what makes it comparable to a batched file built with
  the same ``--seeds``: the same items are attempted the same number of times,
  and the difference between the two scores is the batching and nothing else.
  The order is pool order at every seed, so a given ``batch_index`` names the
  same item in all of them and the seeds can be compared item by item.

The cost side of the comparison follows from the same fact: this strategy sends
one request per item, which is the most expensive way there is to traverse the
corpus. What a batching buys is measured against it.
"""
from __future__ import annotations

from collections.abc import Iterator

from ..registry import BatchingContext, Pass, strategy


@strategy(
    "control",
    description="No batching at all: every item asked on its own (k=1), the "
                "un-batched baseline every strategy is measured against.",
)
def control_batches(ctx: BatchingContext) -> Iterator[Pass]:
    """One pass per seed: the whole pool cut into singletons, in pool order."""
    requested = ctx.int_list_param("k", [1])
    if requested != [1]:
        raise SystemExit(
            "the control strategy simulates no batching at all, so its k is "
            f"always 1; drop --k (or pass --k 1) instead of {requested}"
        )
    n = len(ctx.items)
    yield Pass(
        groups=[[index] for index in range(n)],
        label="k1",
        nominal_k=1,
        info={
            "batching": "none -- one item per prompt",
            "order": "pool order, identical at every seed, so one batch_index "
                     "names the same item across seeds",
            "n_batches": n,
        },
    )
