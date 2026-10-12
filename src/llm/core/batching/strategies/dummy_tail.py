"""Batches whose last position is a dummy question: k-1 actual items, then one throwaway.

Built for measuring per-question thinking tokens with question labels
(``--section-markers``, the per-term-trace stage). A label-cut
trace attributes everything after the last question's label to that last
question -- the answer recap, the format check, the wrap-up -- so the last
position's token count is always inflated. This strategy therefore spends the
last position of every prompt on a *dummy* item whose count is thrown away, and
the partition lives in positions 1..k-1.

The contract, which the engine enforces through the pass's ``actual`` mask:

* **Positions 1..k-1 of one pass partition the pool.** Every item is an actual
  item exactly once per pass, so the dataset can be rebuilt from the actual
  positions alone.
* **Position k is never actual.** The dummy is drawn uniformly from the pool
  minus the items already in its prompt.
* **No short batches.** Every prompt has exactly k items, because a smaller
  prompt would be measured under a different load from the rest. When the pool
  does not divide by k-1, the last batch holds the r leftover actual items at
  r random positions among 1..k-1. The other k-1-r slots take distinct
  *fillers* -- random pool items marked non-actual, which may repeat items
  batched elsewhere in the pass but never one in their own prompt -- and the
  dummy takes position k as usual.

Every row carries ``component_actual`` (one flag per position, in prompt order)
and ``n_actual``, stamped by the engine and lifted onto every trace record, so a
run's own output says which answers count. Grading is untouched: ``acc`` pools
actual, filler and dummy positions alike, and downstream processing filters on
the mask.

Each k is drawn from its own derived stream, as in ``random``, so adding a k to
a later build never re-draws another.
"""
from __future__ import annotations

from collections.abc import Iterator

from ..registry import BatchingContext, Pass, strategy

DEFAULT_K = [3]


@strategy(
    "dummy-tail",
    description="k-1 actual items per prompt, partitioning the pool, plus a "
                "random dummy item in the last position; a short tail is "
                "filled to full size with non-actual fillers.",
)
def dummy_tail(ctx: BatchingContext) -> Iterator[Pass]:
    """One pass per requested prompt size k: chunks of k-1 plus a dummy."""
    n = len(ctx.items)
    for k in ctx.int_list_param("k", DEFAULT_K):
        if k < 2:
            raise SystemExit(f"--k is the prompt size including the dummy, so it "
                             f"must be at least 2, got {k}")
        if k > n:
            raise SystemExit(f"cannot build prompts of {k} distinct items from a "
                             f"pool of {n}")
        rng = ctx.derive(k)
        order = list(range(n))
        rng.shuffle(order)
        width = k - 1
        groups: list[list[int]] = []
        masks: list[list[bool]] = []
        tail_actual = None
        n_fillers = 0
        for start in range(0, n, width):
            chunk = order[start:start + width]
            if len(chunk) == width:
                slots = list(chunk)
                mask = [True] * width
            else:
                # The short tail: r actual items at random slots, the rest
                # filled with distinct non-actual items from outside the chunk.
                tail_actual = len(chunk)
                positions = sorted(rng.sample(range(width), len(chunk)))
                taken = set(chunk)
                fillers = rng.sample([i for i in range(n) if i not in taken],
                                     width - len(chunk))
                n_fillers = len(fillers)
                slots, mask = [], []
                actual_iter, filler_iter = iter(chunk), iter(fillers)
                for slot in range(width):
                    if slot in positions:
                        slots.append(next(actual_iter))
                        mask.append(True)
                    else:
                        slots.append(next(filler_iter))
                        mask.append(False)
            present = set(slots)
            dummy = rng.choice([i for i in range(n) if i not in present])
            groups.append(slots + [dummy])
            masks.append(mask + [False])
        yield Pass(
            groups=groups,
            actual=masks,
            label=f"k{k}",
            nominal_k=k,
            info={
                "actual_per_prompt": width,
                "dummy_position": k,
                "n_batches": len(groups),
                "tail_actual": tail_actual,
                "n_fillers": n_fillers,
            },
        )
