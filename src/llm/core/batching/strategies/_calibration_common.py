"""Tiering a pool by a per-item k, for ``predicted-calibration``.

Given "the largest k this item tolerates" for every item, the pool is cut into
batches tier by tier. The per-item k may be measured (the fragility-measured
selection) or predicted by the probe (the fragility-probe selection); both go
through this same routine, so a difference between their scores is a difference
between a measurement and a prediction, not between two tiering routines.

The algorithm is :func:`assign`, and its one invariant is that the fill runs in
a single direction. Its reasoning is on the function.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

# The one size outside any ladder that a fallback may still name: 1, the
# un-batched control size. It composes nothing, so no measurement or prediction
# is needed to justify it -- which is exactly what is missing for the items a
# calibration cannot speak for. It is also the most expensive fallback there is,
# one request per item, so it is a choice rather than a silent default.
UNBATCHED_K = 1


def one_param(params: Mapping[str, Any], name: str, default, cast, flag: str):
    """One parameter, however it was written on the command line.

    ``--fallback-k 1`` arrives as an int and ``--param fallback_k=1`` as a
    string, and both mean the same thing to a strategy.
    """
    value = params.get(name)
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise SystemExit(f"{flag} takes a single value, got {list(value)!r}")
        value = value[0]
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise SystemExit(f"{flag} must be a number, got {value!r}") from None


def assign(tiers: Mapping[int, Sequence[int]], ks: Sequence[int],
           rng: random.Random) -> tuple[list[list[int]], dict[int, int], list[int]]:
    """Cut each tier into batches of its own k, smallest k first, filling up.

    Returns the groups, how many items each k borrowed from a larger tier, and
    the sizes of any batches that came out short.

    The fill runs one way only and that is the whole invariant: a tier's
    remainder is topped up from the **larger** tiers, nearest first, because an
    item that tolerates k=10 is safe in a batch of 3 while an item that tolerates
    only 3 is not safe in a batch of 10. Nearest-first rather than largest-first
    so that an item is under-batched by as little as possible -- borrowing costs
    tokens, never accuracy.

    A tier of ``k=1`` -- which is where ``--fallback-k 1`` puts the items the
    calibration cannot speak for -- is the same code with nothing to do: batches
    of one always divide the tier evenly, so it never leaves a remainder to fill,
    and being the smallest size it is cut first and is never borrowed *from*.

    The last tier has nothing above it to borrow from, so its remainder is
    emitted short, exactly as ``random`` emits its tail: dropping it would leave
    those items ungraded and padding it by repeating an item would break the
    partition the other way. An earlier tier can end up short too, but only once
    every larger tier is already empty.
    """
    remaining = {int(k): list(tiers.get(int(k), ())) for k in ks}
    for bucket in remaining.values():
        rng.shuffle(bucket)
    groups: list[list[int]] = []
    borrowed = {int(k): 0 for k in ks}
    short: list[int] = []
    for position, k in enumerate(ks):
        bucket = remaining[k]
        while bucket:
            group = [bucket.pop() for _ in range(min(k, len(bucket)))]
            need = k - len(group)
            for larger in ks[position + 1:]:
                while need and remaining[larger]:
                    group.append(remaining[larger].pop())
                    borrowed[k] += 1
                    need -= 1
                if not need:
                    break
            if need:
                short.append(len(group))
            # A borrowed item would otherwise always sit at the end of the batch
            # it filled up, making prompt position a proxy for which tier an
            # item came from; every per-position analysis would then be reading
            # that instead of the batching.
            rng.shuffle(group)
            groups.append(group)
    return groups, borrowed, short


def check_fallback(fallback_k: int, ks: Sequence[int], *, what: str) -> None:
    """Refuse a fallback that is neither one of the pass's sizes nor 1.

    ``what`` names, in the message, which items the fallback is about.
    """
    if fallback_k != UNBATCHED_K and fallback_k not in [int(k) for k in ks]:
        raise SystemExit(
            f"--fallback-k={fallback_k} is neither one of the batch sizes "
            f"{list(ks)} nor {UNBATCHED_K}; an item {what} is either batched at a "
            "size the pass already uses, or asked on its own "
            f"(--fallback-k {UNBATCHED_K}, which composes nothing and so needs no "
            "calibration). Any other size would be a batch size that neither the "
            "ladder nor a measurement put there."
        )
