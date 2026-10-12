"""The batching-strategy registry, and the contract a strategy is written to.

A **batching strategy** answers one question: given the whole item pool, which
items are shown to the model together in one prompt? It is deliberately not
told anything about prompts, grading or lm-eval -- it returns groups of pool
*indices* and nothing else, so the same strategy runs unchanged over integrals,
multiple-choice questions and one-word questions.

Why the whole pool, and why a partition
---------------------------------------
The composed sets built by ``python -m llm.compose`` *draw* compositions:
they sample from the pool of items the model already solves reliably, and one
item may appear in many compositions or in none. That is the right shape for
asking how accuracy falls with k, and the wrong shape for asking which batching
gets the most out of a fixed corpus -- a strategy could then look good by
quietly batching the easy items twice and the hard ones never.

So the unit here is a **pass**: one partition of the entire pool in which every
item appears exactly **once**. Every strategy is then graded on the same items,
and :func:`llm.core.batching.engine.validate_pass` refuses anything else.
Nothing conditions on previous success; the pool is the whole pool.

A strategy yields as many passes as it likes. That is what carries the sweep
dimension a strategy chooses for itself:

* the ``random`` baseline yields one pass per requested k, so ``--k 3 6 10 20``
  puts four full partitions of the pool into one file;
* ``predicted-calibration`` sizes each batch from a per-item k and yields one
  pass per cross-validation fold, whose groups have no common size at all.

k is therefore a *result* of a pass, never an input the engine imposes: a group
is as long as the strategy made it, and ``nominal_k`` is only the label a
strategy that happens to have a target size chooses to record.

A strategy is one function decorated with :func:`strategy`, taking a
:class:`BatchingContext` and yielding :class:`Pass` objects; see
:mod:`llm.core.batching.strategies` for the four the paper uses.
"""
from __future__ import annotations

import importlib
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# The package holding the strategies.
BUILTIN_PACKAGE = "llm.core.batching.strategies"


@dataclass(frozen=True)
class Pass:
    """One partition of the whole pool: every item in exactly one group.

    ``groups`` holds indices into ``BatchingContext.items``, in the order the
    items should appear in the prompt -- position within a group is meaningful
    and is recorded, because every per-position analysis in this repo keys on
    it.

    ``label`` names the pass inside the file. It becomes part of every row's
    id, so it must be unique among the passes of one seed and safe in an
    id-shaped string. ``nominal_k`` is the size the strategy was aiming for, or
    ``None`` when it was not aiming for one; the *actual* k of a composition is
    always ``len(group)``. ``info`` is free-form and lands in the manifest,
    which is where a strategy explains itself -- the thresholds it used, the
    statistics it read.

    ``covers`` and ``stamp`` are for the unusual pass: the first narrows the set
    of items this pass must partition, the second puts extra fields on its rows.
    Both are documented on the fields themselves.
    """

    groups: Sequence[Sequence[int]]
    label: str = ""
    nominal_k: int | None = None
    info: Mapping[str, Any] = field(default_factory=dict)
    # Which pool indices this pass is a partition *of*. ``None`` -- the normal
    # case -- means the whole pool, and that is what the comparison between
    # strategies rests on. A pass that declares a subset is saying "this pass is a complete
    # partition of these items and of nothing else", and the engine holds it to
    # exactly that: every declared index once, no index from outside.
    #
    # It exists for a strategy whose passes are several *separate datasets* that
    # must not be batched into one another -- `predicted-calibration`, whose
    # passes are the held-out sets of the cross-validation folds. Batching across
    # them would put an item into a prompt sized by a predictor that had been
    # trained on that very item.
    covers: Sequence[int] | None = None
    # Extra fields stamped verbatim onto every row this pass produces, on top of
    # the engine's own identity. It is how a pass says what it *is* -- which
    # fold, which arm -- so that a run's own records carry it and no downstream
    # reader has to parse it back out of the label. A name that collides with an
    # engine-owned field or with a record builder's output is refused.
    stamp: Mapping[str, Any] = field(default_factory=dict)
    # Which positions of each group are *actual* items of the partition, as a
    # list of booleans parallel to ``groups``. ``None`` -- every strategy but
    # `dummy-tail` -- means every position is actual, which is the ordinary
    # partition. A masked pass may put non-actual items in a group (a dummy
    # question whose answer nobody reads, a filler that keeps a batch at its
    # full size), and those may repeat items the partition already holds; the
    # engine checks the partition over the actual positions alone and stamps
    # the mask on every row as `component_actual`, so the run's own records say
    # which answers count.
    actual: Sequence[Sequence[bool]] | None = None


@dataclass(frozen=True)
class BatchingContext:
    """Everything a strategy is allowed to look at.

    ``rng`` is seeded by the engine and is the *only* source of randomness a
    strategy may use -- reaching for the global ``random`` module would make a
    seed meaningless. ``items`` is the pool in file order, so an index is a
    stable name for an item within one build.
    """

    items: Sequence[Mapping[str, Any]]
    model: str
    seed: int
    rng: random.Random
    params: Mapping[str, Any]
    # llm.core.batching.stats.StatsBundle, typed loosely so this module stays
    # importable (and testable) without touching the filesystem.
    stats: Any
    # The string `rng` was seeded from, so a strategy can derive further
    # independent streams from it reproducibly; see `derive`.
    key: str = ""

    def derive(self, *parts: Any) -> random.Random:
        """A fresh RNG for one sub-stream of this context.

        Use it whenever a strategy yields several passes: drawing them all from
        ``ctx.rng`` in sequence means adding a pass later silently re-draws
        every pass after it, so a k=3 partition would change on the day k=50 was
        added to the sweep. Keying each pass on its own name instead --
        ``ctx.derive(k)`` -- keeps the passes independent of one another.
        """
        suffix = ":".join(str(part) for part in parts)
        return random.Random(f"{self.key}:{suffix}" if suffix else self.key)

    def int_list_param(self, name: str, default: Sequence[int]) -> list[int]:
        """A parameter read as a list of ints, however it was written.

        ``--k 3 6 10 20`` arrives as a list and ``--param k=10`` as a string;
        both mean the same thing to a strategy that sweeps a size.
        """
        value = self.params.get(name)
        if value is None:
            return [int(v) for v in default]
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        return [int(part) for part in str(value).replace(",", " ").split()]


@dataclass(frozen=True)
class Strategy:
    name: str
    fn: Callable[[BatchingContext], Iterable[Pass]]
    description: str
    uses_model: bool
    uses_stats: bool
    # Optional: turns the parameters of one build into a longer name, so that
    # two settings of one strategy are two conditions rather than one. See
    # `instance_name`.
    name_from_params: Callable[[Mapping[str, Any]], str] | None = None

    def __call__(self, ctx: BatchingContext) -> list[Pass]:
        return list(self.fn(ctx))

    def instance_name(self, params: Mapping[str, Any]) -> str:
        """What *this build* is called: the registered name, plus its settings.

        A strategy whose parameters change the partition -- not how many passes
        it makes, but what the batching *is* -- declares ``name_from_params``
        and gets a name like ``predicted-calibration-lagrangian-fb1-k3-6-10-20``.
        That name is the ``strategy`` field on every row, part of every row id and
        the manifest's ``strategy``, so two settings are never averaged into one
        score afterwards.

        ``--strategy`` still takes the *registered* name, which is why the
        instance name is required to start with it: the name a build is filed
        under has to be recognisable as the name that was asked for. The rest
        of it is checked for the same characters a pass label is, since it ends
        up inside every row id.
        """
        if self.name_from_params is None:
            return self.name
        resolved = str(self.name_from_params(dict(params))).strip()
        if not resolved.startswith(self.name):
            raise SystemExit(
                f"strategy {self.name!r} named a build {resolved!r}, which does "
                f"not start with the strategy's own name; --strategy takes the "
                "registered name, so the build must be recognisable as it"
            )
        bad = [c for c in resolved if not (c.isalnum() or c in "-_.")]
        if bad:
            raise SystemExit(
                f"strategy {self.name!r} named a build {resolved!r}, which holds "
                f"{bad[0]!r}; the name is part of every row id, so it may hold "
                "only letters, digits, '-', '_' and '.'"
            )
        return resolved


_REGISTRY: dict[str, Strategy] = {}


def strategy(name: str, *, description: str = "", uses_model: bool = False,
             uses_stats: bool = False,
             name_from_params: Callable[[Mapping[str, Any]], str] | None = None,
             ) -> Callable[[Callable], Callable]:
    """Register a batching strategy under ``name``.

    Registration refuses name collisions outright: two strategies answering to
    one name would make ``--strategy`` ambiguous, which is the kind of mix-up
    only noticed after the runs have finished.

    ``name_from_params`` is for a strategy whose parameters *are* the condition:
    it turns one build's parameters into the longer name that build is filed
    under. See :meth:`Strategy.instance_name`.
    """

    def decorate(fn: Callable[[BatchingContext], Iterable[Pass]]):
        existing = _REGISTRY.get(name)
        if existing is not None and existing.fn is not fn:
            raise ValueError(
                f"batching strategy {name!r} is already registered by "
                f"{existing.fn.__module__}; names must be unique because they "
                "name every row of a build too"
            )
        _REGISTRY[name] = Strategy(
            name=name,
            fn=fn,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            uses_model=uses_model,
            uses_stats=uses_stats,
            name_from_params=name_from_params,
        )
        return fn

    return decorate


def get(name: str) -> Strategy:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none loaded)"
        raise SystemExit(f"unknown batching strategy {name!r}; known: {known}")
    return _REGISTRY[name]


def load_builtin() -> None:
    """Import (and so register) every strategy."""
    importlib.import_module(BUILTIN_PACKAGE)
