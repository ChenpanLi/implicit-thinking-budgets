"""Batching strategies: how the pool's items are grouped into prompts.

The composed sets (:mod:`llm.compose`) *draw* k-term compositions from the
items a model already solves reliably, to ask how accuracy falls with k. This
package asks a different question -- given a fixed corpus and a model, which
grouping of that corpus into prompts gets the most out of it? -- and so it works
the other way round: no conditioning on previous success, the whole pool, and
every item in exactly one composition per pass, which is what makes two
strategies' scores comparable.

The three pieces:

* :mod:`llm.core.batching.registry` -- the ``@strategy`` decorator, the
  :class:`BatchingContext` a strategy is given and the :class:`Pass` it yields.
* :mod:`llm.core.batching.engine` -- the seed loop, the partition check, the
  identity fields and the manifest.
* :mod:`llm.core.batching.stats` -- the tables a strategy may condition on.

The strategies live in :mod:`llm.core.batching.strategies`: ``control`` (k=1),
``random`` (uniform batches of k), ``dummy-tail`` (the per-term trace batches) and
``predicted-calibration`` (fragility-aware batching). ``python -m llm.batches``
builds a batch file from one of them.
"""
from .engine import manifest_beside
from .stats import StatsBundle

__all__ = ["StatsBundle", "manifest_beside"]
