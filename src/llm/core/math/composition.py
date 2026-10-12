"""The domain fields of one integral composition: the sum, and its antiderivative.

Shared by the composed-set builder (:mod:`llm.compose`, which draws
compositions from a solved pool) and the batching builder (:mod:`llm.batches`,
which partitions the whole pool), so the two cannot disagree about what a
composition record holds.

Two properties of this domain that the question-answering trees do not have, and
that this module is written around:

* **The sum is kept unevaluated.** ``sp.Add(*terms, evaluate=False)`` is what
  puts k visibly separate terms in front of the model; letting SymPy evaluate it
  would collapse ``f + f + f`` to ``3*f`` and erase the very composition being
  tested.
* **The antiderivative is summed per atom, never integrated as a whole.**
  Integrating a k=20 composite directly is orders of magnitude slower and can
  fail outright, whereas by linearity the sum of the parts is exact.

Prompt order is *not* list order here, unlike the other datasets: SymPy re-sorts
an ``Add`` when it renders it, so ``component_integrands`` is draw order while
``integrand_latex`` is SymPy's canonical order. Nothing downstream may assume
they line up -- that is what ``llm.core.analysis`` and the per-term grader are
careful about, and it is why this module returns both rather than one.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import sympy as sp


def reference_for(atom: Mapping[str, Any], variable: sp.Symbol,
                  cache: MutableMapping[str, sp.Expr]) -> sp.Expr:
    """Antiderivative of one atom, integrated once and reused.

    Always per-atom: see the module docstring. The cache is the caller's, so
    one build integrates each atom once however many compositions it lands in --
    which matters here in a way it does not elsewhere, since a batching pass
    puts every atom of the pool into some composition.
    """
    atom_id = atom["id"]
    if atom_id not in cache:
        if atom.get("reference_antiderivative"):
            cache[atom_id] = sp.sympify(atom["reference_antiderivative"])
        else:
            cache[atom_id] = sp.integrate(sp.sympify(atom["integrand"]), variable)
    return cache[atom_id]


def composition_fields(chosen: Sequence[Mapping[str, Any]], *,
                       variable: sp.Symbol,
                       cache: MutableMapping[str, sp.Expr] | None = None) -> dict:
    """The integrand, the provenance and the reference antiderivative.

    ``integrand_latex_integral`` is the only problem-bearing field -- the prompt
    is built from it alone -- while ``integrand`` is what the checker verifies
    against.
    """
    if not chosen:
        raise ValueError("a composition must hold at least one term")
    cache = {} if cache is None else cache

    terms = [sp.sympify(atom["integrand"]) for atom in chosen]
    composed = sp.Add(*terms, evaluate=False)
    antiderivative = sp.Add(*[reference_for(atom, variable, cache) for atom in chosen])

    record = {
        # How the same problem is shown to the model. Recorded so a finished run
        # can be attributed without consulting the build command.
        "presentation": "single_integral",
        "component_ids": [atom["id"] for atom in chosen],
        "component_integrands": [atom["integrand"] for atom in chosen],
        "component_categories": [atom.get("category") for atom in chosen],
        "component_antiderivatives": [
            str(reference_for(atom, variable, cache)) for atom in chosen
        ],
        "category_counts": dict(Counter(atom.get("category") for atom in chosen)),
        "integrand": str(composed),
        "integrand_latex": sp.latex(composed),
        # The prompt is built from this field alone; `integrand` is what the
        # checker verifies against.
        "integrand_latex_integral": f"\\int {sp.latex(composed)} \\, d{variable}",
        "reference_antiderivative": str(antiderivative),
        "variable": str(variable),
    }
    return record
