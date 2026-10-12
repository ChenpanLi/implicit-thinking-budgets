"""Budget-forced thinking: making a reasoning model think for a set token count.

Three pieces, none of which knows what the questions are about:

* :mod:`llm.core.thinking.delimiters` -- where a model's thinking block begins
  and ends, resolved from its tokenizer.
* :mod:`llm.core.thinking.budget` -- the forcing loop itself (continue with
  ``Wait``, close by force, cue the answer), under a :class:`BudgetMode` that
  says whether the budget is a floor, a cap or both.
* :mod:`llm.core.thinking.runner` -- one model, one set of documents each capped
  at its own budget, written out as an ordinary run directory.

``python -m llm.budget_forcing run`` is the entry point. The submodules import
vLLM lazily, so importing this package costs nothing on a login node.
"""
