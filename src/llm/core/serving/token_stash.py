"""Per-response token-usage capture, bridging lm-eval's generation step to trace records.

The model's native tokenizer is only ever close at hand *during* generation --
inside vLLM's ``RequestOutput`` or an OpenRouter response's ``usage`` object --
and neither backend's ``generate_until`` return value carries anything but a
decoded string. By the time a tree's ``process_results`` runs, the raw text is
all it has (see ``lm_tasks/*/utils.py``'s own comment: "process_results
receives only (doc, results)").

Two ``Stash`` instances close that gap, at two different keying layers:

``by_context``
    Written by the low-level hook inside each backend's own generation call
    (``llm.core.serving.lm_eval_vllm.InProcessVLLM._model_generate`` wrapper,
    ``llm.core.serving.lm_eval_openrouter.OpenRouterChatCompletion.amodel_call``),
    keyed by the exact prompt string -- the only identity either hook has
    access to at that depth, since lm-eval's own internal batching has already
    stripped away everything else (the ``Instance``, its ``doc``) by the time a
    response comes back.

``by_doc``
    What each backend's ``generate_until`` override re-keys ``by_context``
    into afterwards, using the original ``Instance.doc`` -- the same identity
    ``lm_tasks/*/utils.py``'s own ``_STASH`` already uses to reunite a
    document's trials with its raw text. Each task module's ``process_results``
    calls :meth:`Stash.drain` on ``by_doc`` and folds the result into the
    ``extra`` dict passed to its record builder (``build_record`` for integrals,
    the ``trace_records`` module of its family otherwise).

Both are FIFO queues per key, not a single value: a document graded with
``repeats > 1`` produces several responses to the same prompt, one after
another, in the same order both backends process them in, so first-recorded is
first-drained. Trials of the same document are exchangeable draws under an
identical prompt, so if two land in the wrong relative order the mismatch is
invisible -- each trial still carries a real, exact count, just possibly the
sibling trial's.
"""
from __future__ import annotations

from collections import deque
from typing import Any


class Stash:
    """A FIFO queue of values per key."""

    def __init__(self) -> None:
        self._queues: dict[str, deque[Any]] = {}

    def record(self, key: Any, value: Any) -> None:
        self._queues.setdefault(str(key), deque()).append(value)

    def drain(self, key: Any, n: int = 1) -> list[Any | None]:
        """Pop up to ``n`` queued values for this key, FIFO. Missing entries are None."""
        queue = self._queues.get(str(key))
        return [(queue.popleft() if queue else None) for _ in range(n)]

    def drain_one(self, key: Any) -> Any | None:
        return self.drain(key, 1)[0]


by_context = Stash()
by_doc = Stash()


def normalize_usage(usage: Any) -> dict[str, int | None]:
    """One ``token_usage`` value, whatever backend wrote it, as prompt/completion/total.

    The two instrumented backends record different shapes on purpose (see the
    module docstring): vLLM writes its own two exact integers,
    OpenRouter writes its ``usage`` object verbatim. Anything that *aggregates*
    token counts must not have to know which
    backend produced the run, so the mapping onto three names lives here, beside
    the code that writes both shapes.

    A missing field stays ``None`` rather than becoming ``0``: a record whose
    backend never reported a prompt count has an *unknown* prompt cost, and a
    caller that summed it as zero would report a dataset as cheaper to traverse
    than it is. ``total`` is the backend's own when it reports one and otherwise
    the sum of the two halves -- only when both are known, for the same reason.
    """
    if usage is None:
        return {"prompt": None, "completion": None, "total": None}
    if not isinstance(usage, dict):
        # A bare integer is how a replayed or hand-written record might spell a
        # completion count; read it as one rather than drop the record.
        try:
            count = int(usage)
        except (TypeError, ValueError):
            return {"prompt": None, "completion": None, "total": None}
        return {"prompt": None, "completion": count, "total": count}

    def _first(*names: str) -> int | None:
        for name in names:
            value = usage.get(name)
            # `bool` is an `int`; a flag in a usage object is not a count.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            return int(value)
        return None

    prompt = _first("vllm_prompt_tokens", "prompt_tokens", "input_tokens")
    completion = _first("vllm_output_tokens", "completion_tokens", "output_tokens")
    total = _first("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {"prompt": prompt, "completion": completion, "total": total}
