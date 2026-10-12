"""Removing a model's thinking block from text that is about to be recorded.

A reasoning model spends most of its response thinking, and only the per-term
trace reads that thinking: the verdict comes from the answer, the stage-by-stage story of how the answer was parsed is in
``stages``, and the length of the response is already recorded as a character
count. What the thinking does do is dominate the run directories -- a
``trace.jsonl`` of half a gigabyte for a few thousand documents is the normal
outcome, and nearly all of it is the same chain of thought stored twice, once
here and once in lm-eval's ``samples_*.jsonl``.

So every writer in this repository strips the thinking block before it writes,
leaving a one-line placeholder that records how much was removed. The exception
is a run with ``--keep-thinking`` (the per-term trace), whose subject *is* the
thinking: it turns stripping off (see ``ENV_THINKING`` in :mod:`llm.core.trace`),
and :mod:`llm.sections` refuses a trace that holds the placeholder.

This module is deliberately free of dependencies -- it is imported by the task
layer, which runs inside lm-eval.

Marker families are the text halves of :data:`llm.core.thinking.delimiters.FAMILIES`,
plus one this module carries alone; the delimiters module resolves the same
markers as *tokens*, which is what budget forcing needs and what this module has
no use for.

The extra one is gpt-oss's **harmony** syntax. Its generation prompt ends at
``<|start|>assistant``, so the response opens with
``<|channel|>analysis<|message|>``, ends the reasoning with ``<|end|>``, and
starts the answer with ``<|start|>assistant<|channel|>final<|message|>``. The
close marker taken here is ``<|channel|>final<|message|>``, so the ``<|end|>``
and the second ``<|start|>assistant`` are removed with the block and what
survives is the answer alone. It is absent from ``delimiters.FAMILIES`` on
purpose: that module accepts a family only when each marker is a *single* token,
and these are three tokens apiece, so budget forcing on gpt-oss would need the
explicit ``--think-open`` / ``--think-close`` overrides. Stripping has no such
constraint, since it matches text.
"""
from __future__ import annotations

import re

# (opening marker, closing marker), searched in this order. Written stripped of
# the surrounding whitespace the tokenizer forms carry, because here they are
# matched against text rather than encoded.
MARKERS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<|channel>thought", "<channel|>"),
    ("<|channel|>analysis<|message|>", "<|channel|>final<|message|>"),
)

# What a removed block leaves behind. Machine-readable on purpose: a reader of a
# stripped trace can tell "the model thought and it was not kept" from "the
# model produced nothing".
PLACEHOLDER = "[thinking removed: {chars} chars]"

_PLACEHOLDER_RE = re.compile(r"\[thinking removed: \d+ chars\]")


def has_placeholder(text: str) -> bool:
    """Has the thinking already been removed from this text?"""
    return bool(_PLACEHOLDER_RE.search(text))


def strip_thinking(text: str) -> tuple[str, int]:
    """`text` with its thinking blocks replaced by placeholders, and the cost.

    Returns the rewritten text and the number of characters removed (zero when
    the text held no thinking, in which case the text is returned unchanged and
    is identical to its input).

    Three shapes occur in practice and all three are handled:

    * ``<think> ... </think> answer`` -- the model opened and closed its block.
    * ``... </think> answer`` -- the chat template ended the prompt with the
      opening marker, so the response *starts* inside the block. Everything up
      to the first close is thinking.
    * ``<think> ...`` with no close -- the generation cap was hit mid-thought.
      Everything after the opener is thinking, and the response has no answer.

    The convention for the second shape is the *first* closing marker: a model
    that writes ``</think>`` again inside its answer -- which several of these
    do -- would otherwise keep a block that had already ended.
    """
    if not text:
        return text, 0
    removed = 0
    for open_marker, close_marker in MARKERS:
        if open_marker not in text and close_marker not in text:
            continue
        out: list[str] = []
        rest = text
        while rest:
            open_at = rest.find(open_marker)
            close_at = rest.find(close_marker)
            if open_at < 0 and close_at < 0:
                break
            if close_at >= 0 and (open_at < 0 or close_at < open_at):
                # A close with no opener before it: the block was opened by the
                # prompt, so the response begins inside it.
                block, rest = rest[:close_at + len(close_marker)], rest[close_at + len(close_marker):]
            else:
                tail = rest[open_at + len(open_marker):]
                end = tail.find(close_marker)
                stop = len(rest) if end < 0 else open_at + len(open_marker) + end + len(close_marker)
                out.append(rest[:open_at])
                block, rest = rest[open_at:stop], rest[stop:]
            removed += len(block)
            out.append(PLACEHOLDER.format(chars=len(block)))
        out.append(rest)
        text = "".join(out)
    return text, removed


def strip_response(text: str | None) -> tuple[str | None, int]:
    """:func:`strip_thinking` over a value that may be absent or not a string.

    ``None`` -- a trial whose response never arrived -- passes through, since
    that is a fact about the run and not a large string.
    """
    if text is None:
        return None, 0
    return strip_thinking(str(text))


def strip_samples(samples: dict[str, list[dict]]) -> int:
    """Strip the thinking out of lm-eval's in-memory samples, in place.

    Only ``resps`` -- the raw generations -- carry thinking; ``filtered_resps``
    is what the extraction ensembles pulled out of them and is an answer
    already. Returns the characters removed, for the line the runner prints.

    Called between ``simple_evaluate`` returning and the evaluation tracker
    writing ``samples_*.jsonl``, which is the only point at which those files
    can be affected: lm-eval writes them itself, from exactly this structure.
    """
    removed = 0
    for task_samples in samples.values():
        for sample in task_samples:
            resps = sample.get("resps")
            if not isinstance(resps, list):
                continue
            for i, per_instance in enumerate(resps):
                if isinstance(per_instance, list):
                    for j, trial in enumerate(per_instance):
                        if isinstance(trial, str):
                            per_instance[j], cost = strip_thinking(trial)
                            removed += cost
                elif isinstance(per_instance, str):
                    resps[i], cost = strip_thinking(per_instance)
                    removed += cost
    return removed
