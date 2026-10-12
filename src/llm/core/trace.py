"""Structured JSONL tracing for the verification pipeline.

Deliberately *additive* to what lm-eval already records. lm-eval's
``samples_*.jsonl`` owns the prompt, the raw generations, the filtered
extraction and the per-doc metrics; this module owns the part nothing else
captures -- the stage-by-stage journey from raw text to verdict.

Every record carries an ordered ``stages`` list that is complete up to the
point of failure, so the last entry is by construction the stage that mangled
the answer.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from llm.core.thinking_text import strip_response

SCHEMA_VERSION = 1

ENV_PATH = "LMEVAL_TRACE"
ENV_RAW = "LMEVAL_TRACE_RAW"
# Off by default: a reasoning model's thinking is most of its response, so keeping
# it multiplies every run directory. Set to 1 by `--keep-thinking`, which the
# per-term trace needs because its subject *is* the thinking -- see
# `llm.core.thinking_text`.
ENV_THINKING = "LMEVAL_TRACE_THINKING"

# How much raw text to keep when raw capture is disabled: enough to recognise a
# truncation or a template bleed, without carrying megabytes of reasoning.
_HEAD_TAIL = 400


def keep_thinking() -> bool:
    """Does this process record thinking blocks verbatim? Read per call.

    Per call rather than once at import, because a runner may set the variable
    after the task module has been imported, in the same process.
    """
    return os.environ.get(ENV_THINKING, "0") in {"1", "true", "yes"}


class TraceWriter:
    """Append-only JSONL writer. Safe to share across threads."""

    def __init__(self, path: str | Path, include_raw: bool = True) -> None:
        self.path = Path(path)
        self.include_raw = include_raw
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8")

    @classmethod
    def from_env(cls) -> "TraceWriter":
        """Build from environment, defaulting to a per-process file.

        Never returns a no-op writer: a missing environment variable means the
        harness was invoked directly rather than through ``python -m llm.run``,
        and silently dropping traces is the failure mode this whole module
        exists to prevent.
        """
        path = os.environ.get(ENV_PATH) or f"logs/trace-{os.getpid()}.jsonl"
        include_raw = os.environ.get(ENV_RAW, "1") not in {"0", "false", "no"}
        return cls(path, include_raw=include_raw)

    def emit(self, record: dict[str, Any]) -> None:
        payload = {"schema": SCHEMA_VERSION,
                   "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
                   **record}
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()  # a crashed run must still leave a usable trace
            os.fsync(self._handle.fileno())


def describe_raw(text: str | None, include_raw: bool = True,
                 include_thinking: bool | None = None) -> dict[str, Any]:
    """Summarise a raw model response, with the full text when enabled.

    The full text duplicates lm-eval's ``resps`` on purpose: it keeps
    ``trace.jsonl`` self-contained, so debugging never requires joining two
    files. ``include_raw=False`` trades that for roughly a 10x size reduction.

    The model's *thinking block* is removed from the recorded text unless
    ``include_thinking`` says otherwise (defaulting to :func:`keep_thinking`,
    i.e. to the environment). Every field beside ``text`` is still measured on
    the whole response -- ``chars`` is its real length, ``sha1`` its real
    identity -- so length figures and the token cache keyed on that digest read
    a stripped trace exactly as they read a full one. ``thinking_chars``
    records what was dropped, and its presence is how a reader knows ``text`` is
    no longer the whole response.
    """
    if text is None:
        return {"text": None, "chars": 0, "empty": True, "sha1": None}
    text = str(text)
    stripped = text.rstrip()
    info: dict[str, Any] = {
        "chars": len(text),
        "sha1": hashlib.sha1(text.encode("utf-8", "replace")).hexdigest(),
        "empty": not text.strip(),
        "has_boxed": "\\boxed" in text,
        # An answer that stops mid-sentence with no closing brace is the
        # signature of hitting the generation cap.
        "truncation_suspected": bool(stripped) and not stripped.endswith(("}", "$", ".", "\\]")),
    }
    if include_thinking is None:
        include_thinking = keep_thinking()
    body, thinking_chars = (text, 0) if include_thinking else strip_response(text)
    if thinking_chars:
        info["thinking_chars"] = thinking_chars
    if include_raw:
        info["text"] = body
    else:
        info["head"] = body[:_HEAD_TAIL]
        info["tail"] = body[-_HEAD_TAIL:] if len(body) > _HEAD_TAIL else ""
    return info


# The identity a batching pass stamps on every row it writes
# (`llm.core.batching.engine.IDENTITY_FIELDS`). All four record builders lift
# these onto the record itself, because a batching run's numbers are read *by*
# them: a run of the `<prefix>_evaluation` task holds every seed and every k in
# one file and one task, so without `batch_seed` on the record there is nothing
# in the trace to group a per-seed total by, and the task name says nothing about
# k there.
# `fold` is not stamped by the engine but by a pass (`Pass.stamp`): it is there
# only for a strategy whose passes are separate datasets -- `predicted-calibration`,
# whose passes are the cross-validation folds' held-out sets. A run over
# such a file has to say which fold each answer belongs to, and the alternative
# is parsing it back out of the pass label, which is exactly the kind of
# re-derivation that is silent when it goes wrong.
BATCHING_FIELDS = ("stage", "strategy", "batch_seed", "batch_pass",
                   "batch_index", "nominal_k", "fold",
                   # a masked pass's actual positions (`dummy-tail`)
                   "component_actual", "n_actual")


def copy_batching_identity(record: dict[str, Any], doc: dict[str, Any]) -> None:
    """Lift whichever of :data:`BATCHING_FIELDS` this document has onto ``record``.

    Absent fields stay absent: a composed-set document lacks most of them, and
    writing nulls would make every reader distinguish "not a batching run" from
    "seed null" itself.
    """
    for name in BATCHING_FIELDS:
        if (value := doc.get(name)) is not None:
            record[name] = value


def build_record(
    *,
    run_id: str,
    task: str,
    doc_id: int | str,
    filter_name: str,
    trial: int,
    doc: dict[str, Any],
    raw_text: str | None,
    result: Any,  # CheckResult, kept loose to avoid a circular import
    include_raw: bool = True,
    include_thinking: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one trace record joining lm-eval's identity keys to our stages."""
    record: dict[str, Any] = {
        "run_id": run_id,
        "task": task,
        "doc_id": doc_id,
        "filter": filter_name,
        "trial": trial,
        "problem_id": doc.get("id"),
        "integrand": doc.get("integrand"),
        "raw": describe_raw(raw_text, include_raw, include_thinking),
    }
    if (k := doc.get("k")) is not None:
        record["k"] = k
    if (components := doc.get("component_ids")) is not None:
        record["component_ids"] = components
    if (category := doc.get("category")) is not None:
        record["category"] = category
    copy_batching_identity(record, doc)
    record.update(result.to_dict())
    if extra:
        record.update(extra)
    return record
