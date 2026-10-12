"""Per-item occurrences in a composition, and the matching isolated accuracies.

Returns frames and writes nothing, so an analysis can join "how often was this item
right *inside* a k-item prompt" against "how often was it right on its own". It reads
``trace.jsonl`` directly.

The unit of :func:`load_composition_items` is the **occurrence**: one row for each
time an item appeared in a composition on a trial, never one row per item. Nothing
is aggregated on the way in.

:func:`load_sanity_items` reads the k=1 runs of an isolated stage, which re-ask the
items on their own. It carries two lengths beside the accuracy -- how long the
model's answer was when it was asked that item on its own, as ``mean_output_tokens``
(the generation's exact token count, where the run recorded one) and
``mean_thinking_chars`` (the thinking block, in characters, which every run has).
:func:`load_composition_items` carries the same information under the same name, but
it is not a per-item quantity there: a composition is answered with one thinking
block for the whole prompt, so an occurrence's ``thinking_chars`` is the length of
*that composition attempt's* trace, identical across every item sharing it.

Reading the trace shapes
------------------------

The trees do not write the same record, and the differences are silent ones -- a
loader that guesses wrong still returns a full frame of plausible numbers:

* **Where the per-position ids live.** Most trees grade in ``component_ids`` order,
  so ``term_correct[i]`` belongs to ``component_ids[i]``. The integrals tree grades
  in **prompt order**, which SymPy re-sorts when it renders the sum, so
  ``term_correct[i]`` belongs to ``term_ids[i]`` and the two orders disagree on the
  overwhelming majority of documents. ``term_ids`` is therefore preferred wherever
  it exists, and ``component_ids`` is the fallback; ``term_categories`` and
  ``component_categories`` follow the same rule.
* **How a solo trial is graded.** Most trees put a one-element ``term_correct`` on
  a k=1 record; integrals runs graded as a whole answer carry only ``verdict``.
  Both readings are accepted, and they agree wherever both are present (a k=1
  ``verdict`` is ``CORRECT`` exactly when its single term is).

Non-obvious constraints
-----------------------

* **``filter_name`` is not optional in practice.** ``trace.jsonl`` holds one
  record per (doc, trial) *per extraction ensemble*; without narrowing to one
  filter every occurrence appears twice.
* **An attempt whose answer list could not be parsed grades no item.** k-1
  answers cannot be attributed to k items, so such an occurrence has
  ``parsed=False`` and ``correct=None`` -- it is kept, never dropped.
* **A ``term_correct`` list not parallel to the ids is skipped, never
  realigned**, since attributing position i to its neighbour's item is worse
  than dropping the attempt.
* **A joint-graded compositional run has no per-item outcome at all**; it
  reaches here as occurrences that are all unparsed, so a load that grades
  nothing raises rather than returning a frame of NaN.
* **The thinking length is read from what the trace kept, not from its text.**
  The thinking block is stripped before a trace is written, and
  ``raw.thinking_chars`` records what was dropped; that field is preferred
  wherever it exists. A run made with ``--keep-thinking`` has no such field and
  its ``raw.text`` still holds the block, so the length is measured from the text
  instead. Either way it is the length of the whole block in characters. A trial
  whose length cannot be read -- a trace written without the raw text --
  contributes nothing to the mean rather than counting as zero.
* **An item id is shared across models but a model's set is not.** Each model's
  composition set is drawn from its own solved pool, so an id need not appear for
  every model: the model is part of every key here, and :func:`atomic_for`
  reindexes rather than joins, leaving NaN where a model never saw the item.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from llm.core.serving.token_stash import normalize_usage
from llm.core.thinking_text import has_placeholder, strip_response

from llm import paths

OCCURRENCE_COLUMNS = ["model", "run_id", "composition_id", "k", "question_id",
                      "category", "position", "trial", "parsed", "correct",
                      "thinking_chars", "output_tokens"]

# --------------------------------------------------------------------------- #
# run discovery
# --------------------------------------------------------------------------- #
def run_dirs(patterns) -> list[Path]:
    """Run directories from globs; a parent directory is searched recursively.

    Only runs with a ``trace.jsonl`` count: an interrupted run leaves the directory
    behind with nothing in it.
    """
    if isinstance(patterns, (str, Path)):
        patterns = [patterns]
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern), recursive=True))
        if not matches:
            raise SystemExit(f"no path matches {pattern!r}")
        for match in matches:
            path = Path(match)
            if (path / "run.json").is_file():
                candidates = [path]
            elif path.is_dir():
                candidates = sorted({p.parent for p in path.rglob("run.json")})
            else:
                continue
            for candidate in candidates:
                resolved = candidate.resolve()
                if resolved in seen or not (candidate / "trace.jsonl").is_file():
                    continue
                seen.add(resolved)
                found.append(candidate)
    if not found:
        raise SystemExit(
            "no run directories with a trace.jsonl found -- pass directories that "
            "contain run.json (e.g. $RUN_DIR/runs/composed/mmlu/*/*)")
    return found


def model_label(meta: dict) -> str:
    """The model config's stem, the label every figure in this repo keys on."""
    config = meta.get("model_config")
    if config:
        return Path(config).stem
    return (meta.get("model") or {}).get("model_id", "unknown")


# --------------------------------------------------------------------------- #
# reading a trace
# --------------------------------------------------------------------------- #
def _records(run_dir: Path, filter_name: str):
    """The records of one run's trace, in file order, narrowed to one ensemble.

    A line that will not parse is skipped and reported by number rather than
    raised on, because the common cause is benign and recurring: a run still
    being written appends its last record in pieces, so reading it while it runs
    finds one truncated line at the end. Raising there makes every analysis of
    every *other* run in the glob fail on an unrelated job's progress. The count
    and the line numbers are printed so the other cause -- a trace genuinely
    damaged in the middle -- is visible rather than silently thinned.
    """
    broken: list[int] = []
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                broken.append(number)
                continue
            if filter_name and record.get("filter") != filter_name:
                continue
            yield record
    if broken:
        where = ", ".join(str(number) for number in broken[:5])
        more = f" (+{len(broken) - 5} more)" if len(broken) > 5 else ""
        print(f"warning: {run_dir.name}: skipped {len(broken)} unparseable "
              f"trace line(s) at {where}{more} -- a run still being written "
              "leaves one truncated line at the end; several, or any in the "
              "middle, mean a damaged trace")


# --------------------------------------------------------------------------- #
# reading one record
# --------------------------------------------------------------------------- #
def _graded_ids(record: dict) -> tuple[list, list]:
    """The item ids and categories of a record's positions, in *graded* order.

    ``term_ids`` wins wherever the tree writes it, because that tree grades in
    prompt order while its ``component_ids`` are in dataset order: the two agree
    on a handful of documents and disagree on the rest, so preferring the wrong
    one attributes almost every position to a neighbour's item.
    """
    term_ids = list(record.get("term_ids") or ())
    if any(value is not None for value in term_ids):
        return term_ids, list(record.get("term_categories") or ())
    return (list(record.get("component_ids") or ()),
            list(record.get("component_categories") or ()))


def _thinking_chars(record: dict) -> int | None:
    """How many characters of thinking one trial spent, or ``None`` if unknown.

    ``raw.thinking_chars`` is the number the writer recorded when it stripped
    the block, and it is the whole block's length even though ``raw.text`` no
    longer holds it. A run made with ``--keep-thinking`` writes no such field
    and keeps the block in the text, so it is measured there. A trace written
    without the raw text (``include_raw=False``) can say nothing either way and
    returns ``None``, which averages as "not measured" rather than as zero.
    """
    raw = record.get("raw") or {}
    recorded = raw.get("thinking_chars")
    if recorded is not None:
        return int(recorded)
    text = raw.get("text")
    if text is None or has_placeholder(text):
        # No text at all, or a stripped text whose writer left no count: both
        # are "unknown", and reading 0 off a placeholder would be a lie.
        return None
    return strip_response(text)[1]


def _output_tokens(record: dict) -> int | None:
    """How many tokens one trial generated, or ``None`` if the run did not say.

    The exact count from the model's own tokenizer, captured during generation
    and written to the trace as ``token_usage``; the two backends spell it
    differently, so it is read through
    :func:`llm.core.serving.token_stash.normalize_usage` rather than by field
    name. The *completion* half is the one meant by "trace length": the prompt
    half is a property of the batch the item sat in, not of the reasoning the
    model spent on it.

    ``None`` where the field is absent -- a run that did not record it, or a
    rerun that regraded a stored trace without regenerating it -- which
    averages as "not measured" rather than as zero, the same convention
    :func:`_thinking_chars` follows. Characters are always available and are the
    fallback length for such a run.
    """
    return normalize_usage(record.get("token_usage"))["completion"]


def _solo_parsed_correct(record: dict) -> tuple[bool, bool]:
    """``(parsed, correct)`` for a record of one item asked on its own.

    Reads ``term_correct`` where the tree writes it and the whole-answer
    ``verdict`` where it does not. The two agree at k=1 by construction -- a
    one-term attempt is correct exactly when its term is -- so which branch runs
    is a fact about the tree, not about the number.

    **An unparseable reply is a wrong trial here**, not an excluded one: a model
    that returned no usable answer did not answer the item. ``parsed`` records how
    many those were, and nothing drops.
    """
    term_correct = record.get("term_correct")
    if term_correct is not None:
        return bool(term_correct), (bool(term_correct[0]) if term_correct else False)
    verdict = record.get("verdict")
    return verdict != "UNPARSEABLE", verdict == "CORRECT"


# --------------------------------------------------------------------------- #
# occurrences
# --------------------------------------------------------------------------- #
def load_composition_items(patterns, filter_name: str) -> pd.DataFrame:
    """Every occurrence of every item: one row per (composition, trial, position).

    Columns: ``model``, ``run_id``, ``composition_id``, ``k``, ``question_id``
    (the atom/question id, whatever the tree calls its items), ``category``,
    ``position`` (1-based, where in the prompt it sat), ``trial``, ``parsed``
    (the attempt returned a usable answer list at all), ``correct`` (``None``
    when it did not), ``thinking_chars`` -- the length, in characters, of
    the *whole* thinking trace produced for that composition attempt (see
    :func:`_thinking_chars`) -- and ``output_tokens``, the completion-token
    count of that same whole attempt (see :func:`_output_tokens`). Both are
    repeated on every position's row, since one trial's answer is produced for
    the composition and not per item. ``None`` where the length could not be read.
    """
    rows = []
    for run_dir in run_dirs(patterns):
        meta = json.loads((run_dir / "run.json").read_text())
        model, run_id = model_label(meta), meta.get("run_id", run_dir.name)
        for record in _records(run_dir, filter_name):
            atoms, categories = _graded_ids(record)
            if not atoms:
                # A record with no component ids: no per-item occurrence to
                # attribute either way.
                continue
            term_correct = list(record.get("term_correct") or ())
            if term_correct and len(term_correct) != len(atoms):
                continue
            record_k = int(record.get("k") or len(atoms))
            record_thinking_chars = _thinking_chars(record)
            record_output_tokens = _output_tokens(record)
            for position, atom in enumerate(atoms):
                rows.append({
                    "model": model,
                    "run_id": run_id,
                    "composition_id": record.get("problem_id") or record.get("doc_id"),
                    "k": record_k,
                    "question_id": atom,
                    "category": categories[position] if position < len(categories) else None,
                    "position": position + 1,
                    "trial": record.get("trial"),
                    "parsed": bool(term_correct),
                    "correct": bool(term_correct[position]) if term_correct else None,
                    "thinking_chars": record_thinking_chars,
                    "output_tokens": record_output_tokens,
                })

    if not rows:
        raise SystemExit(
            f"no compositional records found (filter={filter_name!r}). Check the "
            "filter name and that the runs are compositional.")

    frame = pd.DataFrame(rows, columns=OCCURRENCE_COLUMNS)
    if not frame["parsed"].any():
        raise SystemExit("every occurrence is unparsed: these runs carry no per-item grading.")
    return frame.sort_values(["model", "question_id", "k", "composition_id", "trial"],
                             kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# isolated accuracy: the k=1 runs
# --------------------------------------------------------------------------- #
def _solo_frame(rows: list[dict], what: str, patterns, filter_name) -> pd.DataFrame:
    """Trials of an item asked on its own, folded to one row per (run, item).

    ``mean_thinking_chars`` averages :func:`_thinking_chars` over the trials it
    could be read from, so it is NaN for a run whose trace kept no raw text and
    0 for a model that did not think. ``mean_output_tokens`` is the same mean of
    :func:`_output_tokens` -- NaN for a run that recorded no ``token_usage`` at
    all -- and ``n_token_trials`` counts how many trials it was averaged over, so
    a partially instrumented run is visible rather than silently thinner.
    """
    if not rows:
        raise SystemExit(
            f"no {what} records found (filter={filter_name!r}) under {patterns!r}")
    trials = pd.DataFrame(rows)
    # Object dtype otherwise, whenever a trial's length was unknown.
    for column in ("thinking_chars", "output_tokens"):
        trials[column] = pd.to_numeric(trials.get(column), errors="coerce")
    solo = (trials.groupby(["model", "run_id", "run_order", "question_id", "category"],
                           dropna=False)
                  .agg(n_trials=("correct", "size"), n_correct=("correct", "sum"),
                       n_unparsed=("parsed", lambda values: int((~values).sum())),
                       mean_thinking_chars=("thinking_chars", "mean"),
                       mean_output_tokens=("output_tokens", "mean"),
                       n_token_trials=("output_tokens", "count"))
                  .reset_index())
    solo["accuracy"] = solo["n_correct"] / solo["n_trials"]
    return solo


def load_sanity_items(tree: str, filter_name: str, patterns=None) -> pd.DataFrame:
    """Per-item accuracy from the k=1 runs of an isolated stage.

    One row per (run, item): ``model``, ``run_id``, ``run_order``, ``question_id``,
    ``category``, ``n_trials``, ``n_correct``, ``n_unparsed``,
    ``mean_thinking_chars``, ``mean_output_tokens``, ``n_token_trials``,
    ``accuracy``. ``patterns`` defaults to every isolated-a run of ``tree``.

    Non-obvious constraints:

    * **The item id is ``component_ids[0]``, never ``problem_id``**, which is the
      k=1 batch's id and renumbers per model/seed.
    * **k must be 1.** A record at any other k is skipped and counted, so
      pointing this at a compositional run yields an explicit complaint rather
      than a per-item accuracy silently averaged over k.
    """
    patterns = patterns or paths.runs_glob("isolated-a", tree)
    rows = []
    skipped_k = 0
    for run_dir in run_dirs(patterns):
        meta = json.loads((run_dir / "run.json").read_text())
        model, run_id = model_label(meta), meta.get("run_id", run_dir.name)
        order = run_dir.stat().st_mtime
        for record in _records(run_dir, filter_name):
            atoms, categories = _graded_ids(record)
            if len(atoms) != 1 or int(record.get("k") or len(atoms)) != 1:
                skipped_k += 1
                continue
            parsed, correct = _solo_parsed_correct(record)
            rows.append({
                "model": model,
                "run_id": run_id,
                "run_order": order,
                "question_id": atoms[0],
                "category": categories[0] if categories else record.get("category"),
                "trial": record.get("trial"),
                "parsed": parsed,
                "correct": correct,
                "thinking_chars": _thinking_chars(record),
                "output_tokens": _output_tokens(record),
            })
    if skipped_k:
        print(f"note: skipped {skipped_k} record(s) that were not k=1 -- "
              f"sanity items are the k=1 runs only")
    return _solo_frame(rows, "k=1 sanity-check", patterns, filter_name)


def _question_ids(items: pd.DataFrame) -> pd.Index:
    """The item ids of a frame, whether they are a column or an index level."""
    if "question_id" in items.columns:
        values = items["question_id"]
    elif isinstance(items.index, pd.MultiIndex) and "question_id" in items.index.names:
        values = items.index.get_level_values("question_id")
    elif items.index.name == "question_id":
        values = items.index
    else:
        raise SystemExit("frame has no question_id column or index level")
    return pd.Index(pd.unique(values), name="question_id")


def atomic_for(model: str, items: pd.DataFrame, atomic: pd.DataFrame) -> pd.DataFrame:
    """A solo-accuracy table narrowed to one model, reindexed onto ``items``' ids.

    Returns one row per distinct item id in ``items``. An item the model has no
    record for stays as a NaN row rather than disappearing. With several runs for
    the model, the most recent is used and said so.
    """
    subset = atomic[atomic["model"] == model]
    if subset.empty:
        raise SystemExit(f"no atomic run for model {model!r}; have: "
                         f"{sorted(atomic['model'].unique())}")
    if subset["run_id"].nunique() > 1:
        newest = subset.sort_values("run_order")["run_id"].iloc[-1]
        print(f"note: {model} has {subset['run_id'].nunique()} atomic runs; "
              f"using {newest}")
        subset = subset[subset["run_id"] == newest]

    subset = subset.set_index("question_id").reindex(_question_ids(items))
    columns = [column for column in
               ["category", "n_trials", "n_correct", "n_unparsed",
                "mean_thinking_chars", "mean_output_tokens", "n_token_trials",
                "accuracy", "run_id"]
               if column in subset.columns]
    return subset[columns]


def compare_atomic_to_composition(items: pd.DataFrame, solo: pd.DataFrame,
                                  model: str | None = None) -> pd.DataFrame:
    """``items`` with a solo accuracy joined on ``question_id``.

    Every row gains the same-item solo columns, named ``atomic_accuracy`` and so on.
    ``model`` defaults to the single model in the frame; give it explicitly when the
    frame spans several.
    """
    models = sorted(set(_level_or_column(items, "model")))
    if model is None:
        if len(models) != 1:
            raise SystemExit(f"frame spans {len(models)} models {models}; "
                             "pass model= explicitly")
        model = models[0]
    joined = atomic_for(model, items, solo)
    joined = joined.rename(columns={column: f"atomic_{column}" for column in joined.columns})
    return items.join(joined, on="question_id")


def _level_or_column(frame: pd.DataFrame, name: str):
    if name in frame.columns:
        return frame[name]
    if isinstance(frame.index, pd.MultiIndex) and name in frame.index.names:
        return frame.index.get_level_values(name)
    if frame.index.name == name:
        return frame.index
    raise SystemExit(f"frame has no {name} column or index level")
