#!/usr/bin/env python3
"""The fragility-aware batching tables: accuracy and output tokens per policy.

    python experiments/VI-fragility-aware-batching/batching_table.py

Per dataset and model, every policy's item accuracy and output tokens per item, each
averaged over its batching seeds (`llm.core.analysis.fragility.policy_metrics`):

    k1        isolated, pooled over isolated-a and isolated-b (10 seeds)
    k1_a/b    each isolated pass alone
    k3..k20   random batching, random-a + random-b, prompts holding exactly k items
    measured  fragility-aware batching from measured drops (fragility-measured)
    probe     fragility-aware batching from the probe's predicted drops (fragility-probe)

and the change against k1: ``d_acc_pp`` (percentage points), ``d_acc_rel_pct``,
``d_tok_abs`` and ``d_tok_rel_pct``. For Qwen3.8 27B on GPQA-Diamond the isolated and
random columns read the first passes (``isolated-a-v1`` ...), which its fragility-aware
runs were built from (`llm.specs.baseline_stages`). ``mean_k`` and ``frac_alone``
describe the sizes a fragility-aware pass chose.

Writes to $RUN_DIR/results/VI-fragility-aware-batching/: batching_results_<dataset>.csv,
batching_table_main.tex and batching_table_full.tex (the tabulars of Table 2 and of the
appendix table) and batching_text_numbers.csv (the averages quoted in the text).
An Average row is the mean over the models shown of the k=1 values and of the changes;
the greyed R1 Qwen3 8B row of OneWordQA is left out of it.
"""
from __future__ import annotations

import argparse
import sys
from statistics import mean

import pandas as pd

from llm import paths
from llm.core.analysis.fragility import policy_metrics
from llm.specs import BATCHING_DATASETS, MODEL_NAMES, MODELS, OPUS, baseline_stages
from llm.specs import dataset as dataset_spec

OUT = "VI-fragility-aware-batching"
MAIN_DATASETS = ["gpqa-diamond", "math500", "mmlu"]
GREY = {("onewordqa", "r1-qwen3-8b")}
NAMES = {**MODEL_NAMES, "qwen3-4b-thinking": "Qwen3 4B T.", "r1-llama-distill-8b": "R1 Llama 8B"}
TABLE_POLICIES = ["k6", "measured", "probe"]


def policies(dataset: str, model: str) -> dict:
    """condition key -> (label, stages, k).

    The isolated and random columns read the passes the fragility-aware runs were built
    from (`baseline_stages`), which for one pair are the first passes (``-v1``).
    """
    st = baseline_stages(dataset, model)
    out = {"k1": ("k=1 isolated-a + isolated-b", (st["isolated-a"], st["isolated-b"]), None),
           "k1_a": ("k=1 isolated-a", (st["isolated-a"],), None),
           "k1_b": ("k=1 isolated-b", (st["isolated-b"],), None)}
    for k in dataset_spec(dataset).random_ks:
        out[f"k{k}"] = (f"k={k} random-a + random-b", (st["random-a"], st["random-b"]), k)
    out["measured"] = ("fragility-aware, measured", ("fragility-measured",), None)
    out["probe"] = ("fragility-aware, probe", ("fragility-probe",), None)
    return out


def models_of(dataset: str) -> list[str]:
    return MODELS + [OPUS] if dataset == "gpqa-diamond" else list(MODELS)


def measure(unit: tuple[str, str]) -> list[dict]:
    """Every condition row of one (dataset, model)."""
    dataset, model = unit
    rows, base = [], None
    for key, (label, stages, k) in policies(dataset, model).items():
        m = policy_metrics(stages, dataset, model, k)
        if m is None:
            continue
        row = {"dataset": dataset, "model": model, "condition_key": key, "condition": label,
               "n_seeds": m["n_seeds"], "pool_items": m["items"],
               "item_accuracy": 100 * m["acc"], "output_tokens_per_item": m["tok"]}
        if key == "k1":
            base = row
        elif base is not None:
            d_acc = row["item_accuracy"] - base["item_accuracy"]
            d_tok = row["output_tokens_per_item"] - base["output_tokens_per_item"]
            row.update(d_acc_pp=d_acc, d_acc_rel_pct=100 * d_acc / base["item_accuracy"],
                       d_tok_abs=d_tok,
                       d_tok_rel_pct=100 * d_tok / base["output_tokens_per_item"])
        if key in ("measured", "probe"):
            row.update(mean_k=m["mean_k"], frac_alone=m["frac_alone"])
        rows.append(row)
    print(f"[{dataset}] {model}: {len(rows)} conditions", flush=True)
    return rows


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------

def signed(value: float) -> str:
    """One decimal with an explicit sign; a value that rounds to zero prints as 0.0."""
    text = f"{value:+.1f}"
    return "0.0" if text in ("+0.0", "-0.0") else text


def model_row(dataset: str, model: str, cells: dict) -> str:
    k1 = cells["k1"]
    parts = [NAMES[model], f"{k1['item_accuracy']:.1f}\\%", f"{k1['output_tokens_per_item']:.0f}"]
    for key in TABLE_POLICIES:
        c = cells[key]
        parts.append(f"\\ppcell{{{c['item_accuracy']:.1f}\\%}}{{{signed(c['d_acc_pp'])}}}")
        parts.append(f"\\relcell{{{c['output_tokens_per_item']:.0f}}}"
                     f"{{{signed(c['d_tok_rel_pct'])}}}")
    if (dataset, model) in GREY:
        parts = [f"\\color{{gray}}{p}" for p in parts]
    return " & ".join(parts) + " \\\\"


def average(table: pd.DataFrame, dataset: str) -> dict:
    """Mean over the models shown (the grey row excluded) of k=1 and of every change."""
    shown = table[~table["model"].map(lambda m: (dataset, m) in GREY)]
    cell = lambda key, col: shown.loc[shown["condition_key"] == key, col].mean()  # noqa: E731
    out = {"acc": cell("k1", "item_accuracy"), "tok": cell("k1", "output_tokens_per_item")}
    for key in TABLE_POLICIES:
        out[f"{key}_acc"] = cell(key, "d_acc_pp")
        out[f"{key}_tok"] = cell(key, "d_tok_rel_pct")
    return out


def average_row(avg: dict) -> str:
    return ("Average & "
            f"{avg['acc']:.1f}\\% & {avg['tok']:.0f} & "
            f"${signed(avg['k6_acc'])}$ & \\multicolumn{{1}}{{l}}{{${signed(avg['k6_tok'])}\\%$}} & "
            f"\\multicolumn{{1}}{{|l}}{{${signed(avg['measured_acc'])}$}} & "
            f"${signed(avg['measured_tok'])}\\%$ & "
            f"${signed(avg['probe_acc'])}$ & ${signed(avg['probe_tok'])}\\%$ \\\\")


HEADER = r"""\begin{tabular}{lllll|llll}
\toprule
 & \multicolumn{2}{c}{Isolated, $k{=}1$} & \multicolumn{2}{c|}{Random batching, $k{=}6$} & \multicolumn{2}{c}{Fragility-aware (measured)} & \multicolumn{2}{c}{Fragility-aware (probe)} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
 & \multicolumn{1}{c}{Acc.} & \multicolumn{1}{c}{Tok.} & \multicolumn{1}{c}{Acc.} & \multicolumn{1}{c|}{Tok.} & \multicolumn{1}{c}{Acc.} & \multicolumn{1}{c}{Tok.} & \multicolumn{1}{c}{Acc.} & \multicolumn{1}{c}{Tok.} \\
\midrule"""


def tabular(tables: dict[str, pd.DataFrame], datasets: list[str]) -> str:
    lines = [r"\providecommand{\relcell}[2]{{\small #1}\,{\scriptsize$(#2\%)$}}",
             r"\providecommand{\ppcell}[2]{{\small #1}\,{\scriptsize$(#2)$}}", HEADER]
    for i, dataset in enumerate(datasets):
        table = tables[dataset]
        if i:
            lines.append(r"\midrule")
        lines += [rf"\multicolumn{{9}}{{l}}{{\textit{{{dataset_spec(dataset).title}}}}} \\",
                  r"\midrule"]
        for model in models_of(dataset):
            cells = {r["condition_key"]: r for r in table[table["model"] == model].to_dict("records")}
            lines.append(model_row(dataset, model, cells))
        lines += [r"\midrule[0.08em]", r"\rowcolor{gray!12}", average_row(average(table, dataset))]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def text_numbers(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The averages the text of Section 4 quotes, each with how it is averaged."""
    avg = {d: average(tables[d], d) for d in BATCHING_DATASETS}
    rest = [d for d in BATCHING_DATASETS if d not in ("onewordqa", "strategyqa")]
    opus = tables["gpqa-diamond"].query("model == @OPUS").set_index("condition_key")
    rows = [
        ("measured: token change, mean of the dataset Averages", "-43%",
         mean(avg[d]["measured_tok"] for d in BATCHING_DATASETS)),
        ("measured: accuracy change (pp), mean of the dataset Averages", "+0.6",
         mean(avg[d]["measured_acc"] for d in BATCHING_DATASETS)),
        ("measured: token change, OneWordQA Average", "-68..-74%", avg["onewordqa"]["measured_tok"]),
        ("measured: token change, StrategyQA Average", "-68..-74%", avg["strategyqa"]["measured_tok"]),
        ("measured: token change, mean of the other five dataset Averages", "roughly -32%",
         mean(avg[d]["measured_tok"] for d in rest)),
        ("probe: token change, mean of the dataset Averages", "-31.0%",
         mean(avg[d]["probe_tok"] for d in BATCHING_DATASETS)),
        ("probe: worst dataset-Average accuracy change (pp)", "at most -0.5",
         min(avg[d]["probe_acc"] for d in BATCHING_DATASETS)),
        ("Opus isolated accuracy (%)", "89.5", opus.loc["k1", "item_accuracy"]),
        ("Opus isolated output tokens per task", "689", opus.loc["k1", "output_tokens_per_item"]),
        ("Opus random k=6: accuracy change (pp)", "-0.9", opus.loc["k6", "d_acc_pp"]),
        ("Opus random k=6: token change (%)", "-18%", opus.loc["k6", "d_tok_rel_pct"]),
        ("Opus measured: accuracy change (pp)", "-0.4", opus.loc["measured", "d_acc_pp"]),
        ("Opus measured: token change (%)", "-24%", opus.loc["measured", "d_tok_rel_pct"]),
        ("Opus probe: accuracy change (pp)", "-2.0", opus.loc["probe", "d_acc_pp"]),
        ("Opus probe: token change (%)", "almost none", opus.loc["probe", "d_tok_rel_pct"]),
    ]
    return pd.DataFrame(rows, columns=["quantity", "paper", "value"])


ROUND = {"item_accuracy": 3, "output_tokens_per_item": 1, "d_acc_pp": 3, "d_acc_rel_pct": 2,
         "d_tok_abs": 1, "d_tok_rel_pct": 2, "mean_k": 3, "frac_alone": 4}


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    units = [(d, m) for d in BATCHING_DATASETS for m in models_of(d)]
    rows = [row for unit in units for row in measure(unit)]
    frame = pd.DataFrame(rows)
    tables = {d: frame[frame["dataset"] == d].reset_index(drop=True) for d in BATCHING_DATASETS}

    out = paths.results(OUT)
    for dataset, table in tables.items():
        table.drop(columns="dataset").round(ROUND).to_csv(
            out / f"batching_results_{dataset}.csv", index=False)
    (out / "batching_table_main.tex").write_text(tabular(tables, MAIN_DATASETS))
    (out / "batching_table_full.tex").write_text(tabular(tables, BATCHING_DATASETS))
    numbers = text_numbers(tables)
    numbers.to_csv(out / "batching_text_numbers.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(numbers.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
