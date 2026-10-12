"""How often a tagged thinking trace names every task of its prompt.

Per-task token counts rest on the model opening a section for each task it thinks
about. A trace *names every task* when each of its actual positions (the dummy and the
fillers left out) has at least one section. A *tagged* trace is one whose thinking
closed and holds at least one section. Writes the share of tagged traces that name every
task, per (dataset, model, k), to `tag-coverage.csv` and prints the shares the appendix
reports ("Measuring per-task thinking length").

    python experiments/II-token-ratio/tag_coverage.py
"""
from __future__ import annotations

import pandas as pd

from llm import paths
from llm.specs import MODEL_NAMES, MODELS, TRACE_DATASETS

from trace_data import frames


def load() -> pd.DataFrame:
    rows = []
    for ds in TRACE_DATASETS:
        for model in MODELS:
            a = frames(ds, model)["attempts"]
            a = a[a.think_closed & (a.n_sections > 0)]
            rows.append(a.assign(dataset=ds, model=model,
                                 named_all=a.n_actual_named == a.n_actual))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    a = load()
    table = (a.groupby(["dataset", "model", "k"])
             .agg(traces=("named_all", "size"), named_every_task=("named_all", "mean"))
             .reset_index())
    out = paths.results("II-token-ratio") / "tag-coverage.csv"
    table.to_csv(out, index=False, float_format="%.6g")
    print(a.pivot_table(index="dataset", columns="model", values="named_all", aggfunc="mean")
          .mul(100).round(1).to_string())
    print(f"-> {out}\n")

    gpqa = a[a.dataset == "gpqa-diamond"]
    r1 = gpqa[(gpqa.model == "r1-qwen-distill-7b") & (gpqa.k >= 3)]
    for label, sub in [("every tagged trace", a), ("GPQA-Diamond", gpqa),
                       (f"{MODEL_NAMES['r1-qwen-distill-7b']} on GPQA-Diamond, k >= 3", r1)]:
        print(f"  {label:40s} {100 * sub.named_all.mean():5.1f}% of {len(sub)} traces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
