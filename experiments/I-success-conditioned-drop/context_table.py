"""Appendix table "Context limits and parsing failures in the composed sets".

Reads context-records.csv (from context_records.py) and writes context-parsing.tex. Per
dataset: the median response length and its median share of the budget at the largest k,
and the truncation and parsing-failure rates pooled over every model and k, with the worst
(model, k) cell in brackets.
"""
import pandas as pd

from llm import paths

ROWS = [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("aime", "AIME"), ("integrals", "Integrals"),
        ("mmlu", r"MMLU-Redux$^{\dagger}$"), ("gpqa-diamond", "GPQA-Diamond"),
        ("onewordqa", "OneWordQA"), ("strategyqa", "StrategyQA")]
CAPTION = (
    r"Context limits and parsing failures in the composed sets. Each response has a token "
    r"budget equal to the context window minus the prompt, capped at the model's maximum "
    r"generation length; a response that exhausts it is truncated. Truncation and "
    r"parsing-failure rates are pooled over all models and $k$, with the worst model and $k$ "
    r"in brackets. $^\dagger$Most unparseable responses come from R1 Qwen3 8B, which often "
    r"breaks the answering contract under composition.")


def table_rows(rec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, name in ROWS:
        r = rec[rec["dataset"] == dataset]
        cells = r.groupby(["model", "k"])[["truncated", "unparseable"]].mean()
        top = r[r["k"] == r["k"].max()]
        rows.append({
            "dataset": dataset, "name": name, "k": int(top["k"].iloc[0]),
            "tokens": top["out_tokens"].median(), "share": top["budget_share"].median(),
            "hit": 100 * r["truncated"].mean(), "hit_worst": 100 * cells["truncated"].max(),
            "unparseable": 100 * r["unparseable"].mean(),
            "unparseable_worst": 100 * cells["unparseable"].max(),
        })
    return pd.DataFrame(rows)


def latex(t: pd.DataFrame) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{CAPTION}}}",
             r"\label{tab:context-parsing}", "", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Response at largest $k$} & \multicolumn{2}{c}{Responses (\%)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Dataset & $k$ & tokens & of budget & hit limit & unparseable \\", r"\midrule"]
    for _, x in t.iterrows():
        lines.append(f"{x['name']} & {x.k} & {x.tokens / 1000:.1f}k & {100 * x.share:.0f}\\% & "
                     f"{x.hit:.1f} [{x.hit_worst:.0f}] & "
                     f"{x.unparseable:.1f} [{x.unparseable_worst:.0f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", "", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    out = paths.results("I-success-conditioned-drop")
    t = table_rows(pd.read_csv(out / "context-records.csv"))
    tex = latex(t)
    (out / "context-parsing.tex").write_text(tex)
    print(t.round(3).to_string(index=False))
    print(tex)
    print(f"wrote {out / 'context-parsing.tex'}")


if __name__ == "__main__":
    main()
