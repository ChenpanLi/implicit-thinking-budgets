"""The fragility probe and the per-item composition size of fragility-aware batching.

Each stage is a module with its own entry point, run in this order:

    python -m llm.probe.embed_items    every item alone through Qwen3-Embedding-8B (GPU)
    python -m llm.probe.build_labels   per-item accuracy drop and the per-k survival label
    python -m llm.probe.make_splits    the five outer folds of each dataset
    python -m llm.probe.train          one ordinal head per (dataset, model, fold, inner fold)
    python -m llm.probe.tokens         output tokens per item at each k (token-cost.csv)
    python -m llm.probe.assign_k       the per-item k, from measurements or from the heads

Everything is written under ``$RUN_DIR/probe/`` (see :mod:`llm.probe.common`), and
``assign_k`` writes the two CSVs the ``fragility-measured`` and ``fragility-probe``
batchings are built from.
"""
