"""The batching strategies the paper uses.

A strategy sees only pool indices and the statistics bundle, so the same file
batches integrands, multiple-choice questions and one-word questions. Import is
what registers them, so every module that defines a strategy is imported here.
``_calibration_common`` registers nothing and is imported by the strategy that
uses it.
"""
from . import control  # noqa: F401  (imported for its registration)
from . import dummy_tail  # noqa: F401  (imported for its registration)
from . import predicted_calibration  # noqa: F401  (imported for its registration)
from . import random_batches  # noqa: F401  (imported for its registration)

__all__ = ["control", "dummy_tail", "predicted_calibration", "random_batches"]
