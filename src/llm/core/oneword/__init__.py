"""Short written answers: answer parsing, normalised matching, prompt rendering.

The counterpart of :mod:`llm.core.mcq`, and kept separate from it the same way
that one is kept separate from :mod:`llm.core.math`.

Two task layers import it, the way :mod:`llm.core.mcq` serves MMLU and GPQA and
:mod:`llm.core.shortmath` serves GSM8K, MATH-500 and AIME. They differ in the
corpus and in nothing else here:

* one-word QA -- a committed local pool (``datasets/onewordqa/``)
* StrategyQA -- yes/no questions from the Hub (:mod:`llm.core.oneword.strategyqa`)

What they share is the claim that makes them one family: an answer is a short
string a model wrote in its own house style, and deciding whether it *is* the
reference answer is a matter of how the string is written
(:mod:`llm.core.oneword.normalization`), never of what it means. A domain that
needs more than that -- a yes/no question's ``true``, a date's other word order
-- widens its *pool*, where the extra spellings are visible and reviewable, and
not the matching rules, where they would loosen every other domain too.
"""
