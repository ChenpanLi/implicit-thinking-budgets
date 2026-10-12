from __future__ import annotations

import importlib
import re
from typing import Any


def _load_latex2sympy() -> Any:
	for module_name in ("latex2sympy2_extended", "latex2sympy"):
		try:
			module = importlib.import_module(module_name)
			latex2sympy = getattr(module, "latex2sympy")
			return latex2sympy
		except (ImportError, AttributeError):
			continue
	raise ImportError("latex2sympy is required to use MathLatexParser.")


# Byte-level BPE sentinels and invisible characters. Roughly half of the
# recorded malformed generations arrive with whitespace replaced by these:
# "Ġ" is a space and "Ċ" a newline in GPT-2 byte-level vocabularies.
_SENTINELS = {
	"\u0120": " ",   # G-with-stroke: space
	"\u010a": "\n",  # C-with-stroke: newline
	"\u0109": "\t",
	"\u200b": "",    # zero-width space
	"\u200c": "",
	"\u00a0": " ",   # non-breaking space
	"\ufeff": "",
}

# Models routinely spell a function name as \operatorname{arctan}, \text{arctan}
# or \mathrm{arctan} instead of the \arctan the grammar expects. \operatorname
# only recognizes a small hard-coded whitelist (mostly the ar*sinh/cosh/tanh
# trio) and raises on everything else -- an honest UNPARSEABLE. \text and
# \mathrm are worse: latex2sympy2_extended happily builds an *undefined*
# Function literally named "\text{arctan}" and keeps going, so the answer
# silently stops meaning arctan(x) instead of failing loudly. Rewriting the
# wrapper to the bare command the grammar does understand fixes both.
#
# Deliberately excludes coth/sech/csch (and their inverses): the bare \coth,
# \sech, \csch commands are themselves mis-tokenized by the grammar (e.g.
# "\coth(x)" parses as cot(h(x)), not coth(x)), so rewriting the wrapper to
# them would trade a safe UNPARSEABLE for a silently wrong parse. None of the
# generators in this repo produce hyperbolic-reciprocal integrands anyway.
_FUNCTION_WRAPPER = re.compile(
	r"\\(?:operatorname\*?|text|mathrm|mathop)\s*\{\s*([A-Za-z]+)\s*\}"
)
_FUNCTION_NAME_MAP = {
	# trig
	"sin": "sin", "cos": "cos", "tan": "tan", "tg": "tan",
	"cot": "cot", "ctg": "cot", "cotan": "cot",
	"sec": "sec", "csc": "csc", "cosec": "csc",
	# inverse trig (both "arc*" and short "a*" spellings)
	"arcsin": "arcsin", "asin": "arcsin",
	"arccos": "arccos", "acos": "arccos",
	"arctan": "arctan", "atan": "arctan",
	"arccot": "arccot", "acot": "arccot", "arcctg": "arccot",
	"arcsec": "arcsec", "asec": "arcsec",
	"arccsc": "arccsc", "acsc": "arccsc",
	# hyperbolic
	"sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
	# inverse hyperbolic
	"arcsinh": "arcsinh", "arsinh": "arcsinh", "asinh": "arcsinh",
	"arccosh": "arccosh", "arcosh": "arccosh", "acosh": "arccosh",
	"arctanh": "arctanh", "artanh": "arctanh", "atanh": "arctanh",
	# log / exp
	"ln": "ln", "log": "log", "exp": "exp",
}


def _rewrite_function_wrapper(match: re.Match[str]) -> str:
	target = _FUNCTION_NAME_MAP.get(match.group(1).lower())
	return f"\\{target}" if target else match.group(0)


# The same aliases written as bare commands rather than as wrappers: models
# produce "\asin{\left(\sqrt{x}\right)}" and "\atan(3x + 7)" as readily as
# "\operatorname{asin}{...}", and the grammar knows neither, so the answer is
# lost to "I don't understand this" over a spelling. Restricted to the aliases
# that are unambiguous -- \sin, \cos and friends are already commands, and the
# hyperbolic reciprocals are excluded for the reason _FUNCTION_NAME_MAP gives.
_BARE_FUNCTION_ALIAS = re.compile(
	r"\\(asin|acos|atan|acot|asec|acsc|arcctg|arctg|arcsinh|arsinh|asinh"
	r"|arccosh|arcosh|acosh|arctanh|artanh|atanh|cotan|cosec)(?![A-Za-z])"
)

# `\left` and `\right` with the delimiter pushed off by whitespace. Harmless to
# a LaTeX renderer, fatal to the grammar: "\left( ... \right )" loses its whole
# expression to a parse error over the space before the ")".
_DELIMITER_SPACE = re.compile(r"\\(left|right)[ \t]+(?=[(\[\]{|.<>/)])")

# `\|` (the norm delimiter) used where `|` was meant. Models pair "\left|" with
# "\right\|" often enough to matter, and for a real scalar the two mean the same
# absolute value, so reading the norm as an absolute value changes nothing.
_NORM_DELIMITER = re.compile(r"\\(left|right)\s*\\\|")


# latex2sympy2_extended treats a bare \log (no explicit subscript base) as
# base 10, the ISO/engineering convention. The integrands were generated with
# sympy's sp.ln, and models writing plain \log in a
# calculus context mean the natural log too. Rewriting bare \log to \ln before
# parsing fixes the default; \log_{10}, \log_2, etc. keep their explicit base
# untouched since the underscore lookahead excludes them.
_BARE_LOG = re.compile(r"\\log(?!\s*_)")

# \ln|u| binds more loosely than latex2sympy assumes: it parses "\ln|2x+3|/2"
# as log(Abs(2*x + 3)/2), absorbing the divisor into the argument. Wrapping the
# absolute-value group in explicit parentheses restores the intended reading.
_LN_ABS_DELIMITED = re.compile(r"\\(ln|log)\s*\\left\|(.*?)\\right\|")
_LN_ABS_BARE = re.compile(r"\\(ln|log)\s*\|([^|]*)\|")

# A trailing constant of integration. The prompt says it may be omitted, but
# models routinely include it. Differentiation removes it anyway; stripping it
# here only keeps the recorded trace readable.
_TRAILING_CONSTANT = re.compile(
	r"[+\-]\s*(?:\\(?:text|mathrm|mathit)\s*\{\s*[Cc](?:onst(?:ant)?)?\s*\}|[Cc](?:_\{?\d+\}?)?)\s*$"
)


def strip_bpe_sentinels(text: str) -> tuple[str, bool]:
	"""Restore whitespace that arrived as byte-level BPE sentinels.

	Split out of ``canonicalize_latex`` because it is the one rule that applies
	to prose as well as to LaTeX: anything displaying a stored generation wants
	the whitespace back without also rewriting the mathematics.
	"""
	if not any(ch in text for ch in _SENTINELS):
		return text, False
	for sentinel, replacement in _SENTINELS.items():
		text = text.replace(sentinel, replacement)
	return text, True


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# \big[, \Big[, \bigg[, \Bigg[ -> (
# \big], \Big], \bigg], \Bigg] -> )
_BIG_LBRACKET = re.compile(r"\\(?:big|Big|bigg|Bigg)\[")
_BIG_RBRACKET = re.compile(r"\\(?:big|Big|bigg|Bigg)\]")

# \bigl(
_BIGL = re.compile(r"\\bigl\(")
# \bigr)
_BIGR = re.compile(r"\\bigr\)")

# A genuinely stray doubled backslash.
#
# We deliberately DO NOT replace every `\\`, because:
#
#     \\        -> valid LaTeX line break
#     \\[1em]   -> valid LaTeX line break with spacing
#
# Instead, this targets doubled backslashes which are surrounded by
# expression-like content and are therefore much more likely to be
# accidental model output.
_STRAY_DOUBLE_BACKSLASH = re.compile(
    r"(?<=[\w\)\]\}])\s*\\\\(?=\s*[\w\(\[\{\\])"
)

# Random standalone !.
#
# Keep:
#   !=
#   \!
#   \not
#
# Remove a literal `!` when it appears as an isolated punctuation artifact.
_RANDOM_BANG = re.compile(
    r"(?<!\\)!(?![=])"
)

# Alignment artifacts such as:
#     x &+ y
#     x &- y
#     x &= y
#     x &. y
#
# Only remove the ampersand when it is directly associated with the
# operator, optionally with whitespace.
_STRAY_AMPERSAND = re.compile(
    r"&(?=\s*[+\-=.])"
)


def _normalize_big_brackets(text: str) -> str:
    """Normalize oversized square delimiters to ordinary parentheses."""
    text = _BIG_LBRACKET.sub("(", text)
    text = _BIG_RBRACKET.sub(")", text)
    text = _BIGL.sub("(", text)
    text = _BIGR.sub(")", text)
    return text


def _remove_stray_double_backslashes(text: str) -> str:
    """
    Remove doubled backslashes only when they look like accidental
    expression separators.

    Important:
        Do NOT globally replace `\\\\`.

    A LaTeX `\\\\` can legitimately mean a line break, particularly inside
    aligned environments, arrays, matrices, tables, etc.
    """
    return _STRAY_DOUBLE_BACKSLASH.sub(" ", text)


def _remove_random_bangs(text: str) -> str:
    """
    Remove accidental literal exclamation marks while preserving:
        \\!
        !=
    """
    return _RANDOM_BANG.sub("", text)


def _remove_stray_ampersands(text: str) -> str:
    """
    Remove alignment artifacts before arithmetic/comparison punctuation.

    Examples:
        x &+ 1  -> x + 1
        x &- 1  -> x - 1
        x &= 1  -> x = 1
        x &.    -> x .
    """
    return _STRAY_AMPERSAND.sub("", text)


def canonicalize_latex(
    text: str,
    rules: list[str] | None = None,
) -> str:
    """
    Canonicalize common malformed/generated LaTeX without intentionally
    changing mathematical semantics.

    Every transformation that actually changes the string records its name
    in `rules`, allowing malformed answers to be attributed to a specific
    normalization rather than simply to the parser.
    """
    if rules is None:
        rules = []

    original = text

    # ------------------------------------------------------------------
    # 1. BPE / tokenizer artifacts
    # ------------------------------------------------------------------
    text, stripped = strip_bpe_sentinels(text)
    if stripped:
        rules.append("strip_bpe_sentinels")

    # ------------------------------------------------------------------
    # 2. Basic command normalization
    # ------------------------------------------------------------------
    before = text

    text = re.sub(r"\\[dt]frac", r"\\frac", text)
    text = text.replace("\\lparen", "(")
    text = text.replace("\\rparen", ")")

    if text != before:
        rules.append("normalize_commands")

    # ------------------------------------------------------------------
    # 3. Normalize oversized square brackets
    #
    # Examples:
    #   \big[ x \big]       -> ( x )
    #   \Big[ x \Big]       -> ( x )
    #   \bigg[ x \bigg]     -> ( x )
    #   \Bigg[ x \Bigg]     -> ( x )
    #
    # Do this before delimiter balancing.
    # ------------------------------------------------------------------
    before = text
    text = _normalize_big_brackets(text)

    if text != before:
        rules.append("normalize_big_brackets")

    # ------------------------------------------------------------------
    # 3b. Delimiter spelling: "\right )" -> "\right)", "\right\|" -> "\right|"
    #
    # Both are cosmetic to a renderer and fatal to the grammar. Done before the
    # \left/\right balance check below, which counts the commands and would
    # otherwise strip a pair that is in fact matched.
    # ------------------------------------------------------------------
    before = text
    text = _DELIMITER_SPACE.sub(r"\\\1", text)

    if text != before:
        rules.append("close_delimiter_spacing")

    before = text
    text = _NORM_DELIMITER.sub(r"\\\1|", text)

    if text != before:
        rules.append("norm_to_abs_delimiter")

    # ------------------------------------------------------------------
    # 4. Function wrappers
    # ------------------------------------------------------------------
    before = text
    text = _FUNCTION_WRAPPER.sub(_rewrite_function_wrapper, text)

    if text != before:
        rules.append("rewrite_function_wrapper")

    before = text
    text = _BARE_FUNCTION_ALIAS.sub(
        lambda m: "\\" + _FUNCTION_NAME_MAP.get(m.group(1).lower(), m.group(1)), text)

    if text != before:
        rules.append("rewrite_bare_function_alias")

    # ------------------------------------------------------------------
    # 5. Bare log -> ln
    # ------------------------------------------------------------------
    before = text
    text = _BARE_LOG.sub(r"\\ln", text)

    if text != before:
        rules.append("bare_log_to_ln")

	# remove aligned blocks
    text = re.sub(r"\\begin\{aligned\}.*?\\end\{aligned\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{align\}.*?\\end\{align\}", "", text, flags=re.DOTALL)

    # ------------------------------------------------------------------
    # 6. Normalize ln(abs(...))
    # ------------------------------------------------------------------
    before = text

    text = _LN_ABS_DELIMITED.sub(
        r"\\\1\\left(\\left|\2\\right|\\right)",
        text,
    )

    text = _LN_ABS_BARE.sub(
        r"\\\1\\left(\\left|\2\\right|\\right)",
        text,
    )

    if text != before:
        rules.append("wrap_ln_abs_arg")

    # ------------------------------------------------------------------
    # 7. Remove stray alignment ampersands
    #
    #   x &+ 1  -> x + 1
    #   x &- 1  -> x - 1
    #   x &= 1  -> x = 1
    #   x &.    -> x .
    # ------------------------------------------------------------------
    before = text
    text = _remove_stray_ampersands(text)

    if text != before:
        rules.append("remove_stray_ampersands")

    # ------------------------------------------------------------------
    # 8. Remove accidental ! characters
    # ------------------------------------------------------------------
    before = text
    text = _remove_random_bangs(text)

    if text != before:
        rules.append("remove_random_bangs")

    # ------------------------------------------------------------------
    # 9. Remove clearly stray doubled backslashes
    #
    # This is intentionally conservative. We must NOT turn every valid
    # LaTeX `\\` line break into whitespace.
    # ------------------------------------------------------------------
    before = text
    text = _remove_stray_double_backslashes(text)

    if text != before:
        rules.append("remove_stray_double_backslashes")

    # ------------------------------------------------------------------
    # 10. Drop trailing integration constant
    # ------------------------------------------------------------------
    before = text
    text = _TRAILING_CONSTANT.sub("", text.strip())

    if text != before:
        rules.append("drop_integration_constant")

    # ------------------------------------------------------------------
    # 11. Repair unbalanced \left / \right
    #
    # If truncation has left one side unmatched, remove both commands
    # rather than attempting to invent a missing delimiter.
    # ------------------------------------------------------------------
    if text.count("\\left") != text.count("\\right"):
        text = text.replace("\\left", "")
        text = text.replace("\\right", "")
        rules.append("strip_unbalanced_delimiters")

    # ------------------------------------------------------------------
    # 11b. Close a group the generation was cut off inside
    #
    # A response truncated mid-answer leaves "\cos(12x - 2" or "\frac{1}{144",
    # which the grammar rejects outright even though every character the model
    # did emit is unambiguous. Closing the group recovers exactly what was
    # written and invents nothing; a *surplus* of closers is left alone, since
    # that is a malformed expression rather than a cut-off one.
    #
    # Capped at three of each: past that the shape is noise, not truncation,
    # and guessing further would be inventing an answer.
    # ------------------------------------------------------------------
    for opener, closer in (("(", ")"), ("{", "}"), ("[", "]")):
        missing = text.count(opener) - text.count(closer)
        if 0 < missing <= 3:
            text = text + closer * missing
            if "close_truncated_group" not in rules:
                rules.append("close_truncated_group")

    # ------------------------------------------------------------------
    # 12. Final whitespace normalization
    # ------------------------------------------------------------------
    stripped_text = text.strip()

    if stripped_text != text:
        text = stripped_text
        if "whitespace" not in rules:
            rules.append("whitespace")

    if text != original and not rules:
        rules.append("whitespace")

    return text


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------
# The non-elementary answers models actually write. These are not exotic: the
# antiderivative of sin(x)/x *is* Si(x), of e^x/x *is* Ei(x), of e^{-x^2} *is*
# (sqrt(pi)/2) erf(x) -- correct answers, every one, that the grammar rejects
# because it knows no \operatorname{Si}. They are the single largest class of
# UNPARSEABLE terms on record, so refusing to read them would score the model's
# mathematics as our formatting failure.
#
# Deliberately absent: the elliptic E/F and the hypergeometric {}_2F_1, whose
# argument conventions differ between sources (E(phi, m) vs E(phi | m^2)) --
# reading one convention as the other would turn an UNPARSEABLE into a silently
# wrong expression, which is the trade this module never makes. Ti_2 has no
# SymPy function of its own and is built from the polylogarithms it is defined
# by, so its derivative, atan(x)/x, comes out exactly.
def _special_functions() -> dict[str, Any]:
	import sympy as sp

	return {
		"si": sp.Si, "ci": sp.Ci, "shi": sp.Shi, "chi": sp.Chi,
		"ei": sp.Ei, "li": sp.li, "logint": sp.li,
		"erf": sp.erf, "erfi": sp.erfi, "erfc": sp.erfc,
		"fresnels": sp.fresnels, "fresnelc": sp.fresnelc,
		"s": sp.fresnels, "c": sp.fresnelc,
		"ti_2": lambda z: (sp.polylog(2, sp.I * z) - sp.polylog(2, -sp.I * z)) / (2 * sp.I),
	}


# `Li` is the offset logarithmic integral when unsubscripted and the polylog
# when subscripted, so it is resolved from its subscript rather than by name.
def _resolve_special(name: str, subscript: str | None) -> Any:
	import sympy as sp

	key = name.lower()
	if key == "li" and subscript:
		return lambda z, order=sp.Integer(subscript): sp.polylog(order, z)
	if key == "li" and name == "Li":
		return sp.Li
	if key == "ti" and subscript == "2":
		return _special_functions()["ti_2"]
	if subscript:
		return None
	return _special_functions().get(key)


_SPECIAL_CALL = re.compile(
	r"\\(?P<wrapper>operatorname\*?|text|mathrm|mathop|mathit)"
	r"\s*\{\s*(?P<name>[A-Za-z]{1,8})\s*\}"
	r"(?:\s*_\s*\{?\s*(?P<sub>\d{1,2})\s*\}?)?",
)

# The same names written bare, as a model writing "2 Ei(\sqrt{x})" does. These
# are worth reading not because they fail -- they do not -- but because they
# succeed wrongly: the grammar makes "Ei(x)" into the product E*i*x, and every
# one of the recorded answers spelled this way would be graded WRONG on the
# strength of that reading rather than on its mathematics. Only the canonical spellings
# are accepted, and only immediately before an argument group, so an ordinary
# product of symbols is never mistaken for a function call.
_BARE_SPECIAL_CALL = re.compile(
	r"(?<![A-Za-z\\])(?P<name>erfi|erfc|erf|Shi|Chi|Si|Ci|Ei|Li|li|Ti)"
	r"(?:\s*_\s*\{?\s*(?P<sub>\d{1,2})\s*\}?)?"
	r"(?=\s*(?:\\left)?[({])"
)

# `\mathop{\text{Ei}}` and friends: two wrappers round one name, which neither
# the grammar nor a single-wrapper pattern reads.
_NESTED_WRAPPER = re.compile(
	r"\\(?:mathop|operatorname\*?)\s*\{\s*"
	r"(\\(?:text|mathrm|mathit|operatorname\*?)\s*\{\s*[A-Za-z]{1,8}\s*\})"
	r"\s*\}"
)

# The delimiters an argument group may open with, in the order they are tried.
_ARG_OPENERS = (("\\left(", "\\right)"), ("(", ")"), ("{", "}"))


def _read_group(text: str, index: int) -> tuple[str, int] | None:
	"""The balanced argument group starting at `index`, and where it ends.

	Returns ``(inner_latex, end)`` or ``None`` when no group opens there. The
	walk counts ``\\left``/``\\right`` as well as bare brackets so that
	``\\left( \\frac{1}{2} (x + 1) \\right)`` closes on its own ``\\right)``.
	"""
	while index < len(text) and text[index] in " \t\n\\,;!":
		if text[index] == "\\" and not text[index:].startswith(("\\,", "\\;", "\\!", "\\ ")):
			break
		index += 2 if text[index] == "\\" else 1
	for opener, closer in _ARG_OPENERS:
		if not text.startswith(opener, index):
			continue
		depth = 0
		walk = index
		while walk < len(text):
			if text.startswith(opener, walk):
				depth += 1
				walk += len(opener)
				continue
			if text.startswith(closer, walk):
				depth -= 1
				walk += len(closer)
				if depth == 0:
					return text[index + len(opener):walk - len(closer)], walk
				continue
			walk += 1
		return None
	return None


def _placeholder(position: int) -> str:
	"""A name that survives latex2sympy as one symbol, spelling included.

	Lowercase and letters only: the grammar lowercases ``\\mathrm{Ei}`` to
	``ei`` and splits some mixed-case names into a product of one-letter
	symbols, either of which would lose the mapping back.
	"""
	letters = ""
	position += 1
	while position:
		position, remainder = divmod(position - 1, 26)
		letters = chr(ord("a") + remainder) + letters
	return f"zqspecial{letters}"


def substitute_special_functions(text: str) -> tuple[str, dict[str, tuple[Any, str]]]:
	r"""Replace each special-function call with a placeholder symbol.

	Returns the rewritten LaTeX and ``{placeholder: (function, argument latex)}``.
	The whole call including its argument is replaced, so the placeholder is
	left standing alone in the expression and parses as a plain symbol whatever
	surrounds it -- which is the property the round trip depends on. The
	argument is parsed separately and the function applied to it afterwards, so
	an argument that is itself unreadable fails as loudly as any other.
	"""
	text = _NESTED_WRAPPER.sub(r"\1", text)
	mapping: dict[str, tuple[Any, str]] = {}
	out: list[str] = []
	cursor = 0
	matches = sorted(
		list(_SPECIAL_CALL.finditer(text)) + list(_BARE_SPECIAL_CALL.finditer(text)),
		key=lambda match: match.start(),
	)
	for match in matches:
		if match.start() < cursor:
			continue
		name_written = match.group("name")
		wrapper = match.groupdict().get("wrapper")
		# A one-letter name is only ever read as a special function when the
		# model marked it as an operator: `\text{C}(x)` is far more likely to
		# be a constant of integration than a Fresnel integral, and misreading
		# it would turn an honest failure into a wrong expression.
		if len(name_written) == 1 and not (wrapper or "").startswith("operatorname"):
			continue
		function = _resolve_special(name_written, match.group("sub"))
		if function is None:
			continue
		group = _read_group(text, match.end())
		if group is None:
			continue
		argument, end = group
		name = _placeholder(len(mapping))
		mapping[name] = (function, argument)
		out.append(text[cursor:match.start()])
		out.append(f"\\mathrm{{{name}}}")
		cursor = end
	if not mapping:
		return text, {}
	out.append(text[cursor:])
	return "".join(out), mapping


class MathLatexParser:
    def __init__(self) -> None:
        self._latex2sympy = _load_latex2sympy()

    def _canonicalize_latex(self, latex: str) -> str:
        return canonicalize_latex(latex)

    def _parse_one(self, latex: str) -> Any:
        import sympy as sp

        canonical = self._canonicalize_latex(latex)
        rewritten, specials = substitute_special_functions(canonical)
        expr = self._latex2sympy(rewritten)
        if not specials:
            return expr
        # Applied after the parse rather than before it, because a placeholder
        # only stays one symbol while it stands alone: `\mathrm{Si}(x)` is read
        # by the grammar as a product in some contexts and as an undefined
        # function in others, and neither survives a round trip.
        substitutions = {}
        for name, (function, argument) in specials.items():
            symbol = sp.Symbol(name)
            if symbol not in expr.free_symbols:
                raise ValueError(f"special-function placeholder {name!r} was lost in parsing")
            substitutions[symbol] = function(self._parse_one(argument))
        return expr.xreplace(substitutions)

    def parse(self, text: str) -> Any:
        return self._parse_one(text)

