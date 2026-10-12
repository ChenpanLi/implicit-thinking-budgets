"""Verification of candidate antiderivatives.

The single source of truth for "is this F an antiderivative of this f?", used
identically by the lm-eval task and the ``--explain`` CLI.

The approach is deliberately *symbolic-first*: differentiate the candidate with
SymPy and compare against the integrand.

Every stage records what it did, so a verdict is always attributable to the
exact step that produced it: ``result.stages[-1]`` is, by construction, the
stage that decided (or broke).
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import mpmath as mp
import sympy as sp

InputKind = Literal["response", "latex", "sympy", "expr"]

# Numeric-screen tolerances. Relative, because a k=20 sum can span ~30 orders
# of magnitude between its smallest and largest term.
_RTOL = mp.mpf("1e-12")
_ATOL = mp.mpf("1e-25")
_DPS = 30
_MIN_USABLE_POINTS = 4

# Generic complex sample points. Non-zero imaginary parts almost surely miss
# poles and branch cuts, which is exactly what a real-axis sampler cannot
# promise.
_COMPLEX_POINTS = (
    (0.7351, 0.3172), (-1.2837, 0.6193), (2.1379, -0.4271), (-3.4517, 0.8123),
    (1.6942, 0.5387), (-0.8261, -0.7519), (3.3078, 0.2946), (-2.5194, -0.3628),
)

# Real fallback, used when the expressions contain non-analytic pieces (Abs,
# sign, Piecewise). Straddles zero, avoids small integers and common
# singularities.
_REAL_POINTS = (-5.13, -3.31, -1.27, -0.41, 0.67, 1.53, 3.09, 4.77)

# Nodes whose presence makes complex sampling invalid. `sign`/`Abs` are the
# important ones: for real x, sign(x)/Abs(x) == 1/x, but for complex z it
# equals 1/conj(z), so complex sampling would reject a correct log|x| answer.
_NON_ANALYTIC = (sp.Abs, sp.sign, sp.re, sp.im, sp.conjugate, sp.Piecewise, sp.Heaviside)


class Verdict(str, Enum):
    CORRECT = "CORRECT"
    WRONG = "WRONG"
    UNPARSEABLE = "UNPARSEABLE"  # never reached a SymPy Expr
    TIMEOUT = "TIMEOUT"          # symbolic confirmation exceeded its budget
    CHECKER_ERROR = "CHECKER_ERROR"  # a bug in us, not in the model


@dataclass
class StageRecord:
    """One step of the pipeline. `status` is ok | fail | skipped."""

    stage: str
    status: str = "ok"
    ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "status": self.status, "ms": round(self.ms, 3), **self.data}


@dataclass
class CheckResult:
    verdict: Verdict
    stage: str
    stages: list[StageRecord]
    detail: dict[str, Any] = field(default_factory=dict)
    confidence: str = "high"

    @property
    def correct(self) -> bool:
        return self.verdict is Verdict.CORRECT

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "verdict_stage": self.stage,
            "confidence": self.confidence,
            "detail": self.detail,
            "stages": [s.to_dict() for s in self.stages],
        }

    def render(self) -> str:
        """Human-readable stage dump, used by --explain."""
        lines = [f"verdict={self.verdict.value} stage={self.stage} confidence={self.confidence}"]
        if self.detail:
            lines.append(f"  detail: {self.detail}")
        for s in self.stages:
            mark = {"ok": "+", "fail": "!", "skipped": "-"}.get(s.status, "?")
            lines.append(f"  {mark} {s.stage} ({s.ms:.1f}ms)")
            for key, value in s.data.items():
                text = str(value)
                if len(text) > 300:
                    text = text[:300] + f"... [{len(text)} chars]"
                lines.append(f"      {key}: {text}")
        return "\n".join(lines)


class _Unparseable(Exception):
    """Raised internally when input never becomes a usable expression."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason
        self.message = message


def _bind_real(expr: sp.Expr, name: str, real_symbol: sp.Symbol) -> sp.Expr:
    """Rebind the integration variable to a real symbol.

    Without ``real=True``, ``diff(log(Abs(x)), x)`` returns unevaluated
    ``re``/``im`` derivatives instead of ``sign(x)/Abs(x)``, which makes every
    ``ln|x|`` answer -- the single most common form -- unverifiable.
    """
    return expr.xreplace({s: real_symbol for s in expr.free_symbols if s.name == name})


def _normalize(expr: sp.Expr, rules: list[str]) -> sp.Expr:
    """Canonicalize an expression, appending the name of each rule that fires."""
    # Two-argument log, e.g. latex2sympy's log(u, E). Usually already collapsed
    # by the real-symbol rebind (which re-evaluates the node), but an explicit
    # base such as log(u, 10) survives and must be rewritten.
    if expr.has(sp.log):
        rewritten = expr.replace(
            lambda t: isinstance(t, sp.log) and len(t.args) == 2,
            lambda t: sp.log(t.args[0]) / sp.log(t.args[1]),
        )
        if rewritten != expr:
            rules.append("two_arg_log")
            expr = rewritten

    # log(Abs(u)) -> log(u). These differ by a locally constant i*pi, which
    # vanishes under d/dx. Applied ONLY inside log arguments, so a bare |x|
    # answer is never silently rescued.
    if expr.has(sp.Abs) and expr.has(sp.log):
        rewritten = expr.replace(
            lambda t: isinstance(t, sp.log) and isinstance(t.args[0], sp.Abs),
            lambda t: sp.log(t.args[0].args[0]),
        )
        if rewritten != expr:
            rules.append("log_abs")
            expr = rewritten

    # A model that echoed "\int ... dx" instead of answering leaves an
    # unevaluated Integral. Never .doit() that -- it would solve the problem on
    # the model's behalf and score a non-answer as CORRECT.
    if expr.has(sp.Integral):
        raise _Unparseable("echoed_integral", "answer is an unevaluated Integral")
    if expr.has(sp.Derivative):
        expr = expr.doit()
        rules.append("doit_derivative")

    return expr


def _zero_almost_everywhere(expr: sp.Expr | None) -> bool | None:
    """True when expr is zero except possibly on a measure-zero set.

    ``simplify(diff(log(Abs(x))) - 1/x)`` yields
    ``Piecewise((zoo, Eq(x, 0)), (0, True))`` -- zero everywhere except the
    singularity. A plain ``is_zero`` returns None for that, so the generic
    branch has to be inspected directly.
    """
    if expr is None:
        return None
    try:
        if expr == 0 or expr.is_zero:
            return True
    except (TypeError, AttributeError):
        pass
    if isinstance(expr, sp.Piecewise):
        generic = [e for e, cond in expr.args if cond == sp.true]
        if generic and all(e == 0 or bool(e.is_zero) for e in generic):
            return True
    return None


def _numeric_screen(derivative: sp.Expr, integrand: sp.Expr, symbol: sp.Symbol) -> dict[str, Any]:
    """Evaluate F' and f *separately* and compare pointwise.

    Separately, rather than lambdifying a simplified residual: `simplify` can
    produce a Piecewise containing `zoo`, which lambdify cannot compile at all
    (KeyError: 'ComplexInfinity').
    """
    use_real = any(e.has(*_NON_ANALYTIC) for e in (derivative, integrand))
    mode = "real" if use_real else "complex"

    previous_dps = mp.mp.dps
    mp.mp.dps = _DPS
    try:
        try:
            f_derivative = sp.lambdify(symbol, derivative, "mpmath")
            f_integrand = sp.lambdify(symbol, integrand, "mpmath")
        except Exception as exc:  # unlambdifiable node
            return {"mode": mode, "decision": "inconclusive", "n_usable": 0,
                    "error": f"{type(exc).__name__}: {exc}"}

        points = (
            [mp.mpf(str(p)) for p in _REAL_POINTS]
            if use_real
            else [mp.mpc(str(re_), str(im_)) for re_, im_ in _COMPLEX_POINTS]
        )

        usable = 0
        worst_residual = mp.mpf(0)
        worst_point = None
        for point in points:
            try:
                value_d = mp.mpmathify(f_derivative(point))
                value_f = mp.mpmathify(f_integrand(point))
            except Exception:
                continue
            if not (mp.isfinite(value_d) and mp.isfinite(value_f)):
                continue
            usable += 1
            scale = max(abs(value_d), abs(value_f), mp.mpf(1))
            residual = abs(value_d - value_f) / scale
            if residual > worst_residual:
                worst_residual = residual
                worst_point = point

        out: dict[str, Any] = {
            "mode": mode,
            "n_points": len(points),
            "n_usable": usable,
            "worst_rel_residual": mp.nstr(worst_residual, 6),
            "worst_point": mp.nstr(worst_point, 8) if worst_point is not None else None,
        }
        if usable < _MIN_USABLE_POINTS:
            out["decision"] = "inconclusive"
        elif worst_residual > _RTOL + _ATOL:
            out["decision"] = "nonzero"
        else:
            out["decision"] = "zero"
        return out
    finally:
        mp.mp.dps = previous_dps


def _symbolic_confirm_worker(derivative_srepr: str, integrand_srepr: str) -> bool | None:
    """Cheapest-first ladder. Returns True/False/None (undecided).

    Runs in a subprocess; takes srepr strings so no Expr graph is pickled.
    """
    derivative = sp.sympify(derivative_srepr)
    integrand = sp.sympify(integrand_srepr)
    residual = derivative - integrand

    if _zero_almost_everywhere(sp.expand(residual)):
        return True
    simplified = sp.simplify(residual)
    if _zero_almost_everywhere(simplified):
        return True
    for transform in (sp.trigsimp, sp.logcombine, sp.powsimp, sp.radsimp):
        try:
            if _zero_almost_everywhere(transform(simplified)):
                return True
        except Exception:
            continue
    try:
        # NOTE: a False from .equals() is NOT decisive. SymPy returns False for
        # diff(log(Abs(x))) vs 1/x, which is a genuine antiderivative pair. The
        # caller treats False as "undecided" whenever the numeric screen was
        # clean.
        return derivative.equals(integrand)
    except Exception:
        return None


_MP_CONTEXT = multiprocessing.get_context("forkserver")
_POOL: concurrent.futures.ProcessPoolExecutor | None = None


def _symbolic_confirm(
    derivative: sp.Expr, integrand: sp.Expr, timeout_s: float, use_subprocess: bool
) -> tuple[bool | None, str, dict[str, Any] | None]:
    """Return (result, status, error_detail) where status is ok | timeout | error.

    error_detail is populated (type/message/traceback) only when status == "error",
    so a checker crash is diagnosable from trace.jsonl without reproducing it by hand.
    """
    if not use_subprocess:
        try:
            return _symbolic_confirm_worker(sp.srepr(derivative), sp.srepr(integrand)), "ok", None
        except Exception as exc:
            return None, "error", {"type": type(exc).__name__, "message": str(exc),
                                    "traceback": traceback.format_exc()}

    # A hard timeout needs a killable process: SIGALRM is main-thread-only and
    # cannot interrupt a C-level SymPy loop. forkserver (not fork) because the
    # lm-eval API backend leaves worker threads alive, and forking a threaded
    # process risks inheriting held locks.
    global _POOL
    if _POOL is None:
        _POOL = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=_MP_CONTEXT)
    try:
        future = _POOL.submit(_symbolic_confirm_worker, sp.srepr(derivative), sp.srepr(integrand))
        return future.result(timeout=timeout_s), "ok", None
    except concurrent.futures.TimeoutError:
        # cancel_futures only drops pending work; the worker that timed out is
        # still executing and must be killed explicitly, or its manager thread
        # blocks forever joining it -- at the latest, at interpreter exit.
        for proc in _POOL._processes.values():
            if proc.is_alive():
                proc.kill()
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None
        return None, "timeout", None
    except Exception as exc:
        # ProcessPoolExecutor chains the worker's exception in via __cause__
        # (_RemoteTraceback), so format_exc() here recovers the child's actual
        # stack, not just "future.result() raised".
        _POOL = None
        return None, "error", {"type": type(exc).__name__, "message": str(exc),
                                "traceback": traceback.format_exc()}


def _to_expression(
    value: str | sp.Expr, kind: InputKind, symbol_name: str, stages: list[StageRecord], role: str
) -> sp.Expr:
    """Ingest raw input into a SymPy expression, recording each step."""
    from llm.core.parser.extraction import ExtractionStatus, extract_boxed

    if isinstance(value, sp.Expr) or (kind == "expr" and value is not None):
        stages.append(StageRecord(f"{role}_ingest", data={"kind": "expr", "str": str(value)}))
        return value  # type: ignore[return-value]

    if kind == "response":
        start = time.perf_counter()
        extraction = extract_boxed(value)
        stages.append(
            StageRecord(
                f"{role}_boxed_extract",
                status="ok" if extraction.status is ExtractionStatus.OK else "fail",
                ms=(time.perf_counter() - start) * 1000,
                data={
                    "n_matches": extraction.n_matches,
                    "chosen_index": extraction.chosen_index,
                    "chosen": extraction.value,
                    **({} if extraction.status is ExtractionStatus.OK else {"reason": extraction.status.value}),
                },
            )
        )
        if extraction.status is not ExtractionStatus.OK:
            raise _Unparseable(extraction.status.value, extraction.detail)
        value = extraction.value
        kind = "latex"

    if value is None:
        raise _Unparseable("empty", "no candidate text")
    text = value if isinstance(value, str) else str(value)
    if not text.strip():
        raise _Unparseable("empty", "no candidate text")

    if kind == "latex":
        from llm.core.parser.math_parser import MathLatexParser, canonicalize_latex

        start = time.perf_counter()
        rules: list[str] = []
        canonical = canonicalize_latex(text, rules)
        stages.append(
            StageRecord(
                f"{role}_latex_canonicalize",
                ms=(time.perf_counter() - start) * 1000,
                data={"rules_applied": rules, "in": text, "out": canonical},
            )
        )
        if not canonical.strip():
            raise _Unparseable("empty", "nothing left after canonicalization")

        start = time.perf_counter()
        try:
            expr = MathLatexParser().parse(canonical)
        except Exception as exc:
            stages.append(
                StageRecord(
                    f"{role}_latex2sympy",
                    status="fail",
                    ms=(time.perf_counter() - start) * 1000,
                    data={"error": {"type": type(exc).__name__, "msg": str(exc)}},
                )
            )
            raise _Unparseable("latex_error", f"{type(exc).__name__}: {exc}") from exc
        stages.append(
            StageRecord(
                f"{role}_latex2sympy",
                ms=(time.perf_counter() - start) * 1000,
                data={"str": str(expr), "srepr": sp.srepr(expr)[:500],
                      "free_symbols": sorted(s.name for s in expr.free_symbols)},
            )
        )
        return expr

    # kind == "sympy": a Python/SymPy-syntax string such as "exp(3*x)/3 + C"
    start = time.perf_counter()
    try:
        expr = sp.sympify(text, locals={symbol_name: sp.Symbol(symbol_name)})
    except Exception as exc:
        stages.append(
            StageRecord(f"{role}_sympify", status="fail",
                        ms=(time.perf_counter() - start) * 1000,
                        data={"error": {"type": type(exc).__name__, "msg": str(exc)}, "in": text})
        )
        raise _Unparseable("sympify_error", f"{type(exc).__name__}: {exc}") from exc
    stages.append(
        StageRecord(f"{role}_sympify", ms=(time.perf_counter() - start) * 1000,
                    data={"str": str(expr), "free_symbols": sorted(s.name for s in expr.free_symbols)})
    )
    return expr


def check_antiderivative(
    candidate: str | sp.Expr,
    integrand: str | sp.Expr,
    *,
    symbol: str = "x",
    timeout_s: float = 10.0,
    input_kind: InputKind = "latex",
    integrand_kind: InputKind = "sympy",
    use_subprocess: bool = True,
) -> CheckResult:
    """Decide whether ``candidate`` is an antiderivative of ``integrand``.

    ``input_kind`` selects how ``candidate`` is read:
      response -- raw model output; the \\boxed{} is extracted first
      latex    -- already-extracted LaTeX
      sympy    -- a Python/SymPy-syntax string
      expr     -- an existing sympy Expr

    Pure: no I/O, no globals, no environment reads. Callers that want a
    persisted trace wrap this and emit ``result.to_dict()``.
    """
    stages: list[StageRecord] = []
    started = time.perf_counter()

    def finish(verdict: Verdict, stage: str, detail: dict[str, Any] | None = None,
               confidence: str = "high") -> CheckResult:
        return CheckResult(verdict, stage, stages, dict(detail or {}, total_ms=round(
            (time.perf_counter() - started) * 1000, 3)), confidence)

    # --- S0 ingest -----------------------------------------------------------
    try:
        candidate_expr = _to_expression(candidate, input_kind, symbol, stages, "candidate")
        integrand_expr = _to_expression(integrand, integrand_kind, symbol, stages, "integrand")
    except _Unparseable as exc:
        return finish(Verdict.UNPARSEABLE, stages[-1].stage if stages else "ingest",
                      {"reason": exc.reason, "message": exc.message})
    except Exception as exc:
        return finish(Verdict.CHECKER_ERROR, "ingest",
                      {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})

    try:
        # --- S1 real symbol + S2 normalize -----------------------------------
        start = time.perf_counter()
        real_symbol = sp.Symbol(symbol, real=True)
        rules = ["bind_real_symbol"]
        candidate_expr = _bind_real(candidate_expr, symbol, real_symbol)
        integrand_expr = _bind_real(integrand_expr, symbol, real_symbol)
        try:
            candidate_expr = _normalize(candidate_expr, rules)
            integrand_expr = _normalize(integrand_expr, [])
        except _Unparseable as exc:
            stages.append(StageRecord("normalize", status="fail",
                                      ms=(time.perf_counter() - start) * 1000,
                                      data={"rules_applied": rules, "reason": exc.reason}))
            return finish(Verdict.UNPARSEABLE, "normalize",
                          {"reason": exc.reason, "message": exc.message})
        stages.append(StageRecord("normalize", ms=(time.perf_counter() - start) * 1000,
                                  data={"rules_applied": rules, "out": str(candidate_expr)}))

        # --- S3 differentiate ------------------------------------------------
        # Symbolic, not numeric: exact across branch cuts, needs no analyticity,
        # and additive integration constants vanish here for free.
        start = time.perf_counter()
        derivative = sp.diff(candidate_expr, real_symbol)
        stages.append(StageRecord("differentiate", ms=(time.perf_counter() - start) * 1000,
                                  data={"out": str(derivative)}))

        # --- S4 residual free symbols ----------------------------------------
        extra = {s for s in derivative.free_symbols if s != real_symbol}
        if extra:
            # "+ C" is already gone (d/dx C == 0); anything left is a genuine
            # free parameter, e.g. C*x.
            stages.append(StageRecord("residual_symbols", status="fail",
                                      data={"symbols": sorted(s.name for s in extra)}))
            return finish(Verdict.WRONG, "residual_symbols",
                          {"residual_symbols": sorted(s.name for s in extra)})

        # --- S5 structural zero ----------------------------------------------
        start = time.perf_counter()
        residual = derivative - integrand_expr
        structural = _zero_almost_everywhere(sp.expand(residual))
        stages.append(StageRecord("structural", ms=(time.perf_counter() - start) * 1000,
                                  data={"is_zero": bool(structural)}))
        if structural:
            return finish(Verdict.CORRECT, "structural")

        # --- S6 numeric screen -----------------------------------------------
        start = time.perf_counter()
        screen = _numeric_screen(derivative, integrand_expr, real_symbol)
        stages.append(StageRecord("numeric_screen", ms=(time.perf_counter() - start) * 1000,
                                  data=screen))
        if screen["decision"] == "nonzero":
            return finish(Verdict.WRONG, "numeric_screen",
                          {"worst_rel_residual": screen["worst_rel_residual"]})

        # --- S7 symbolic confirm ---------------------------------------------
        start = time.perf_counter()
        confirmed, status, error_detail = _symbolic_confirm(
            derivative, integrand_expr, timeout_s, use_subprocess)
        stages.append(StageRecord("symbolic_confirm",
                                  status="ok" if status == "ok" else "fail",
                                  ms=(time.perf_counter() - start) * 1000,
                                  data={"result": confirmed, "outcome": status}))
        if status == "timeout":
            if screen["decision"] == "zero":
                # The screen was clean; a timeout is not evidence of wrongness.
                return finish(Verdict.CORRECT, "numeric_screen",
                              {"note": "symbolic confirmation timed out"}, "numeric_only")
            return finish(Verdict.TIMEOUT, "symbolic_confirm", {"timeout_s": timeout_s})
        if status == "error":
            return finish(Verdict.CHECKER_ERROR, "symbolic_confirm",
                          {"error": f"{error_detail['type']}: {error_detail['message']}",
                           "traceback": error_detail["traceback"]})
        if confirmed:
            return finish(Verdict.CORRECT, "symbolic_confirm")
        if screen["decision"] == "zero":
            # SymPy could not prove it, but 8 independent points agree to 1e-12.
            # .equals() is known to return False for genuine pairs such as
            # diff(log|x|) vs 1/x, so a negative here does not override a clean
            # screen -- it only lowers confidence.
            return finish(Verdict.CORRECT, "numeric_screen",
                          {"note": "symbolic confirmation returned non-positive"}, "numeric_only")
        return finish(Verdict.WRONG, "symbolic_confirm", {"screen": screen["decision"]}, "low")

    except Exception as exc:
        return finish(Verdict.CHECKER_ERROR, "unexpected",
                      {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Explain a single antiderivative check.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--integrand", required=True)
    parser.add_argument("--kind", default="response",
                        choices=["response", "latex", "sympy", "expr"])
    parser.add_argument("--integrand-kind", default="sympy",
                        choices=["latex", "sympy", "expr"])
    parser.add_argument("--symbol", default="x")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    result = check_antiderivative(
        args.candidate, args.integrand, symbol=args.symbol, timeout_s=args.timeout,
        input_kind=args.kind, integrand_kind=args.integrand_kind, use_subprocess=False,
    )
    print(result.render())
    return 0 if result.verdict is Verdict.CORRECT else 1


if __name__ == "__main__":
    raise SystemExit(_main())
