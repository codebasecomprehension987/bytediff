"""
TracedScalar — operator-overloading proxy for tracing through control flow.

When ``BytecodeRewriter`` detects a function with conditionals or loops,
it falls back to the "shadow tracer" strategy: wrap each input in a
``TracedScalar`` and let Python's dunder methods intercept every arithmetic
operation, recording VJPs on the tape as they happen.

This is conceptually similar to how PyTorch's autograd works (via
``__torch_function__``), but applies to arbitrary Python scalars rather
than tensors.  The trade-off vs the bytecode-rewrite path is:
  - Slower per-op (Python method dispatch)
  - Handles loops / conditionals correctly without bytecode analysis
  - Cannot differentiate through C-extension functions that bypass dunders
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

from bytediff.bytecode.vjp_registry import lookup, lookup_math
from bytediff.tape import Tape


class TracedScalar:
    """
    A scalar proxy that records arithmetic operations on a ``Tape``.

    Wraps a Python ``float`` (or ``int``) and overloads every arithmetic
    dunder to:
      1. Compute the primal result.
      2. Look up the VJP in the registry.
      3. Push a ``TapeEntry`` onto the tape.
      4. Return a new ``TracedScalar`` wrapping the primal result.

    This allows the *original, unmodified* Python control-flow to run while
    the tape records the mathematical dependencies.
    """

    __slots__ = ("value", "_tape", "_id")
    _id_counter = 0

    def __init__(self, value: float, tape: Optional[Tape] = None):
        self.value = float(value)
        self._tape = tape
        TracedScalar._id_counter += 1
        self._id = TracedScalar._id_counter

    def __repr__(self) -> str:
        return f"TracedScalar({self.value})"

    def __float__(self) -> float:
        return self.value

    def __int__(self) -> int:
        return int(self.value)

    def __bool__(self) -> bool:
        return bool(self.value)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _binary_op(self, other: Any, op: str) -> "TracedScalar":
        x = self.value
        y = other.value if isinstance(other, TracedScalar) else float(other)
        tape = self._tape or (other._tape if isinstance(other, TracedScalar) else None)

        entry = lookup(op)
        if entry is None or tape is None:
            import operator as _o
            fn = {
                "+": _o.add, "-": _o.sub, "*": _o.mul, "/": _o.truediv,
                "//": _o.floordiv, "%": _o.mod, "**": _o.pow,
            }.get(op, _o.add)
            return TracedScalar(fn(x, y), tape)

        primal_fn, vjp_factory = entry
        result_val = primal_fn(x, y)
        result = TracedScalar(result_val, tape)
        vjp_fn = vjp_factory((x, y))
        tape.record(vjp_fn, primal_inputs=(x, y), output=result_val)
        # Keep track of the TracedScalar ↔ primal value for gradient lookup
        tape._var_ids[id(self)] = self
        tape._var_ids[id(result)] = result
        return result

    def _rbinary_op(self, other: Any, op: str) -> "TracedScalar":
        """Right-hand version: other OP self."""
        x = other.value if isinstance(other, TracedScalar) else float(other)
        y = self.value
        tape = self._tape

        entry = lookup(op)
        if entry is None or tape is None:
            import operator as _o
            fn = {"+": _o.add, "-": _o.sub, "*": _o.mul, "/": _o.truediv}.get(op, _o.add)
            return TracedScalar(fn(x, y), tape)

        primal_fn, vjp_factory = entry
        result_val = primal_fn(x, y)
        result = TracedScalar(result_val, tape)
        vjp_fn = vjp_factory((x, y))
        tape.record(vjp_fn, primal_inputs=(x, y), output=result_val)
        return result

    def _unary_op(self, op: str) -> "TracedScalar":
        x = self.value
        tape = self._tape
        entry = lookup(op)
        if entry is None or tape is None:
            fn = {"UNARY_NEGATIVE": lambda v: -v, "UNARY_POSITIVE": lambda v: +v}.get(op)
            return TracedScalar(fn(x) if fn else x, tape)

        primal_fn, vjp_factory = entry
        result_val = primal_fn(x)
        result = TracedScalar(result_val, tape)
        vjp_fn = vjp_factory((x,))
        tape.record(vjp_fn, primal_inputs=(x,), output=result_val)
        return result

    # ------------------------------------------------------------------
    # Arithmetic dunders
    # ------------------------------------------------------------------

    def __add__(self, other): return self._binary_op(other, "+")
    def __radd__(self, other): return self._rbinary_op(other, "+")
    def __sub__(self, other): return self._binary_op(other, "-")
    def __rsub__(self, other): return self._rbinary_op(other, "-")
    def __mul__(self, other): return self._binary_op(other, "*")
    def __rmul__(self, other): return self._rbinary_op(other, "*")
    def __truediv__(self, other): return self._binary_op(other, "/")
    def __rtruediv__(self, other): return self._rbinary_op(other, "/")
    def __floordiv__(self, other): return self._binary_op(other, "//")
    def __mod__(self, other): return self._binary_op(other, "%")
    def __pow__(self, other): return self._binary_op(other, "**")
    def __rpow__(self, other): return self._rbinary_op(other, "**")
    def __neg__(self): return self._unary_op("UNARY_NEGATIVE")
    def __pos__(self): return self._unary_op("UNARY_POSITIVE")
    def __abs__(self):
        sign = 1.0 if self.value >= 0 else -1.0
        tape = self._tape
        result_val = abs(self.value)
        result = TracedScalar(result_val, tape)
        if tape is not None:
            tape.record(lambda g: (g * sign,), primal_inputs=(self.value,), output=result_val)
        return result

    # ------------------------------------------------------------------
    # Comparisons (non-differentiable; return plain bool)
    # ------------------------------------------------------------------

    def __lt__(self, other):
        v = other.value if isinstance(other, TracedScalar) else float(other)
        return self.value < v

    def __le__(self, other):
        v = other.value if isinstance(other, TracedScalar) else float(other)
        return self.value <= v

    def __gt__(self, other):
        v = other.value if isinstance(other, TracedScalar) else float(other)
        return self.value > v

    def __ge__(self, other):
        v = other.value if isinstance(other, TracedScalar) else float(other)
        return self.value >= v

    def __eq__(self, other):
        v = other.value if isinstance(other, TracedScalar) else float(other)
        return self.value == v

    # ------------------------------------------------------------------
    # Math function support
    # ------------------------------------------------------------------

    def sin(self):
        return self._math_fn("sin")

    def cos(self):
        return self._math_fn("cos")

    def exp(self):
        return self._math_fn("exp")

    def log(self):
        return self._math_fn("log")

    def sqrt(self):
        return self._math_fn("sqrt")

    def tanh(self):
        return self._math_fn("tanh")

    def _math_fn(self, name: str) -> "TracedScalar":
        entry = lookup_math(name)
        if entry is None or self._tape is None:
            fn = getattr(math, name)
            return TracedScalar(fn(self.value), self._tape)
        primal_fn, vjp_factory = entry
        result_val = primal_fn(self.value)
        result = TracedScalar(result_val, self._tape)
        vjp_fn = vjp_factory((self.value,))
        self._tape.record(vjp_fn, primal_inputs=(self.value,), output=result_val)
        return result


# ---------------------------------------------------------------------------
# Traced math module shim
# ---------------------------------------------------------------------------

class _TracedMathModule:
    """
    Drop-in replacement for the ``math`` module that returns ``TracedScalar``
    values and records VJPs.  Inject via ``builtins`` or pass explicitly.
    """

    def __init__(self, tape: Tape):
        self._tape = tape

    def _wrap(self, fn_name: str, x):
        if isinstance(x, TracedScalar):
            return x._math_fn(fn_name)
        entry = lookup_math(fn_name)
        if entry is None:
            return getattr(math, fn_name)(x)
        primal_fn, vjp_factory = entry
        result_val = primal_fn(float(x))
        result = TracedScalar(result_val, self._tape)
        vjp_fn = vjp_factory((float(x),))
        self._tape.record(vjp_fn, primal_inputs=(float(x),), output=result_val)
        return result

    def sin(self, x):   return self._wrap("sin", x)
    def cos(self, x):   return self._wrap("cos", x)
    def exp(self, x):   return self._wrap("exp", x)
    def log(self, x):   return self._wrap("log", x)
    def sqrt(self, x):  return self._wrap("sqrt", x)
    def tanh(self, x):  return self._wrap("tanh", x)
    def atanh(self, x): return self._wrap("atanh", x)

    # Passthrough for non-differentiable math functions
    def __getattr__(self, name: str):
        return getattr(math, name)


def trace_fn(fn: Callable, *args, tape: Tape) -> Any:
    """
    Execute ``fn(*args)`` with inputs wrapped as ``TracedScalar`` objects.

    Returns the primal output (unwrapped).
    """
    traced = tuple(
        TracedScalar(a, tape) if isinstance(a, (int, float)) else a
        for a in args
    )
    result = fn(*traced)
    if isinstance(result, TracedScalar):
        return result.value
    return result
