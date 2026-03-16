"""
VJP (vector-Jacobian product) registry for Python primitive operations.

Each entry maps an operator symbol to a tuple:
    (primal_fn, vjp_factory)

where ``vjp_factory(primals) -> vjp_fn`` returns a closure that accepts
a cotangent scalar and returns a tuple of cotangents for each input.

This is the mathematical heart of bytediff.  Every differentiable operator
intercepted in the bytecode must have a registered VJP here.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Primals = tuple
CotangentIn = float
CotangentOut = Tuple[float, ...]
VJPFn = Callable[[CotangentIn], CotangentOut]
VJPFactory = Callable[[Primals], VJPFn]
PrimalFn = Callable[..., float]

_registry: Dict[str, Tuple[PrimalFn, VJPFactory]] = {}


def register(op: str, primal_fn: PrimalFn, vjp_factory: VJPFactory) -> None:
    """Register a primitive VJP."""
    _registry[op] = (primal_fn, vjp_factory)


def lookup(op: str) -> Optional[Tuple[PrimalFn, VJPFactory]]:
    """Return the (primal_fn, vjp_factory) for ``op``, or None."""
    return _registry.get(op)


def has_vjp(op: str) -> bool:
    return op in _registry


# ---------------------------------------------------------------------------
# Binary arithmetic VJPs
# ---------------------------------------------------------------------------

# Addition: f(x,y) = x + y
# df/dx = 1, df/dy = 1
register(
    "+",
    primal_fn=lambda x, y: x + y,
    vjp_factory=lambda primals: (lambda g: (g, g)),
)

register(
    "+=",
    primal_fn=lambda x, y: x + y,
    vjp_factory=lambda primals: (lambda g: (g, g)),
)

# Subtraction: f(x,y) = x - y
# df/dx = 1, df/dy = -1
register(
    "-",
    primal_fn=lambda x, y: x - y,
    vjp_factory=lambda primals: (lambda g: (g, -g)),
)

register(
    "-=",
    primal_fn=lambda x, y: x - y,
    vjp_factory=lambda primals: (lambda g: (g, -g)),
)

# Multiplication: f(x,y) = x * y
# df/dx = y, df/dy = x
def _mul_vjp(primals):
    x, y = primals
    return lambda g: (g * y, g * x)

register("*",  primal_fn=lambda x, y: x * y,  vjp_factory=_mul_vjp)
register("*=", primal_fn=lambda x, y: x * y,  vjp_factory=_mul_vjp)

# True division: f(x,y) = x / y
# df/dx = 1/y, df/dy = -x/y^2
def _div_vjp(primals):
    x, y = primals
    return lambda g: (g / y, -g * x / (y * y))

register("/", primal_fn=lambda x, y: x / y, vjp_factory=_div_vjp)

# Power: f(x,y) = x ** y
# df/dx = y * x^(y-1), df/dy = x^y * ln(x)
def _pow_vjp(primals):
    x, y = primals
    result = x ** y
    def vjp(g):
        dx = g * y * (x ** (y - 1)) if x != 0 else 0.0
        dy = g * result * math.log(abs(x)) if x > 0 else 0.0
        return (dx, dy)
    return vjp

register("**",  primal_fn=lambda x, y: x ** y, vjp_factory=_pow_vjp)
register("**=", primal_fn=lambda x, y: x ** y, vjp_factory=_pow_vjp)

# Floor division: non-differentiable everywhere, return zero gradients
register(
    "//",
    primal_fn=lambda x, y: x // y,
    vjp_factory=lambda primals: (lambda g: (0.0, 0.0)),
)

# Modulo: df/dx = 1, df/dy ≈ 0 (non-smooth, zero for now)
register(
    "%",
    primal_fn=lambda x, y: x % y,
    vjp_factory=lambda primals: (lambda g: (g, 0.0)),
)

# ---------------------------------------------------------------------------
# Unary VJPs
# ---------------------------------------------------------------------------

# Negation: f(x) = -x, df/dx = -1
register(
    "UNARY_NEGATIVE",
    primal_fn=lambda x: -x,
    vjp_factory=lambda primals: (lambda g: (-g,)),
)

# Unary positive: f(x) = x, df/dx = 1
register(
    "UNARY_POSITIVE",
    primal_fn=lambda x: +x,
    vjp_factory=lambda primals: (lambda g: (g,)),
)

# ---------------------------------------------------------------------------
# Math function VJPs (for CALL interception of math.*)
# ---------------------------------------------------------------------------

_math_registry: Dict[str, Tuple[PrimalFn, VJPFactory]] = {}


def register_math(name: str, primal_fn: PrimalFn, vjp_factory: VJPFactory) -> None:
    _math_registry[name] = (primal_fn, vjp_factory)


def lookup_math(name: str) -> Optional[Tuple[PrimalFn, VJPFactory]]:
    return _math_registry.get(name)


# sin: df/dx = cos(x)
register_math("sin", math.sin, lambda p: (lambda g: (g * math.cos(p[0]),)))
# cos: df/dx = -sin(x)
register_math("cos", math.cos, lambda p: (lambda g: (-g * math.sin(p[0]),)))
# exp: df/dx = exp(x)
register_math("exp", math.exp, lambda p: (lambda g: (g * math.exp(p[0]),)))
# log: df/dx = 1/x
register_math("log", math.log, lambda p: (lambda g: (g / p[0],)))
# sqrt: df/dx = 1/(2*sqrt(x))
register_math("sqrt", math.sqrt, lambda p: (lambda g: (g / (2 * math.sqrt(p[0])),)))
# tanh: df/dx = 1 - tanh(x)^2
register_math("tanh", math.tanh, lambda p: (lambda g: (g * (1 - math.tanh(p[0])**2),)))
# atanh: df/dx = 1/(1 - x^2)
register_math("atanh", math.atanh, lambda p: (lambda g: (g / (1 - p[0]**2),)))
