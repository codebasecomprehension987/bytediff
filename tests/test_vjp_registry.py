"""
Tests for the VJP registry — ensures each registered primitive has
mathematically correct derivatives, verified against finite differences.
"""

import math
import pytest
from bytediff.bytecode.vjp_registry import lookup, lookup_math, has_vjp


EPS = 1e-6
RTOL = 1e-4  # relative tolerance for FD comparisons


def finite_diff_binary(op_fn, x, y, eps=EPS):
    """Return (df/dx, df/dy) via central finite differences."""
    dfdx = (op_fn(x + eps, y) - op_fn(x - eps, y)) / (2 * eps)
    dfdy = (op_fn(x, y + eps) - op_fn(x, y - eps)) / (2 * eps)
    return dfdx, dfdy


def finite_diff_unary(op_fn, x, eps=EPS):
    return (op_fn(x + eps) - op_fn(x - eps)) / (2 * eps)


class TestBinaryVJPs:
    @pytest.mark.parametrize("op,x,y", [
        ("+",  3.0, 4.0),
        ("+",  -1.0, 2.5),
        ("-",  5.0, 3.0),
        ("-",  0.0, 1.0),
        ("*",  2.0, 7.0),
        ("*",  -3.0, 4.0),
        ("/",  6.0, 2.0),
        ("/",  1.0, 3.0),
        ("**", 2.0, 3.0),
        ("**", 3.0, 2.0),
    ])
    def test_vjp_matches_finite_diff(self, op, x, y):
        entry = lookup(op)
        assert entry is not None, f"No VJP registered for '{op}'"
        primal_fn, vjp_factory = entry

        # Forward
        result = primal_fn(x, y)

        # VJP with seed g=1
        vjp_fn = vjp_factory((x, y))
        dx, dy = vjp_fn(1.0)

        # Finite difference reference
        fd_dx, fd_dy = finite_diff_binary(primal_fn, x, y)

        assert dx == pytest.approx(fd_dx, rel=RTOL, abs=1e-8), \
            f"{op}: dx={dx} != fd_dx={fd_dx}"
        assert dy == pytest.approx(fd_dy, rel=RTOL, abs=1e-8), \
            f"{op}: dy={dy} != fd_dy={fd_dy}"

    def test_vjp_linearity_with_seed(self):
        """VJP(g) = g * VJP(1) for all g."""
        entry = lookup("*")
        assert entry is not None
        _, vjp_factory = entry
        x, y = 3.0, 5.0
        vjp_fn = vjp_factory((x, y))
        dx1, dy1 = vjp_fn(1.0)
        for g in [0.5, 2.0, -1.0, 100.0]:
            dxg, dyg = vjp_fn(g)
            assert dxg == pytest.approx(g * dx1, rel=1e-12)
            assert dyg == pytest.approx(g * dy1, rel=1e-12)

    def test_addition_vjp_is_identity(self):
        entry = lookup("+")
        _, vjp_factory = entry
        for x, y in [(1.0, 2.0), (-5.0, 3.0), (0.0, 0.0)]:
            vjp_fn = vjp_factory((x, y))
            dx, dy = vjp_fn(1.0)
            assert dx == 1.0
            assert dy == 1.0

    def test_subtraction_vjp_signs(self):
        entry = lookup("-")
        _, vjp_factory = entry
        vjp_fn = vjp_factory((5.0, 3.0))
        dx, dy = vjp_fn(1.0)
        assert dx == 1.0
        assert dy == -1.0

    def test_floor_div_vjp_zero(self):
        entry = lookup("//")
        _, vjp_factory = entry
        vjp_fn = vjp_factory((7.0, 2.0))
        dx, dy = vjp_fn(1.0)
        assert dx == 0.0
        assert dy == 0.0

    def test_has_vjp(self):
        for op in ("+", "-", "*", "/", "**", "//", "%"):
            assert has_vjp(op), f"Expected VJP for '{op}'"
        assert not has_vjp("BITWISE_AND")


class TestUnaryVJPs:
    @pytest.mark.parametrize("op,x", [
        ("UNARY_NEGATIVE", 3.0),
        ("UNARY_NEGATIVE", -5.0),
        ("UNARY_POSITIVE", 2.5),
    ])
    def test_unary_vjp_correct(self, op, x):
        entry = lookup(op)
        assert entry is not None
        primal_fn, vjp_factory = entry
        vjp_fn = vjp_factory((x,))
        (dx,) = vjp_fn(1.0)
        fd_dx = finite_diff_unary(primal_fn, x)
        assert dx == pytest.approx(fd_dx, rel=RTOL, abs=1e-8)


class TestMathVJPs:
    @pytest.mark.parametrize("fn_name,x", [
        ("sin",  0.5),
        ("cos",  0.5),
        ("exp",  1.0),
        ("log",  2.0),
        ("sqrt", 4.0),
        ("tanh", 0.5),
    ])
    def test_math_vjp_matches_fd(self, fn_name, x):
        entry = lookup_math(fn_name)
        assert entry is not None, f"No VJP registered for math.{fn_name}"
        primal_fn, vjp_factory = entry
        vjp_fn = vjp_factory((x,))
        (dx,) = vjp_fn(1.0)
        fd_dx = finite_diff_unary(primal_fn, x)
        assert dx == pytest.approx(fd_dx, rel=RTOL, abs=1e-8), \
            f"math.{fn_name}: dx={dx} != fd_dx={fd_dx}"
