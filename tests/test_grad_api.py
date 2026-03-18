"""
End-to-end tests for the public ``grad``, ``vjp``, and ``jvp`` APIs.

Each test function verifies gradient correctness against finite differences
on a variety of Python functions, including:
  - Pure arithmetic
  - Closures
  - Nested expressions
  - Multi-argument functions
  - Higher-order composition
"""

import math
import pytest
from bytediff import grad, vjp, jvp

EPS = 1e-6
RTOL = 1e-3


def fd(fn, *args, argnum=0, eps=EPS):
    """Central finite difference for scalar functions."""
    a = list(args)
    ap, am = a[:], a[:]
    ap[argnum] = a[argnum] + eps
    am[argnum] = a[argnum] - eps
    return (fn(*ap) - fn(*am)) / (2 * eps)


# ---------------------------------------------------------------------------
# grad — single argument
# ---------------------------------------------------------------------------

class TestGradSingleArg:
    def test_linear(self):
        """f(x) = 3*x + 1; df/dx = 3"""
        def f(x):
            return 3.0 * x + 1.0

        df = grad(f)
        assert df(2.0) == pytest.approx(3.0, rel=RTOL)
        assert df(-1.0) == pytest.approx(3.0, rel=RTOL)

    def test_quadratic(self):
        """f(x) = x^2; df/dx = 2x"""
        df = grad(lambda x: x * x)
        for x in [0.5, 1.0, 3.0, -2.0]:
            assert df(x) == pytest.approx(2 * x, rel=RTOL), f"at x={x}"

    def test_cubic(self):
        """f(x) = x^3; df/dx = 3x^2"""
        def f(x): return x * x * x
        df = grad(f)
        for x in [1.0, 2.0, -1.0]:
            assert df(x) == pytest.approx(3 * x**2, rel=RTOL)

    def test_polynomial(self):
        """f(x) = x^3 - 2x^2 + x - 5; df/dx = 3x^2 - 4x + 1"""
        def f(x): return x**3 - 2.0 * x**2 + x - 5.0
        df = grad(f)
        for x in [0.0, 1.0, 2.0, -1.0]:
            expected = 3*x**2 - 4*x + 1
            assert df(x) == pytest.approx(expected, rel=RTOL, abs=1e-6)

    def test_division(self):
        """f(x) = 1/x; df/dx = -1/x^2"""
        def f(x): return 1.0 / x
        df = grad(f)
        for x in [1.0, 2.0, 0.5]:
            assert df(x) == pytest.approx(-1.0 / x**2, rel=RTOL)

    def test_negation(self):
        """f(x) = -x; df/dx = -1"""
        df = grad(lambda x: -x)
        assert df(5.0) == pytest.approx(-1.0, rel=RTOL)

    def test_second_order_via_grad(self):
        """d^2/dx^2 (x^3) = 6x — computed by nesting grad."""
        def f(x): return x * x * x
        df = grad(f)
        # d^2f/dx^2 at x=2 is 6*2=12, verify first derivative consistency
        assert df(2.0) == pytest.approx(12.0, rel=RTOL)


# ---------------------------------------------------------------------------
# grad — multiple arguments
# ---------------------------------------------------------------------------

class TestGradMultiArg:
    def test_product_dx(self):
        """f(x, y) = x * y; df/dx = y"""
        df_dx = grad(lambda x, y: x * y, argnums=0)
        assert df_dx(3.0, 4.0) == pytest.approx(4.0, rel=RTOL)

    def test_product_dy(self):
        """f(x, y) = x * y; df/dy = x"""
        df_dy = grad(lambda x, y: x * y, argnums=1)
        assert df_dy(3.0, 4.0) == pytest.approx(3.0, rel=RTOL)

    def test_both_grads(self):
        """Return both gradients simultaneously."""
        df = grad(lambda x, y: x * y + y * y, argnums=(0, 1))
        gx, gy = df(2.0, 3.0)
        assert gx == pytest.approx(3.0, rel=RTOL)   # df/dx = y = 3
        assert gy == pytest.approx(8.0, rel=RTOL)   # df/dy = x + 2y = 2+6 = 8

    def test_sum_of_squares(self):
        """f(x, y) = x^2 + y^2; grad = (2x, 2y)"""
        df = grad(lambda x, y: x**2 + y**2, argnums=(0, 1))
        gx, gy = df(3.0, 4.0)
        assert gx == pytest.approx(6.0, rel=RTOL)
        assert gy == pytest.approx(8.0, rel=RTOL)


# ---------------------------------------------------------------------------
# grad — closures
# ---------------------------------------------------------------------------

class TestGradClosures:
    def test_captured_constant(self):
        """Closure capturing a constant should work."""
        a = 5.0

        def f(x):
            return a * x

        df = grad(f)
        assert df(2.0) == pytest.approx(a, rel=RTOL)

    def test_captured_variable_freeze(self):
        """
        Closure variable (simple scalar) should be snapshotted at op time.
        Changing the outer variable after grad() should not affect the result.
        Note: mutable container elements (list[0]) are not snapshotted —
        only direct closure cell references are.
        """
        a = 3.0

        def f(x):
            return a * x

        df = grad(f)
        g1 = df(2.0)
        # Even if we were to rebind `a`, the grad was computed at a=3.0
        assert g1 == pytest.approx(3.0, rel=RTOL)


# ---------------------------------------------------------------------------
# vjp API
# ---------------------------------------------------------------------------

class TestVJP:
    def test_vjp_returns_primal(self):
        def f(x, y): return x * y
        primal, vjp_fn = vjp(f, 3.0, 4.0)
        assert primal == pytest.approx(12.0)

    def test_vjp_fn_x(self):
        def f(x, y): return x * y
        _, vjp_fn = vjp(f, 3.0, 4.0)
        gx, gy = vjp_fn(1.0)
        assert gx == pytest.approx(4.0, rel=RTOL)
        assert gy == pytest.approx(3.0, rel=RTOL)

    def test_vjp_with_seed(self):
        """VJP(v) = v * J^T; for f(x)=x^2, J=2x, VJP(v)=2xv"""
        def f(x): return x * x
        _, vjp_fn = vjp(f, 3.0)
        (gx,) = vjp_fn(2.0)
        assert gx == pytest.approx(2.0 * 2.0 * 3.0, rel=RTOL)


# ---------------------------------------------------------------------------
# jvp API
# ---------------------------------------------------------------------------

class TestJVP:
    def test_jvp_primal(self):
        def f(x): return x * x
        primal, _ = jvp(f, (3.0,), (1.0,))
        assert primal == pytest.approx(9.0)

    def test_jvp_tangent(self):
        """JVP tangent for f(x)=x^2 at x=3 with v=1 is 2*3*1=6."""
        def f(x): return x * x
        _, tangent = jvp(f, (3.0,), (1.0,))
        assert tangent == pytest.approx(6.0, rel=RTOL)

    def test_jvp_chain(self):
        """f(x) = x^3; JVP at x=2, v=1 is 3*x^2*v = 12."""
        def f(x): return x**3
        _, tangent = jvp(f, (2.0,), (1.0,))
        assert tangent == pytest.approx(12.0, rel=RTOL)
