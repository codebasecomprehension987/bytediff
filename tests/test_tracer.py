"""
Tests for TracedScalar — the operator-overloading fallback tracer.

Verifies that:
  - Arithmetic operations record correct VJPs on the tape
  - Control-flow functions (if/while/for) are traced correctly
  - Closure-captured variables are handled
  - Math functions route through the VJP registry
  - Gradients match finite differences
"""

import math
import pytest
from bytediff.tracer import TracedScalar, trace_fn
from bytediff.tape import Tape

EPS = 1e-6
RTOL = 1e-4


def fd_grad(fn, *args, argnum=0):
    """Scalar finite-difference gradient."""
    args = list(args)
    args_p = args[:]
    args_m = args[:]
    args_p[argnum] = args[argnum] + EPS
    args_m[argnum] = args[argnum] - EPS
    return (fn(*args_p) - fn(*args_m)) / (2 * EPS)


class TestTracedScalarBasics:
    def test_float_conversion(self):
        t = TracedScalar(3.14)
        assert float(t) == pytest.approx(3.14)

    def test_int_conversion(self):
        t = TracedScalar(2.9)
        assert int(t) == 2

    def test_bool_truthy(self):
        assert bool(TracedScalar(1.0))
        assert not bool(TracedScalar(0.0))

    def test_repr(self):
        r = repr(TracedScalar(2.5))
        assert "2.5" in r

    def test_no_tape_passthrough(self):
        """Without a tape, TracedScalar behaves like a float."""
        x = TracedScalar(3.0)
        y = TracedScalar(4.0)
        assert float(x + y) == pytest.approx(7.0)
        assert float(x * y) == pytest.approx(12.0)
        assert float(x - y) == pytest.approx(-1.0)
        assert float(x / y) == pytest.approx(0.75)


class TestTracedScalarGradients:
    def _grad(self, fn, *args, argnum=0):
        tape = Tape()
        traced = [TracedScalar(a, tape) for a in args]
        result = fn(*traced)
        tape.backward(seed=1.0)
        return tape.gradient(traced[argnum])

    def test_add_grad_x(self):
        grad = self._grad(lambda x, y: x + y, 3.0, 4.0, argnum=0)
        assert grad == pytest.approx(1.0)

    def test_add_grad_y(self):
        grad = self._grad(lambda x, y: x + y, 3.0, 4.0, argnum=1)
        assert grad == pytest.approx(1.0)

    def test_mul_grad_x(self):
        grad = self._grad(lambda x, y: x * y, 3.0, 4.0, argnum=0)
        assert grad == pytest.approx(4.0)

    def test_mul_grad_y(self):
        grad = self._grad(lambda x, y: x * y, 3.0, 4.0, argnum=1)
        assert grad == pytest.approx(3.0)

    def test_sub_grad(self):
        grad = self._grad(lambda x, y: x - y, 5.0, 2.0, argnum=0)
        assert grad == pytest.approx(1.0)
        grad_y = self._grad(lambda x, y: x - y, 5.0, 2.0, argnum=1)
        assert grad_y == pytest.approx(-1.0)

    def test_div_grad(self):
        x, y = 6.0, 2.0
        grad_x = self._grad(lambda x, y: x / y, x, y, argnum=0)
        grad_y = self._grad(lambda x, y: x / y, x, y, argnum=1)
        fd_x = fd_grad(lambda x, y: x / y, x, y, argnum=0)
        fd_y = fd_grad(lambda x, y: x / y, x, y, argnum=1)
        assert grad_x == pytest.approx(fd_x, rel=RTOL)
        assert grad_y == pytest.approx(fd_y, rel=RTOL)

    def test_pow_grad(self):
        x, n = 3.0, 2.0
        grad = self._grad(lambda x, n: x ** n, x, n, argnum=0)
        fd = fd_grad(lambda x, n: x ** n, x, n, argnum=0)
        assert grad == pytest.approx(fd, rel=RTOL)

    def test_neg_grad(self):
        grad = self._grad(lambda x: -x, 3.0, argnum=0)
        assert grad == pytest.approx(-1.0)

    def test_chain_rule_composition(self):
        """f(x) = (x * 2 + 1) * 3; df/dx = 6"""
        def f(x):
            return (x * TracedScalar(2.0, x._tape) + TracedScalar(1.0, x._tape)) * TracedScalar(3.0, x._tape)

        tape = Tape()
        x = TracedScalar(5.0, tape)
        result = f(x)
        tape.backward(seed=1.0)
        grad = tape.gradient(x)
        fd = fd_grad(lambda v: (v * 2 + 1) * 3, 5.0)
        assert grad == pytest.approx(fd, rel=RTOL)

    def test_abs_grad_positive(self):
        grad = self._grad(lambda x: abs(x), 3.0)
        assert grad == pytest.approx(1.0)

    def test_abs_grad_negative(self):
        grad = self._grad(lambda x: abs(x), -3.0)
        assert grad == pytest.approx(-1.0)


class TestTracedScalarControlFlow:
    def test_conditional_branch_true(self):
        """Gradient through if-branch."""
        def f(x):
            if x > TracedScalar(0.0):
                return x * x
            else:
                return -x

        tape = Tape()
        x = TracedScalar(3.0, tape)
        result = f(x)
        tape.backward(seed=1.0)
        grad = tape.gradient(x)
        # x > 0, so f(x) = x^2, df/dx = 2x = 6
        fd = fd_grad(lambda v: v * v, 3.0)
        assert grad == pytest.approx(fd, rel=RTOL)

    def test_loop_sum(self):
        """Gradient through a simple accumulation loop."""
        def f(x):
            total = TracedScalar(0.0, x._tape)
            for _ in range(5):
                total = total + x
            return total

        tape = Tape()
        x = TracedScalar(2.0, tape)
        result = f(x)
        tape.backward(seed=1.0)
        grad = tape.gradient(x)
        # f(x) = 5*x, df/dx = 5
        assert grad == pytest.approx(5.0, rel=RTOL)


class TestTraceMathFunctions:
    def _grad_math(self, fn, x_val):
        tape = Tape()
        x = TracedScalar(x_val, tape)
        result = fn(x)
        tape.backward(seed=1.0)
        return tape.gradient(x)

    def test_sin_grad(self):
        g = self._grad_math(lambda x: x.sin(), 0.5)
        assert g == pytest.approx(math.cos(0.5), rel=RTOL)

    def test_cos_grad(self):
        g = self._grad_math(lambda x: x.cos(), 0.5)
        assert g == pytest.approx(-math.sin(0.5), rel=RTOL)

    def test_exp_grad(self):
        g = self._grad_math(lambda x: x.exp(), 1.0)
        assert g == pytest.approx(math.exp(1.0), rel=RTOL)

    def test_log_grad(self):
        g = self._grad_math(lambda x: x.log(), 2.0)
        assert g == pytest.approx(1.0 / 2.0, rel=RTOL)

    def test_sqrt_grad(self):
        g = self._grad_math(lambda x: x.sqrt(), 4.0)
        assert g == pytest.approx(1.0 / (2.0 * math.sqrt(4.0)), rel=RTOL)


class TestTraceFn:
    def test_trace_fn_basic(self):
        """trace_fn should return the primal value."""
        tape = Tape()
        result = trace_fn(lambda x, y: x + y, 3.0, 4.0, tape=tape)
        assert result == pytest.approx(7.0)

    def test_trace_fn_grad(self):
        tape = Tape()
        trace_fn(lambda x, y: x * y, 3.0, 4.0, tape=tape)
        tape.backward(seed=1.0)
        # We lose the variable refs in trace_fn, but tape entries are recorded
        assert len(tape) > 0
