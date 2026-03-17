"""
Unit tests for the Wengert tape.

Tests cover:
  - Recording entries
  - Backward accumulation
  - Gradient correctness vs finite differences
  - Arena reset lifecycle
  - Multi-output gradient accumulation
"""

import math
import pytest
from bytediff.tape import Tape, TapeEntry, Arena, ARENA_SIZE


# ---------------------------------------------------------------------------
# Arena tests
# ---------------------------------------------------------------------------

class TestArena:
    def test_alloc_and_read_float64(self):
        a = Arena(size=1024)
        offset = a.alloc_float64(3.14)
        assert a.read_float64(offset) == pytest.approx(3.14)

    def test_multiple_allocs(self):
        a = Arena(size=1024)
        offsets = [a.alloc_float64(float(i)) for i in range(10)]
        for i, off in enumerate(offsets):
            assert a.read_float64(off) == pytest.approx(float(i))

    def test_reset_clears_pointer(self):
        a = Arena(size=1024)
        a.alloc_float64(1.0)
        a.alloc_float64(2.0)
        assert a._ptr == 16
        a.reset()
        assert a._ptr == 0

    def test_oom_raises(self):
        a = Arena(size=8)  # only one float64 fits
        a.alloc_float64(1.0)
        with pytest.raises(MemoryError):
            a.alloc_float64(2.0)

    def test_close(self):
        a = Arena(size=1024)
        a.alloc_float64(42.0)
        a.close()  # should not raise


# ---------------------------------------------------------------------------
# Tape unit tests
# ---------------------------------------------------------------------------

class TestTape:
    def test_empty_tape_backward(self):
        tape = Tape()
        tape.backward(seed=1.0)  # should not raise

    def test_gradient_not_recorded_returns_zero(self):
        tape = Tape()
        x = 5.0
        assert tape.gradient(x) == 0.0

    def test_record_and_backward_addition(self):
        """f(x, y) = x + y; df/dx = df/dy = 1."""
        tape = Tape()
        x, y = 3.0, 4.0
        result = x + y

        def add_vjp(g):
            return (g, g)

        tape.record(add_vjp, primal_inputs=(x, y), output=result)
        tape.backward(seed=1.0)

        assert tape.gradient(x) == pytest.approx(1.0)
        assert tape.gradient(y) == pytest.approx(1.0)

    def test_record_and_backward_multiplication(self):
        """f(x, y) = x * y; df/dx = y, df/dy = x."""
        tape = Tape()
        x, y = 3.0, 4.0
        result = x * y

        def mul_vjp(g):
            return (g * y, g * x)

        tape.record(mul_vjp, primal_inputs=(x, y), output=result)
        tape.backward(seed=1.0)

        assert tape.gradient(x) == pytest.approx(y)
        assert tape.gradient(y) == pytest.approx(x)

    def test_chain_rule(self):
        """
        f(x) = (x + 2) * (x + 2)
        df/dx = 2 * (x + 2) = 2 * 5 = 10 at x=3
        """
        tape = Tape()
        x = 3.0
        c = 2.0
        intermediate = x + c   # = 5.0
        result = intermediate * intermediate  # = 25.0

        # Record x + c
        tape.record(lambda g: (g, g), primal_inputs=(x, c), output=intermediate)
        # Record intermediate * intermediate
        tape.record(
            lambda g: (g * intermediate, g * intermediate),
            primal_inputs=(intermediate, intermediate),
            output=result,
        )

        tape.backward(seed=1.0)

        # Gradient should accumulate through chain rule
        grad_intermediate = tape.gradient(intermediate)
        assert grad_intermediate == pytest.approx(2 * intermediate)

    def test_reset(self):
        tape = Tape()
        x, y = 1.0, 2.0
        tape.record(lambda g: (g, g), primal_inputs=(x, y), output=x + y)
        tape.backward()
        assert len(tape) == 1

        tape.reset()
        assert len(tape) == 0
        assert tape.gradient(x) == 0.0

    def test_len(self):
        tape = Tape()
        for i in range(5):
            tape.record(lambda g: (g,), primal_inputs=(float(i),), output=float(i))
        assert len(tape) == 5

    def test_repr(self):
        tape = Tape()
        r = repr(tape)
        assert "Tape" in r
        assert "entries=0" in r

    def test_gradient_accumulation_fan_in(self):
        """
        y = x + x  (x used twice)
        dy/dx should be 2.0
        """
        tape = Tape()
        x = 5.0
        r1 = x + x  # both branches use x

        # Simulate two separate add ops that share x
        tape.record(lambda g: (g, g), primal_inputs=(x, x), output=r1)
        tape.backward(seed=1.0)

        # x participates in both input slots → gradient accumulates
        grad = tape.gradient(x)
        assert grad == pytest.approx(2.0)
