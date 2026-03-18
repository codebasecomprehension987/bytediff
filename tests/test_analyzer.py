"""
Tests for the CPython 3.12 bytecode analyzer.

Verifies that BINARY_OP, UNARY_OP, LOAD_DEREF and CALL sites
are correctly identified across a variety of function shapes.
"""

import pytest
from bytediff.bytecode.analyzer import analyze_code, iter_differentiable


class TestAnalyzeCode:
    def test_simple_addition(self):
        def f(x, y):
            return x + y

        a = analyze_code(f.__code__)
        assert len(a.differentiable_indices) >= 1
        ops = [i.binary_op_name for i in iter_differentiable(a)]
        assert any(o in ("+", "+=") for o in ops)

    def test_polynomial(self):
        def f(x):
            return x * x + 2.0 * x + 1.0

        a = analyze_code(f.__code__)
        ops = [i.binary_op_name for i in iter_differentiable(a)]
        assert any(o in ("*", "*=") for o in ops)
        assert any(o in ("+", "+=") for o in ops)

    def test_no_math_no_diff(self):
        def f(x):
            return x

        a = analyze_code(f.__code__)
        assert len(a.differentiable_indices) == 0

    def test_closure_freevars_detected(self):
        outer = 5.0
        def f(x):
            return x * outer

        a = analyze_code(f.__code__)
        assert "outer" in a.freevars

    def test_closure_load_indices(self):
        outer = 5.0
        def f(x):
            return x * outer

        a = analyze_code(f.__code__)
        assert len(a.closure_load_indices) > 0

    def test_no_freevars_for_simple_fn(self):
        def f(x, y):
            return x + y

        a = analyze_code(f.__code__)
        assert len(a.freevars) == 0

    def test_unary_neg_detected(self):
        def f(x):
            return -x

        a = analyze_code(f.__code__)
        # UNARY_NEGATIVE should be in differentiable ops
        names = [a.instructions[i].instr.opname for i in a.differentiable_indices]
        assert "UNARY_NEGATIVE" in names

    def test_call_sites_detected(self):
        import math
        def f(x):
            return math.sin(x)

        a = analyze_code(f.__code__)
        assert len(a.call_indices) >= 1

    def test_cellvars(self):
        def outer():
            x = 1.0
            def inner():
                return x + 1.0
            return inner

        a = analyze_code(outer.__code__)
        assert "x" in a.cellvars

    def test_analysis_is_non_destructive(self):
        """analyze_code should not modify the original function."""
        def f(x):
            return x * x

        original_result = f(3.0)
        _ = analyze_code(f.__code__)
        assert f(3.0) == original_result
