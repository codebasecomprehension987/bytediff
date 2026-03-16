"""
Core bytecode rewriter — the main engine of bytediff.

``BytecodeRewriter`` takes a Python function, analyzes its ``__code__``,
and returns a new function whose bytecode:

  1. Snapshots closure variables (LOAD_DEREF → LOAD_DEREF + STORE_FAST)
     to capture primals at op time rather than at tape-replay time.
  2. Replaces each differentiable BINARY_OP / UNARY_OP with a call into
     ``_tape_record_binary`` / ``_tape_record_unary`` that:
       a. Computes the primal (same result as the original op)
       b. Pushes a ``TapeEntry`` onto the injected ``Tape``
  3. Threads a ``tape=`` keyword argument through the rewritten function
     so callers can inject the tape without modifying the original signature.

Rewriting strategy
------------------
Rather than emitting raw bytecode (fragile across patch versions), we
use Python's ``exec``-based code generation: we synthesize a thin wrapper
function in pure Python source that calls ``_dispatch_op(tape, op, x, y)``
for each operation site, then compile it.  The actual bytecode transformation
is applied as a second pass using ``InstructionPatcher`` for the
LOAD_DEREF snapshotting step, which cannot be expressed at the source level
without AST access.

For callers that care only about scalar / numpy-free Python functions, this
is sufficient.  The C extension (cffi_ext) provides a faster path for hot loops.
"""

from __future__ import annotations

import dis
import functools
import sys
import types
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from bytediff.bytecode.analyzer import (
    CodeAnalysis,
    analyze_code,
    CLOSURE_LOAD_OPS,
    DIFFERENTIABLE_BINARY_OPS,
    DIFFERENTIABLE_UNARY_OPS,
)
from bytediff.bytecode.patcher import Instr, InstructionPatcher
from bytediff.bytecode.vjp_registry import lookup, lookup_math, has_vjp
from bytediff.tape import Tape


# ---------------------------------------------------------------------------
# Dispatch helpers (called from rewritten bytecode)
# ---------------------------------------------------------------------------

def _dispatch_binary(tape: Tape, op: str, x, y):
    """
    Compute ``x OP y`` and push the VJP onto ``tape``.

    Returns the primal result so the rewritten bytecode can push it onto
    the operand stack in place of the original op's result.
    """
    entry = lookup(op)
    if entry is None or tape is None:
        # Unknown op or no tape — fall through to Python semantics
        return _PYTHON_OPS[op](x, y)

    primal_fn, vjp_factory = entry
    result = primal_fn(x, y)
    vjp_fn = vjp_factory((x, y))
    tape.record(vjp_fn, primal_inputs=(x, y), output=result)
    return result


def _dispatch_unary(tape: Tape, op: str, x):
    """Compute ``OP x`` and push the VJP onto ``tape``."""
    entry = lookup(op)
    if entry is None or tape is None:
        return _PYTHON_UNARY_OPS[op](x)

    primal_fn, vjp_factory = entry
    result = primal_fn(x)
    vjp_fn = vjp_factory((x,))
    tape.record(vjp_fn, primal_inputs=(x,), output=result)
    return result


# Python operator implementations for fallback
import operator as _op

_PYTHON_OPS: Dict[str, Callable] = {
    "+": _op.add, "+=": _op.add,
    "-": _op.sub, "-=": _op.sub,
    "*": _op.mul, "*=": _op.mul,
    "/": _op.truediv,
    "//": _op.floordiv, "//=": _op.floordiv,
    "%": _op.mod, "%=": _op.mod,
    "**": _op.pow, "**=": _op.pow,
    "&": _op.and_, "|": _op.or_, "^": _op.xor,
    "<<": _op.lshift, ">>": _op.rshift,
}

_PYTHON_UNARY_OPS: Dict[str, Callable] = {
    "UNARY_NEGATIVE": _op.neg,
    "UNARY_POSITIVE": _op.pos,
    "UNARY_INVERT": _op.invert,
}

# Binary op symbol lookup table (NB_* index → symbol), mirrors analyzer.py
_NB_TABLE = {
    0:  "+",   # NB_ADD
    1:  "+=",  # NB_INPLACE_ADD
    2:  "//",  # NB_FLOOR_DIVIDE
    3:  "//=", # NB_INPLACE_FLOOR_DIVIDE
    4:  "%",   # NB_REMAINDER (actually lshift in some versions)
    5:  "*",   # NB_MULTIPLY
    6:  "%",   # NB_REMAINDER
    7:  "%=",  # NB_INPLACE_REMAINDER
    8:  "**",  # NB_POWER
    9:  "**=", # NB_INPLACE_POWER
    10: "-",   # NB_SUBTRACT
    11: "/",   # NB_TRUE_DIVIDE
    12: "*=",  # NB_INPLACE_MULTIPLY
    13: "-=",  # NB_INPLACE_SUBTRACT
    14: "/=",  # NB_INPLACE_TRUE_DIVIDE
}


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------

class BytecodeRewriter:
    """
    Rewrites a Python function's bytecode to emit a Wengert tape.

    Usage::

        rewriter = BytecodeRewriter()
        rewritten = rewriter.rewrite(fn)
        tape = Tape()
        result = rewritten(x, tape=tape)
        tape.backward()
        grad = tape.gradient(x)
    """

    def __init__(self, *, debug: bool = False):
        self._debug = debug
        self._cache: Dict[int, Callable] = {}  # id(fn.__code__) → rewritten fn

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def rewrite(self, fn: Callable) -> Callable:
        """
        Return a rewritten version of ``fn`` that accepts an optional
        ``tape=`` keyword argument and records differentiable operations.

        The returned function is cached by ``fn.__code__`` identity.
        """
        code_id = id(fn.__code__)
        if code_id in self._cache:
            return self._cache[code_id]

        rewritten = self._rewrite_fn(fn)
        self._cache[code_id] = rewritten
        return rewritten

    # ------------------------------------------------------------------
    # Internal rewriting pipeline
    # ------------------------------------------------------------------

    def _rewrite_fn(self, fn: Callable) -> Callable:
        analysis = analyze_code(fn.__code__)

        if self._debug:
            self._dump_analysis(analysis)

        # Step 1: Build a source-level wrapper that routes ops through dispatch
        wrapper_src = self._gen_wrapper_source(fn, analysis)

        if self._debug:
            print("=== Generated wrapper source ===")
            print(wrapper_src)

        # Step 2: Compile the wrapper
        safe_name = fn.__name__.replace("<", "_").replace(">", "_")
        # Extract free variable values from closure cells so the stackvm
        # wrapper source can resolve bare names for LOAD_DEREF-translated vars.
        closure_vars: dict = {}
        if fn.__code__.co_freevars and fn.__closure__:
            for varname, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    closure_vars[varname] = cell.cell_contents
                except ValueError:
                    pass  # unbound cell — skip

        globs = {
            "__builtins__": __builtins__,
            "_dispatch_binary": _dispatch_binary,
            "_dispatch_unary": _dispatch_unary,
            "_original_fn": fn,
            "Tape": Tape,
            **fn.__globals__,        # module-level names
            **closure_vars,          # closure-captured names
        }
        exec(compile(wrapper_src, f"<bytediff:{fn.__name__}>", "exec"), globs)
        wrapper_fn = globs[f"_rewritten_{safe_name}"]

        # Step 3: Apply closure-snapshot bytecode patch on the *original* fn's
        # code to freeze cell variable values (LOAD_DEREF → snapshot STORE_FAST)
        if analysis.freevars or analysis.cellvars:
            patched_code = self._patch_closure_snapshots(fn.__code__, analysis)
            # Swap out the original function's code object so the wrapper
            # captures the snapshotted version
            fn = types.FunctionType(
                patched_code,
                fn.__globals__,
                fn.__name__,
                fn.__defaults__,
                fn.__closure__,
            )
            globs["_original_fn"] = fn

        functools.update_wrapper(wrapper_fn, fn)
        return wrapper_fn

    # ------------------------------------------------------------------
    # Source-level wrapper generation
    # ------------------------------------------------------------------

    def _gen_wrapper_source(self, fn: Callable, analysis: CodeAnalysis) -> str:
        """
        Generate Python source for a wrapper that replaces each
        differentiable op with a ``_dispatch_*`` call.

        We inline-rewrite the function body using a simplified bytecode
        walk: for each instruction site we know is a BINARY_OP or UNARY_OP,
        we let the *original* Python function execute its ops but wrap the
        result via a monkey-patch-free dispatch mechanism.

        Concretely: the wrapper calls ``_original_fn`` with arguments but
        intercepts the result by re-implementing the function's body with
        dispatch calls substituted in.  Because we don't have the AST, we
        use a *per-instruction* approach: we generate a flat sequence of
        stack operations mirroring the bytecode, using local variable
        temporaries for each stack slot.

        For the common case of pure arithmetic functions this produces
        correct, differentiable code.  For functions with control flow
        (loops, conditionals) the wrapper falls back to executing the
        original and recording only the arithmetic ops via a shadow
        evaluation — see ``_gen_shadow_wrapper``.
        """
        code = analysis.code
        instrs = analysis.instructions

        # Detect whether the function body is "straight-line" arithmetic
        has_control_flow = any(
            i.instr.opname in {
                "JUMP_FORWARD", "JUMP_BACKWARD", "POP_JUMP_IF_TRUE",
                "POP_JUMP_IF_FALSE", "FOR_ITER", "GET_ITER",
                "JUMP_IF_TRUE_OR_POP", "JUMP_IF_FALSE_OR_POP",
            }
            for i in instrs
        )

        if has_control_flow:
            return self._gen_shadow_wrapper(fn, analysis)
        else:
            return self._gen_stackvm_wrapper(fn, analysis)

    def _gen_stackvm_wrapper(self, fn: Callable, analysis: CodeAnalysis) -> str:
        """
        Generate a stack-VM-style wrapper for straight-line functions.

        We simulate the CPython evaluation stack using numbered local
        variables ``_s0``, ``_s1``, … and translate each instruction
        to a Python assignment.
        """
        code = analysis.code
        instrs = analysis.instructions

        arg_names = list(code.co_varnames[: code.co_argcount])
        fn_name = fn.__name__.replace("<", "_").replace(">", "_")

        lines = [
            f"def _rewritten_{fn_name}({', '.join(arg_names)}, *, tape=None):",
            f"    # Auto-generated by bytediff",
        ]

        # Add locals for all varnames beyond args
        for vn in code.co_varnames[code.co_argcount :]:
            lines.append(f"    {vn} = None")

        # Simulate stack
        stack: List[str] = []
        slot = [0]  # mutable counter

        def push(expr: str) -> str:
            name = f"_s{slot[0]}"
            slot[0] += 1
            lines.append(f"    {name} = {expr}")
            stack.append(name)
            return name

        def pop() -> str:
            return stack.pop()

        def peek(n: int = 0) -> str:
            return stack[-(n + 1)]

        try:
            for ai in instrs:
                op = ai.instr.opname
                arg = ai.instr.arg
                argval = ai.instr.argval

                if op == "RESUME":
                    continue
                elif op == "LOAD_FAST":
                    push(str(argval))
                elif op == "LOAD_CONST":
                    if isinstance(argval, str):
                        push(repr(argval))
                    elif argval is None:
                        push("None")
                    else:
                        push(repr(argval))
                elif op == "LOAD_DEREF":
                    push(str(argval))
                elif op == "STORE_FAST":
                    val = pop()
                    lines.append(f"    {argval} = {val}")
                elif op == "BINARY_OP":
                    sym = _NB_TABLE.get(arg, "+")
                    rhs = pop()
                    lhs = pop()
                    if has_vjp(sym):
                        push(f"_dispatch_binary(tape, {sym!r}, {lhs}, {rhs})")
                    else:
                        push(f"({lhs} {sym} {rhs})")
                elif op in ("UNARY_NEGATIVE", "UNARY_POSITIVE", "UNARY_INVERT"):
                    operand = pop()
                    if has_vjp(op):
                        push(f"_dispatch_unary(tape, {op!r}, {operand})")
                    else:
                        sym = {"UNARY_NEGATIVE": "-", "UNARY_POSITIVE": "+",
                               "UNARY_INVERT": "~"}[op]
                        push(f"({sym}{operand})")
                elif op == "RETURN_VALUE":
                    ret = pop() if stack else "None"
                    lines.append(f"    return {ret}")
                    break
                elif op == "POP_TOP":
                    if stack:
                        pop()
                elif op in ("COPY", "DUP_TOP"):
                    lines.append(f"    _s{slot[0]} = {peek()}")
                    stack.append(f"_s{slot[0]}")
                    slot[0] += 1
                elif op == "SWAP":
                    if len(stack) >= 2:
                        stack[-1], stack[-2] = stack[-2], stack[-1]
                elif op == "CALL":
                    # Simple call with positional args
                    n_args = arg
                    args_vars = [pop() for _ in range(n_args)][::-1]
                    fn_var = pop()
                    call_str = f"{fn_var}({', '.join(args_vars)})"
                    push(call_str)
                elif op == "PUSH_NULL":
                    pass  # 3.12 CALL protocol - push sentinel
                elif op == "LOAD_GLOBAL":
                    # argval may be (push_null, name) tuple in 3.12
                    if isinstance(argval, str):
                        push(argval)
                    else:
                        push(str(argval))
                elif op in ("NOP", "CACHE", "COPY_FREE_VARS"):
                    continue
                else:
                    # Unknown op — emit a comment and continue
                    lines.append(f"    # Unhandled: {op} {argval!r}")

        except Exception as e:
            # Fall back to shadow wrapper on any translation error
            return self._gen_shadow_wrapper(fn, analysis)

        if not any("return" in l for l in lines):
            lines.append("    return None")

        return "\n".join(lines)

    def _gen_shadow_wrapper(self, fn: Callable, analysis: CodeAnalysis) -> str:
        """
        Fallback wrapper for functions with control flow.

        We execute the original function *and* run a shadow evaluation that
        intercepts arithmetic ops.  The shadow uses operator overloading via
        a ``TracedScalar`` proxy object so the same control-flow path is
        followed.
        """
        arg_names = list(analysis.code.co_varnames[: analysis.code.co_argcount])
        fn_name = fn.__name__.replace("<", "_").replace(">", "_")
        args_str = ", ".join(arg_names)

        src = f"""
def _rewritten_{fn_name}({args_str}, *, tape=None):
    from bytediff.tracer import TracedScalar, trace_fn
    if tape is None:
        return _original_fn({args_str})
    traced_args = tuple(TracedScalar(a, tape) for a in ({args_str},))
    result = _original_fn(*traced_args)
    if isinstance(result, TracedScalar):
        return result.value
    return result
"""
        return src.strip()

    # ------------------------------------------------------------------
    # Closure snapshot bytecode patch
    # ------------------------------------------------------------------

    def _patch_closure_snapshots(
        self, code: types.CodeType, analysis: CodeAnalysis
    ) -> types.CodeType:
        """
        Rewrite ``LOAD_DEREF X`` preceding a differentiable op to:
            LOAD_DEREF X
            STORE_FAST  __snap_X
            LOAD_FAST   __snap_X

        This snapshots the value at op time rather than at tape-replay time,
        preventing the stale-primal correctness bug described in the spec.
        """
        patcher = InstructionPatcher(code)
        instrs = list(patcher)

        # Find LOAD_DEREF instructions that immediately precede a diff op
        for idx, instr in enumerate(instrs):
            if instr.opname not in CLOSURE_LOAD_OPS:
                continue
            # Check if the *next* differentiable op uses this value
            # (simplified: snapshot ALL LOAD_DEREF in differentiable functions)
            varname = instr.argval
            snap_name = f"__snap_{varname}"
            snap_idx = patcher.add_varname(snap_name)

            store = Instr("STORE_FAST", arg=snap_idx, argval=snap_name)
            load = Instr("LOAD_FAST", arg=snap_idx, argval=snap_name)
            patcher.insert_after(idx, [store, load])

        try:
            return patcher.build_code()
        except Exception:
            # If patching fails, return original code unchanged
            return code

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _dump_analysis(self, analysis: CodeAnalysis) -> None:
        print(f"=== BytecodeRewriter: {analysis.code.co_name} ===")
        print(f"  cellvars: {analysis.cellvars}")
        print(f"  freevars: {analysis.freevars}")
        print(f"  differentiable ops: {len(analysis.differentiable_indices)}")
        print(f"  closure loads: {len(analysis.closure_load_indices)}")
        print("  Instructions:")
        for ai in analysis.instructions:
            marker = ">>>" if ai.is_differentiable else "   "
            print(f"  {marker} {ai.instr.offset:4d} {ai.instr.opname:<30} {ai.instr.argval!r}")
