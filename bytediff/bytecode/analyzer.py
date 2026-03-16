"""
CPython 3.12 bytecode analyzer.

Wraps ``dis`` to produce a structured view of the instruction stream,
identifying:
  - BINARY_OP / UNARY_OP / CALL sites to intercept
  - LOAD_DEREF / STORE_DEREF for closure variable tracking
  - RESUME, PUSH_EXC_INFO and other 3.12-specific instructions
  - co_cellvars / co_freevars metadata for closure snapshotting
"""

from __future__ import annotations

import dis
import sys
from dataclasses import dataclass, field
from types import CodeType
from typing import Iterator, List, Optional, Set

if sys.version_info < (3, 12):
    raise RuntimeError("bytediff requires CPython 3.12+")

# ---------------------------------------------------------------------------
# CPython 3.12 opcodes we care about
# ---------------------------------------------------------------------------

# Arithmetic / unary ops whose VJPs we know
DIFFERENTIABLE_BINARY_OPS: Set[str] = {
    "BINARY_OP",       # covers +, -, *, /, **, //, %
}

DIFFERENTIABLE_UNARY_OPS: Set[str] = {
    "UNARY_NEGATIVE",
    "UNARY_POSITIVE",
    "UNARY_INVERT",    # only differentiable for floats via neg
}

# 3.12-specific specialised opcodes that alias BINARY_OP
SPECIALIZED_BINARY: Set[str] = {
    "BINARY_OP",
    "BINARY_SUBSCR",    # not differentiable but must be preserved
}

CLOSURE_LOAD_OPS: Set[str] = {"LOAD_DEREF", "COPY_FREE_VARS"}
CLOSURE_STORE_OPS: Set[str] = {"STORE_DEREF"}

# Ops that may call user functions (potential higher-order diff)
CALL_OPS: Set[str] = {
    "CALL",
    "CALL_FUNCTION_EX",
    "CALL_INTRINSIC_1",
}

# Ops we must NOT touch
CONTROL_FLOW_OPS: Set[str] = {
    "RESUME",
    "PUSH_EXC_INFO",
    "POP_EXCEPT",
    "RERAISE",
    "JUMP_BACKWARD",
    "JUMP_FORWARD",
    "FOR_ITER",
    "GET_ITER",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnalyzedInstruction:
    """Wrapper around ``dis.Instruction`` with extra metadata."""
    instr: dis.Instruction
    index: int                 # position in the flat instruction list
    is_differentiable: bool = False
    is_closure_load: bool = False
    is_closure_store: bool = False
    is_call: bool = False
    binary_op_name: Optional[str] = None   # e.g. "+", "-", "*"


@dataclass
class CodeAnalysis:
    """Result of analyzing one code object."""
    code: CodeType
    instructions: List[AnalyzedInstruction]
    differentiable_indices: List[int]       # indices into `instructions`
    closure_load_indices: List[int]
    call_indices: List[int]
    cellvars: tuple[str, ...]
    freevars: tuple[str, ...]
    has_nested_functions: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_code(code: CodeType) -> CodeAnalysis:
    """
    Analyze a code object and return a ``CodeAnalysis`` describing every
    instruction site that bytediff needs to handle.

    Parameters
    ----------
    code:
        The ``__code__`` attribute of a Python function.

    Returns
    -------
    CodeAnalysis
    """
    raw_instrs = list(dis.get_instructions(code))
    analyzed: List[AnalyzedInstruction] = []
    diff_indices: List[int] = []
    closure_indices: List[int] = []
    call_indices: List[int] = []

    for idx, instr in enumerate(raw_instrs):
        ai = AnalyzedInstruction(instr=instr, index=idx)

        if instr.opname in DIFFERENTIABLE_BINARY_OPS:
            ai.is_differentiable = True
            # argval encodes the operator symbol in 3.12
            ai.binary_op_name = _binary_op_symbol(instr)
            diff_indices.append(idx)

        elif instr.opname in DIFFERENTIABLE_UNARY_OPS:
            ai.is_differentiable = True
            diff_indices.append(idx)

        if instr.opname in CLOSURE_LOAD_OPS:
            ai.is_closure_load = True
            closure_indices.append(idx)

        if instr.opname in CLOSURE_STORE_OPS:
            ai.is_closure_store = True

        if instr.opname in CALL_OPS:
            ai.is_call = True
            call_indices.append(idx)

        analyzed.append(ai)

    has_nested = any(
        isinstance(i.instr.argval, CodeType) for i in analyzed
        if i.instr.opname in ("MAKE_FUNCTION", "LOAD_CONST")
    )

    return CodeAnalysis(
        code=code,
        instructions=analyzed,
        differentiable_indices=diff_indices,
        closure_load_indices=closure_indices,
        call_indices=call_indices,
        cellvars=code.co_cellvars,
        freevars=code.co_freevars,
        has_nested_functions=has_nested,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Map CPython 3.12 BINARY_OP arg values to operator symbols
# (arg is an int index into the NB_* table)
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


def _binary_op_symbol(instr: dis.Instruction) -> str:
    """Return the operator symbol for a BINARY_OP instruction."""
    if instr.argval in _NB_TABLE:
        return _NB_TABLE[instr.argval]
    # Fallback for specialised opnames
    name = instr.opname
    mapping = {
        "BINARY_ADD": "+",
        "BINARY_SUBTRACT": "-",
        "BINARY_MULTIPLY": "*",
        "BINARY_TRUE_DIVIDE": "/",
        "BINARY_FLOOR_DIVIDE": "//",
        "BINARY_POWER": "**",
        "BINARY_MODULO": "%",
    }
    return mapping.get(name, "?")


def iter_differentiable(analysis: CodeAnalysis) -> Iterator[AnalyzedInstruction]:
    """Yield only the differentiable instructions."""
    for idx in analysis.differentiable_indices:
        yield analysis.instructions[idx]
