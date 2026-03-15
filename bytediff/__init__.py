# bytediff/__init__.py
"""
bytediff: Differentiable Python bytecode rewriter for closed-source autodiff.

Differentiates arbitrary Python functions by rewriting CPython 3.12+ bytecode
to emit a Wengert tape — no source code, no library cooperation required.
"""

from bytediff.grad import grad, vjp, jvp
from bytediff.tape import Tape, TapeEntry
from bytediff.bytecode.rewriter import BytecodeRewriter

__version__ = "0.1.0"
__all__ = ["grad", "vjp", "jvp", "Tape", "TapeEntry", "BytecodeRewriter"]
