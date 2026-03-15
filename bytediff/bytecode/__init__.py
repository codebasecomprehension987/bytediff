# bytediff/bytecode/__init__.py
"""
bytediff.bytecode — CPython 3.12 bytecode analysis and rewriting.
"""

from bytediff.bytecode.rewriter import BytecodeRewriter
from bytediff.bytecode.analyzer import analyze_code
from bytediff.bytecode.patcher import InstructionPatcher

__all__ = ["BytecodeRewriter", "analyze_code", "InstructionPatcher"]
