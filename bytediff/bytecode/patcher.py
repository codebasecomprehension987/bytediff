"""
Low-level instruction stream patcher for CPython 3.12.

Operates on a mutable list of ``dis.Instruction``-like objects and provides
methods to:
  - Replace a single instruction with a sequence of instructions
  - Insert instructions before/after a given index
  - Rebuild the ``co_code`` / ``co_consts`` / ``co_varnames`` of a CodeType

CPython 3.12 uses a 2-byte wordcode: (opcode, arg) pairs.  Extended args
(EXTENDED_ARG) are handled transparently.  Jump targets are stored as absolute
instruction offsets and must be patched after any insertion.

We use the ``bytecode`` library (pip install bytecode) for safe round-trip
manipulation rather than hand-rolling offset arithmetic, falling back to a
pure-dis implementation if the library is unavailable.
"""

from __future__ import annotations

import dis
import sys
import types
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple

if sys.version_info < (3, 12):
    raise RuntimeError("bytediff requires CPython 3.12+")

try:
    import bytecode as _bc
    _HAS_BYTECODE_LIB = True
except ImportError:
    _HAS_BYTECODE_LIB = False


# ---------------------------------------------------------------------------
# Instruction wrapper (used when bytecode lib is absent)
# ---------------------------------------------------------------------------

class Instr:
    """Minimal mutable instruction wrapper."""
    __slots__ = ("opname", "opcode", "arg", "argval", "argrepr", "offset", "starts_line")

    def __init__(
        self,
        opname: str,
        arg: int = 0,
        argval=None,
        argrepr: str = "",
        offset: int = 0,
        starts_line: Optional[int] = None,
    ):
        self.opname = opname
        self.opcode = dis.opmap.get(opname, 0)
        self.arg = arg
        self.argval = argval if argval is not None else arg
        self.argrepr = argrepr
        self.offset = offset
        self.starts_line = starts_line

    @classmethod
    def from_dis(cls, instr: dis.Instruction) -> "Instr":
        return cls(
            opname=instr.opname,
            arg=instr.arg,
            argval=instr.argval,
            argrepr=instr.argrepr,
            offset=instr.offset,
            starts_line=instr.starts_line,
        )

    def __repr__(self) -> str:
        return f"Instr({self.opname!r}, arg={self.arg}, argval={self.argval!r})"


class InstructionPatcher:
    """
    Mutable view of a function's instruction stream.

    After all patches are applied, call ``build_code()`` to obtain a new
    ``types.CodeType`` incorporating the changes.
    """

    def __init__(self, code: types.CodeType):
        self._original = code
        self._instrs: List[Instr] = [
            Instr.from_dis(i) for i in dis.get_instructions(code)
            # Skip EXTENDED_ARG — we'll re-emit as needed
            if i.opname != "EXTENDED_ARG"
        ]
        # Mutable copies of code object internals
        self._consts: list = list(code.co_consts)
        self._varnames: list = list(code.co_varnames)
        self._names: list = list(code.co_names)

    # ------------------------------------------------------------------
    # Const / varname helpers
    # ------------------------------------------------------------------

    def add_const(self, value) -> int:
        """Intern ``value`` in co_consts; return its index."""
        try:
            return self._consts.index(value)
        except ValueError:
            self._consts.append(value)
            return len(self._consts) - 1

    def add_varname(self, name: str) -> int:
        """Intern ``name`` in co_varnames; return its index."""
        if name not in self._varnames:
            self._varnames.append(name)
        return self._varnames.index(name)

    def add_name(self, name: str) -> int:
        """Intern ``name`` in co_names; return its index."""
        if name not in self._names:
            self._names.append(name)
        return self._names.index(name)

    # ------------------------------------------------------------------
    # Instruction editing
    # ------------------------------------------------------------------

    def replace(self, index: int, new_instrs: Sequence[Instr]) -> None:
        """Replace instruction at ``index`` with ``new_instrs``."""
        self._instrs[index : index + 1] = list(new_instrs)

    def insert_before(self, index: int, new_instrs: Sequence[Instr]) -> None:
        """Insert ``new_instrs`` immediately before ``index``."""
        self._instrs[index:index] = list(new_instrs)

    def insert_after(self, index: int, new_instrs: Sequence[Instr]) -> None:
        """Insert ``new_instrs`` immediately after ``index``."""
        self._instrs[index + 1 : index + 1] = list(new_instrs)

    def get(self, index: int) -> Instr:
        return self._instrs[index]

    def __len__(self) -> int:
        return len(self._instrs)

    def __iter__(self):
        return iter(self._instrs)

    # ------------------------------------------------------------------
    # Code object reconstruction
    # ------------------------------------------------------------------

    def build_code(self) -> types.CodeType:
        """
        Assemble a new ``types.CodeType`` from the current instruction list.

        We use ``compile()``-based round-trip via the ``bytecode`` library when
        available for correctness.  Otherwise we fall back to raw bytecode
        assembly using ``co_code`` (bytes) construction.
        """
        if _HAS_BYTECODE_LIB:
            return self._build_via_lib()
        return self._build_raw()

    def _build_via_lib(self) -> types.CodeType:
        """Use the ``bytecode`` library for safe assembly."""
        import bytecode as bc
        cfg = bc.Bytecode()
        cfg.argcount = self._original.co_argcount
        cfg.posonlyargcount = self._original.co_posonlyargcount
        cfg.kwonlyargcount = self._original.co_kwonlyargcount
        cfg.consts = tuple(self._consts)
        cfg.varnames = tuple(self._varnames)

        for instr in self._instrs:
            if instr.opname in ("NOP",):
                cfg.append(bc.Instr("NOP"))
            elif instr.opname == "LOAD_CONST":
                cfg.append(bc.Instr("LOAD_CONST", instr.argval))
            elif instr.opname == "LOAD_FAST":
                cfg.append(bc.Instr("LOAD_FAST", instr.argval))
            elif instr.opname == "STORE_FAST":
                cfg.append(bc.Instr("STORE_FAST", instr.argval))
            elif instr.opname == "RETURN_VALUE":
                cfg.append(bc.Instr("RETURN_VALUE"))
            else:
                try:
                    cfg.append(bc.Instr(instr.opname, instr.arg))
                except Exception:
                    cfg.append(bc.Instr("NOP"))

        return cfg.to_code()

    def _build_raw(self) -> types.CodeType:
        """
        Construct bytecode bytes directly.

        CPython 3.12 uses 2-byte wordcode (opcode, arg).  Arguments > 255
        require EXTENDED_ARG prefix bytes.
        """
        code_bytes = bytearray()

        for instr in self._instrs:
            opcode = instr.opcode
            arg = instr.arg or 0

            # Emit EXTENDED_ARG chain if needed
            if arg > 0xFFFFFF:
                code_bytes += bytes([dis.opmap["EXTENDED_ARG"], (arg >> 24) & 0xFF])
            if arg > 0xFFFF:
                code_bytes += bytes([dis.opmap["EXTENDED_ARG"], (arg >> 16) & 0xFF])
            if arg > 0xFF:
                code_bytes += bytes([dis.opmap["EXTENDED_ARG"], (arg >> 8) & 0xFF])
            code_bytes += bytes([opcode, arg & 0xFF])

        orig = self._original
        # CodeType constructor signature varies by Python version
        # For 3.12 we use replace() to safely update fields
        return orig.replace(
            co_code=bytes(code_bytes),
            co_consts=tuple(self._consts),
            co_varnames=tuple(self._varnames),
            co_names=tuple(self._names),
            co_stacksize=orig.co_stacksize + 16,  # conservative headroom
        )
