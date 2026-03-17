"""
Python-level wrapper around the compiled CFFI tape extension.

Provides ``FastArena`` — a drop-in replacement for ``bytediff.tape.Arena``
that delegates allocation and reset to the C layer, completely bypassing
Python's memory allocator and GC during the hot backward pass.
"""

from __future__ import annotations

import struct
from typing import Optional

try:
    from bytediff.cffi_ext._tape_ext import ffi, lib as _lib
    _HAS_LIB = True
except ImportError:
    _HAS_LIB = False
    _lib = None
    ffi = None


class FastArena:
    """
    mmap bump-pointer arena backed by a C extension.

    Falls back to a pure-Python bytearray implementation if the C extension
    is not compiled.  The API is identical to ``bytediff.tape.Arena``.
    """

    def __init__(self, size: int = 64 * 1024 * 1024):
        self._size = size
        if _HAS_LIB:
            self._arena = _lib.bd_arena_create(size)
            if self._arena == ffi.NULL:
                raise MemoryError(f"bd_arena_create({size}) failed")
            self._use_c = True
        else:
            # Pure-Python fallback
            self._buf = bytearray(size)
            self._ptr = 0
            self._use_c = False

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def alloc_float64(self, value: float) -> int:
        """Store one float64; return its byte offset."""
        if self._use_c:
            offset = _lib.bd_arena_alloc_f64(self._arena, value)
            if offset < 0:
                raise MemoryError("FastArena exhausted")
            return int(offset)
        else:
            if self._ptr + 8 > self._size:
                raise MemoryError("FastArena (Python fallback) exhausted")
            offset = self._ptr
            struct.pack_into("d", self._buf, offset, value)
            self._ptr += 8
            return offset

    def read_float64(self, offset: int) -> float:
        if self._use_c:
            return float(_lib.bd_arena_read_f64(self._arena, offset))
        return struct.unpack_from("d", self._buf, offset)[0]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """O(1) reset — memset + bump-pointer to zero."""
        if self._use_c:
            _lib.bd_arena_reset(self._arena)
        else:
            self._buf = bytearray(self._size)
            self._ptr = 0

    def close(self) -> None:
        if self._use_c and self._arena != ffi.NULL:
            _lib.bd_arena_destroy(self._arena)
            self._arena = ffi.NULL

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def bytes_used(self) -> int:
        if self._use_c:
            return int(self._arena.ptr)
        return self._ptr


# ---------------------------------------------------------------------------
# Fast primal helpers (thin Python wrappers around C primitives)
# ---------------------------------------------------------------------------

if _HAS_LIB:
    def c_add(x: float, y: float) -> float: return float(_lib.bd_add(x, y))
    def c_sub(x: float, y: float) -> float: return float(_lib.bd_sub(x, y))
    def c_mul(x: float, y: float) -> float: return float(_lib.bd_mul(x, y))
    def c_div(x: float, y: float) -> float: return float(_lib.bd_div(x, y))
    def c_pow(x: float, y: float) -> float: return float(_lib.bd_pow(x, y))
    def c_neg(x: float) -> float: return float(_lib.bd_neg(x))
else:
    def c_add(x, y): return x + y
    def c_sub(x, y): return x - y
    def c_mul(x, y): return x * y
    def c_div(x, y): return x / y
    def c_pow(x, y): return x ** y
    def c_neg(x):    return -x
