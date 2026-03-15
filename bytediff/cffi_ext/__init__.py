# bytediff/cffi_ext/__init__.py
"""
bytediff.cffi_ext — C extension for hot-path tape operations.

Provides a bump-pointer arena allocator and fast primal computation
via CFFI, bypassing Python's memory allocator for tape entries in
tight loops.

The extension is optional: if CFFI or the compiled shared object is
unavailable, bytediff falls back to the pure-Python tape implementation
transparently.
"""

try:
    from bytediff.cffi_ext._tape_ext import (
        TapeArena,
        arena_alloc_float64,
        arena_reset,
    )
    HAS_CFFI_EXT = True
except ImportError:
    HAS_CFFI_EXT = False

    class TapeArena:  # type: ignore[no-redef]
        """Stub when CFFI extension is not compiled."""
        def __init__(self, size_bytes: int = 64 * 1024 * 1024):
            self._buf = bytearray(size_bytes)
            self._ptr = 0

        def alloc_float64(self, value: float) -> int:
            import struct
            offset = self._ptr
            struct.pack_into("d", self._buf, offset, value)
            self._ptr += 8
            return offset

        def reset(self) -> None:
            self._ptr = 0

    def arena_alloc_float64(arena: TapeArena, value: float) -> int:
        return arena.alloc_float64(value)

    def arena_reset(arena: TapeArena) -> None:
        arena.reset()


__all__ = ["TapeArena", "arena_alloc_float64", "arena_reset", "HAS_CFFI_EXT"]
