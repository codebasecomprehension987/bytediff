"""
Build script for the bytediff C extension via CFFI.

Compiles a small C library that provides:
  - A mmap-backed bump-pointer arena for tape entries
  - Fast primal computation for the four basic arithmetic ops
  - Atomic tape entry writing to minimize GIL interaction

Run with: python build_ext.py
Or via: pip install -e ".[dev]"
"""

from __future__ import annotations

import os
import sys

try:
    from cffi import FFI
except ImportError:
    print("CFFI not installed — skipping C extension build.")
    print("Install with: pip install cffi")
    sys.exit(0)

ffi = FFI()

# ---------------------------------------------------------------------------
# C declarations
# ---------------------------------------------------------------------------

ffi.cdef("""
    /* Arena allocator */
    typedef struct {
        char   *base;     /* mmap base pointer */
        size_t  size;     /* total arena size in bytes */
        size_t  ptr;      /* current bump pointer offset */
    } BDArena;

    BDArena *bd_arena_create(size_t size_bytes);
    void     bd_arena_destroy(BDArena *arena);
    void     bd_arena_reset(BDArena *arena);

    /* Allocate one float64 in the arena; return byte offset or -1 on OOM */
    ssize_t  bd_arena_alloc_f64(BDArena *arena, double value);

    /* Read a float64 from the arena at a given byte offset */
    double   bd_arena_read_f64(const BDArena *arena, size_t offset);

    /* Primal computation helpers */
    double bd_add(double x, double y);
    double bd_sub(double x, double y);
    double bd_mul(double x, double y);
    double bd_div(double x, double y);
    double bd_pow(double x, double y);
    double bd_neg(double x);
""")

# ---------------------------------------------------------------------------
# C source
# ---------------------------------------------------------------------------

_C_SOURCE = r"""
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>

typedef struct {
    char   *base;
    size_t  size;
    size_t  ptr;
} BDArena;

BDArena *bd_arena_create(size_t size_bytes) {
    BDArena *a = malloc(sizeof(BDArena));
    if (!a) return NULL;
    a->size = size_bytes;
    a->ptr  = 0;
    a->base = mmap(NULL, size_bytes,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
    if (a->base == MAP_FAILED) {
        free(a);
        return NULL;
    }
    return a;
}

void bd_arena_destroy(BDArena *a) {
    if (a) {
        munmap(a->base, a->size);
        free(a);
    }
}

void bd_arena_reset(BDArena *a) {
    if (a) {
        a->ptr = 0;
        /* zero the memory so stale values don't leak */
        memset(a->base, 0, a->size);
    }
}

ssize_t bd_arena_alloc_f64(BDArena *a, double value) {
    if (!a || a->ptr + sizeof(double) > a->size) return -1;
    size_t offset = a->ptr;
    *((double *)(a->base + offset)) = value;
    a->ptr += sizeof(double);
    return (ssize_t)offset;
}

double bd_arena_read_f64(const BDArena *a, size_t offset) {
    return *((double *)(a->base + offset));
}

double bd_add(double x, double y) { return x + y; }
double bd_sub(double x, double y) { return x - y; }
double bd_mul(double x, double y) { return x * y; }
double bd_div(double x, double y) { return x / y; }
double bd_pow(double x, double y) { return pow(x, y); }
double bd_neg(double x)           { return -x; }
"""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

ffi.set_source(
    "bytediff.cffi_ext._tape_ext",
    _C_SOURCE,
    libraries=["m"],
    extra_compile_args=["-O3", "-march=native", "-ffast-math"],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
    print("CFFI extension compiled successfully.")
