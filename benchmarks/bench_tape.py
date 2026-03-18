"""
Performance benchmarks for bytediff.

Measures:
  - Tape record + backward throughput (ops/sec)
  - Arena alloc/reset cycle time
  - grad() overhead vs raw Python
  - TracedScalar hot loop throughput

Run with: python -m pytest benchmarks/ -v --tb=short
Or standalone: python benchmarks/bench_tape.py
"""

import time
import math
import sys
from bytediff.tape import Tape, Arena
from bytediff.tracer import TracedScalar
from bytediff import grad

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timeit(fn, n: int = 1000, warmup: int = 10) -> float:
    """Return mean wall time in microseconds per call."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - t0
    return (elapsed / n) * 1e6  # µs/call


# ---------------------------------------------------------------------------
# Tape throughput
# ---------------------------------------------------------------------------

def bench_tape_record(n_entries: int = 1000) -> float:
    """Record n_entries onto a fresh tape; return µs total."""
    tape = Tape()

    def run():
        tape.reset()
        for i in range(n_entries):
            x = float(i)
            y = float(i + 1)
            result = x + y
            tape.record(lambda g: (g, g), primal_inputs=(x, y), output=result)

    return timeit(run, n=100)


def bench_tape_backward(n_entries: int = 1000) -> float:
    """Record n_entries, then run backward(); return µs for backward only."""
    tape = Tape()
    for i in range(n_entries):
        x, y = float(i), float(i + 1)
        tape.record(lambda g: (g, g), primal_inputs=(x, y), output=x + y)

    return timeit(tape.backward, n=500)


def bench_arena_alloc(n_allocs: int = 1000) -> float:
    """Alloc n float64s into the arena; return µs per full cycle."""
    arena = Arena(size=64 * 1024 * 1024)

    def run():
        arena.reset()
        for i in range(n_allocs):
            arena.alloc_float64(float(i))

    return timeit(run, n=200)


# ---------------------------------------------------------------------------
# TracedScalar throughput
# ---------------------------------------------------------------------------

def bench_traced_scalar_loop(n: int = 1000) -> float:
    """Sum n TracedScalars in a loop; return µs per full forward+backward."""

    def run():
        tape = Tape()
        x = TracedScalar(1.0, tape)
        total = TracedScalar(0.0, tape)
        for _ in range(n):
            total = total + x
        tape.backward(seed=1.0)

    return timeit(run, n=50)


def bench_raw_python_loop(n: int = 1000) -> float:
    """Baseline: same loop without tracing."""

    def run():
        x = 1.0
        total = 0.0
        for _ in range(n):
            total = total + x

    return timeit(run, n=200)


# ---------------------------------------------------------------------------
# grad() overhead
# ---------------------------------------------------------------------------

def bench_grad_quadratic() -> float:
    """Time a single grad() call for f(x) = x^2."""
    df = grad(lambda x: x * x)
    return timeit(lambda: df(3.0), n=500)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("bytediff performance benchmarks")
    print(f"Python {sys.version}")
    print("=" * 60)

    N = 500

    t = bench_tape_record(N)
    print(f"Tape.record()  x{N:4d}  : {t:8.1f} µs/cycle  ({1e6/t*N:.0f} entries/sec)")

    t = bench_tape_backward(N)
    print(f"Tape.backward() x{N:4d} : {t:8.1f} µs/call")

    t = bench_arena_alloc(N)
    print(f"Arena.alloc_float64() x{N:4d}: {t:8.1f} µs/cycle")

    t_traced = bench_traced_scalar_loop(N)
    t_raw    = bench_raw_python_loop(N)
    overhead = t_traced / t_raw if t_raw > 0 else float("inf")
    print(f"TracedScalar loop x{N:4d} : {t_traced:8.1f} µs/cycle  ({overhead:.1f}x vs raw Python)")

    t = bench_grad_quadratic()
    print(f"grad(x^2) single call    : {t:8.1f} µs/call")

    print("=" * 60)


if __name__ == "__main__":
    main()
