# bytediff

**Differentiable Python bytecode rewriter for closed-source autodiff**

`bytediff` differentiates arbitrary Python functions by rewriting their CPython 3.12+ bytecode to emit a [Wengert tape](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation) — no source code access, no library cooperation required.

```python
from bytediff import grad

def f(x):
    return x * x + 2.0 * x   # no special types, no decorators on ops

df = grad(f)
df(3.0)   # → 8.0  (= 2*3 + 2)
```

---

## Why bytediff?

| Approach | Requires | Python version |
|---|---|---|
| JAX | `jax.numpy` ops | Any |
| PyTorch Autograd | `torch.Tensor` | Any |
| Tangent (Google, 2017) | Source code (AST) | 2.x era, unmaintained |
| **bytediff** | **Nothing — any Python function** | **CPython 3.12+** |

CPython 3.12 introduced a richer bytecode format with `RESUME`, `PUSH_EXC_INFO`, and typed `BINARY_OP` specializations that make bytecode-level AD practical for the first time.

---

## How it works

```
                  fn.__code__
                       │
              ┌────────▼──────────┐
              │  BytecodeRewriter │
              │  (analyzer.py)    │
              └────────┬──────────┘
                       │  CodeAnalysis
              ┌────────▼──────────┐
              │  InstructionPatcher│   ← LOAD_DEREF snapshot
              │  (patcher.py)     │
              └────────┬──────────┘
                       │  patched CodeType
       ┌───────────────▼────────────────────┐
       │  Stackvm wrapper (straight-line)   │  ← BINARY_OP → _dispatch_binary()
       │  Shadow wrapper  (control-flow)    │  ← TracedScalar proxy
       └───────────────┬────────────────────┘
                       │  rewritten_fn(*args, tape=tape)
              ┌────────▼──────────┐
              │  Tape             │   ← bump-pointer mmap arena
              │  (tape.py)        │
              └────────┬──────────┘
                       │  tape.backward(seed=1.0)
              ┌────────▼──────────┐
              │  VJP Registry     │   ← per-op VJP closures
              │  (vjp_registry.py)│
              └───────────────────┘
```

### Tape memory model

The tape is backed by a single `mmap(MAP_ANONYMOUS)` bump-pointer arena (default 64 MiB). Primal float snapshots are written directly into the arena; `pymalloc` is not touched during the backward pass. After `backward()`, a single `arena.reset()` zeros the pointer — O(1) free.

### Closure variable correctness

A closure capturing a mutable outer variable via `LOAD_DEREF` would normally record a *binding* on the tape, not a *value* — meaning the gradient would be computed with whatever value the variable holds at backward time. `bytediff` prevents this by rewriting every `LOAD_DEREF` to a `LOAD_DEREF + STORE_FAST` snapshot pair, freezing the primal at op time.

---

## Installation

```bash
pip install bytediff                    # pure-Python mode
pip install "bytediff[dev]"             # + pytest, cffi, bytecode lib
```

To compile the optional C extension (faster arena):

```bash
python bytediff/cffi_ext/build_ext.py
```

**Requirements:** CPython 3.12+. No NumPy, no Torch, no JAX.

---

## Usage

### `grad`

```python
from bytediff import grad

# Single argument
df = grad(lambda x: x ** 3)
df(2.0)   # → 12.0

# Multiple arguments
df = grad(lambda x, y: x * y + y ** 2, argnums=(0, 1))
df(2.0, 3.0)   # → (3.0, 8.0)
```

### `vjp` — vector-Jacobian product

```python
from bytediff import vjp

primal, vjp_fn = vjp(lambda x, y: x * y, 3.0, 4.0)
primal         # → 12.0
vjp_fn(1.0)   # → (4.0, 3.0)
```

### `jvp` — Jacobian-vector product (forward mode)

```python
from bytediff import jvp

primal, tangent = jvp(lambda x: x ** 2, (3.0,), (1.0,))
tangent   # → 6.0
```

### Control-flow functions

Functions with `if`/`while`/`for` are handled via the `TracedScalar` fallback:

```python
def relu(x):
    return x if x > 0.0 else 0.0

grad(relu)(2.0)    # → 1.0
grad(relu)(-1.0)   # → 0.0
```

---

## Supported primitives

| Category | Ops |
|---|---|
| Binary arithmetic | `+ - * / // % **` and in-place variants |
| Unary | `-x  +x` |
| Math functions | `sin cos exp log sqrt tanh atanh` |

---

## Architecture

```
bytediff/
├── __init__.py          # Public API: grad, vjp, jvp
├── grad.py              # grad / vjp / jvp implementation
├── tape.py              # Wengert tape + mmap Arena
├── tracer.py            # TracedScalar operator-overloading fallback
├── bytecode/
│   ├── analyzer.py      # CPython 3.12 instruction stream analysis
│   ├── patcher.py       # Mutable instruction list + CodeType reconstruction
│   ├── rewriter.py      # Core rewriting engine
│   └── vjp_registry.py  # Per-op VJP closures
└── cffi_ext/
    ├── build_ext.py     # CFFI build script
    └── tape_ext.py      # Python wrapper around C arena
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Limitations

- **Scalars only** (float/int). Tensor support requires extending the VJP registry and TracedScalar to handle ndarray shapes.
- **C extensions** that bypass Python's dunder methods (e.g. NumPy ufuncs called directly) cannot be traced via TracedScalar. Use the VJP registry to add explicit rules.
- **Higher-order derivatives** work by nesting `grad()` calls but may be slow for deep nesting.
- **In-place mutation** (`x += y` where `x` is a list element) may produce incorrect gradients if the same object appears in multiple tape entries.

---

## License

MIT
