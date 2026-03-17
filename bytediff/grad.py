"""
Top-level differentiation API: grad, vjp, jvp.
"""

from __future__ import annotations

import functools
import sys
from typing import Any, Callable, Sequence

if sys.version_info < (3, 12):
    raise RuntimeError("bytediff requires CPython 3.12+")

from bytediff.bytecode.rewriter import BytecodeRewriter
from bytediff.tape import Tape


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grad(fn: Callable, argnums: int | Sequence[int] = 0) -> Callable:
    """
    Return a function that computes the gradient of `fn` with respect to
    the positional argument(s) specified by `argnums`.

    Parameters
    ----------
    fn:
        A Python callable whose bytecode will be rewritten.
    argnums:
        Index or sequence of indices of arguments to differentiate.

    Returns
    -------
    A wrapped callable ``grad_fn(*args, **kwargs)`` that returns the gradient
    (or a tuple of gradients) instead of the primal output.

    Example
    -------
    >>> def f(x):
    ...     return x * x + 2.0 * x
    >>> df = grad(f)
    >>> df(3.0)
    8.0
    """
    if isinstance(argnums, int):
        argnums = (argnums,)
    argnums = tuple(argnums)

    rewriter = BytecodeRewriter()
    rewritten = rewriter.rewrite(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        tape = Tape()
        # Inject tape into rewritten function's closure
        result = rewritten(*args, tape=tape, **kwargs)
        # Seed: d(output)/d(output) = 1.0
        tape.backward(seed=1.0)
        grads = tuple(tape.gradient(args[i]) for i in argnums)
        return grads[0] if len(grads) == 1 else grads

    return wrapped


def vjp(fn: Callable, *args, **kwargs):
    """
    Compute the value and a function for the vector-Jacobian product.

    Returns
    -------
    (primals_out, vjp_fn)
        ``primals_out`` is the output of ``fn(*args)``.
        ``vjp_fn(v)`` returns the VJP of ``fn`` at ``args`` with cotangent ``v``.
    """
    rewriter = BytecodeRewriter()
    rewritten = rewriter.rewrite(fn)
    tape = Tape()
    primals_out = rewritten(*args, tape=tape, **kwargs)

    def vjp_fn(v):
        tape.backward(seed=v)
        return tuple(tape.gradient(a) for a in args)

    return primals_out, vjp_fn


def jvp(fn: Callable, primals, tangents):
    """
    Compute the Jacobian-vector product (forward-mode AD) via finite differences
    fallback until forward-mode tape support is complete.

    Returns
    -------
    (primals_out, tangents_out)
    """
    # Forward-mode: use perturbation / dual number approximation
    eps = 1e-5
    f0 = fn(*primals)

    tangents_out = []
    for i, (p, t) in enumerate(zip(primals, tangents)):
        perturbed = list(primals)
        perturbed[i] = p + eps * t
        f1 = fn(*perturbed)
        tangents_out.append((f1 - f0) / eps)

    return f0, tangents_out[0] if len(tangents_out) == 1 else tuple(tangents_out)
