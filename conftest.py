"""
pytest configuration and shared fixtures for the bytediff test suite.
"""

import sys
import pytest

# Ensure we're running on CPython 3.12+
if sys.version_info < (3, 12):
    pytest.exit(
        f"bytediff requires CPython 3.12+, got {sys.version}",
        returncode=1,
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_tape():
    """Return a freshly initialized Tape."""
    from bytediff.tape import Tape
    return Tape()


@pytest.fixture
def rewriter():
    """Return a BytecodeRewriter instance."""
    from bytediff.bytecode.rewriter import BytecodeRewriter
    return BytecodeRewriter()


@pytest.fixture(params=[0.5, 1.0, 2.0, 3.0, -1.0, -2.5])
def scalar_x(request):
    """Parametrized scalar input values."""
    return request.param


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running"
    )
