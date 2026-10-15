"""Project entry point for running the test suite.

Registered as the ``ann-test`` console script in ``pyproject.toml``::

    uv run ann-test
    uv run ann-test -k search -vv
"""

from __future__ import annotations

import sys


def run() -> None:
    """Run the project's pytest suite, forwarding any extra CLI arguments."""
    try:
        import pytest
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise SystemExit(
            "pytest is required to run the test suite. Install the dev "
            "dependencies with `uv sync`."
        ) from exc

    sys.exit(pytest.main(["-q", *sys.argv[1:]]))
