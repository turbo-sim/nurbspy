#!/usr/bin/env python3
"""Run all tests against the installed package, including editable installs."""

from pathlib import Path
import sys

import pytest


if __name__ == "__main__":
    tests_dir = Path(__file__).resolve().parent
    # Locate the tests independently of the working directory. Install the
    # checkout in editable mode to test local changes without modifying sys.path.
    raise SystemExit(pytest.main([str(tests_dir), "-vv", *sys.argv[1:]]))
