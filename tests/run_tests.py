#!/usr/bin/env python3
from pathlib import Path
import sys

import pytest


if __name__ == "__main__":
    tests_dir = Path(__file__).resolve().parent
    # Test this checkout even when launched from another working directory.
    sys.path.insert(0, str(tests_dir.parent))
    raise SystemExit(pytest.main([str(tests_dir), "-vv", *sys.argv[1:]]))
