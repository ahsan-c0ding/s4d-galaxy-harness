"""
Makes both `model` (repo root -- the original richer-stem package) and
`s4d_harness` (src/ -- the new unified harness) importable for pytest,
without requiring PYTHONPATH to be set by hand. Scripts run outside pytest
still need `export PYTHONPATH="$PWD:$PWD/src"` (see README) since this file
is pytest-specific.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
