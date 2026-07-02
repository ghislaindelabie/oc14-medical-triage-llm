"""Put the repo root on sys.path so tests can import the top-level `scripts/` package.

The package under test (`oc14_triage`) lives in `src/` (on the path via the editable install),
but the one-off build scripts live in `scripts/` at the repo root, which pytest does not add to
sys.path by default. Tests that exercise a script's pure helpers import `scripts.<module>`.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
