"""Make the repo root and the tests directory importable.

``synthetic.py`` is a helper module beside the tests, not a package, and the
package itself is used from the source tree during development.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (str(ROOT), str(ROOT / "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)
