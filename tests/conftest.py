"""pytest configuration shared across all test modules.

Adds the repository root to ``sys.path`` so that ``import src`` and
``import scripts.*`` resolve correctly regardless of the working directory
used to invoke pytest.
"""

import sys
from pathlib import Path

# Repository root = the directory that contains this conftest.py's parent.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

