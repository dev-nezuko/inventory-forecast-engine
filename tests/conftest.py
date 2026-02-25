"""
Pytest configuration. Ensures repo root is on PYTHONPATH so 'src' is importable.
Run from repo root: pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root (parent of tests/)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
