"""Makes the project root importable so tests can `from classifier import ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
