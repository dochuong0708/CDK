"""Agent runtime and tools, separate from the legacy detection API."""
import os
from pathlib import Path

# Numba otherwise probes site-packages/user cache, which may be read-only on Windows.
_cache_root = Path(__file__).resolve().parents[1] / "outputs"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache_root / "numba-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "mpl-cache"))
