"""Compatibility entry point; implementation lives in the adjacent package."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    # Keep native packages named libero/habitat visible to external environments.
    integration_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != integration_dir]
    sys.path.insert(0, str(integration_dir.parent))

from integrations.habitat import *


if __name__ == "__main__":
    raise SystemExit(main())
