"""Compatibility launcher for the standalone Python 3.9 Habitat worker."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("habitat") / "worker.py"), run_name="__main__")
else:
    # Loading this legacy module on Python 3.9 must not import the 3.10+ core.
    globals().update({name: value for name, value in
                      runpy.run_path(str(Path(__file__).with_name("habitat") / "worker.py")).items()
                      if not name.startswith("__")})
