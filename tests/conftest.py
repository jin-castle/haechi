import os
from pathlib import Path

# Subprocess tests run scripts/*.py with sys.executable. pytest's `pythonpath`
# setting only affects this process, so export src/ for child processes too.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_SRC, os.environ.get("PYTHONPATH", "")) if p
)
