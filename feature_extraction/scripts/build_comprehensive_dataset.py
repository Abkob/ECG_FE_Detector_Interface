"""Command-line entry point for the comprehensive branch-validation matrix."""

from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_cascade.comprehensive_dataset import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
