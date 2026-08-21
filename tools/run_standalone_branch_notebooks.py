"""Execute the standalone ECG branch notebooks in place and fail on any error."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import nbformat


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "feature_extraction"
NOTEBOOK_DIR = PROJECT / "notebooks" / "standalone_branches"


def main() -> int:
    # The bundled python3 kernelspec intentionally uses the generic `python`
    # executable. Put this runner's virtual-environment Scripts directory first
    # so notebook kernels use the same tested environment as the runner.
    interpreter_dir = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = interpreter_dir + os.pathsep + os.environ.get("PATH", "")

    notebooks = sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb"))
    if len(notebooks) != 4:
        raise RuntimeError(f"Expected four notebooks, found {len(notebooks)}")

    failures: list[tuple[Path, str]] = []
    for path in notebooks:
        print(f"\nEXECUTING {path.name}", flush=True)
        started = time.perf_counter()
        notebook = nbformat.read(path, as_version=4)
        for cell_number, cell in enumerate(notebook.cells, start=1):
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:cell-{cell_number}", "exec")
        notebook.metadata.setdefault("ecg_cascade", {})
        notebook.metadata["ecg_cascade"]["execution_started_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )
        client = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(PROJECT)}},
            record_timing=True,
        )
        try:
            client.execute()
        except CellExecutionError as exc:
            failures.append((path, str(exc)))
            print(f"FAILED {path.name}: {exc}", flush=True)
        finally:
            elapsed = time.perf_counter() - started
            notebook.metadata["ecg_cascade"]["execution_finished_utc"] = (
                datetime.now(timezone.utc).isoformat()
            )
            notebook.metadata["ecg_cascade"]["execution_elapsed_s"] = round(elapsed, 3)
            nbformat.write(notebook, path)
            print(f"SAVED {path.name} ({elapsed:.1f} s)", flush=True)

    if failures:
        print("\nNotebook failures:")
        for path, message in failures:
            print(f"- {path.name}: {message.splitlines()[-1]}")
        return 1
    print("\nPASS — all four standalone branch notebooks executed without errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
