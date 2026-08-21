"""Execute all four branch walkthroughs with one fresh kernel per notebook."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
NOTEBOOKS = [
    OUTPUT / "branches" / "B1_peak_rr_hrv" / "B1_Peak_RR_HRV_Walkthrough.ipynb",
    OUTPUT / "branches" / "B2_morphology" / "B2_Morphology_Walkthrough.ipynb",
    OUTPUT / "branches" / "B3_conduction_repolarization" / "B3_Conduction_Repolarization_Walkthrough.ipynb",
    OUTPUT / "branches" / "B4_signal_quality" / "B4_Signal_Quality_Walkthrough.ipynb",
]


def execute(path: Path) -> dict[str, object]:
    started = time.perf_counter()
    record: dict[str, object] = {"notebook": path.relative_to(ROOT).as_posix()}
    executed_dir = OUTPUT / "qa" / "executed_notebooks"
    executed_dir.mkdir(parents=True, exist_ok=True)
    try:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata":{"path":str(ROOT)}})
        client.execute(cwd=str(ROOT))
        destination = executed_dir / path.name
        nbformat.write(notebook, destination)
        record.update({"status":"passed","executed_copy":destination.relative_to(ROOT).as_posix(),"exception":""})
    except Exception:
        record.update({"status":"failed","executed_copy":"","exception":traceback.format_exc()})
    record["elapsed_seconds"] = round(time.perf_counter()-started, 3)
    return record


def main() -> int:
    results = [execute(path) for path in NOTEBOOKS]
    expected = {
        "B1_peak_rr_hrv":["jeppesen_feature_arrays.csv","b1_diagnostic.png"],
        "B2_morphology":["beat_context.csv","varon_features.csv","beat_waveforms.npy","b2_diagnostic.png"],
        "B3_conduction_repolarization":["timing_v1_beats.csv","timing_v1_rolling.csv","information_v2_windows.csv","qt_nine_beat_context.csv","b3_diagnostic.png"],
        "B4_signal_quality":["signal_quality_records.csv","b4_diagnostic.png"],
    }
    for record in results:
        branch = next(name for name in expected if name in record["notebook"])
        missing = [name for name in expected[branch] if not (OUTPUT/"notebook_runs"/branch/name).exists()]
        record["expected_artifacts"] = expected[branch]
        record["missing_artifacts"] = missing
        if missing: record["status"] = "failed"
    report={"status":"passed" if all(r["status"]=="passed" for r in results) else "failed","notebooks":results}
    (OUTPUT/"qa"/"notebook_execution_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"status":report["status"],"notebooks":[{"notebook":r["notebook"],"status":r["status"],"elapsed_seconds":r["elapsed_seconds"],"missing_artifacts":r["missing_artifacts"]} for r in results]},indent=2))
    return 0 if report["status"]=="passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
