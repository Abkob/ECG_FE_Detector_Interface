"""Vercel adapter for the ECG Cascade clinician-facing research demo."""

from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "feature_extraction" / "src"
sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault("ECG_DEMO_HOSTED", "1")

from ecg_cascade.clinical_demo_web import ClinicalDemoHandler, DemoState  # noqa: E402


_STATE = DemoState()


class handler(ClinicalDemoHandler):
    """Attach the local handler's state object to Vercel's HTTP server."""

    def __init__(self, request, client_address, server):  # type: ignore[no-untyped-def]
        server.state = _STATE
        super().__init__(request, client_address, server)
