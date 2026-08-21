from __future__ import annotations

import gzip
import json
from typing import Any

from ecg_cascade.clinical_demo_persistence import DemoPersistence


class _FakeConnection:
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self.calls = calls

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, params: object = None) -> None:
        self.calls.append((query, params))


def _fake_store() -> tuple[DemoPersistence, list[tuple[str, Any]]]:
    calls: list[tuple[str, Any]] = []
    store = DemoPersistence(
        "postgresql://example.invalid/demo",
        connect_factory=lambda _: _FakeConnection(calls),
    )
    return store, calls


def test_disabled_store_reports_no_database() -> None:
    store = DemoPersistence("")

    assert store.health()["configured"] is False
    assert store.record_session(
        "browser-1", source_token="bundled", source_metadata={}
    ) is False


def test_session_upsert_initializes_schema_once() -> None:
    store, calls = _fake_store()

    assert store.record_session(
        "browser-1",
        source_token="bundled",
        source_metadata={"name": "100.hea", "kind": "WFDB"},
    ) is True
    assert store.record_session(
        "browser-1",
        source_token="bundled",
        source_metadata={"name": "100.hea", "kind": "WFDB"},
    ) is True

    schema_calls = [query for query, _ in calls if "CREATE TABLE" in query]
    upserts = [params for query, params in calls if "INSERT INTO" in query]
    assert len(schema_calls) == 1
    assert len(upserts) == 2
    assert json.loads(upserts[0][2])["kind"] == "WFDB"


def test_analysis_is_compressed_and_replaces_latest_result() -> None:
    store, calls = _fake_store()
    analysis = {
        "source": {"name": "100.hea", "channel": "MLII"},
        "track": {"key": "neurokit"},
        "summary": {"r_peak_count": 10},
        "interpretation": {"classifier_applied": False},
        "signal": {"values": [0.0, 1.0, 0.0]},
    }

    saved, analysis_id = store.record_analysis(
        "browser-1",
        source_token="bundled",
        request_metadata={"duration_s": 12},
        analysis=analysis,
    )

    assert saved is True
    assert analysis_id is not None
    params = next(params for query, params in calls if "INSERT INTO" in query)
    assert json.loads(params[2])["channel"] == "MLII"
    assert json.loads(params[4])["duration_s"] == 12
    assert json.loads(gzip.decompress(params[6]).decode("utf-8")) == analysis


def test_database_failure_does_not_escape_to_demo() -> None:
    def fail(_: str) -> _FakeConnection:
        raise RuntimeError("database unavailable")

    store = DemoPersistence(
        "postgresql://example.invalid/demo", connect_factory=fail
    )

    assert store.record_session(
        "browser-1", source_token="bundled", source_metadata={}
    ) is False
    status = store.public_status()
    assert status["configured"] is True
    assert status["last_error_type"] == "RuntimeError"
