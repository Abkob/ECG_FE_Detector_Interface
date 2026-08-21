"""Optional Neon persistence for the hosted ECG Cascade demonstration.

The public demo stores only session metadata and the most recent compressed
analysis result for each browser.  Raw ECG files are deliberately not stored in
Postgres; hosted uploads remain disabled until dedicated object storage is
configured.
"""

from __future__ import annotations

from collections.abc import Callable
import gzip
import json
import os
import re
import threading
from typing import Any, Protocol
from uuid import uuid4


_SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ecg_demo_sessions (
    session_id varchar(80) PRIMARY KEY,
    source_token text,
    source_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    last_analysis_id uuid,
    last_analysis_at timestamptz,
    last_analysis_request jsonb,
    last_analysis_summary jsonb,
    last_analysis_payload_gzip bytea
)
"""


class _Connection(Protocol):
    def __enter__(self) -> "_Connection": ...

    def __exit__(self, *args: object) -> object: ...

    def execute(self, query: str, params: object = None) -> Any: ...


ConnectFactory = Callable[[str], _Connection]


class DemoPersistence:
    """Best-effort PostgreSQL storage that never blocks the research UI.

    Every public method catches database failures and reports a small status
    object without returning credentials or database error text to the browser.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        connect_factory: ConnectFactory | None = None,
    ) -> None:
        self._database_url = (
            os.environ.get("DATABASE_URL", "")
            if database_url is None
            else database_url
        ).strip()
        self._connect_factory = connect_factory or _connect_postgres
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._last_error_type: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self._database_url)

    def public_status(self, *, reachable: bool | None = None) -> dict[str, Any]:
        status: dict[str, Any] = {
            "configured": self.configured,
            "provider": "neon_postgres" if self.configured else None,
            "storage_scope": "latest_analysis_per_browser_session",
            "raw_ecg_files_stored": False,
        }
        if reachable is not None:
            status["reachable"] = reachable
        if self._last_error_type is not None:
            status["last_error_type"] = self._last_error_type
        return status

    def health(self) -> dict[str, Any]:
        if not self.configured:
            return self.public_status(reachable=False)
        try:
            self._ensure_schema()
            with self._connect_factory(self._database_url) as connection:
                connection.execute("SELECT 1")
            self._last_error_type = None
            return self.public_status(reachable=True)
        except Exception as exc:  # pragma: no cover - depends on external service
            self._last_error_type = type(exc).__name__
            return self.public_status(reachable=False)

    def record_session(
        self,
        session_id: str,
        *,
        source_token: str | None,
        source_metadata: dict[str, Any],
    ) -> bool:
        if not self.configured:
            return False
        try:
            _validate_session_id(session_id)
            self._ensure_schema()
            source_json = _json_text(source_metadata)
            with self._connect_factory(self._database_url) as connection:
                connection.execute(
                    """
                    INSERT INTO ecg_demo_sessions (
                        session_id, source_token, source_metadata
                    ) VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (session_id) DO UPDATE SET
                        source_token = EXCLUDED.source_token,
                        source_metadata = EXCLUDED.source_metadata,
                        updated_at = now()
                    """,
                    (session_id, source_token, source_json),
                )
            self._last_error_type = None
            return True
        except Exception as exc:  # database is optional; keep the demo available
            self._last_error_type = type(exc).__name__
            return False

    def record_analysis(
        self,
        session_id: str,
        *,
        source_token: str | None,
        request_metadata: dict[str, Any],
        analysis: dict[str, Any],
    ) -> tuple[bool, str | None]:
        if not self.configured:
            return False, None
        analysis_id = str(uuid4())
        try:
            _validate_session_id(session_id)
            self._ensure_schema()
            source = analysis.get("source")
            summary = {
                "source": source if isinstance(source, dict) else {},
                "track": analysis.get("track", {}),
                "summary": analysis.get("summary", {}),
                "interpretation": analysis.get("interpretation", {}),
            }
            payload = gzip.compress(
                _json_text(analysis).encode("utf-8"), compresslevel=6
            )
            with self._connect_factory(self._database_url) as connection:
                connection.execute(
                    """
                    INSERT INTO ecg_demo_sessions (
                        session_id,
                        source_token,
                        source_metadata,
                        last_analysis_id,
                        last_analysis_at,
                        last_analysis_request,
                        last_analysis_summary,
                        last_analysis_payload_gzip
                    ) VALUES (
                        %s, %s, %s::jsonb, %s, now(), %s::jsonb, %s::jsonb, %s
                    )
                    ON CONFLICT (session_id) DO UPDATE SET
                        source_token = EXCLUDED.source_token,
                        source_metadata = EXCLUDED.source_metadata,
                        updated_at = now(),
                        last_analysis_id = EXCLUDED.last_analysis_id,
                        last_analysis_at = EXCLUDED.last_analysis_at,
                        last_analysis_request = EXCLUDED.last_analysis_request,
                        last_analysis_summary = EXCLUDED.last_analysis_summary,
                        last_analysis_payload_gzip = EXCLUDED.last_analysis_payload_gzip
                    """,
                    (
                        session_id,
                        source_token,
                        _json_text(summary["source"]),
                        analysis_id,
                        _json_text(request_metadata),
                        _json_text(summary),
                        payload,
                    ),
                )
            self._last_error_type = None
            return True, analysis_id
        except Exception as exc:  # database is optional; keep analysis available
            self._last_error_type = type(exc).__name__
            return False, None

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            with self._connect_factory(self._database_url) as connection:
                connection.execute(_SCHEMA_SQL)
            self._schema_ready = True


def _connect_postgres(database_url: str) -> _Connection:
    import psycopg

    return psycopg.connect(
        database_url,
        autocommit=True,
        connect_timeout=8,
        prepare_threshold=None,
    )


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_PATTERN.fullmatch(session_id):
        raise ValueError("Invalid browser session identifier.")
