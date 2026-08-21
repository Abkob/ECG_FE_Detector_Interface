"""Local web server for the clinician-facing ECG branch demonstration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json
import mimetypes
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .clinical_demo_data import (
    SignalSource,
    build_preview_payload,
    bundled_demo_record,
    inspect_signal_source,
    load_signal_segment,
    run_demo_analysis,
)
from .clinical_demo_persistence import DemoPersistence


MAX_UPLOAD_BYTES = 512 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".edf", ".hea", ".dat", ".atr", ".csv", ".txt"}
HOSTED_MAX_DURATION_S = 120.0


@dataclass
class DemoSession:
    identifier: str
    directory: Path
    source: SignalSource | None = None


class DemoState:
    def __init__(self) -> None:
        self.static_root = Path(__file__).with_name("web_demo")
        self._temporary_root = Path(tempfile.mkdtemp(prefix="ecg_clinical_demo_"))
        self._sessions: dict[str, DemoSession] = {}
        self._lock = threading.Lock()
        self.persistence = DemoPersistence()

    def session(self, identifier: str, *, create: bool = True) -> DemoSession:
        safe = "".join(character for character in identifier if character.isalnum() or character in "-_")
        if not safe or len(safe) > 80:
            raise ValueError("Invalid browser session identifier.")
        with self._lock:
            existing = self._sessions.get(safe)
            if existing is not None:
                return existing
            if not create:
                raise ValueError("This browser session has expired. Reload the page.")
            directory = self._temporary_root / safe
            directory.mkdir(parents=True, exist_ok=True)
            created = DemoSession(safe, directory)
            self._sessions[safe] = created
            return created

    def close(self) -> None:
        shutil.rmtree(self._temporary_root, ignore_errors=True)


class ClinicalDemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: DemoState):
        super().__init__(address, ClinicalDemoHandler)
        self.state = state


class ClinicalDemoHandler(BaseHTTPRequestHandler):
    server: ClinicalDemoServer

    def log_message(self, format: str, *args: object) -> None:
        print(f"[ECG demo] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        route = _api_route(parsed.path, parse_qs(parsed.query))
        if route == "/api/health":
            self._json(
                200,
                {
                    "ok": True,
                    "service": "ECG Cascade Clinical Demo",
                    "database": self.server.state.persistence.health(),
                    "capabilities": _runtime_capabilities(),
                },
            )
            return
        route = "/index.html" if parsed.path == "/" else parsed.path
        self._serve_static(route)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        route = _api_route(parsed.path, query)
        try:
            if route == "/api/upload":
                self._upload(query)
            elif route == "/api/inspect":
                self._inspect(self._json_body())
            elif route == "/api/demo":
                self._demo(self._json_body())
            elif route == "/api/preview":
                self._preview(self._json_body())
            elif route == "/api/analyze":
                self._analyze(self._json_body())
            else:
                self._json(404, {"error": "Unknown API route."})
        except (ValueError, FileNotFoundError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # keep a readable message in the local research UI
            self._json(500, {"error": f"Analysis failed: {exc}"})

    def _serve_static(self, route: str) -> None:
        relative = Path(unquote(route.lstrip("/")))
        if ".." in relative.parts:
            self._json(403, {"error": "Invalid static path."})
            return
        target = (self.server.state.static_root / relative).resolve()
        root = self.server.state.static_root.resolve()
        if root not in target.parents and target != root:
            self._json(403, {"error": "Invalid static path."})
            return
        if not target.is_file():
            self._json(404, {"error": "Page asset not found."})
            return
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        content = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") else mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _upload(self, query: dict[str, list[str]]) -> None:
        if _hosted_mode():
            raise ValueError(
                "Uploads are disabled on the public research demo. "
                "Run the application locally for private or patient ECG files."
            )
        session_id = _required_query(query, "session")
        filename = Path(_required_query(query, "filename")).name
        suffix = Path(filename).suffix.casefold()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise ValueError(f"Unsupported uploaded file type: {suffix or '(none)' }.")
        length = self._content_length()
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("One uploaded file exceeded the 512 MB local-demo limit.")
        session = self.server.state.session(session_id)
        destination = (session.directory / filename).resolve()
        if session.directory.resolve() not in destination.parents:
            raise ValueError("Invalid uploaded filename.")
        remaining = length
        with destination.open("wb") as handle:
            while remaining:
                block = self.rfile.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError("The uploaded file ended unexpectedly.")
                handle.write(block)
                remaining -= len(block)
        self._json(200, {"ok": True, "filename": filename, "bytes": length})

    def _inspect(self, body: dict[str, Any]) -> None:
        if _hosted_mode():
            raise ValueError(
                "Uploads are disabled on the public research demo. "
                "Run the application locally for private or patient ECG files."
            )
        session = self.server.state.session(_string(body, "session"), create=False)
        filenames = body.get("files")
        if not isinstance(filenames, list) or not filenames:
            raise ValueError("Select an EDF, WFDB, or CSV signal file.")
        candidates = [(session.directory / Path(str(name)).name) for name in filenames]
        primary = _choose_primary_source(candidates)
        source = inspect_signal_source(
            primary,
            csv_sampling_rate_hz=_optional_float(body.get("csv_sampling_rate_hz")),
        )
        session.source = source
        self._json(200, {"session": session.identifier, "source": source.to_public_dict()})

    def _demo(self, body: dict[str, Any]) -> None:
        session_id = _string(body, "session")
        session = self.server.state.session(session_id)
        path = bundled_demo_record()
        if not path.is_file():
            raise FileNotFoundError("The bundled MIT-BIH demo record is not available.")
        source = inspect_signal_source(path)
        session.source = source
        payload = {
            "session": session.identifier,
            "source": source.to_public_dict(),
            "source_token": "bundled",
            "capabilities": _runtime_capabilities(),
        }
        segment = load_signal_segment(
            source,
            channel="MLII" if "MLII" in source.channels else source.channels[0],
            start_s=0.0,
            duration_s=min(12.0, source.duration_s or 12.0),
        )
        payload["preview"] = build_preview_payload(segment)
        saved = self.server.state.persistence.record_session(
            session_id,
            source_token="bundled",
            source_metadata=payload["source"],
        )
        payload["persistence"] = {
            **self.server.state.persistence.public_status(),
            "session_saved": saved,
        }
        self._json(200, payload)

    def _preview(self, body: dict[str, Any]) -> None:
        source = self._source_for_body(body)
        segment = load_signal_segment(
            source,
            channel=_string(body, "channel"),
            start_s=_float(body, "start_s", default=0.0),
            duration_s=min(12.0, _float(body, "duration_s", default=12.0)),
            csv_sampling_rate_hz=_optional_float(body.get("csv_sampling_rate_hz")),
        )
        self._json(200, {"preview": build_preview_payload(segment)})

    def _analyze(self, body: dict[str, Any]) -> None:
        session_id = _string(body, "session")
        source = self._source_for_body(body)
        duration_s = _float(body, "duration_s", default=120.0)
        if _hosted_mode() and duration_s > HOSTED_MAX_DURATION_S:
            raise ValueError(
                f"The hosted demo supports segments up to "
                f"{HOSTED_MAX_DURATION_S:.0f} seconds."
            )
        segment = load_signal_segment(
            source,
            channel=_string(body, "channel"),
            start_s=_float(body, "start_s", default=0.0),
            duration_s=duration_s,
            csv_sampling_rate_hz=_optional_float(body.get("csv_sampling_rate_hz")),
        )
        payload = run_demo_analysis(
            segment,
            timestamp_track=str(body.get("timestamp_track", "neurokit")),
            calibration_beats=_integer(
                body, "calibration_beats", default=20, minimum=5, maximum=100
            ),
        )
        saved, analysis_id = self.server.state.persistence.record_analysis(
            session_id,
            source_token=(
                str(body.get("source_token"))
                if body.get("source_token") is not None
                else None
            ),
            request_metadata={
                "channel": body.get("channel"),
                "start_s": body.get("start_s", 0.0),
                "duration_s": duration_s,
                "timestamp_track": body.get("timestamp_track", "neurokit"),
                "calibration_beats": body.get("calibration_beats", 20),
            },
            analysis=payload,
        )
        self._json(
            200,
            {
                "analysis": payload,
                "persistence": {
                    **self.server.state.persistence.public_status(),
                    "analysis_saved": saved,
                    "analysis_id": analysis_id,
                },
            },
        )

    def _source_for_body(self, body: dict[str, Any]) -> SignalSource:
        if body.get("source_token") == "bundled":
            path = bundled_demo_record()
            if not path.is_file():
                raise FileNotFoundError("The bundled MIT-BIH demo record is not available.")
            return inspect_signal_source(path)
        session = self.server.state.session(_string(body, "session"), create=False)
        return _session_source(session)

    def _json_body(self) -> dict[str, Any]:
        length = self._content_length()
        if length > 2 * 1024 * 1024:
            raise ValueError("The JSON request was unexpectedly large.")
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("The browser sent invalid JSON.") from exc
        if not isinstance(body, dict):
            raise ValueError("The browser request must be a JSON object.")
        return body

    def _content_length(self) -> int:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if length < 0:
            raise ValueError("Invalid Content-Length header.")
        return length

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        content_encoding: str | None = None
        if len(encoded) >= 1024 and "gzip" in self.headers.get("Accept-Encoding", "").lower():
            compressed = gzip.compress(encoded, compresslevel=6)
            if len(compressed) < len(encoded):
                encoded = compressed
                content_encoding = "gzip"
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if content_encoding is not None:
            self.send_header("Content-Encoding", content_encoding)
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def _api_route(path: str, query: dict[str, list[str]]) -> str:
    """Support one Vercel function at /api while preserving local API routes."""

    if path != "/api":
        return path
    values = query.get("route")
    route = values[0].strip("/") if values and values[0] else ""
    return f"/api/{route}" if route else path


def _hosted_mode() -> bool:
    return os.environ.get("ECG_DEMO_HOSTED", "").strip().casefold() in {
        "1",
        "true",
        "yes",
    }


def _runtime_capabilities() -> dict[str, Any]:
    hosted = _hosted_mode()
    return {
        "uploads": not hosted,
        "max_duration_s": HOSTED_MAX_DURATION_S if hosted else 300.0,
        "deployment": "vercel" if hosted else "local",
        "upload_reason": (
            "secure_object_storage_required"
            if hosted
            else "private_local_filesystem"
        ),
    }


def _choose_primary_source(paths: list[Path]) -> Path:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        raise ValueError("No uploaded signal file was found in this browser session.")
    for suffix in (".edf", ".hea", ".csv", ".txt", ".dat"):
        matches = [path for path in existing if path.suffix.casefold() == suffix]
        if matches:
            return sorted(matches, key=lambda item: item.name.casefold())[0]
    raise ValueError("Select an EDF, WFDB (.hea + .dat), or CSV/TXT signal.")


def _session_source(session: DemoSession) -> SignalSource:
    if session.source is None:
        raise ValueError("Load a signal before requesting a preview or analysis.")
    return session.source


def _required_query(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key)
    if not values or not values[0]:
        raise ValueError(f"Missing query parameter: {key}.")
    return values[0]


def _string(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing browser field: {key}.")
    return value.strip()


def _float(body: dict[str, Any], key: str, *, default: float) -> float:
    value = body.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric.") from exc


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sampling rate must be numeric.") from exc


def _integer(
    body: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = body.get(key, default)
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer.") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}.")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = DemoState()
    server = ClinicalDemoServer((args.host, args.port), state)
    actual_port = int(server.server_address[1])
    url = f"http://{args.host}:{actual_port}/"
    print(f"ECG Cascade Clinical Demo is running at {url}")
    print("Press Ctrl+C in this window to stop it.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping ECG demo.")
    finally:
        server.server_close()
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
