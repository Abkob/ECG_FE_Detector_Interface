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
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse, urlsplit
from urllib.request import Request, urlopen
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
HOSTED_BLOB_FILE_BYTES = 16 * 1024 * 1024
HOSTED_BLOB_FILE_COUNT = 3
ALLOWED_UPLOAD_SUFFIXES = {".edf", ".hea", ".dat", ".atr", ".csv", ".txt"}
DEFAULT_DURATION_S = 120.0


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
        session_id = _string(body, "session")
        session = self.server.state.session(session_id, create=_hosted_mode())
        source_token: dict[str, Any] | None = None
        if _hosted_mode():
            source_token = _validated_blob_source_token(body.get("blobs"), session_id)
            candidates = _materialize_blob_files(session, source_token)
        else:
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
        payload: dict[str, Any] = {
            "session": session.identifier,
            "source": source.to_public_dict(),
            "source_token": source_token,
        }
        if source_token is not None:
            saved = self.server.state.persistence.record_session(
                session_id,
                source_token="private_blob",
                source_metadata=payload["source"],
            )
            payload["persistence"] = {
                **self.server.state.persistence.public_status(),
                "session_saved": saved,
            }
        self._json(200, payload)

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
        blob_source_token = (
            _validated_blob_source_token(body.get("source_token"), session_id)
            if isinstance(body.get("source_token"), dict)
            else None
        )
        source = self._source_for_body(body)
        duration_s = _float(body, "duration_s", default=DEFAULT_DURATION_S)
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
            source_token=("private_blob" if blob_source_token else body.get("source_token")),
            request_metadata={
                "channel": body.get("channel"),
                "start_s": body.get("start_s", 0.0),
                "duration_s": duration_s,
                "timestamp_track": body.get("timestamp_track", "neurokit"),
                "calibration_beats": body.get("calibration_beats", 20),
            },
            analysis=payload,
        )
        source_released = _delete_blob_files(blob_source_token) if blob_source_token else False
        self._json(
            200,
            {
                "analysis": payload,
                "persistence": {
                    **self.server.state.persistence.public_status(),
                    "analysis_saved": saved,
                    "analysis_id": analysis_id,
                },
                "source_released": source_released,
            },
        )

    def _source_for_body(self, body: dict[str, Any]) -> SignalSource:
        if body.get("source_token") == "bundled":
            path = bundled_demo_record()
            if not path.is_file():
                raise FileNotFoundError("The bundled MIT-BIH demo record is not available.")
            return inspect_signal_source(path)
        if isinstance(body.get("source_token"), dict):
            session_id = _string(body, "session")
            source_token = _validated_blob_source_token(body.get("source_token"), session_id)
            session = self.server.state.session(session_id)
            candidates = _materialize_blob_files(session, source_token)
            primary = _choose_primary_source(candidates)
            source = inspect_signal_source(
                primary,
                csv_sampling_rate_hz=_optional_float(body.get("csv_sampling_rate_hz")),
            )
            session.source = source
            return source
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
    blob_configured = bool(os.environ.get("BLOB_STORE_ID", "").strip())
    uploads = not hosted or blob_configured
    return {
        "uploads": uploads,
        "default_duration_s": DEFAULT_DURATION_S,
        "max_duration_s": None,
        "deployment": "vercel" if hosted else "local",
        "upload_mode": (
            "private_blob"
            if hosted and blob_configured
            else "unavailable"
            if hosted
            else "local_filesystem"
        ),
        "max_upload_bytes": HOSTED_BLOB_FILE_BYTES if hosted else MAX_UPLOAD_BYTES,
        "max_upload_files": HOSTED_BLOB_FILE_COUNT if hosted else None,
        "upload_reason": (
            "private_blob_connected"
            if hosted and blob_configured
            else "secure_object_storage_required"
            if hosted
            else "private_local_filesystem"
        ),
    }


def _validated_blob_source_token(value: Any, session_id: str) -> dict[str, Any]:
    if not isinstance(value, (dict, list)):
        raise ValueError("The private ECG upload reference is missing.")
    raw_files = value.get("files") if isinstance(value, dict) else value
    if not isinstance(raw_files, list) or not 1 <= len(raw_files) <= HOSTED_BLOB_FILE_COUNT:
        raise ValueError("Upload between one and three matching ECG files.")
    prefix = f"ecg-uploads/{session_id}/"
    validated: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    total_bytes = 0
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ValueError("The private ECG upload reference is invalid.")
        filename = Path(str(raw.get("filename", ""))).name
        if not filename or filename in seen_names:
            raise ValueError("Uploaded ECG filenames must be unique.")
        suffix = Path(filename).suffix.casefold()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise ValueError(f"Unsupported uploaded file type: {suffix or '(none)' }.")
        try:
            byte_count = int(raw.get("bytes", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("The private ECG upload size is invalid.") from exc
        if not 0 < byte_count <= HOSTED_BLOB_FILE_BYTES:
            raise ValueError("Each hosted ECG file must be 16 MB or smaller.")
        total_bytes += byte_count
        if total_bytes > HOSTED_BLOB_FILE_BYTES * HOSTED_BLOB_FILE_COUNT:
            raise ValueError("The hosted ECG upload is too large.")
        pathname = str(raw.get("pathname", ""))
        if not pathname.startswith(prefix) or ".." in pathname:
            raise ValueError("The private ECG object is outside this browser session.")
        get_url = _validated_private_blob_url(raw.get("get_url"), pathname)
        delete_url = _validated_blob_delete_url(raw.get("delete_url"), pathname)
        validated.append(
            {
                "filename": filename,
                "pathname": pathname,
                "bytes": byte_count,
                "get_url": get_url,
                "delete_url": delete_url,
            }
        )
        seen_names.add(filename)
    return {"kind": "vercel_blob", "files": validated}


def _validated_private_blob_url(value: Any, pathname: str) -> str:
    url = str(value or "")
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").casefold()
    if (
        parsed.scheme != "https"
        or not hostname.endswith(".private.blob.vercel-storage.com")
        or unquote(parsed.path).lstrip("/") != pathname
        or "vercel-blob-signature=" not in parsed.query
    ):
        raise ValueError("The signed private ECG download URL is invalid.")
    return url


def _validated_blob_delete_url(value: Any, pathname: str) -> str:
    url = str(value or "")
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").casefold() != "vercel.com"
        or parsed.path.rstrip("/") != "/api/blob"
        or query.get("pathname", [""])[0] != pathname
        or "vercel-blob-signature" not in query
    ):
        raise ValueError("The signed private ECG deletion URL is invalid.")
    return url


def _materialize_blob_files(session: DemoSession, source_token: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for descriptor in source_token["files"]:
        destination = (session.directory / descriptor["filename"]).resolve()
        if session.directory.resolve() not in destination.parents:
            raise ValueError("Invalid private ECG filename.")
        request = Request(
            descriptor["get_url"],
            method="GET",
            headers={"User-Agent": "ECG-Cascade-Demo/1.0"},
        )
        try:
            with urlopen(request, timeout=60) as response, destination.open("wb") as handle:
                reported = response.headers.get("Content-Length")
                if reported and int(reported) > descriptor["bytes"]:
                    raise ValueError("The stored ECG file is larger than its signed upload size.")
                received = 0
                while True:
                    block = response.read(min(1024 * 1024, descriptor["bytes"] + 1 - received))
                    if not block:
                        break
                    handle.write(block)
                    received += len(block)
                    if received > descriptor["bytes"]:
                        raise ValueError("The stored ECG file exceeded its signed upload size.")
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            destination.unlink(missing_ok=True)
            raise ValueError(f"Could not read private ECG file {descriptor['filename']}.") from exc
        if destination.stat().st_size != descriptor["bytes"]:
            destination.unlink(missing_ok=True)
            raise ValueError(f"Private ECG file {descriptor['filename']} was incomplete.")
        candidates.append(destination)
    return candidates


def _delete_blob_files(source_token: dict[str, Any]) -> bool:
    deleted = True
    for descriptor in source_token["files"]:
        request = Request(
            descriptor["delete_url"],
            method="DELETE",
            headers={"User-Agent": "ECG-Cascade-Demo/1.0"},
        )
        try:
            with urlopen(request, timeout=30):
                pass
        except (HTTPError, URLError, TimeoutError, OSError):
            deleted = False
    return deleted


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
