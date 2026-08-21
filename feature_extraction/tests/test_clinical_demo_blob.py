from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

from ecg_cascade import clinical_demo_web as web


def _descriptor(session: str = "browser-123", *, content: bytes = b"ecg") -> dict[str, object]:
    pathname = f"ecg-uploads/{session}/id-signal.csv"
    encoded = quote(pathname, safe="")
    return {
        "filename": "signal.csv",
        "pathname": pathname,
        "bytes": len(content),
        "get_url": (
            f"https://store.private.blob.vercel-storage.com/{pathname}"
            "?vercel-blob-delegation=test&vercel-blob-signature=test"
        ),
        "delete_url": (
            f"https://vercel.com/api/blob/?pathname={encoded}"
            "&vercel-blob-delegation=test&vercel-blob-signature=test"
        ),
    }


def test_hosted_capabilities_enable_private_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ECG_DEMO_HOSTED", "1")
    monkeypatch.setenv("BLOB_STORE_ID", "store_test")
    capabilities = web._runtime_capabilities()
    assert capabilities["uploads"] is True
    assert capabilities["upload_mode"] == "private_blob"
    assert capabilities["max_upload_bytes"] == 16 * 1024 * 1024


def test_blob_source_token_is_scoped_to_browser_session() -> None:
    token = web._validated_blob_source_token([_descriptor()], "browser-123")
    assert token["kind"] == "vercel_blob"
    assert token["files"][0]["filename"] == "signal.csv"

    with pytest.raises(ValueError, match="outside this browser session"):
        web._validated_blob_source_token([_descriptor("another")], "browser-123")


def test_materialize_private_blob_downloads_exact_signed_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b"time,ecg\n0,1\n"
    descriptor = _descriptor(content=content)
    token = web._validated_blob_source_token([descriptor], "browser-123")

    class FakeResponse:
        headers = {"Content-Length": str(len(content))}

        def __init__(self) -> None:
            self.offset = 0

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            chunk = content[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    monkeypatch.setattr(web, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    session = web.DemoSession("browser-123", tmp_path)
    paths = web._materialize_blob_files(session, token)
    assert paths == [tmp_path / "signal.csv"]
    assert paths[0].read_bytes() == content


def test_delete_private_blob_uses_only_validated_signed_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = web._validated_blob_source_token([_descriptor()], "browser-123")
    methods: list[str] = []

    class FakeDeleteResponse:
        def __enter__(self) -> FakeDeleteResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_open(request: object, **_kwargs: object) -> FakeDeleteResponse:
        methods.append(request.get_method())  # type: ignore[attr-defined]
        return FakeDeleteResponse()

    monkeypatch.setattr(web, "urlopen", fake_open)
    assert web._delete_blob_files(token) is True
    assert methods == ["DELETE"]
