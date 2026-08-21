"""Explicit WFDB loading for annotated MIT--BIH validation records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb


@dataclass(frozen=True)
class WFDBSegment:
    record_path: str
    channel_index: int
    channel_name: str
    sampling_rate_hz: float
    start_sample: int
    samples: np.ndarray

    @property
    def start_s(self) -> float:
        return self.start_sample / self.sampling_rate_hz

    @property
    def duration_s(self) -> float:
        return self.samples.size / self.sampling_rate_hz


def load_wfdb_segment(
    record_path: str | Path,
    *,
    channel: int | str = 0,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> WFDBSegment:
    """Load one verified WFDB channel; ``record_path`` omits the extension."""

    record_path = Path(record_path).expanduser().resolve()
    header_path = record_path.with_suffix(".hea")
    if not header_path.is_file():
        raise FileNotFoundError(header_path)
    header = wfdb.rdheader(str(record_path))
    fs = float(header.fs)
    if start_s < 0:
        raise ValueError("start_s must be non-negative")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if isinstance(channel, str):
        matches = [
            index
            for index, name in enumerate(header.sig_name)
            if name.casefold() == channel.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Channel {channel!r} matched {len(matches)} channels; "
                f"available={header.sig_name}"
            )
        channel_index = matches[0]
    else:
        channel_index = int(channel)
        if not 0 <= channel_index < int(header.n_sig):
            raise ValueError(
                f"channel index {channel_index} outside 0..{int(header.n_sig)-1}"
            )
    start_sample = int(np.rint(start_s * fs))
    if start_sample >= int(header.sig_len):
        raise ValueError("start_s lies beyond the WFDB record")
    if duration_s is None:
        end_sample = int(header.sig_len)
    else:
        end_sample = min(
            int(header.sig_len), start_sample + int(np.rint(duration_s * fs))
        )
    record = wfdb.rdrecord(
        str(record_path),
        sampfrom=start_sample,
        sampto=end_sample,
        channels=[channel_index],
        physical=True,
    )
    samples = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    if not np.all(np.isfinite(samples)):
        raise ValueError("WFDB segment contains non-finite ECG samples")
    return WFDBSegment(
        record_path=str(record_path),
        channel_index=channel_index,
        channel_name=str(header.sig_name[channel_index]),
        sampling_rate_hz=fs,
        start_sample=start_sample,
        samples=samples,
    )


def load_wfdb_beat_annotations(
    record_path: str | Path,
    *,
    extension: str = "atr",
    start_sample: int = 0,
    end_sample: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load reference beat samples and symbols, excluding non-beat markers."""

    record_path = Path(record_path).expanduser().resolve()
    annotation = wfdb.rdann(str(record_path), extension)
    samples = np.asarray(annotation.sample, dtype=np.int64)
    symbols = np.asarray(annotation.symbol, dtype=object)
    # WFDB beat labels documented for the MIT--BIH annotations.  Rhythm,
    # waveform, comment, and signal-quality markers are intentionally absent.
    beat_symbols = {
        "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F",
        "e", "j", "n", "E", "/", "f", "Q", "?",
    }
    keep = np.array(
        [symbol in beat_symbols for symbol in symbols],
        dtype=bool,
    )
    keep &= samples >= start_sample
    if end_sample is not None:
        keep &= samples < end_sample
    return samples[keep] - start_sample, symbols[keep]
