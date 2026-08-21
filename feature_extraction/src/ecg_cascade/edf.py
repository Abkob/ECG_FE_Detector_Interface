"""EDF inspection and deliberately explicit ECG-channel loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import numpy as np
import pyedflib


@dataclass(frozen=True)
class ChannelInfo:
    index: int
    label: str
    sampling_rate_hz: float
    sample_count: int
    physical_dimension: str
    physical_min: float
    physical_max: float


@dataclass(frozen=True)
class RecordingInfo:
    path: str
    duration_s: float
    start_datetime: str
    channels: tuple[ChannelInfo, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["channels"] = [asdict(channel) for channel in self.channels]
        return result


@dataclass(frozen=True)
class SignalSegment:
    path: str
    channel: ChannelInfo
    start_s: float
    duration_s: float
    samples: np.ndarray


def inspect_edf(path: str | Path) -> RecordingInfo:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    reader = pyedflib.EdfReader(str(path))
    try:
        labels = reader.getSignalLabels()
        sample_counts = reader.getNSamples()
        channels = tuple(
            ChannelInfo(
                index=index,
                label=label.strip(),
                sampling_rate_hz=float(reader.getSampleFrequency(index)),
                sample_count=int(sample_counts[index]),
                physical_dimension=reader.getPhysicalDimension(index).strip(),
                physical_min=float(reader.getPhysicalMinimum(index)),
                physical_max=float(reader.getPhysicalMaximum(index)),
            )
            for index, label in enumerate(labels)
        )
        start = reader.getStartdatetime().isoformat()
        duration = float(reader.getFileDuration())
    finally:
        reader.close()

    return RecordingInfo(
        path=str(path),
        duration_s=duration,
        start_datetime=start,
        channels=channels,
    )


def choose_ecg_channel(
    recording: RecordingInfo, requested_label: str | None = None
) -> ChannelInfo:
    """Choose an ECG channel without silently resolving an ambiguous header."""

    if requested_label is not None:
        matches = [
            channel
            for channel in recording.channels
            if channel.label.casefold() == requested_label.strip().casefold()
        ]
        if len(matches) != 1:
            labels = ", ".join(repr(channel.label) for channel in recording.channels)
            raise ValueError(
                f"Requested ECG channel {requested_label!r} matched {len(matches)} "
                f"channels. Available labels: {labels}"
            )
        return matches[0]

    pattern = re.compile(r"(^|[^A-Z])(ECG|EKG)([^A-Z]|$)")
    candidates = [
        channel
        for channel in recording.channels
        if pattern.search(channel.label.upper())
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "No channel label contains the token ECG or EKG. Specify --channel "
            "only after verifying the EDF header and signal morphology."
        )
    raise ValueError(
        "Multiple ECG/EKG-labelled channels were found; specify --channel exactly: "
        + ", ".join(repr(channel.label) for channel in candidates)
    )


def load_ecg_segment(
    path: str | Path,
    *,
    channel_label: str | None,
    start_s: float,
    duration_s: float,
) -> SignalSegment:
    if start_s < 0:
        raise ValueError("start_s must be non-negative")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    recording = inspect_edf(path)
    channel = choose_ecg_channel(recording, channel_label)
    if start_s >= recording.duration_s:
        raise ValueError(
            f"start_s={start_s} is outside the {recording.duration_s:.3f}-s recording"
        )

    fs = channel.sampling_rate_hz
    start_sample = int(round(start_s * fs))
    requested_count = int(round(duration_s * fs))
    sample_count = min(requested_count, channel.sample_count - start_sample)
    if sample_count <= 0:
        raise ValueError("Selected segment contains no samples")

    reader = pyedflib.EdfReader(recording.path)
    try:
        samples = np.asarray(
            reader.readSignal(channel.index, start=start_sample, n=sample_count),
            dtype=np.float64,
        )
    finally:
        reader.close()

    if samples.size != sample_count:
        raise RuntimeError(
            f"EDF reader returned {samples.size} samples; expected {sample_count}"
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("ECG segment contains non-finite samples; no silent imputation applied")

    return SignalSegment(
        path=recording.path,
        channel=channel,
        start_s=float(start_s),
        duration_s=float(samples.size / fs),
        samples=samples,
    )

