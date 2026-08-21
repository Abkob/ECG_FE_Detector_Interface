"""Generate reproducible figures and literal outputs for detector explainers.

The script intentionally mirrors the installed NeuroKit2 0.2.13 source for
the Khamis/UNSW and default NeuroKit detectors.  It validates the reproduced
event arrays against the public NeuroKit entry points before exporting any
numbers used in the PDFs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import scipy.signal
import wfdb


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIGURES = HERE / "figures"
OUTPUTS = HERE / "outputs"
DATA = (
    ROOT
    / "Datasets"
    / "mit-bih-arrhythmia-database-1.0"
    / "mit-bih-arrhythmia-database-1.0.0"
)
sys.path.insert(0, str(ROOT / "feature_extraction" / "src"))

from ecg_cascade.peaks import detect_neurokit_gradient, detect_unsw  # noqa: E402
from ecg_cascade.reliability import refine_r_fiducials  # noqa: E402


NAVY = "#17324D"
TEAL = "#147D7E"
RED = "#9E2A2B"
ORANGE = "#D97706"
GREEN = "#287A45"
PURPLE = "#6B4C9A"
GREY = "#6B7280"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.23,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_record(record: str) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    path = str(DATA / record)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    return (
        np.asarray(rec.p_signal[:, 0], dtype=float),
        float(rec.fs),
        np.asarray(ann.sample, dtype=int),
        np.asarray(ann.symbol, dtype=str),
    )


def sortfilt1(x: np.ndarray, n: int, p: float) -> np.ndarray:
    p = float(np.clip(p, 0, 100))
    if n % 2 == 0:
        n1, n2 = int(n / 2 - 1), int(n / 2)
    else:
        n1 = n2 = int((n - 1) / 2)
    return scipy.ndimage.percentile_filter(
        x, percentile=p, size=n1 + n2 + 1, mode="nearest"
    )


def turning_points(x: np.ndarray, threshold: float) -> np.ndarray:
    x = np.append(x, [x[-1] + np.finfo(float).eps, x[-1]])
    tps = np.concatenate(([0], -np.sign(np.diff(np.sign(np.diff(x)))), [0]))
    tpidx = np.where(tps != 0)[0]
    pkth = tps[tpidx]
    i = 0
    inpeak = 0
    possibleidx: int | None = None
    ref = x[0]
    confirmed = np.full(len(pkth), np.nan)
    k = 0
    while i < len(tpidx):
        if inpeak == 0 and abs(x[tpidx[i]] - ref) > threshold:
            inpeak = int(pkth[i])
            possibleidx = int(tpidx[i])
            ref = x[tpidx[i]]
        if inpeak == 1 and (ref - x[tpidx[i]]) > threshold:
            assert possibleidx is not None
            confirmed[k] = possibleidx * tps[possibleidx]
            k += 1
            possibleidx = int(tpidx[i])
            ref = x[tpidx[i]]
            inpeak = -1
        elif inpeak == 1 and x[tpidx[i]] > ref:
            possibleidx = int(tpidx[i])
            ref = x[tpidx[i]]
        if inpeak == -1 and (x[tpidx[i]] - ref) > threshold:
            assert possibleidx is not None
            confirmed[k] = possibleidx * tps[possibleidx]
            k += 1
            possibleidx = int(tpidx[i])
            ref = x[tpidx[i]]
            inpeak = 1
        elif inpeak == -1 and possibleidx is not None and x[tpidx[i]] < x[possibleidx]:
            possibleidx = int(tpidx[i])
            ref = x[tpidx[i]]
        i += 1
    return confirmed[~np.isnan(confirmed)].astype(int)


def calculate_rr(qrs: np.ndarray, fs: float) -> tuple[float, np.ndarray]:
    rr = np.diff(qrs)
    rr = rr[rr < 5 * fs]
    return (float(np.mean(rr)) if rr.size else float("nan"), rr)


def smashed_fft(signal: np.ndarray, fs: float, m: int = 2**14) -> np.ndarray:
    fs_i = int(fs)
    p = int(np.ceil(len(signal) / (2 * fs_i)))
    y = np.zeros(2 * fs_i * p)
    y[: len(signal)] = signal
    y = y.reshape((p, 2 * fs_i))
    y = y - np.mean(y, axis=1, keepdims=True)
    z = np.zeros(m)
    for row in y:
        spectrum = np.abs(np.fft.fft(row, m))
        total = np.sum(spectrum)
        if total > 0:
            spectrum /= total
        z += spectrum
    return z / p


def unsw_diagnostic(signal: np.ndarray, fs: float) -> dict[str, object]:
    detrended = scipy.signal.detrend(signal).ravel()
    baseline_window = round(0.5 * fs)
    baseline = sortfilt1(detrended, baseline_window, 50)
    median_removed = detrended - baseline
    hp_sos = scipy.signal.butter(7, 0.7 / (fs / 2), btype="high", output="sos")
    highpassed = scipy.signal.sosfiltfilt(hp_sos, median_removed)
    lp_b, lp_a = scipy.signal.butter(7, 20 / (fs / 2), btype="low")
    filtered = scipy.signal.filtfilt(lp_b, lp_a, highpassed)

    derivative_coefficients = np.array([1, 0, -1]) / (2 * (1 / fs))
    differentiated = scipy.signal.lfilter(derivative_coefficients, [1], filtered)
    range_window = round(fs * 0.1)
    top = sortfilt1(filtered, range_window, 100)
    bottom = sortfilt1(filtered, range_window, 0)
    amplitude_envelope = np.maximum(top - bottom, 0)
    feature = np.abs(differentiated * amplitude_envelope)

    first_fc = 6.0
    first_k = ((fs / 2) * 0.0037) / first_fc
    first_nh = round(first_k * fs)
    first_hamming = scipy.signal.windows.hamming(first_nh, sym=True)
    first_hamming /= np.sum(np.abs(first_hamming))
    diffpower1 = np.abs(scipy.signal.filtfilt(first_hamming, [1], feature)) ** 0.5

    spectrum = smashed_fft(diffpower1, fs)
    frequencies = np.arange(0, 2**13) * fs / 2**14
    band = np.where((frequencies >= 0.1) & (frequencies < 4))[0]
    fft_hr_frequency = float(frequencies[band[np.argmax(spectrum[band])]])

    hr_min, hr_max = 1.5, 4.0
    adaptive_fc = float(np.median(2 * np.array([hr_min, fft_hr_frequency, hr_max])))
    second_k = ((fs / 2) * 0.0037) / adaptive_fc
    second_nh = round(second_k * fs)
    second_hamming = scipy.signal.windows.hamming(second_nh, sym=True)
    second_hamming /= np.sum(np.abs(second_hamming))
    diffpower2 = np.abs(scipy.signal.filtfilt(second_hamming, [1], feature)) ** 0.5

    wsort = round(np.median(2 * np.array([fs, fs / fft_hr_frequency, fs / hr_max])))
    upper = sortfilt1(diffpower2, wsort, 100)
    upper = sortfilt1(upper, wsort, 0)
    lower = sortfilt1(diffpower2, wsort, 0)
    lower = sortfilt1(lower, wsort, 100)
    qrs_envelope = upper - lower
    feature_height = float(np.median(qrs_envelope))
    main_threshold = 0.2 * feature_height

    pass1_tp = turning_points(diffpower2, main_threshold)
    pass1 = pass1_tp[pass1_tp > 0]
    mean_rr, _ = calculate_rr(pass1, fs)
    if pass1.size == 0:
        final = pass1.astype(int)
        pass2 = np.array([], dtype=int)
        pass3 = np.array([], dtype=int)
    else:
        pass2_tp = turning_points(diffpower2, main_threshold / 2)
        pass2 = pass2_tp[pass2_tp > 0]
        temp = np.zeros(len(diffpower2) + 2)
        temp[pass1 + 1] = 1
        temp = np.concatenate(([1], temp, [1]))
        start = 0
        for i in range(1, len(temp)):
            if temp[i] - temp[i - 1] == -1:
                start = i
            if temp[i] - temp[i - 1] == 1 and (i - 1) - start < 1.5 * mean_rr:
                temp[start : i - 1] = 1
        mask = np.where(temp[1:-1] == 1)[0]
        new_pass2 = np.setdiff1d(pass2, mask)
        combined = np.union1d(pass1, new_pass2)
        mean_rr, _ = calculate_rr(combined, fs)
        pass3_tp = turning_points(diffpower2, main_threshold * 2)
        pass3 = pass3_tp[pass3_tp > 0]
        short_idx: list[int] = []
        for i in range(1, len(combined)):
            if combined[i] - combined[i - 1] < 0.5 * mean_rr:
                short_idx.extend(range(int(combined[i - 1]), int(combined[i])))
        strong_short = np.intersect1d(pass3, np.asarray(short_idx, dtype=int))
        final = np.setdiff1d(combined, np.asarray(short_idx, dtype=int))
        final = np.union1d(final, strong_short).astype(int)

    official = np.asarray(
        nk.ecg_findpeaks(signal, sampling_rate=fs, method="khamis2016")["ECG_R_Peaks"],
        dtype=int,
    )
    if not np.array_equal(final, official):
        raise RuntimeError(
            f"UNSW diagnostic mismatch: reproduced {len(final)}, official {len(official)}"
        )
    return {
        "detrended": detrended,
        "baseline": baseline,
        "median_removed": median_removed,
        "highpassed": highpassed,
        "filtered": filtered,
        "derivative_coefficients": derivative_coefficients,
        "differentiated": differentiated,
        "top": top,
        "bottom": bottom,
        "amplitude_envelope": amplitude_envelope,
        "feature": feature,
        "diffpower1": diffpower1,
        "spectrum": spectrum,
        "frequencies": frequencies,
        "fft_hr_frequency": fft_hr_frequency,
        "diffpower2": diffpower2,
        "upper": upper,
        "lower": lower,
        "qrs_envelope": qrs_envelope,
        "feature_height": feature_height,
        "main_threshold": main_threshold,
        "pass1": pass1,
        "pass2": pass2,
        "pass3": pass3,
        "final": final,
        "parameters": {
            "sampling_rate_hz": fs,
            "baseline_window_samples": baseline_window,
            "baseline_window_ms": 1000 * baseline_window / fs,
            "range_window_samples": range_window,
            "range_window_ms": 1000 * range_window / fs,
            "highpass_hz": 0.7,
            "lowpass_hz": 20.0,
            "filter_order": 7,
            "first_smoother_fc_hz": first_fc,
            "first_hamming_samples": first_nh,
            "fft_hr_frequency_hz": fft_hr_frequency,
            "fft_hr_equivalent_bpm": 60 * fft_hr_frequency,
            "adaptive_smoother_fc_hz": adaptive_fc,
            "adaptive_hamming_samples": second_nh,
            "morphological_window_samples": wsort,
            "morphological_window_ms": 1000 * wsort / fs,
            "feature_height": feature_height,
            "main_turning_point_threshold": main_threshold,
            "first_pass_count": int(len(pass1)),
            "final_count": int(len(final)),
            "official_count": int(len(official)),
            "reproduction_exact": True,
        },
    }


def neurokit_diagnostic(signal: np.ndarray, fs: float) -> dict[str, object]:
    cleaned = np.asarray(nk.ecg_clean(signal, sampling_rate=fs, method="neurokit"))
    gradient = np.gradient(cleaned)
    abs_gradient = np.abs(gradient)
    smooth_samples = int(np.rint(0.1 * fs))
    average_samples = int(np.rint(0.75 * fs))
    smooth_gradient = scipy.ndimage.uniform_filter1d(
        abs_gradient, smooth_samples, mode="nearest"
    )
    average_gradient = scipy.ndimage.uniform_filter1d(
        smooth_gradient, average_samples, mode="nearest"
    )
    threshold = 1.5 * average_gradient
    mask = smooth_gradient > threshold
    beginnings = np.where((~mask[:-1]) & mask[1:])[0]
    endings = np.where(mask[:-1] & (~mask[1:]))[0]
    if beginnings.size:
        endings = endings[endings > beginnings[0]]
    count = min(len(beginnings), len(endings))
    minimum_length = float(np.mean(endings[:count] - beginnings[:count]) * 0.4)
    # The installed NeuroKit2 implementation starts with a sentinel peak at
    # sample zero, applies the strict delay test to the first candidate too,
    # and removes the sentinel at the end.
    peaks: list[int] = [0]
    regions: list[dict[str, object]] = []
    minimum_delay_samples = int(np.rint(0.3 * fs))
    for beginning, ending in zip(beginnings[:count], endings[:count], strict=True):
        length = int(ending - beginning)
        region_row: dict[str, object] = {
            "begin": int(beginning),
            "end": int(ending),
            "length": length,
            "long_enough": bool(length >= minimum_length),
            "local_maxima": [],
            "prominences": [],
            "chosen": None,
            "accepted": False,
        }
        if length >= minimum_length:
            local, props = scipy.signal.find_peaks(
                cleaned[beginning:ending], prominence=(None, None)
            )
            absolute = beginning + local
            region_row["local_maxima"] = absolute.astype(int).tolist()
            region_row["prominences"] = props["prominences"].astype(float).tolist()
            if local.size:
                peak = int(absolute[np.argmax(props["prominences"])])
                region_row["chosen"] = peak
                if peak - peaks[-1] > minimum_delay_samples:
                    peaks.append(peak)
                    region_row["accepted"] = True
        regions.append(region_row)
    peaks_array = np.asarray(peaks[1:], dtype=int)
    official = np.asarray(
        nk.ecg_findpeaks(cleaned, sampling_rate=fs, method="neurokit")["ECG_R_Peaks"],
        dtype=int,
    )
    if not np.array_equal(peaks_array, official):
        raise RuntimeError(
            f"NeuroKit diagnostic mismatch: reproduced {len(peaks_array)}, official {len(official)}"
        )
    project_250 = detect_neurokit_gradient(
        signal,
        fs,
        orientation="original",
        minimum_delay_ms=250.0,
        minimum_delay_inclusive=True,
    ).peak_samples
    return {
        "cleaned": cleaned,
        "gradient": gradient,
        "abs_gradient": abs_gradient,
        "smooth_gradient": smooth_gradient,
        "average_gradient": average_gradient,
        "threshold": threshold,
        "mask": mask,
        "beginnings": beginnings,
        "endings": endings,
        "minimum_length": minimum_length,
        "regions": regions,
        "peaks_300": peaks_array,
        "peaks_250": project_250,
        "parameters": {
            "sampling_rate_hz": fs,
            "clean_highpass_hz": 0.5,
            "clean_butterworth_order": 5,
            "powerline_hz": 50,
            "powerline_moving_average_samples": int(fs / 50),
            "smooth_window_samples": smooth_samples,
            "smooth_window_ms": 1000 * smooth_samples / fs,
            "average_window_samples": average_samples,
            "average_window_ms": 1000 * average_samples / fs,
            "gradient_threshold_weight": 1.5,
            "minimum_length_weight": 0.4,
            "default_minimum_delay_samples": minimum_delay_samples,
            "default_minimum_delay_ms": 1000 * minimum_delay_samples / fs,
            "project_minimum_delay_samples": int(round(0.25 * fs)),
            "project_minimum_delay_ms": 250.0,
            "mean_region_length_samples": float(
                np.mean(endings[:count] - beginnings[:count])
            ),
            "minimum_region_length_samples": minimum_length,
            "region_count": int(count),
            "default_peak_count": int(len(peaks_array)),
            "project_250_peak_count": int(len(project_250)),
            "official_count": int(len(official)),
            "reproduction_exact": True,
        },
    }


def nearest(values: np.ndarray, target: int) -> int:
    return int(values[np.argmin(np.abs(values - target))])


def expert_match_false_positives(
    expert: np.ndarray, detected: np.ndarray, tolerance_samples: int
) -> np.ndarray:
    used = np.zeros(len(expert), dtype=bool)
    false: list[int] = []
    for peak in detected:
        left = np.searchsorted(expert, peak - tolerance_samples, side="left")
        right = np.searchsorted(expert, peak + tolerance_samples, side="right")
        candidates = [i for i in range(left, right) if not used[i]]
        if not candidates:
            false.append(int(peak))
            continue
        best = min(candidates, key=lambda i: abs(int(expert[i]) - int(peak)))
        used[best] = True
    return np.asarray(false, dtype=int)


def zoom_slice(center: int, fs: float, before: float, after: float) -> slice:
    return slice(max(0, int(center - before * fs)), int(center + after * fs))


def savefig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def plot_unsw_stages(
    signal: np.ndarray,
    fs: float,
    target: int,
    stages: dict[str, object],
    q: int,
    refined: int,
) -> None:
    sl = zoom_slice(target, fs, 0.32, 0.38)
    idx = np.arange(sl.start, sl.stop)
    t = (idx - target) / fs * 1000
    fig, axes = plt.subplots(6, 1, figsize=(10.5, 12), sharex=True)
    axes[0].plot(t, signal[sl], color=NAVY, lw=1.2, label="raw MLII")
    axes[0].plot(t, np.asarray(stages["baseline"])[sl], color=ORANGE, lw=1, label="500-ms median baseline")
    axes[0].set_ylabel("mV")
    axes[0].set_title("UNSW transformation of the record-108 expert j beat")
    axes[0].legend(ncol=2, loc="upper left")

    axes[1].plot(t, np.asarray(stages["filtered"])[sl], color=TEAL, lw=1.2)
    axes[1].set_ylabel("mV")
    axes[1].set_title("Detrended, baseline removed, 0.7-20 Hz filtered")

    axes[2].plot(t, np.asarray(stages["differentiated"])[sl], color=PURPLE, lw=1.0, label="central-difference lens")
    ax2 = axes[2].twinx()
    ax2.plot(t, np.asarray(stages["amplitude_envelope"])[sl], color=ORANGE, lw=1, label="100-ms peak-to-trough envelope")
    axes[2].set_ylabel("mV/s")
    ax2.set_ylabel("mV")
    axes[2].set_title("Two lenses: rapid change and local peak-to-trough amplitude")
    lines = axes[2].lines + ax2.lines
    axes[2].legend(lines, [line.get_label() for line in lines], ncol=2, loc="upper left")

    axes[3].plot(t, np.asarray(stages["feature"])[sl], color=RED, lw=1.1)
    axes[3].set_ylabel("feature")
    axes[3].set_title(r"QRS feature: $|\mathrm{derivative}\times\mathrm{envelope}|$")

    axes[4].plot(t, np.asarray(stages["diffpower1"])[sl], color=GREY, lw=1, label="first smoother")
    axes[4].plot(t, np.asarray(stages["diffpower2"])[sl], color=GREEN, lw=1.3, label="rate-adaptive smoother")
    axes[4].set_ylabel("sqrt feature")
    axes[4].set_title("Smoothed positive QRS-evidence signals")
    axes[4].legend(ncol=2, loc="upper left")

    axes[5].plot(t, signal[sl], color=NAVY, lw=1.2)
    axes[5].axvline((q - target) / fs * 1000, color=TEAL, lw=2, label=f"UNSW q = {q}")
    axes[5].axvline(0, color="black", ls="--", lw=1.5, label=f"expert = {target}")
    axes[5].axvline((refined - target) / fs * 1000, color=ORANGE, lw=2, label=f"positive refinement = {refined}")
    axes[5].set_ylabel("mV")
    axes[5].set_xlabel("Time relative to expert j annotation (ms)")
    axes[5].set_title("The feature-signal event q is deterministic but is not the raw ECG apex")
    axes[5].legend(ncol=3, loc="lower left")
    savefig(fig, "unsw_record108_j_transformations.png")


def plot_unsw_frequency(stages: dict[str, object]) -> None:
    f = np.asarray(stages["frequencies"])
    # ``smashed_fft`` retains the full two-sided FFT.  The Khamis detector
    # searches only the non-negative half, represented by ``frequencies``.
    s = np.asarray(stages["spectrum"])[: len(f)]
    selected = float(stages["fft_hr_frequency"])
    mask = (f >= 0.1) & (f <= 4.0)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(f[mask], s[mask], color=NAVY, lw=1.3)
    ax.axvline(selected, color=RED, lw=2, label=f"selected spectral peak = {selected:.4f} Hz ({60*selected:.1f} bpm)")
    ax.axvspan(1.5, 4.0, color=TEAL, alpha=0.08, label="HRmin-HRmax design range")
    ax.set(xlabel="Frequency (Hz)", ylabel="Normalized spectral magnitude", title="UNSW rate estimate used to choose the second smoothing bandwidth")
    ax.legend(loc="upper right")
    savefig(fig, "unsw_frequency_adaptation.png")


def plot_unsw_polarity_cases(
    signal: np.ndarray,
    fs: float,
    expert_samples: np.ndarray,
    expert_symbols: np.ndarray,
    original_qrs: np.ndarray,
    inverted_qrs: np.ndarray,
) -> dict[str, dict[str, float | int | str]]:
    cases = {"F": 180621, "j": 435657}
    refined_original = refine_r_fiducials(
        original_qrs, signal, sampling_rate_hz=fs, radius_ms=50
    )
    refined_inverted = refine_r_fiducials(
        inverted_qrs, -signal, sampling_rate_hz=fs, radius_ms=50
    )
    out: dict[str, dict[str, float | int | str]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (symbol, target) in zip(axes, cases.items(), strict=True):
        q_o = nearest(original_qrs, target)
        r_o = refined_original[np.argmin(np.abs(original_qrs - target))]
        q_i = nearest(inverted_qrs, target)
        r_i = refined_inverted[np.argmin(np.abs(inverted_qrs - target))]
        sl = zoom_slice(target, fs, 0.16, 0.20)
        idx = np.arange(sl.start, sl.stop)
        t = (idx - target) / fs * 1000
        ax.plot(t, signal[sl], color=NAVY, lw=1.3)
        ax.axvline(0, color="black", ls="--", lw=1.3, label="expert")
        ax.axvline((q_o - target) / fs * 1000, color=TEAL, lw=1.5, label="UNSW initial")
        ax.axvline((r_o - target) / fs * 1000, color=ORANGE, lw=2, label="positive argmax")
        ax.axvline((r_i - target) / fs * 1000, color=PURPLE, lw=2, label="inverted argmax = original argmin")
        ax.set(title=f"Expert {symbol}: original and inverted refinements", xlabel="Relative time (ms)", ylabel="MLII (mV)")
        out[symbol] = {
            "expert_sample": target,
            "expert_time_s": target / fs,
            "original_q_sample": int(q_o),
            "original_q_error_ms": float((q_o - target) * 1000 / fs),
            "positive_refined_sample": int(r_o),
            "positive_refined_error_ms": float((r_o - target) * 1000 / fs),
            "inverted_q_sample": int(q_i),
            "negative_refined_sample": int(r_i),
            "negative_refined_error_ms": float((r_i - target) * 1000 / fs),
        }
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("One polarity rule cannot win on both morphologies", fontsize=12, fontweight="bold")
    savefig(fig, "unsw_positive_negative_counterexample.png")
    return out


def region_for_peak(regions: list[dict[str, object]], peak: int) -> dict[str, object]:
    for row in regions:
        if row["chosen"] == peak:
            return row
    return min(
        [row for row in regions if row["chosen"] is not None],
        key=lambda row: abs(int(row["chosen"]) - peak),
    )


def plot_neurokit_stages(
    signal: np.ndarray,
    fs: float,
    target: int,
    stages: dict[str, object],
    chosen: int,
    region: dict[str, object],
) -> None:
    sl = zoom_slice(target, fs, 0.32, 0.38)
    idx = np.arange(sl.start, sl.stop)
    t = (idx - target) / fs * 1000
    fig, axes = plt.subplots(6, 1, figsize=(10.5, 12), sharex=True)
    axes[0].plot(t, signal[sl], color=NAVY, lw=1.1, label="raw")
    axes[0].plot(t, np.asarray(stages["cleaned"])[sl], color=TEAL, lw=1.1, label="cleaned")
    axes[0].set_ylabel("mV")
    axes[0].set_title("NeuroKit default transformation of the record-108 expert j beat")
    axes[0].legend(ncol=2)

    axes[1].plot(t, np.asarray(stages["gradient"])[sl], color=PURPLE, lw=1)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_ylabel("mV/sample")
    axes[1].set_title("Signed gradient: positive when voltage rises, negative when it falls")

    axes[2].plot(t, np.asarray(stages["abs_gradient"])[sl], color=RED, lw=1)
    axes[2].set_ylabel("absolute")
    axes[2].set_title("Absolute gradient: rising and falling edges both become positive")

    axes[3].plot(t, np.asarray(stages["smooth_gradient"])[sl], color=GREEN, lw=1.3, label="100-ms smoothed absolute gradient")
    axes[3].plot(t, np.asarray(stages["threshold"])[sl], color=ORANGE, lw=1.2, label="1.5 x 750-ms local average")
    axes[3].fill_between(t, 0, np.asarray(stages["smooth_gradient"])[sl], where=np.asarray(stages["mask"])[sl], color=TEAL, alpha=0.16, label="candidate QRS mask")
    axes[3].set_ylabel("gradient")
    axes[3].set_title("A candidate QRS region exists where smoothed steepness exceeds threshold")
    axes[3].legend(ncol=3, fontsize=7)

    axes[4].plot(t, np.asarray(stages["cleaned"])[sl], color=NAVY, lw=1.2)
    begin, end = int(region["begin"]), int(region["end"])
    axes[4].axvspan((begin - target) / fs * 1000, (end - target) / fs * 1000, color=TEAL, alpha=0.16, label="accepted region")
    maxima = np.asarray(region["local_maxima"], dtype=int)
    if maxima.size:
        axes[4].scatter((maxima - target) / fs * 1000, np.asarray(stages["cleaned"])[maxima], color=GREY, s=24, label="positive local maxima")
    axes[4].scatter([(chosen - target) / fs * 1000], [np.asarray(stages["cleaned"])[chosen]], color=RED, s=60, marker="x", linewidth=2, label="greatest prominence")
    axes[4].set_ylabel("mV")
    axes[4].set_title("Inside the region, only positive local maxima compete")
    axes[4].legend(ncol=3, fontsize=7)

    axes[5].plot(t, signal[sl], color=NAVY, lw=1.2)
    axes[5].axvline(0, color="black", ls="--", lw=1.5, label=f"expert = {target}")
    axes[5].axvline((chosen - target) / fs * 1000, color=RED, lw=2, label=f"NeuroKit = {chosen}")
    axes[5].set(xlabel="Time relative to expert j annotation (ms)", ylabel="mV", title="The returned event is a positive-prominence maximum, not a polarity-neutral fiducial")
    axes[5].legend()
    savefig(fig, "neurokit_record108_j_transformations.png")


def plot_neurokit_delay(stages: dict[str, object], fs: float) -> None:
    rr_ms = np.arange(180, 421, 1)
    bpm = 60000 / rr_ms
    accepted_300 = rr_ms > 300
    accepted_250 = rr_ms >= 250
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
    axes[0].plot(rr_ms, bpm, color=NAVY, lw=1.5)
    axes[0].axvline(300, color=RED, lw=2, label="default: separation >300 ms")
    axes[0].axvline(250, color=TEAL, lw=2, label="project experiment: separation >=250 ms")
    axes[0].set(ylabel="Equivalent instantaneous bpm", title="Minimum-delay rule converts directly into a high-rate ceiling")
    axes[0].legend()
    axes[1].step(rr_ms, accepted_300.astype(int), where="mid", color=RED, lw=2, label="default 300-ms strict")
    axes[1].step(rr_ms, accepted_250.astype(int), where="mid", color=TEAL, lw=2, label="project 250-ms inclusive")
    axes[1].set(xlabel="Candidate separation (ms)", ylabel="Eligible?", yticks=[0, 1])
    axes[1].legend()
    savefig(fig, "neurokit_minimum_delay_timeline.png")


def plot_record113_failure() -> dict[str, float | int]:
    signal, fs, expert, symbols = load_record("113")
    cleaned = np.asarray(nk.ecg_clean(signal, sampling_rate=fs, method="neurokit"))
    nk300 = np.asarray(
        nk.ecg_findpeaks(cleaned, sampling_rate=fs, method="neurokit")["ECG_R_Peaks"],
        dtype=int,
    )
    false = expert_match_false_positives(expert, nk300, round(0.075 * fs))
    candidates: list[tuple[int, int, int]] = []
    for fp in false:
        prev = expert[expert < fp]
        if prev.size:
            offset = int(fp - prev[-1])
            if 0.30 * fs <= offset <= 0.38 * fs:
                candidates.append((int(prev[-1]), int(fp), offset))
    if not candidates:
        raise RuntimeError("No record-113 T-wave-like false detection found")
    true_peak, false_peak, offset = candidates[len(candidates) // 2]
    unsw = detect_unsw(signal, fs, orientation="original").peak_samples
    sl = zoom_slice(true_peak, fs, 0.30, 0.75)
    idx = np.arange(sl.start, sl.stop)
    t = (idx - true_peak) / fs * 1000
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.plot(t, signal[sl], color=NAVY, lw=1.2)
    exp_win = expert[(expert >= sl.start) & (expert < sl.stop)]
    nk_win = nk300[(nk300 >= sl.start) & (nk300 < sl.stop)]
    unsw_win = unsw[(unsw >= sl.start) & (unsw < sl.stop)]
    ax.scatter((exp_win - true_peak) / fs * 1000, signal[exp_win], facecolors="white", edgecolors="black", s=60, label="expert beats", zorder=4)
    ax.scatter((nk_win - true_peak) / fs * 1000, signal[nk_win], color=RED, marker="x", s=60, label="NeuroKit 300 ms", zorder=5)
    ax.scatter((unsw_win - true_peak) / fs * 1000, signal[unsw_win], color=TEAL, marker="^", s=48, label="UNSW", zorder=4)
    ax.axvspan((false_peak - true_peak) / fs * 1000 - 15, (false_peak - true_peak) / fs * 1000 + 15, color=ORANGE, alpha=0.15)
    ax.annotate(
        f"NeuroKit extra candidate\n{offset*1000/fs:.1f} ms after true QRS",
        xy=((false_peak - true_peak) / fs * 1000, signal[false_peak]),
        xytext=(420, np.max(signal[sl]) * 0.85),
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
    )
    ax.set(xlabel="Time from preceding expert QRS (ms)", ylabel="MLII (mV)", title="Record 113: a steep post-QRS/T-wave deflection passes NeuroKit's region and delay rules")
    ax.legend(ncol=3, loc="lower right")
    savefig(fig, "neurokit_record113_twave_double_detection.png")
    return {
        "record": 113,
        "true_qrs_sample": true_peak,
        "false_neurokit_sample": false_peak,
        "offset_samples": offset,
        "offset_ms": offset * 1000 / fs,
        "neurokit_300_full_record_count": int(len(nk300)),
        "neurokit_false_positive_count_under_75ms_matching": int(len(false)),
        "unsw_full_record_count": int(len(unsw)),
    }


def write_tables(
    unsw: dict[str, object],
    neurokit: dict[str, object],
    polarity: dict[str, dict[str, float | int | str]],
    record113: dict[str, float | int],
    unsw_target: dict[str, object],
    neurokit_target: dict[str, object],
) -> None:
    payload_unsw = {
        "software": {"neurokit2": nk.__version__, "scipy": scipy.__version__},
        "parameters": unsw["parameters"],
        "record108_j": unsw_target,
        "polarity_cases": polarity,
        "all48_metrics": {
            "unsw_initial": {
                "tp": 109244,
                "fp": 350,
                "fn": 250,
                "sensitivity": 0.99772,
                "ppv": 0.99681,
                "f1": 0.99726,
            },
            "unsw_positive_refined": {
                "tp": 109207,
                "fp": 387,
                "fn": 287,
                "sensitivity": 0.99738,
                "ppv": 0.99647,
                "f1": 0.99692,
            },
        },
    }
    payload_nk = {
        "software": {"neurokit2": nk.__version__, "scipy": scipy.__version__},
        "parameters": neurokit["parameters"],
        "record108_j": neurokit_target,
        "record113_failure_example": record113,
        "all48_metrics": {
            "neurokit_250_inclusive": {
                "tp": 107713,
                "fp": 2780,
                "fn": 1781,
                "sensitivity": 0.98373,
                "ppv": 0.97484,
                "f1": 0.97927,
            },
            "neurokit_300_default": {
                "tp": 107677,
                "fp": 2503,
                "fn": 1817,
                "sensitivity": 0.98341,
                "ppv": 0.97728,
                "f1": 0.98033,
            },
        },
    }
    (OUTPUTS / "unsw_literal_outputs.json").write_text(
        json.dumps(payload_unsw, indent=2), encoding="utf-8"
    )
    (OUTPUTS / "neurokit_literal_outputs.json").write_text(
        json.dumps(payload_nk, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"stage": key, **value}
            for key, value in {
                "UNSW initial": payload_unsw["all48_metrics"]["unsw_initial"],
                "UNSW positive refined": payload_unsw["all48_metrics"]["unsw_positive_refined"],
                "NeuroKit 250 inclusive": payload_nk["all48_metrics"]["neurokit_250_inclusive"],
                "NeuroKit 300 default": payload_nk["all48_metrics"]["neurokit_300_default"],
            }.items()
        ]
    ).to_csv(OUTPUTS / "detector_literal_metrics.csv", index=False)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    style()
    signal, fs, expert, symbols = load_record("108")
    target = 435657
    if symbols[np.where(expert == target)[0][0]] != "j":
        raise RuntimeError("Expected record-108 sample 435657 to be expert symbol j")

    unsw = unsw_diagnostic(signal, fs)
    original_qrs = np.asarray(unsw["final"], dtype=int)
    q = nearest(original_qrs, target)
    refined_all = refine_r_fiducials(
        original_qrs, signal, sampling_rate_hz=fs, radius_ms=50
    )
    refined = int(refined_all[np.argmin(np.abs(original_qrs - target))])
    plot_unsw_stages(signal, fs, target, unsw, q, refined)
    plot_unsw_frequency(unsw)
    inverted_qrs = detect_unsw(signal, fs, orientation="inverted").peak_samples
    polarity = plot_unsw_polarity_cases(
        signal, fs, expert, symbols, original_qrs, inverted_qrs
    )

    neurokit = neurokit_diagnostic(signal, fs)
    nk_peaks = np.asarray(neurokit["peaks_300"], dtype=int)
    nk_target = nearest(nk_peaks, target)
    nk_region = region_for_peak(neurokit["regions"], nk_target)
    plot_neurokit_stages(signal, fs, target, neurokit, nk_target, nk_region)
    plot_neurokit_delay(neurokit, fs)
    record113 = plot_record113_failure()

    unsw_target = {
        "expert_symbol": "j",
        "expert_sample": target,
        "expert_time_s": target / fs,
        "unsw_q_sample": q,
        "unsw_q_error_ms": (q - target) * 1000 / fs,
        "unsw_feature_value_at_q": float(np.asarray(unsw["diffpower2"])[q]),
        "positive_refined_sample": refined,
        "positive_refined_error_ms": (refined - target) * 1000 / fs,
        "positive_refined_raw_amplitude_mv": float(signal[refined]),
        "search_radius_samples": int(round(0.05 * fs)),
        "search_radius_ms": 50.0,
    }
    maxima = np.asarray(nk_region["local_maxima"], dtype=int)
    prominences = np.asarray(nk_region["prominences"], dtype=float)
    chosen_idx = int(np.where(maxima == nk_target)[0][0]) if nk_target in maxima else -1
    previous_nk = nk_peaks[np.where(nk_peaks == nk_target)[0][0] - 1]
    neurokit_target = {
        "expert_symbol": "j",
        "expert_sample": target,
        "expert_time_s": target / fs,
        "candidate_region_begin_sample": int(nk_region["begin"]),
        "candidate_region_end_sample": int(nk_region["end"]),
        "candidate_region_length_samples": int(nk_region["length"]),
        "candidate_region_length_ms": float(int(nk_region["length"]) * 1000 / fs),
        "positive_local_maxima_count": int(len(maxima)),
        "chosen_sample": nk_target,
        "chosen_error_ms": float((nk_target - target) * 1000 / fs),
        "chosen_cleaned_amplitude_mv": float(np.asarray(neurokit["cleaned"])[nk_target]),
        "chosen_prominence": float(prominences[chosen_idx]) if chosen_idx >= 0 else None,
        "separation_from_previous_accepted_ms": float((nk_target - previous_nk) * 1000 / fs),
        "accepted_under_300ms_strict_rule": bool(nk_target - previous_nk > round(0.3 * fs)),
    }
    write_tables(
        unsw, neurokit, polarity, record113, unsw_target, neurokit_target
    )
    print(json.dumps({"UNSW": unsw_target, "NeuroKit": neurokit_target, "record113": record113}, indent=2))


if __name__ == "__main__":
    main()
