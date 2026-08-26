from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy import signal, stats

from grinder_diagnostics_model.constants import (
    EXTRACTED_SENSORS,
    SELECTED_SENSORS,
    STATISTIC_NAMES,
)
from grinder_diagnostics_model.data import RingRecord, discover_rings

ANALOG_SAMPLE_RATE = 100_000
DIGITAL_SAMPLE_RATE = 10_000


@dataclass(frozen=True)
class FeatureConfig:
    minimum_idle_seconds: float = 0.25
    vibration_cutoff_hz: float = 43_000.0
    acoustic_emission_cutoff_hz: float = 43_000.0
    temperature_cutoff_hz: float = 25.0
    filter_order: int = 4

    def cutoff_for(self, channel: str) -> float:
        if channel.startswith("Temp_"):
            return self.temperature_cutoff_hz
        if channel.startswith("AE_"):
            return self.acoustic_emission_cutoff_hz
        return self.vibration_cutoff_hz


CHANNEL_ALIASES = {"Vib_WH_X": "Vib_WH_Z"}


def feature_names(sensors: tuple[str, ...] = SELECTED_SENSORS) -> list[str]:
    return [
        f"{sensor}__{domain}__{statistic}"
        for sensor in sensors
        for domain in ("time", "frequency")
        for statistic in STATISTIC_NAMES
    ]


def _contact_index(marker: np.ndarray, minimum_idle_samples: int) -> int:
    if marker.ndim != 1 or len(marker) <= minimum_idle_samples:
        raise ValueError("AE_limit marker is too short to identify an idle segment")
    initial_value = marker[0]
    changes = np.flatnonzero(marker[minimum_idle_samples:] != initial_value)
    if len(changes) == 0:
        raise ValueError("AE_limit marker has no contact transition")
    return int(changes[0] + minimum_idle_samples)


def _low_pass(values: np.ndarray, cutoff_hz: float, order: int) -> np.ndarray:
    nyquist = ANALOG_SAMPLE_RATE / 2
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(f"Low-pass cutoff must be between 0 and {nyquist:g} Hz")
    sos = signal.butter(order, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return signal.sosfiltfilt(sos, values)


def _statistics(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("At least three samples are required for feature extraction")
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    rms = float(np.sqrt(np.mean(np.square(values))))
    if std == 0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        skewness = float(stats.skew(values, bias=False))
        kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))
    result = {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "rms": rms,
        "peak_to_peak": float(np.ptp(values)),
        "crest_factor": float(np.max(np.abs(values)) / rms) if rms else 0.0,
        "band_power": float(np.mean(np.square(values))),
        "energy": float(np.trapezoid(values)),
        "percentile_90": float(np.percentile(values, 90)),
    }
    non_finite = [name for name, value in result.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError(f"Non-finite features: {non_finite}")
    return result


def extract_ring_features(
    record: RingRecord,
    config: FeatureConfig | None = None,
) -> dict[str, float | int | str]:
    config = config or FeatureConfig()
    with TdmsFile.open(record.path) as tdms:
        marker = np.asarray(tdms["Digital"]["AE_limit"][:])
        digital_contact = _contact_index(
            marker,
            round(config.minimum_idle_seconds * DIGITAL_SAMPLE_RATE),
        )
        analog_contact = round(digital_contact * ANALOG_SAMPLE_RATE / DIGITAL_SAMPLE_RATE)
        output: dict[str, float | int | str] = {
            "ring_id": record.ring_id,
            "test": record.test,
            "condition": record.condition,
            "has_fault": record.has_fault,
            "dressing_cycle": record.dressing_cycle,
            "ring": record.ring,
            "idle_samples": analog_contact,
            "idle_seconds": digital_contact / DIGITAL_SAMPLE_RATE,
            "source_path": str(record.path),
        }
        for channel_name in EXTRACTED_SENSORS:
            source_channel_name = CHANNEL_ALIASES.get(channel_name, channel_name)
            channel = tdms["Analogue"][source_channel_name]
            if len(channel) < analog_contact:
                raise ValueError(
                    f"{record.ring_id} channel {channel_name} ends before contact"
                )
            idle = np.asarray(channel[:analog_contact], dtype=np.float64)
            filtered = _low_pass(
                idle,
                cutoff_hz=config.cutoff_for(channel_name),
                order=config.filter_order,
            )
            time_features = _statistics(filtered)
            detrended = signal.detrend(filtered, type="linear")
            spectrum = np.abs(np.fft.rfft(detrended)) / len(detrended)
            frequency_features = _statistics(spectrum)
            output.update(
                {
                    f"{channel_name}__time__{name}": value
                    for name, value in time_features.items()
                }
            )
            output.update(
                {
                    f"{channel_name}__frequency__{name}": value
                    for name, value in frequency_features.items()
                }
            )
    return output


def build_feature_table(
    data_root: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    records = discover_rings(data_root.resolve())
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        records = records[:limit]
    partial_path = output_path.with_suffix(f"{output_path.suffix}.partial")
    if partial_path.is_file():
        frame = pd.read_parquet(partial_path)
        completed = set(frame["ring_id"])
        print(f"features: resuming after {len(completed)} completed rings")
        rows = frame.to_dict(orient="records")
    else:
        completed = set()
        rows = []
    pending = [record for record in records if record.ring_id not in completed]
    for index, record in enumerate(pending, start=1):
        rows.append(extract_ring_features(record, config))
        total_completed = len(completed) + index
        if total_completed == 1 or total_completed % 10 == 0 or index == len(pending):
            frame = pd.DataFrame(rows)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(partial_path, index=False)
            print(f"features: {total_completed}/{len(records)} rings")
    frame = pd.DataFrame(rows)
    missing = sorted(set(feature_names(EXTRACTED_SENSORS)) - set(frame.columns))
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    partial_path.unlink(missing_ok=True)
    return frame
