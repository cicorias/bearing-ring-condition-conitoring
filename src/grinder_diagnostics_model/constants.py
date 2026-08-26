from pathlib import Path

DEFAULT_DATA_ROOT = Path("data/source")
DEFAULT_FEATURE_PATH = Path("data/generated/grinder-diagnostics-model/ring-features.parquet")
DEFAULT_ARTIFACT_DIR = Path("artifacts/grinder-diagnostics-model")

TEST_LABELS = {
    1: "baseline",
    2: "workhead_drive_belt_damage",
    3: "workhead_spindle_unbalance",
    4: "drive_plate_setup_fault",
    5: "workhead_tooling_setup_fault",
    6: "worn_workhead_tooling_support",
    7: "baseline",
}

FAULT_LABELS = {test: label for test, label in TEST_LABELS.items() if test not in {1, 7}}

SELECTED_SENSORS = (
    "AE_Dittel_lp30",
    "AE_WH",
    "Vib_Grind_Motor",
    "Vib_WH_Motor",
    "Temp_WH_Tooling",
)

PROCESS_CONTROL_SENSORS = (
    "AE_Dittel_lp30",
    "AE_WH",
    "Force_WH",
)

CONDITION_MONITORING_SENSORS = (
    "Vib_WH_Motor",
    "Vib_WH_X",
    "Vib_WH_Y",
)

EXTRACTED_SENSORS = tuple(
    dict.fromkeys(
        SELECTED_SENSORS + PROCESS_CONTROL_SENSORS + CONDITION_MONITORING_SENSORS
    )
)

STATISTIC_NAMES = (
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "rms",
    "peak_to_peak",
    "crest_factor",
    "band_power",
    "energy",
    "percentile_90",
)
