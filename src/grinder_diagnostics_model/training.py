from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler

from grinder_diagnostics_model.constants import (
    CONDITION_MONITORING_SENSORS,
    FAULT_LABELS,
    PROCESS_CONTROL_SENSORS,
)
from grinder_diagnostics_model.features import feature_names
from grinder_diagnostics_model.torch_forest import from_sklearn


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int = 20220809
    test_size: float = 0.30
    selected_feature_count: int = 58
    trees: int = 30
    binary_threshold: float = 0.5
    nca_max_iterations: int = 100


def _feature_table_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _paper_split(frame: pd.DataFrame, config: TrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    train, test = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=frame["test"].to_numpy(),
    )
    return np.sort(train), np.sort(test)


def _grouped_split(frame: pd.DataFrame, config: TrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.random_seed)
    test_groups: set[tuple[int, int]] = set()
    for test in range(1, 8):
        cycles = np.arange(1, 8)
        selected = rng.choice(cycles, size=2, replace=False)
        test_groups.update((test, int(cycle)) for cycle in selected)
    is_test = np.array(
        [
            (int(row.test), int(row.dressing_cycle)) in test_groups
            for row in frame.itertuples(index=False)
        ]
    )
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def _select_features(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    config: TrainingConfig,
) -> tuple[list[str], list[dict[str, float | str]]]:
    candidates = feature_names()
    values = frame.loc[train_indices, candidates].to_numpy(dtype=np.float64)
    labels = frame.loc[train_indices, "has_fault"].to_numpy(dtype=np.int64)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    selector = NeighborhoodComponentsAnalysis(
        init="identity",
        max_iter=config.nca_max_iterations,
        random_state=config.random_seed,
        tol=1e-5,
    )
    selector.fit(scaled, labels)
    weights = np.linalg.norm(selector.components_, axis=0)
    ranking = sorted(
        (
            {"feature": name, "weight": float(weight)}
            for name, weight in zip(candidates, weights, strict=True)
        ),
        key=lambda item: (-float(item["weight"]), str(item["feature"])),
    )
    selected = [str(item["feature"]) for item in ranking[: config.selected_feature_count]]
    return selected, ranking


def _new_forest(config: TrainingConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.trees,
        criterion="gini",
        bootstrap=True,
        max_features="sqrt",
        random_state=config.random_seed,
        n_jobs=-1,
    )


def _metrics(
    model: RandomForestClassifier,
    values: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, Any]:
    predictions = model.predict(values)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def _evaluate_protocol(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    selected_features: list[str],
    config: TrainingConfig,
) -> dict[str, Any]:
    train = frame.iloc[train_indices]
    test = frame.iloc[test_indices]
    binary = _new_forest(config)
    binary.fit(train[selected_features], train["has_fault"])
    binary_metrics = _metrics(
        binary,
        test[selected_features],
        test["has_fault"].to_numpy(),
    )

    fault_train = train[train["has_fault"] == 1]
    fault_test = test[test["has_fault"] == 1]
    fault = _new_forest(config)
    fault.fit(fault_train[selected_features], fault_train["test"])
    fault_metrics = _metrics(
        fault,
        fault_test[selected_features],
        fault_test["test"].to_numpy(),
    )

    reduced: dict[str, Any] = {}
    for name, sensors in {
        "process_control_only": PROCESS_CONTROL_SENSORS,
        "condition_monitoring_only": CONDITION_MONITORING_SENSORS,
    }.items():
        sensor_features = feature_names(sensors)
        reduced_binary = _new_forest(config)
        reduced_binary.fit(train[sensor_features], train["has_fault"])
        reduced_fault = _new_forest(config)
        reduced_fault.fit(fault_train[sensor_features], fault_train["test"])
        reduced[name] = {
            "feature_count": len(sensor_features),
            "binary": _metrics(
                reduced_binary,
                test[sensor_features],
                test["has_fault"].to_numpy(),
            ),
            "fault": _metrics(
                reduced_fault,
                fault_test[sensor_features],
                fault_test["test"].to_numpy(),
            ),
        }
    return {"binary": binary_metrics, "fault": fault_metrics, "reduced_sensors": reduced}


def train_and_export(
    feature_path: Path,
    artifact_dir: Path,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or TrainingConfig()
    frame = pd.read_parquet(feature_path)
    expected = set(feature_names())
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise ValueError(f"Feature table is missing {len(missing)} expected columns")
    if len(frame) != 735:
        raise ValueError(f"Expected 735 feature rows, found {len(frame)}")

    paper_train, paper_test = _paper_split(frame, config)
    selected, ranking = _select_features(frame, paper_train, config)
    grouped_train, grouped_test = _grouped_split(frame, config)
    metrics = {
        "paper_like": _evaluate_protocol(
            frame, paper_train, paper_test, selected, config
        ),
        "dressing_cycle_grouped": _evaluate_protocol(
            frame, grouped_train, grouped_test, selected, config
        ),
    }

    final_selected, final_ranking = _select_features(frame, np.arange(len(frame)), config)
    binary = _new_forest(config)
    binary.fit(frame[final_selected], frame["has_fault"])
    fault_frame = frame[frame["has_fault"] == 1]
    fault = _new_forest(config)
    fault.fit(fault_frame[final_selected], fault_frame["test"])

    torch_binary = from_sklearn(binary)
    torch_fault = from_sklearn(fault)
    all_values = torch.tensor(frame[final_selected].to_numpy(), dtype=torch.float64)
    with torch.no_grad():
        binary_delta = np.max(
            np.abs(torch_binary(all_values).numpy() - binary.predict_proba(frame[final_selected]))
        )
        fault_values = torch.tensor(
            fault_frame[final_selected].to_numpy(), dtype=torch.float64
        )
        fault_delta = np.max(
            np.abs(
                torch_fault(fault_values).numpy()
                - fault.predict_proba(fault_frame[final_selected])
            )
        )
    tolerance = 1e-12
    if binary_delta > tolerance or fault_delta > tolerance:
        raise ValueError(
            f"PyTorch export parity failed: binary={binary_delta}, fault={fault_delta}"
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_table_sha256 = _feature_table_hash(feature_path)
    implementation_sha256 = _implementation_hash()
    version_payload = json.dumps(
        {
            "config": asdict(config),
            "feature_names": final_selected,
            "feature_table_sha256": feature_table_sha256,
            "implementation_sha256": implementation_sha256,
        },
        sort_keys=True,
    ).encode()
    model_version = f"rf-{hashlib.sha256(version_payload).hexdigest()[:16]}"
    metadata = {
        "format_version": 1,
        "model_version": model_version,
        "feature_table_sha256": feature_table_sha256,
        "implementation_sha256": implementation_sha256,
        "feature_names": final_selected,
        "binary_classes": [int(value) for value in binary.classes_],
        "fault_classes": [int(value) for value in fault.classes_],
        "fault_labels": FAULT_LABELS,
        "binary_threshold": config.binary_threshold,
        "training": asdict(config),
        "pytorch_max_probability_delta": {
            "binary": float(binary_delta),
            "fault": float(fault_delta),
        },
    }
    payload = {
        "format_version": 1,
        "metadata": metadata,
        "binary": torch_binary.to_payload(),
        "fault": torch_fault.to_payload(),
    }
    torch.save(payload, artifact_dir / "model.pt")
    joblib.dump(
        {"binary": binary, "fault": fault, "feature_names": final_selected},
        artifact_dir / "reference-models.joblib",
    )
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "feature-ranking.json").write_text(
        json.dumps(
            {"paper_split": ranking, "production": final_ranking},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    split_manifest = {
        "paper_like": {
            "train": frame.iloc[paper_train]["ring_id"].tolist(),
            "test": frame.iloc[paper_test]["ring_id"].tolist(),
        },
        "dressing_cycle_grouped": {
            "train": frame.iloc[grouped_train]["ring_id"].tolist(),
            "test": frame.iloc[grouped_test]["ring_id"].tolist(),
        },
    }
    (artifact_dir / "split-manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    sample_dir = artifact_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    binary_class_indices = {
        int(class_id): index for index, class_id in enumerate(binary.classes_)
    }
    fault_class_indices = {
        int(class_id): index for index, class_id in enumerate(fault.classes_)
    }

    def write_sample(row_index: int, request_id: str, file_name: str) -> dict[str, Any]:
        sample = {
            "request_id": request_id,
            "features": {
                name: float(frame.loc[row_index, name]) for name in final_selected
            },
        }
        (sample_dir / file_name).write_text(
            json.dumps(sample, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return sample

    baseline = frame[frame["has_fault"] == 0]
    baseline_probabilities = binary.predict_proba(baseline[final_selected])
    baseline_row = int(
        baseline.index[
            np.argmax(baseline_probabilities[:, binary_class_indices[0]])
        ]
    )
    default_sample = write_sample(baseline_row, "sample-baseline", "baseline.json")
    (artifact_dir / "sample-request.json").write_text(
        json.dumps(default_sample, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for class_id, label in FAULT_LABELS.items():
        candidates = frame[frame["test"] == class_id]
        binary_probabilities = binary.predict_proba(candidates[final_selected])
        fault_probabilities = fault.predict_proba(candidates[final_selected])
        joint_probability = (
            binary_probabilities[:, binary_class_indices[1]]
            * fault_probabilities[:, fault_class_indices[class_id]]
        )
        selected_row = int(candidates.index[np.argmax(joint_probability)])
        write_sample(
            selected_row,
            f"sample-{label.replace('_', '-')}",
            f"{label}.json",
        )
    return {"metadata": metadata, "metrics": metrics}
