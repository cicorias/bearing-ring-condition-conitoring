from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from grinder_diagnostics_model.torch_forest import TorchRandomForest


@dataclass(frozen=True)
class Prediction:
    model_version: str
    has_fault: bool
    binary_probability: float
    binary_probabilities: dict[str, float]
    fault_type: str | None
    fault_probabilities: dict[str, float]
    downstream_analysis_required: bool
    warnings: list[str]
    provenance: dict[str, str]


class InferenceEngine:
    def __init__(
        self,
        *,
        metadata: dict[str, Any],
        binary: TorchRandomForest,
        fault: TorchRandomForest,
        artifact_path: Path,
    ) -> None:
        self.metadata = metadata
        self.binary = binary.eval()
        self.fault = fault.eval()
        self.artifact_path = artifact_path
        self.feature_names = [str(value) for value in metadata["feature_names"]]

    @classmethod
    def load(cls, artifact_path: Path) -> InferenceEngine:
        artifact_path = artifact_path.resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported model artifact format")
        return cls(
            metadata=payload["metadata"],
            binary=TorchRandomForest.from_payload(payload["binary"]),
            fault=TorchRandomForest.from_payload(payload["fault"]),
            artifact_path=artifact_path,
        )

    def _vector(self, features: dict[str, float]) -> torch.Tensor:
        expected = set(self.feature_names)
        supplied = set(features)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            raise ValueError(f"Feature schema mismatch: missing={missing}, extra={extra}")
        values = [float(features[name]) for name in self.feature_names]
        non_finite = [
            name for name, value in zip(self.feature_names, values, strict=True)
            if not math.isfinite(value)
        ]
        if non_finite:
            raise ValueError(f"Features must be finite: {non_finite}")
        return torch.tensor([values], dtype=torch.float64)

    def predict(self, features: dict[str, float]) -> Prediction:
        values = self._vector(features)
        with torch.no_grad():
            binary_values = self.binary(values)[0].tolist()
            fault_values = self.fault(values)[0].tolist()
        binary_classes = [int(value) for value in self.metadata["binary_classes"]]
        binary_by_class = dict(zip(binary_classes, binary_values, strict=True))
        fault_probability = float(binary_by_class[1])
        threshold = float(self.metadata["binary_threshold"])
        has_fault = fault_probability >= threshold

        fault_classes = [int(value) for value in self.metadata["fault_classes"]]
        fault_labels = {
            int(key): str(value) for key, value in self.metadata["fault_labels"].items()
        }
        fault_by_label = {
            fault_labels[class_id]: float(probability)
            for class_id, probability in zip(fault_classes, fault_values, strict=True)
        }
        fault_type = max(fault_by_label, key=fault_by_label.get) if has_fault else None
        return Prediction(
            model_version=str(self.metadata["model_version"]),
            has_fault=has_fault,
            binary_probability=fault_probability,
            binary_probabilities={
                "normal": float(binary_by_class[0]),
                "fault": fault_probability,
            },
            fault_type=fault_type,
            fault_probabilities=fault_by_label if has_fault else {},
            downstream_analysis_required=has_fault,
            warnings=[],
            provenance={
                "model_artifact": str(self.artifact_path),
                "feature_table_sha256": str(self.metadata["feature_table_sha256"]),
            },
        )
