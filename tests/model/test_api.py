from pathlib import Path

from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from grinder_diagnostics_model.api import create_app
from grinder_diagnostics_model.inference import InferenceEngine
from grinder_diagnostics_model.torch_forest import from_sklearn


def _engine() -> InferenceEngine:
    values, binary_labels = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )
    binary = RandomForestClassifier(n_estimators=3, random_state=7).fit(values, binary_labels)
    fault_labels = (binary_labels % 2) + 2
    fault = RandomForestClassifier(n_estimators=3, random_state=7).fit(values, fault_labels)
    metadata = {
        "model_version": "test-model",
        "feature_names": ["f0", "f1", "f2", "f3"],
        "binary_classes": [0, 1],
        "fault_classes": [2, 3],
        "fault_labels": {2: "fault_a", 3: "fault_b"},
        "binary_threshold": 0.5,
        "feature_table_sha256": "test",
    }
    return InferenceEngine(
        metadata=metadata,
        binary=from_sklearn(binary),
        fault=from_sklearn(fault),
        artifact_path=Path("/test/model.pt"),
    )


def test_api_health_model_and_prediction() -> None:
    with TestClient(create_app(_engine())) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "model_version": "test-model",
        }
        model = client.get("/v1/model")
        assert model.status_code == 200
        assert model.json()["feature_count"] == 4

        response = client.post(
            "/v1/predict",
            json={
                "request_id": "test-request",
                "features": {"f0": 0, "f1": 0, "f2": 0, "f3": 0},
            },
        )
        assert response.status_code == 200
        assert response.json()["request_id"] == "test-request"
        assert "downstream_analysis_required" in response.json()


def test_api_rejects_feature_schema_mismatch() -> None:
    with TestClient(create_app(_engine())) as client:
        response = client.post(
            "/v1/predict",
            json={"request_id": "bad-request", "features": {"f0": 0}},
        )
        assert response.status_code == 422
        assert "Feature schema mismatch" in response.json()["detail"]
