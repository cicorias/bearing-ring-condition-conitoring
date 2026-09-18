import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from grinder_diagnostics_model.inference import InferenceEngine


def test_inference_loads_sklearn_forest_artifact(tmp_path) -> None:
    values, labels = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        random_state=42,
    )
    feature_names = [f"f{index}" for index in range(values.shape[1])]
    frame = pd.DataFrame(values, columns=feature_names)
    binary = RandomForestClassifier(
        n_estimators=7,
        bootstrap=True,
        random_state=42,
    ).fit(frame, labels)
    fault = RandomForestClassifier(n_estimators=7, random_state=42).fit(frame, labels + 2)
    artifact_path = tmp_path / "model.joblib"
    metadata = {
        "model_version": "test-model",
        "feature_names": feature_names,
        "binary_classes": [0, 1],
        "fault_classes": [2, 3],
        "fault_labels": {2: "fault_a", 3: "fault_b"},
        "binary_threshold": 0.5,
        "feature_table_sha256": "test",
    }
    joblib.dump(
        {"format_version": 1, "metadata": metadata, "binary": binary, "fault": fault},
        artifact_path,
    )

    engine = InferenceEngine.load(artifact_path)
    row_index = int(np.argmax(binary.predict_proba(frame)[:, 1]))
    features = dict(zip(metadata["feature_names"], values[row_index], strict=True))
    prediction = engine.predict(features)

    np.testing.assert_allclose(
        list(prediction.binary_probabilities.values()),
        binary.predict_proba(frame.iloc[[row_index]])[0],
    )
    np.testing.assert_allclose(
        list(prediction.fault_probabilities.values()),
        fault.predict_proba(frame.iloc[[row_index]])[0],
    )
