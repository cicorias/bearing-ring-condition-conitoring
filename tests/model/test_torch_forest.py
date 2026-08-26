import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from grinder_diagnostics_model.torch_forest import from_sklearn


def test_torch_forest_matches_sklearn_probabilities() -> None:
    values, labels = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        n_classes=3,
        random_state=42,
    )
    reference = RandomForestClassifier(
        n_estimators=7,
        bootstrap=True,
        random_state=42,
    ).fit(values, labels)
    exported = from_sklearn(reference)

    with torch.no_grad():
        actual = exported(torch.tensor(values, dtype=torch.float64)).numpy()

    np.testing.assert_allclose(actual, reference.predict_proba(values), atol=1e-12, rtol=0)
