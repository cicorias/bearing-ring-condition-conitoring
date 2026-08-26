import numpy as np
import pytest

from grinder_diagnostics_model.constants import EXTRACTED_SENSORS
from grinder_diagnostics_model.features import _contact_index, _statistics, feature_names


def test_feature_schema_has_one_hundred_candidates() -> None:
    names = feature_names()
    assert len(names) == 100
    assert len(set(names)) == 100
    assert len(feature_names(EXTRACTED_SENSORS)) == 160


def test_contact_index_finds_first_marker_change_after_minimum() -> None:
    marker = np.ones(20, dtype=np.uint8)
    marker[8:] = 0
    assert _contact_index(marker, minimum_idle_samples=5) == 8


def test_statistics_are_finite_for_constant_signal() -> None:
    result = _statistics(np.ones(10))
    assert result["std"] == 0
    assert result["skewness"] == 0
    assert result["kurtosis"] == 0
    assert result["crest_factor"] == 1


def test_contact_index_rejects_missing_transition() -> None:
    with pytest.raises(ValueError, match="no contact transition"):
        _contact_index(np.ones(20), minimum_idle_samples=5)
