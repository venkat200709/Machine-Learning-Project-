"""Feature engineering contract tests.

These matter more than they look: the transform runs inside the fitted
pipeline, so a silent change here corrupts a model that was trained under the
old maths. Each test pins one property the rest of the system depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskradar import config as C
from riskradar.features import EPS, engineer, feature_names, pretty


@pytest.fixture
def frame(sample_payload) -> pd.DataFrame:
    return pd.DataFrame([sample_payload])


def test_output_shape_and_order(frame):
    out = engineer(frame)
    assert list(out.columns) == feature_names()
    assert len(out) == 1
    assert out.dtypes.unique().tolist() == [np.dtype("float32")]


def test_no_nan_or_inf_even_with_zeros():
    """Every denominator is smoothed, so an all-zero record must still work."""
    zeros = dict.fromkeys(C.NUMERIC_COLUMNS, 0)
    zeros |= {"Day": "Mon", "Weather": "Sunny", "Visibility": "Good",
              "Previous_Risk": "Low", "Crime_Trend": "Stable"}
    out = engineer(pd.DataFrame([zeros]))
    assert np.isfinite(out.to_numpy()).all()


def test_row_independence(frame):
    """No cross-row statistics — scoring one record alone must equal scoring
    it inside a batch. If this ever fails, the API and training would diverge."""
    batch = pd.concat([frame] * 5, ignore_index=True)
    batch.loc[1:, "Crime_Count"] = [1, 500, 12, 99]
    single = engineer(frame).to_numpy()
    inside = engineer(batch).iloc[[0]].to_numpy()
    np.testing.assert_allclose(single, inside, rtol=1e-6)


def test_target_never_leaks(frame):
    """The transform must ignore Risk_Level even when handed it."""
    with_target = frame.assign(Risk_Level="High")
    np.testing.assert_allclose(
        engineer(frame).to_numpy(), engineer(with_target).to_numpy()
    )


def test_area_id_is_excluded():
    """Area_ID is unique per row; leaking it would let a tree memorise rows."""
    assert "Area_ID" not in feature_names()
    assert "Area_ID" not in C.RAW_FEATURE_COLUMNS


def test_ordinal_maps_are_ordered(frame):
    """Poor < Medium < Good visibility, Low < Medium < High previous risk."""
    vals = [
        float(engineer(frame.assign(Visibility=v))["Visibility_Score"].iloc[0])
        for v in ["Poor", "Medium", "Good"]
    ]
    assert vals == sorted(vals) and len(set(vals)) == 3

    prev = [float(engineer(frame.assign(Previous_Risk=r))["Previous_Risk_Ord"].iloc[0])
            for r in ["Low", "Medium", "High"]]
    assert prev == [0.0, 1.0, 2.0]


def test_unknown_category_does_not_crash(frame):
    """Serving must survive a category the training data never contained."""
    out = engineer(frame.assign(Weather="Hailstorm", Day="Funday"))
    assert np.isfinite(out.to_numpy()).all()


def test_lighting_health_is_a_ratio(frame):
    out = engineer(frame.assign(Streetlight_Count=100, Working_Streetlights=75,
                                Broken_Streetlights=25))
    assert out["Lighting_Health"].iloc[0] == pytest.approx(75 / (100 + EPS), rel=1e-4)
    assert 0.0 <= out["Broken_Light_Ratio"].iloc[0] <= 1.0


def test_darkness_exposure_only_fires_at_night(frame):
    """Broken lights are irrelevant at noon and dangerous at 2am — the feature
    must encode that interaction, not just the raw lamp count."""
    dark = frame.assign(Hour=2, Visibility="Poor", Broken_Streetlights=30,
                        Working_Streetlights=10, Streetlight_Count=40)
    day = dark.assign(Hour=13)
    assert engineer(dark)["Darkness_Exposure"].iloc[0] > 0
    assert engineer(day)["Darkness_Exposure"].iloc[0] == 0


def test_night_flag_wraps_midnight(frame):
    night_hours = [22, 23, 0, 1, 3, 5]
    day_hours = [8, 12, 15, 19]
    for h in night_hours:
        assert engineer(frame.assign(Hour=h))["Is_Night"].iloc[0] == 1.0, h
    for h in day_hours:
        assert engineer(frame.assign(Hour=h))["Is_Night"].iloc[0] == 0.0, h


def test_cyclical_hour_encoding_is_continuous(frame):
    """23:00 and 00:00 must be neighbours, not maximally distant."""
    a = engineer(frame.assign(Hour=23))[["Hour_Sin", "Hour_Cos"]].to_numpy()
    b = engineer(frame.assign(Hour=0))[["Hour_Sin", "Hour_Cos"]].to_numpy()
    far = engineer(frame.assign(Hour=11))[["Hour_Sin", "Hour_Cos"]].to_numpy()
    assert np.linalg.norm(a - b) < np.linalg.norm(a - far)


def test_crime_normalisation_reflects_exposure(frame):
    """Identical counts in a dense vs. sparse area must not look identical."""
    dense = engineer(frame.assign(Population_Density=50_000))["Crime_Per_1k_Population"].iloc[0]
    sparse = engineer(frame.assign(Population_Density=500))["Crime_Per_1k_Population"].iloc[0]
    assert sparse > dense


def test_missing_columns_are_tolerated(frame):
    """A partial payload must not raise — the API repairs, it does not reject."""
    out = engineer(frame.drop(columns=["CCTV_Count", "Footfall"]))
    assert np.isfinite(out.to_numpy()).all()


def test_pretty_labels_exist_for_top_features():
    for f in ["Threat_Score", "Darkness_Exposure", "Surveillance_Deficit"]:
        assert pretty(f) != f and "_" not in pretty(f)
