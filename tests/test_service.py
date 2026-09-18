"""Service-layer and model-quality tests.

The API tests prove the wiring works. These prove the *model* is good enough
to ship, and that the artefact on disk matches the metrics we advertise.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from tests.conftest import requires_model

from riskradar import config as C
from riskradar import data as D
from riskradar.models import SoftVoteEnsemble, ensemble_weights
from riskradar.service import RiskService

pytestmark = requires_model


@pytest.fixture(scope="module")
def svc() -> RiskService:
    return RiskService.instance()


# ── artefact integrity ────────────────────────────────────────────────
def test_metadata_matches_artefact(svc):
    meta = json.loads(C.METADATA_PATH.read_text(encoding="utf-8"))
    assert meta["classes"] == C.CLASS_ORDER
    assert len(meta["feature_names"]) == meta["n_engineered_features"]
    assert meta["metrics"]["accuracy"] > 0.9


def test_beats_the_previous_project_baseline(svc):
    """The whole point of the rebuild: the reported number must be real."""
    meta = svc.metadata
    previous = meta["baseline_reference"]["previous_project_accuracy"]
    assert meta["metrics"]["accuracy"] > previous


def test_reported_accuracy_is_reproducible(svc):
    """Re-score the hold-out split from scratch and confirm the metadata."""
    df = D.load_dataset()
    X, y = D.split_xy(df)
    _, X_test, _, y_test = D.stratified_split(X, y)

    sample = X_test.head(4000)
    truth = y_test[:4000]
    predicted = np.asarray(svc.model.predict(sample))
    measured = float((predicted == truth).mean())

    claimed = svc.metadata["metrics"]["accuracy"]
    assert abs(measured - claimed) < 0.02, f"claimed {claimed}, measured {measured}"


def test_pipeline_accepts_raw_records(svc):
    """The artefact must consume raw columns — no preprocessing in the caller."""
    df = D.load_dataset().head(50)
    out = svc.model.predict(df[C.RAW_FEATURE_COLUMNS])
    assert len(out) == 50
    assert set(np.unique(out)) <= {0, 1, 2}


# ── ensemble mechanics ────────────────────────────────────────────────
def test_ensemble_weights_exclude_weak_models():
    scores = {"lgbm": 0.983, "lgbm_wide": 0.980, "histgb": 0.979,
              "logreg": 0.974, "rf": 0.945, "extratrees": 0.939}
    w = ensemble_weights(scores)
    assert "rf" not in w and "extratrees" not in w
    assert w["lgbm"] == max(w.values())
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)


def test_ensemble_weighting_is_sharp():
    """Near-uniform weights would let the weakest member drag the vote down."""
    w = ensemble_weights({"lgbm": 0.983, "lgbm_wide": 0.980,
                          "histgb": 0.979, "logreg": 0.974})
    assert w["lgbm"] > 4 * w["logreg"]


def test_shared_feature_transform_matches_naive_path(svc):
    """The single-transform optimisation must be numerically identical to
    letting every member run its own pipeline."""
    if not isinstance(svc.model, SoftVoteEnsemble):
        pytest.skip("deployed model is not an ensemble")
    frame = D.load_dataset().head(20)[C.RAW_FEATURE_COLUMNS]
    fast = svc.model.predict_proba(frame)
    naive = np.average(
        np.stack([m.predict_proba(frame) for m in svc.model.estimators]),
        axis=0, weights=svc.model._w)
    np.testing.assert_allclose(fast, naive, rtol=1e-6, atol=1e-8)


# ── service behaviour ─────────────────────────────────────────────────
def test_safety_score_bounds():
    assert RiskService.safety_score(np.array([1.0, 0.0, 0.0])) == 100.0
    assert RiskService.safety_score(np.array([0.0, 0.0, 1.0])) == 0.0
    assert RiskService.safety_score(np.array([0.0, 1.0, 0.0])) == 50.0


def test_to_frame_fills_missing_columns():
    frame = RiskService.to_frame({"Hour": 3})
    assert list(frame.columns) == C.RAW_FEATURE_COLUMNS
    assert len(frame) == 1


def test_explanation_contributions_are_signed(svc, sample_payload):
    res = svc.predict(sample_payload)
    impacts = [d["impact"] for d in res["explanation"]["drivers"]]
    assert any(i > 0 for i in impacts), "no risk-increasing factor found"
    assert impacts == sorted(impacts, key=abs, reverse=True)


def test_batch_matches_single_scoring(svc, sample_payload):
    """Vectorised and single-record paths must never disagree."""
    single = svc.predict(sample_payload, explain=False, recommend=False)
    batch = svc.predict_batch(pd.DataFrame([sample_payload] * 3))
    assert batch["labels"] == [single["risk_level"]] * 3
    assert batch["confidence"][0] == pytest.approx(single["confidence"], abs=1e-4)


def test_predictions_counter_increments(svc, sample_payload):
    before = svc.predictions_served
    svc.predict(sample_payload, explain=False, recommend=False)
    assert svc.predictions_served == before + 1


def test_night_increases_risk_all_else_equal(svc):
    """A domain sanity check — the same street at 3am should not look safer
    than at 3pm once lighting is degraded."""
    base = {
        "Crime_Count": 45, "Violent_Crime": 9, "Theft_Count": 22,
        "Assault_Count": 7, "Harassment_Count": 7, "Emergency_Calls": 22,
        "Day": "Sat", "Month": 8, "Weekend": 1,
        "Working_Streetlights": 30, "Broken_Streetlights": 25,
        "Streetlight_Count": 55, "CCTV_Count": 12,
        "Population_Density": 9000, "Footfall": 400,
        "Bus_Stop_Count": 4, "Metro_Distance_km": 6.0,
        "Police_Distance_km": 5.5, "Hospital_Distance_km": 5.0,
        "School_Count": 1, "Commercial_Area": 0, "Residential_Area": 1,
        "Weather": "Fog", "Visibility": "Poor",
        "Previous_Risk": "High", "Crime_Trend": "Increasing",
        "Latitude": 13.05, "Longitude": 80.2,
    }
    day = svc.predict(base | {"Hour": 14}, explain=False, recommend=False)
    night = svc.predict(base | {"Hour": 2}, explain=False, recommend=False)
    assert night["safety_score"] <= day["safety_score"]


def test_what_if_returns_aligned_series(svc, sample_payload):
    values = [0, 30, 60, 120]
    out = svc.what_if(sample_payload, "CCTV_Count", values)
    assert out["values"] == values
    assert len(out["risk_levels"]) == len(out["safety_scores"]) == len(values)


# ── data contract ─────────────────────────────────────────────────────
def test_dataset_quality_is_clean():
    q = D.data_quality_report(D.load_dataset())
    assert q["missing_values"] == 0
    assert q["duplicate_rows"] == 0
    assert q["constant_columns"] == []
    assert set(q["class_distribution"]) == set(C.CLASS_ORDER)


def test_stratified_split_preserves_class_balance():
    df = D.load_dataset()
    X, y = D.split_xy(df)
    _, _, y_tr, y_te = D.stratified_split(X, y)
    for cls in range(3):
        assert abs((y_tr == cls).mean() - (y_te == cls).mean()) < 0.01
