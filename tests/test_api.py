"""End-to-end API tests against the real trained artefact."""

from __future__ import annotations

import base64
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from tests.conftest import requires_model

from riskradar import config as C
from riskradar.api import app

pytestmark = requires_model


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # `with` triggers the lifespan warm-up
        yield c


# ── system ────────────────────────────────────────────────────────────
def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "operational"
    assert body["model_loaded"] is True
    assert body["accuracy"] > 0.9


def test_model_card_exposes_leaderboard(client):
    body = client.get("/api/model").json()
    assert body["metadata"]["classes"] == C.CLASS_ORDER
    assert len(body["leaderboard"]) >= 5
    assert body["metadata"]["n_engineered_features"] > body["metadata"]["n_raw_features"]


def test_openapi_docs_served(client):
    assert client.get("/docs").status_code == 200
    assert "RiskRadar" in client.get("/openapi.json").json()["info"]["title"]


def test_frontend_is_mounted(client):
    assert "<title>RiskRadar" in client.get("/").text
    for asset in ("/styles.css", "/app.js"):
        assert client.get(asset).status_code == 200


def test_timing_header_present(client):
    assert "X-Process-Time-ms" in client.get("/api/health").headers


# ── prediction ────────────────────────────────────────────────────────
def test_predict_contract(client, sample_payload):
    r = client.post("/api/predict", json=sample_payload)
    assert r.status_code == 200
    body = r.json()

    assert body["risk_level"] in C.CLASS_ORDER
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["safety_score"] <= 100.0
    assert len(body["probabilities"]) == 3
    assert sum(p["probability"] for p in body["probabilities"]) == pytest.approx(1.0, abs=1e-3)
    assert body["explanation"]["drivers"], "explanation must not be empty"
    assert body["latency_ms"] > 0


def test_confidence_matches_argmax(client, sample_payload):
    body = client.post("/api/predict", json=sample_payload).json()
    top = max(body["probabilities"], key=lambda p: p["probability"])
    assert top["label"] == body["risk_level"]
    assert top["probability"] == pytest.approx(body["confidence"], abs=1e-4)


def test_prediction_is_deterministic(client, sample_payload):
    a = client.post("/api/predict", json=sample_payload).json()
    b = client.post("/api/predict", json=sample_payload).json()
    assert a["risk_level"] == b["risk_level"]
    assert a["confidence"] == pytest.approx(b["confidence"], abs=1e-9)


def test_safe_and_dangerous_areas_separate(client, sample_payload):
    """The headline claim of the project: the model must actually distinguish
    a well-lit, patrolled, low-crime street from a dark isolated one."""
    safe = sample_payload | {
        "Crime_Count": 6, "Violent_Crime": 1, "Theft_Count": 3, "Assault_Count": 0,
        "Harassment_Count": 1, "Emergency_Calls": 2, "Hour": 14,
        "Working_Streetlights": 78, "Broken_Streetlights": 2, "Streetlight_Count": 80,
        "CCTV_Count": 95, "Police_Distance_km": 0.5, "Hospital_Distance_km": 0.8,
        "Footfall": 5200, "Weather": "Sunny", "Visibility": "Good",
        "Previous_Risk": "Low", "Crime_Trend": "Decreasing",
    }
    risky = sample_payload | {
        "Crime_Count": 78, "Violent_Crime": 19, "Theft_Count": 30, "Assault_Count": 16,
        "Harassment_Count": 13, "Emergency_Calls": 46, "Hour": 2,
        "Working_Streetlights": 12, "Broken_Streetlights": 34, "Streetlight_Count": 46,
        "CCTV_Count": 2, "Police_Distance_km": 11.0, "Hospital_Distance_km": 9.0,
        "Footfall": 60, "Weather": "Fog", "Visibility": "Poor",
        "Previous_Risk": "High", "Crime_Trend": "Increasing",
    }
    a = client.post("/api/predict", json=safe).json()
    b = client.post("/api/predict", json=risky).json()
    assert a["risk_level"] == "Low"
    assert b["risk_level"] == "High"
    assert a["safety_score"] > b["safety_score"] + 40


def test_explanation_can_be_disabled(client, sample_payload):
    body = client.post("/api/predict?explain=false&recommend=false",
                       json=sample_payload).json()
    assert body["explanation"] is None
    assert body["recommendations"] == []


def test_recommendations_actually_lower_risk(client, sample_payload):
    """Every suggested intervention must be verified by re-scoring, not asserted."""
    risky = sample_payload | {
        "Crime_Count": 55, "Violent_Crime": 11, "Assault_Count": 9,
        "Harassment_Count": 8, "Hour": 1, "Working_Streetlights": 20,
        "Broken_Streetlights": 25, "Streetlight_Count": 45, "CCTV_Count": 6,
        "Police_Distance_km": 6.0, "Footfall": 200, "Visibility": "Poor",
    }
    body = client.post("/api/predict", json=risky).json()
    order = {c: i for i, c in enumerate(C.CLASS_ORDER)}
    for rec in body["recommendations"]:
        assert order[rec["new_risk"]] < order[body["risk_level"]]
        assert rec["bands_improved"] >= 1


# ── validation ────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [
    {"Hour": 24}, {"Hour": -1}, {"Month": 13},
    {"Weather": "Snow"}, {"Visibility": "Excellent"},
    {"Previous_Risk": "Extreme"}, {"Crime_Trend": "Exploding"},
    {"Latitude": 900}, {"Crime_Count": -5}, {"Weekend": 7},
])
def test_invalid_input_is_rejected(client, sample_payload, bad):
    assert client.post("/api/predict", json=sample_payload | bad).status_code == 422


def test_streetlight_totals_are_repaired(client, sample_payload):
    """Users edit working/broken independently; the total is derived, not trusted."""
    payload = sample_payload | {"Working_Streetlights": 30,
                                "Broken_Streetlights": 10,
                                "Streetlight_Count": 3}
    assert client.post("/api/predict", json=payload).status_code == 200


def test_defaults_alone_are_a_valid_request(client):
    assert client.post("/api/predict", json={}).status_code == 200


# ── batch ─────────────────────────────────────────────────────────────
def test_batch_scoring(client):
    df = pd.read_csv(C.RAW_DATASET, nrows=120)
    buf = io.StringIO()
    df.to_csv(buf, index=False)

    r = client.post("/api/predict/batch",
                    files={"file": ("sample.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200
    body = r.json()

    assert body["rows_processed"] == 120
    assert sum(body["distribution"].values()) == 120
    assert 0 <= body["average_safety_score"] <= 100

    out = pd.read_csv(io.StringIO(base64.b64decode(body["csv_base64"]).decode()))
    assert len(out) == 120
    assert {"Predicted_Risk", "Confidence", "Safety_Score"} <= set(out.columns)
    assert out["Predicted_Risk"].isin(C.CLASS_ORDER).all()


def test_batch_rejects_wrong_schema(client):
    r = client.post("/api/predict/batch",
                    files={"file": ("bad.csv", "a,b,c\n1,2,3\n", "text/csv")})
    assert r.status_code == 422
    assert "missing" in r.json()["detail"].lower()


def test_batch_rejects_non_csv(client):
    r = client.post("/api/predict/batch",
                    files={"file": ("notes.txt", "hello", "text/plain")})
    assert r.status_code == 400


def test_batch_rejects_empty_file(client):
    r = client.post("/api/predict/batch",
                    files={"file": ("empty.csv", "Hour\n", "text/csv")})
    assert r.status_code in (400, 422)


# ── what-if ───────────────────────────────────────────────────────────
def test_what_if_sweep_is_monotone_in_cctv(client, sample_payload):
    """More cameras must never make an area look more dangerous."""
    base = sample_payload | {"CCTV_Count": 0, "Crime_Count": 55,
                             "Harassment_Count": 9, "Hour": 23}
    r = client.post("/api/what-if", json={
        "base": base, "field": "CCTV_Count", "values": [0, 25, 50, 100, 150]})
    scores = r.json()["safety_scores"]
    assert scores == sorted(scores), scores


def test_what_if_rejects_unknown_field(client, sample_payload):
    r = client.post("/api/what-if", json={
        "base": sample_payload, "field": "Not_A_Column", "values": [1, 2]})
    assert r.status_code == 422


# ── insight endpoints ─────────────────────────────────────────────────
def test_geo_endpoint_filters(client):
    everything = client.get("/api/geo?limit=500").json()
    assert everything["count"] > 0
    high = client.get("/api/geo?risk=High&limit=500").json()
    assert all(p["risk"] == "High" for p in high["points"])


def test_analytics_shape(client):
    a = client.get("/api/analytics").json()
    assert len(a["high_risk_share_by_hour"]) == 24
    assert len(a["avg_crime_by_hour"]) == 24
    assert set(a["class_distribution"]) == set(C.CLASS_ORDER)
    assert a["dataset"]["missing_values"] == 0


def test_template_is_scoreable(client):
    """The template we hand users must round-trip through batch scoring."""
    t = client.get("/api/template.csv").json()
    r = client.post("/api/predict/batch",
                    files={"file": (t["filename"], t["content"], "text/csv")})
    assert r.status_code == 200
    assert r.json()["rows_processed"] == 3
