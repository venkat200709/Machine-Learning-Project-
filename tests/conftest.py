"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from riskradar import config as C  # noqa: E402


@pytest.fixture(scope="session")
def sample_payload() -> dict:
    """A realistic mid-risk record used across the suite."""
    return {
        "Latitude": 13.0827, "Longitude": 80.2707,
        "Crime_Count": 38, "Violent_Crime": 7, "Theft_Count": 21,
        "Assault_Count": 5, "Harassment_Count": 5, "Emergency_Calls": 17,
        "Hour": 19, "Day": "Sat", "Month": 8, "Weekend": 1,
        "Streetlight_Count": 63, "Working_Streetlights": 52,
        "Broken_Streetlights": 11, "CCTV_Count": 34,
        "Population_Density": 14500, "Footfall": 6800,
        "Bus_Stop_Count": 19, "Metro_Distance_km": 1.8,
        "Police_Distance_km": 2.4, "Hospital_Distance_km": 2.9,
        "School_Count": 4, "Commercial_Area": 1, "Residential_Area": 1,
        "Weather": "Cloudy", "Visibility": "Medium",
        "Previous_Risk": "Medium", "Crime_Trend": "Stable",
    }


@pytest.fixture(scope="session")
def has_model() -> bool:
    return C.MODEL_PATH.exists()


requires_model = pytest.mark.skipif(
    not C.MODEL_PATH.exists(),
    reason="No trained model artefact — run `python run.py --train` first.",
)
