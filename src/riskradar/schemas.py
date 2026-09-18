"""Pydantic contracts for the RiskRadar API.

Validation lives at the edge: a request that reaches the model has already been
proven type-correct and range-correct, so the inference path contains no
defensive branching.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from . import config as C


class AreaFeatures(BaseModel):
    """One area-hour observation to be scored."""

    # -- location -------------------------------------------------------
    Latitude: float = Field(13.0827, ge=-90, le=90, description="Latitude in decimal degrees")
    Longitude: float = Field(80.2707, ge=-180, le=180, description="Longitude in decimal degrees")

    # -- crime ----------------------------------------------------------
    Crime_Count: int = Field(40, ge=0, le=10_000, description="Total recorded incidents")
    Violent_Crime: int = Field(8, ge=0, le=10_000)
    Theft_Count: int = Field(18, ge=0, le=10_000)
    Assault_Count: int = Field(6, ge=0, le=10_000)
    Harassment_Count: int = Field(5, ge=0, le=10_000)
    Emergency_Calls: int = Field(20, ge=0, le=10_000, description="Distress calls logged")

    # -- time -----------------------------------------------------------
    Hour: int = Field(21, ge=0, le=23)
    Day: Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] = "Fri"
    Month: int = Field(6, ge=1, le=12)
    Weekend: int = Field(0, ge=0, le=1)

    # -- infrastructure -------------------------------------------------
    Streetlight_Count: int = Field(70, ge=0, le=100_000)
    Working_Streetlights: int = Field(55, ge=0, le=100_000)
    Broken_Streetlights: int = Field(15, ge=0, le=100_000)
    CCTV_Count: int = Field(45, ge=0, le=100_000)

    # -- population -----------------------------------------------------
    Population_Density: int = Field(5000, ge=0, le=1_000_000, description="People per km²")
    Footfall: int = Field(1200, ge=0, le=1_000_000, description="Pedestrians per hour")

    # -- access ---------------------------------------------------------
    Bus_Stop_Count: int = Field(10, ge=0, le=1000)
    Metro_Distance_km: float = Field(3.0, ge=0, le=500)
    Police_Distance_km: float = Field(2.5, ge=0, le=500)
    Hospital_Distance_km: float = Field(2.0, ge=0, le=500)
    School_Count: int = Field(5, ge=0, le=1000)

    # -- land use -------------------------------------------------------
    Commercial_Area: int = Field(1, ge=0, le=1)
    Residential_Area: int = Field(1, ge=0, le=1)

    # -- environment ----------------------------------------------------
    Weather: Literal["Sunny", "Cloudy", "Rain", "Fog"] = "Sunny"
    Visibility: Literal["Good", "Medium", "Poor"] = "Good"
    Previous_Risk: Literal["Low", "Medium", "High"] = "Medium"
    Crime_Trend: Literal["Decreasing", "Stable", "Increasing"] = "Stable"

    @field_validator("Working_Streetlights", "Broken_Streetlights")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        return max(0, v)

    @model_validator(mode="after")
    def _coherent(self):
        """Reconcile inputs that must agree with each other.

        Users edit these fields independently in the UI, so we repair rather
        than reject: the lamp total is defined as working + broken, and the
        itemised crime counts can never exceed the headline total.
        """
        self.Streetlight_Count = self.Working_Streetlights + self.Broken_Streetlights
        itemised = (self.Violent_Crime + self.Theft_Count
                    + self.Assault_Count + self.Harassment_Count)
        if itemised > self.Crime_Count:
            self.Crime_Count = itemised
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "Latitude": 13.0827, "Longitude": 80.2707,
                "Crime_Count": 68, "Violent_Crime": 14, "Theft_Count": 30,
                "Assault_Count": 12, "Harassment_Count": 9, "Emergency_Calls": 34,
                "Hour": 23, "Day": "Sat", "Month": 7, "Weekend": 1,
                "Streetlight_Count": 60, "Working_Streetlights": 32,
                "Broken_Streetlights": 28, "CCTV_Count": 8,
                "Population_Density": 9500, "Footfall": 300,
                "Bus_Stop_Count": 3, "Metro_Distance_km": 7.4,
                "Police_Distance_km": 8.1, "Hospital_Distance_km": 6.6,
                "School_Count": 1, "Commercial_Area": 0, "Residential_Area": 1,
                "Weather": "Fog", "Visibility": "Poor",
                "Previous_Risk": "High", "Crime_Trend": "Increasing",
            }
        }
    }


class ClassProbability(BaseModel):
    label: str
    probability: float


class Driver(BaseModel):
    feature: str
    label: str
    value: float
    impact: float
    direction: str


class Explanation(BaseModel):
    method: str
    predicted_class: str
    baseline: float
    total_contribution: float
    drivers: list[Driver]
    risk_factors: list[Driver]
    protective_factors: list[Driver]
    narrative: str


class Recommendation(BaseModel):
    action: str
    detail: str
    new_risk: str
    bands_improved: int


class PredictionResponse(BaseModel):
    risk_level: str = Field(description="Low | Medium | High")
    risk_index: int
    confidence: float = Field(description="Probability of the predicted class")
    probabilities: list[ClassProbability]
    safety_score: float = Field(description="0–100, higher is safer")
    color: str
    advice: str
    latency_ms: float
    model_name: str
    model_version: str
    explanation: Explanation | None = None
    recommendations: list[Recommendation] = []


class BatchRow(BaseModel):
    row: int
    risk_level: str
    confidence: float
    safety_score: float


class BatchResponse(BaseModel):
    rows_processed: int
    latency_ms: float
    distribution: dict[str, int]
    average_safety_score: float
    results: list[BatchRow]
    csv_base64: str | None = None


class HealthResponse(BaseModel):
    status: str
    detail: str | None = None
    model_loaded: bool
    model_name: str | None = None
    accuracy: float | None = None
    version: str
    uptime_seconds: float
    predictions_served: int


class WhatIfRequest(BaseModel):
    base: AreaFeatures
    field: str
    values: list[float] = Field(min_length=1, max_length=40)

    @field_validator("field")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in C.NUMERIC_COLUMNS:
            raise ValueError(f"'{v}' is not a sweepable numeric field")
        return v
