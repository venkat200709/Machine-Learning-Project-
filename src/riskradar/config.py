"""Central configuration: paths, schema, constants.

Everything that another module might need to "know" about the project lives
here, so there is exactly one source of truth.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

RAW_DATASET = DATA_DIR / "riskradar_dataset.csv"

MODEL_PATH = MODELS_DIR / "riskradar_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
LEADERBOARD_PATH = REPORTS_DIR / "model_leaderboard.json"
ANALYTICS_PATH = REPORTS_DIR / "analytics.json"
GEO_PATH = REPORTS_DIR / "geo_sample.json"
EXPLAINER_PATH = MODELS_DIR / "shap_background.joblib"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Target schema
# --------------------------------------------------------------------------
TARGET = "Risk_Level"

# Ordered low -> high so that the encoded integer is itself meaningful.
CLASS_ORDER: list[str] = ["Low", "Medium", "High"]
CLASS_TO_INT: dict[str, int] = {c: i for i, c in enumerate(CLASS_ORDER)}
INT_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_INT.items()}

CLASS_COLOR = {"Low": "#22D3A7", "Medium": "#F5B14C", "High": "#FF4D6D"}
CLASS_ADVICE = {
    "Low": "Area shows healthy safety indicators. Standard precautions are sufficient.",
    "Medium": "Elevated risk signals detected. Prefer well-lit main routes and avoid isolated stretches.",
    "High": "Strong risk signals detected. Avoid travelling alone; share live location and stay near patrolled zones.",
}

# --------------------------------------------------------------------------
# Input schema (raw columns expected from the user / dataset)
# --------------------------------------------------------------------------
ID_COLUMNS = ["Area_ID"]

NUMERIC_COLUMNS = [
    "Latitude",
    "Longitude",
    "Crime_Count",
    "Violent_Crime",
    "Theft_Count",
    "Assault_Count",
    "Harassment_Count",
    "Hour",
    "Month",
    "Weekend",
    "Streetlight_Count",
    "Working_Streetlights",
    "Broken_Streetlights",
    "Population_Density",
    "Footfall",
    "Bus_Stop_Count",
    "Metro_Distance_km",
    "Police_Distance_km",
    "Hospital_Distance_km",
    "School_Count",
    "Commercial_Area",
    "Residential_Area",
    "CCTV_Count",
    "Emergency_Calls",
]

CATEGORICAL_COLUMNS = ["Day", "Weather", "Visibility", "Previous_Risk", "Crime_Trend"]

RAW_FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# Ordinal encodings — deliberately hand-mapped (not LabelEncoder) so the
# ordering carries real-world meaning and is reproducible at serving time.
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_MAP = {d: i for i, d in enumerate(DAY_ORDER)}

WEATHER_ORDER = ["Sunny", "Cloudy", "Rain", "Fog"]  # increasing obstruction
WEATHER_MAP = {w: i for i, w in enumerate(WEATHER_ORDER)}

VISIBILITY_ORDER = ["Poor", "Medium", "Good"]  # increasing visibility
VISIBILITY_MAP = {v: i for i, v in enumerate(VISIBILITY_ORDER)}

PREVIOUS_RISK_ORDER = ["Low", "Medium", "High"]
PREVIOUS_RISK_MAP = {r: i for i, r in enumerate(PREVIOUS_RISK_ORDER)}

CRIME_TREND_ORDER = ["Decreasing", "Stable", "Increasing"]
CRIME_TREND_MAP = {t: i for i, t in enumerate(CRIME_TREND_ORDER)}

CATEGORICAL_CHOICES = {
    "Day": DAY_ORDER,
    "Weather": WEATHER_ORDER,
    "Visibility": ["Good", "Medium", "Poor"],
    "Previous_Risk": PREVIOUS_RISK_ORDER,
    "Crime_Trend": ["Decreasing", "Stable", "Increasing"],
}

# --------------------------------------------------------------------------
# Training constants
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Hours considered "night" for the darkness-exposure features.
NIGHT_HOURS = set(range(21, 24)) | set(range(6))
LATE_NIGHT_HOURS = set(range(5))
