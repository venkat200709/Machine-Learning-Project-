"""Domain-driven, leakage-free feature engineering for RiskRadar.

Why a module-level function instead of inline notebook code?
------------------------------------------------------------
The transformation is wrapped in a ``FunctionTransformer`` and becomes step 1
of the fitted ``Pipeline``. That means the *exact* same maths runs at training
time and at serving time — the single most common source of "great notebook
accuracy, broken production model" bugs is eliminated by construction.

Design rules followed here:

1. **No target usage.** Nothing in this file touches ``Risk_Level``.
2. **No cross-row statistics.** Every feature is a pure row-wise function, so
   there is zero train/test information bleed and a single record can be
   scored in isolation by the API.
3. **Identifiers dropped.** ``Area_ID`` is unique per row; keeping it lets a
   tree memorise rows instead of learning structure.
4. **Ratios over raw counts.** "40 crimes" means something very different in a
   dense commercial hub than in a quiet residential lane, so counts are
   normalised by exposure (population, footfall, total crime).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

EPS = 1.0  # additive smoothing to keep every denominator strictly positive


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ordinal(series: pd.Series, mapping: dict, default: int = 0) -> pd.Series:
    """Map a categorical column to its ordinal code, tolerating unseen values."""
    if series.dtype.kind in "iufb":  # already numeric — trust the caller
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return series.astype(str).str.strip().map(mapping).fillna(default).astype(float)


def _cyclical(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Encode a cyclical quantity so that 23:00 sits next to 00:00."""
    radians = 2.0 * np.pi * series.astype(float) / period
    return np.sin(radians), np.cos(radians)


# --------------------------------------------------------------------------
# main transform
# --------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw RiskRadar records into the numeric matrix the models consume.

    Parameters
    ----------
    df : DataFrame containing (at minimum) the columns in
         ``config.RAW_FEATURE_COLUMNS``.

    Returns
    -------
    DataFrame of float features with a deterministic column order.
    """
    d = df.copy()

    # -- 0. Defensive coercion -------------------------------------------
    for col in C.NUMERIC_COLUMNS:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    for col, mapping in (
        ("Day", C.DAY_MAP),
        ("Weather", C.WEATHER_MAP),
        ("Visibility", C.VISIBILITY_MAP),
        ("Previous_Risk", C.PREVIOUS_RISK_MAP),
        ("Crime_Trend", C.CRIME_TREND_MAP),
    ):
        if col not in d.columns:
            d[col] = next(iter(mapping))

    out = pd.DataFrame(index=d.index)

    # ------------------------------------------------------------------
    # 1. Raw signals worth keeping as-is
    # ------------------------------------------------------------------
    passthrough = [
        "Latitude", "Longitude",
        "Crime_Count", "Violent_Crime", "Theft_Count",
        "Assault_Count", "Harassment_Count",
        "Hour", "Month", "Weekend",
        "Streetlight_Count", "Working_Streetlights", "Broken_Streetlights",
        "Population_Density", "Footfall",
        "Bus_Stop_Count", "Metro_Distance_km",
        "Police_Distance_km", "Hospital_Distance_km",
        "School_Count", "Commercial_Area", "Residential_Area",
        "CCTV_Count", "Emergency_Calls",
    ]
    for col in passthrough:
        out[col] = d[col].astype(float)

    # ------------------------------------------------------------------
    # 2. Ordinal categoricals (meaningful ordering, not arbitrary codes)
    # ------------------------------------------------------------------
    out["Day_Index"] = _ordinal(d["Day"], C.DAY_MAP)
    out["Weather_Severity"] = _ordinal(d["Weather"], C.WEATHER_MAP)
    out["Visibility_Score"] = _ordinal(d["Visibility"], C.VISIBILITY_MAP)
    out["Previous_Risk_Ord"] = _ordinal(d["Previous_Risk"], C.PREVIOUS_RISK_MAP)
    out["Crime_Trend_Ord"] = _ordinal(d["Crime_Trend"], C.CRIME_TREND_MAP)

    # ------------------------------------------------------------------
    # 3. Temporal structure — night-time is the dominant risk modifier
    # ------------------------------------------------------------------
    hour = out["Hour"]
    out["Hour_Sin"], out["Hour_Cos"] = _cyclical(hour, 24)
    out["Month_Sin"], out["Month_Cos"] = _cyclical(out["Month"], 12)
    out["Is_Night"] = hour.isin(C.NIGHT_HOURS).astype(float)
    out["Is_Late_Night"] = hour.isin(C.LATE_NIGHT_HOURS).astype(float)
    out["Is_Rush_Hour"] = hour.between(8, 10).astype(float) + hour.between(17, 20).astype(float)
    # Distance from 14:00, the statistically safest point of the day.
    out["Hours_From_Midday"] = (hour - 14).abs().clip(upper=12)
    out["Night_Weekend"] = out["Is_Night"] * out["Weekend"]

    # ------------------------------------------------------------------
    # 4. Crime composition — *what kind* of crime, not just how much
    # ------------------------------------------------------------------
    total_crime = out["Crime_Count"] + EPS
    out["Violent_Ratio"] = out["Violent_Crime"] / total_crime
    out["Theft_Ratio"] = out["Theft_Count"] / total_crime
    out["Assault_Ratio"] = out["Assault_Count"] / total_crime
    out["Harassment_Ratio"] = out["Harassment_Count"] / total_crime
    # Crimes that disproportionately target women, weighted by severity.
    out["Gendered_Crime_Load"] = (
        2.0 * out["Assault_Count"] + 1.5 * out["Harassment_Count"] + 1.0 * out["Violent_Crime"]
    )
    out["Gendered_Crime_Ratio"] = out["Gendered_Crime_Load"] / total_crime
    out["Crime_Severity_Index"] = (
        3.0 * out["Violent_Crime"] + 2.5 * out["Assault_Count"]
        + 2.0 * out["Harassment_Count"] + 1.0 * out["Theft_Count"]
    )
    out["Reported_Crime_Sum"] = (
        out["Violent_Crime"] + out["Theft_Count"] + out["Assault_Count"] + out["Harassment_Count"]
    )
    # Gap between headline count and itemised crimes = unclassified incidents.
    out["Unclassified_Crime"] = out["Crime_Count"] - out["Reported_Crime_Sum"]
    out["Log_Crime_Count"] = np.log1p(out["Crime_Count"].clip(lower=0))

    # ------------------------------------------------------------------
    # 5. Exposure normalisation — risk per person, not per area
    # ------------------------------------------------------------------
    pop = out["Population_Density"] + EPS
    foot = out["Footfall"] + EPS
    out["Crime_Per_1k_Population"] = 1000.0 * out["Crime_Count"] / pop
    out["Crime_Per_1k_Footfall"] = 1000.0 * out["Crime_Count"] / foot
    out["Severity_Per_1k_Population"] = 1000.0 * out["Crime_Severity_Index"] / pop
    out["Emergency_Calls_Per_1k_Pop"] = 1000.0 * out["Emergency_Calls"] / pop
    out["Calls_Per_Crime"] = out["Emergency_Calls"] / total_crime
    # Jane Jacobs' "eyes on the street": more natural surveillance = safer.
    out["Guardianship_Index"] = foot / pop

    # ------------------------------------------------------------------
    # 6. Lighting & surveillance infrastructure
    # ------------------------------------------------------------------
    lights = out["Streetlight_Count"] + EPS
    out["Lighting_Health"] = out["Working_Streetlights"] / lights
    out["Broken_Light_Ratio"] = out["Broken_Streetlights"] / lights
    out["Effective_Lighting"] = out["Working_Streetlights"] * out["Visibility_Score"] / 2.0
    # Darkness exposure: broken lights only matter after sunset.
    out["Darkness_Exposure"] = (
        out["Is_Night"] * out["Broken_Streetlights"] / lights * (3.0 - out["Visibility_Score"])
    )
    out["CCTV_Per_1k_Population"] = 1000.0 * out["CCTV_Count"] / pop
    out["CCTV_Per_Crime"] = out["CCTV_Count"] / total_crime
    out["Surveillance_Index"] = (out["CCTV_Count"] + out["Working_Streetlights"]) / 2.0
    out["Surveillance_Deficit"] = out["Crime_Severity_Index"] / (out["Surveillance_Index"] + EPS)

    # ------------------------------------------------------------------
    # 7. Emergency reachability — how fast can help arrive?
    # ------------------------------------------------------------------
    out["Police_Proximity"] = 1.0 / (1.0 + out["Police_Distance_km"])
    out["Hospital_Proximity"] = 1.0 / (1.0 + out["Hospital_Distance_km"])
    out["Metro_Proximity"] = 1.0 / (1.0 + out["Metro_Distance_km"])
    out["Nearest_Help_km"] = out[["Police_Distance_km", "Hospital_Distance_km"]].min(axis=1)
    out["Total_Help_Distance"] = out["Police_Distance_km"] + out["Hospital_Distance_km"]
    out["Emergency_Access_Score"] = (
        0.6 * out["Police_Proximity"] + 0.4 * out["Hospital_Proximity"]
    )
    # Crimes committed where police are far away are the dangerous ones.
    out["Unpoliced_Crime_Load"] = out["Crime_Severity_Index"] * out["Police_Distance_km"]

    # ------------------------------------------------------------------
    # 8. Mobility & escape options
    # ------------------------------------------------------------------
    out["Transit_Access_Score"] = (
        np.log1p(out["Bus_Stop_Count"]) + 2.0 * out["Metro_Proximity"]
    )
    out["Isolation_Index"] = out["Nearest_Help_km"] / (out["Transit_Access_Score"] + EPS)
    out["Land_Use_Mix"] = out["Commercial_Area"] + out["Residential_Area"]
    out["Amenity_Density"] = (
        out["School_Count"] + out["Bus_Stop_Count"] + out["Land_Use_Mix"]
    )

    # ------------------------------------------------------------------
    # 9. Composite safety scores — the "expert rules" a domain analyst
    #    would hand-write, given to the model as strong priors.
    # ------------------------------------------------------------------
    out["Infrastructure_Score"] = (
        0.35 * out["Lighting_Health"]
        + 0.30 * np.tanh(out["CCTV_Per_1k_Population"])
        + 0.20 * out["Emergency_Access_Score"]
        + 0.15 * np.tanh(out["Transit_Access_Score"] / 5.0)
    )
    out["Threat_Score"] = (
        np.tanh(out["Crime_Per_1k_Population"])
        + 0.8 * out["Gendered_Crime_Ratio"]
        + 0.5 * out["Previous_Risk_Ord"] / 2.0
        + 0.5 * out["Crime_Trend_Ord"] / 2.0
        + 0.4 * out["Darkness_Exposure"]
    )
    out["Net_Safety_Score"] = out["Infrastructure_Score"] - out["Threat_Score"]
    out["Risk_Momentum"] = out["Previous_Risk_Ord"] * (1.0 + out["Crime_Trend_Ord"]) / 2.0
    out["Night_Threat"] = out["Threat_Score"] * (1.0 + out["Is_Night"])
    out["Weather_Adjusted_Visibility"] = out["Visibility_Score"] - 0.5 * out["Weather_Severity"]

    # ------------------------------------------------------------------
    # 10. Sanitisation — models never see inf/NaN
    # ------------------------------------------------------------------
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype("float32")


def feature_names() -> list[str]:
    """Column order produced by :func:`engineer` (derived, never hard-coded)."""
    probe = pd.DataFrame([dict.fromkeys(C.NUMERIC_COLUMNS, 0) | {
        "Day": "Mon", "Weather": "Sunny", "Visibility": "Good",
        "Previous_Risk": "Low", "Crime_Trend": "Stable",
    }])
    return list(engineer(probe).columns)


# Human-readable labels used by the explainability UI.
FEATURE_GLOSSARY = {
    "Threat_Score": "Composite threat level",
    "Net_Safety_Score": "Net safety balance",
    "Infrastructure_Score": "Safety infrastructure quality",
    "Crime_Severity_Index": "Weighted crime severity",
    "Gendered_Crime_Load": "Crimes targeting women",
    "Gendered_Crime_Ratio": "Share of gendered crime",
    "Crime_Per_1k_Population": "Crime rate per 1k residents",
    "Crime_Per_1k_Footfall": "Crime rate per 1k visitors",
    "Darkness_Exposure": "Night-time darkness exposure",
    "Lighting_Health": "Working streetlight ratio",
    "Broken_Light_Ratio": "Broken streetlight ratio",
    "Surveillance_Deficit": "Crime vs. surveillance gap",
    "Surveillance_Index": "CCTV + lighting coverage",
    "CCTV_Per_1k_Population": "CCTV per 1k residents",
    "Police_Proximity": "Closeness to police",
    "Unpoliced_Crime_Load": "Crime far from police",
    "Emergency_Access_Score": "Emergency reachability",
    "Isolation_Index": "Geographic isolation",
    "Guardianship_Index": "Natural surveillance (footfall)",
    "Risk_Momentum": "Historical risk momentum",
    "Previous_Risk_Ord": "Previously recorded risk",
    "Crime_Trend_Ord": "Crime trend direction",
    "Night_Threat": "Night-amplified threat",
    "Is_Night": "Night-time flag",
    "Is_Late_Night": "Late-night flag",
    "Visibility_Score": "Visibility conditions",
    "Weather_Severity": "Weather obstruction",
    "Nearest_Help_km": "Distance to nearest help",
    "Transit_Access_Score": "Public transport access",
    "Calls_Per_Crime": "Emergency calls per crime",
    "Unclassified_Crime": "Unclassified incidents",
    "Hour_Sin": "Time-of-day cycle (sine)",
    "Hour_Cos": "Time-of-day cycle (cosine)",
    "Month_Sin": "Season cycle (sine)",
    "Month_Cos": "Season cycle (cosine)",
    "Hours_From_Midday": "Hours away from midday",
    "Night_Weekend": "Weekend night flag",
    "CCTV_Per_Crime": "CCTV cameras per incident",
    "Log_Crime_Count": "Crime volume (log scale)",
    "Reported_Crime_Sum": "Itemised crime total",
    "Severity_Per_1k_Population": "Crime severity per 1k residents",
    "Emergency_Calls_Per_1k_Pop": "Distress calls per 1k residents",
    "Effective_Lighting": "Usable lighting after visibility",
    "Total_Help_Distance": "Combined distance to help",
    "Weather_Adjusted_Visibility": "Visibility adjusted for weather",
    "Amenity_Density": "Neighbourhood amenity density",
    "Land_Use_Mix": "Mixed-use land profile",
    "Theft_Ratio": "Theft share",
    "Violent_Ratio": "Violent crime share",
    "Harassment_Ratio": "Harassment share",
    "Assault_Ratio": "Assault share",
}


def pretty(name: str) -> str:
    """Map an internal feature name to a display label."""
    return FEATURE_GLOSSARY.get(name, name.replace("_", " "))
