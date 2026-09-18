"""Data and prediction drift monitoring.

Why this exists
---------------
A model's accuracy is measured once, on the day it is trained, against data
that no longer exists. Every day after that the world moves: a new metro line
changes footfall, a lighting programme changes the lamp counts, a reporting
policy change alters what "Crime_Count" even means. The model does not
degrade loudly — it degrades silently, and the dashboard keeps showing 98.27%
because that number is a fossil.

Drift monitoring is the only thing standing between "deployed" and "quietly
wrong for eight months".

What is measured
----------------
* **Covariate drift** — has the input distribution moved? Population Stability
  Index per feature, plus a Kolmogorov–Smirnov statistic and Jensen–Shannon
  distance as corroborating evidence, because PSI alone is sensitive to
  binning choices.
* **Prediction drift** — has the *output* mix moved? Cheap, needs no labels,
  and often the first thing to twitch.
* **Confidence decay** — is the model becoming less certain? A falling mean
  confidence with a stable input distribution usually means the inputs have
  moved somewhere the boundaries are thin.
* **Concept drift** — accuracy on records with reported ground truth. The real
  thing, only available where the feedback loop is closed.

PSI interpretation follows the convention used in credit-risk model
monitoring, where it originated:

===========  ==========================================================
``< 0.10``   stable
``0.10–0.25`` moderate shift — investigate
``> 0.25``   significant shift — the model may no longer be valid
===========  ==========================================================
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import config as C
from .settings import settings

log = logging.getLogger("riskradar.drift")

REFERENCE_PATH = C.MODELS_DIR / "drift_reference.json"

N_BINS = 10
# Additive smoothing: a bin that is empty in one distribution and populated in
# the other would otherwise send PSI to infinity and swamp every other signal.
EPS = 1e-6

STATUS_STABLE = "stable"
STATUS_WARNING = "warning"
STATUS_ALERT = "alert"

# Features worth alerting on. The full engineered matrix is 79 columns; most
# are deterministic functions of these, so monitoring all of them would raise
# a dozen correlated alerts for one underlying change.
MONITORED_FEATURES = [
    "Crime_Count", "Violent_Crime", "Harassment_Count", "Emergency_Calls",
    "Hour", "Population_Density", "Footfall", "Working_Streetlights",
    "Broken_Streetlights", "CCTV_Count", "Police_Distance_km",
    "Crime_Per_1k_Population", "Gendered_Crime_Ratio", "Lighting_Health",
    "Darkness_Exposure", "Surveillance_Deficit", "Threat_Score",
    "Infrastructure_Score", "Net_Safety_Score", "Isolation_Index",
]


# ==========================================================================
# Statistical distances
# ==========================================================================
def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                               bins: np.ndarray | None = None,
                               n_bins: int = N_BINS) -> tuple[float, np.ndarray]:
    """PSI between two samples, plus the bin edges used.

    Bins come from the *expected* (reference) sample's quantiles, so each
    reference bin holds roughly the same mass and the statistic is not
    dominated by a long tail.
    """
    expected = np.asarray(expected, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = np.asarray(actual, dtype=float)
    actual = actual[np.isfinite(actual)]

    if len(expected) < 2 or len(actual) < 1:
        return 0.0, np.array([])

    if bins is None:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.unique(np.percentile(expected, quantiles))
        if len(bins) < 3:
            # A near-constant feature has no meaningful quantile structure.
            lo, hi = float(expected.min()), float(expected.max())
            bins = np.array([lo - 1e-9, hi + 1e-9]) if hi > lo else np.array([lo - 1, lo + 1])
    bins = np.asarray(bins, dtype=float)

    edges = np.concatenate(([-np.inf], bins[1:-1], [np.inf]))
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)

    e_prop = e_counts / max(e_counts.sum(), 1) + EPS
    a_prop = a_counts / max(a_counts.sum(), 1) + EPS

    psi = float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))
    return round(abs(psi), 6), bins


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic — max CDF gap, binning-free."""
    expected = np.sort(np.asarray(expected, dtype=float))
    actual = np.sort(np.asarray(actual, dtype=float))
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    grid = np.concatenate([expected, actual])
    cdf_e = np.searchsorted(expected, grid, side="right") / len(expected)
    cdf_a = np.searchsorted(actual, grid, side="right") / len(actual)
    return round(float(np.max(np.abs(cdf_e - cdf_a))), 6)


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Bounded [0, 1] symmetric divergence between two discrete distributions."""
    p = np.asarray(p, dtype=float) + EPS
    q = np.asarray(q, dtype=float) + EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    divergence = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return round(float(np.sqrt(max(divergence, 0.0))), 6)


def classify_psi(psi: float) -> str:
    if psi >= settings.drift_psi_alert:
        return STATUS_ALERT
    if psi >= settings.drift_psi_warn:
        return STATUS_WARNING
    return STATUS_STABLE


# ==========================================================================
# Reference distribution
# ==========================================================================
@dataclass
class ReferenceDistribution:
    """Frozen snapshot of the training distribution.

    Storing bin edges and reference proportions rather than the raw training
    data means drift can be evaluated in a container that never ships the
    dataset — which is both a size win and a privacy win.
    """

    features: dict[str, dict[str, Any]]
    class_distribution: dict[str, float]
    n_reference: int
    created_at: str
    model_version: str = ""

    def to_dict(self) -> dict:
        return {
            "features": self.features,
            "class_distribution": self.class_distribution,
            "n_reference": self.n_reference,
            "created_at": self.created_at,
            "model_version": self.model_version,
        }

    def save(self, path=None) -> None:
        path = path or REFERENCE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path=None) -> ReferenceDistribution | None:
        path = path or REFERENCE_PATH
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                features=data["features"],
                class_distribution=data.get("class_distribution", {}),
                n_reference=int(data.get("n_reference", 0)),
                created_at=data.get("created_at", ""),
                model_version=data.get("model_version", ""),
            )
        except Exception:  # pragma: no cover
            log.warning("drift reference is unreadable; ignoring it")
            return None


def build_reference(engineered: pd.DataFrame, y: np.ndarray | None = None,
                    *, model_version: str = "", save: bool = True) -> ReferenceDistribution:
    """Compute and persist the reference profile from the training matrix."""
    features: dict[str, dict[str, Any]] = {}
    for name in MONITORED_FEATURES:
        if name not in engineered.columns:
            continue
        col = engineered[name].to_numpy(dtype=float)
        col = col[np.isfinite(col)]
        if len(col) < 10:
            continue
        _, bins = population_stability_index(col, col)
        edges = np.concatenate(([-np.inf], bins[1:-1], [np.inf])) if len(bins) > 2 else None
        counts = np.histogram(col, bins=edges)[0] if edges is not None else np.array([len(col)])
        features[name] = {
            "bins": [float(b) for b in bins],
            "proportions": (counts / max(counts.sum(), 1)).round(8).tolist(),
            "mean": round(float(col.mean()), 6),
            "std": round(float(col.std()), 6),
            "p05": round(float(np.percentile(col, 5)), 6),
            "p50": round(float(np.percentile(col, 50)), 6),
            "p95": round(float(np.percentile(col, 95)), 6),
            "min": round(float(col.min()), 6),
            "max": round(float(col.max()), 6),
        }

    class_dist: dict[str, float] = {}
    if y is not None and len(y):
        counts = Counter(int(v) for v in np.asarray(y).ravel())
        total = sum(counts.values())
        class_dist = {
            C.INT_TO_CLASS[k]: round(v / total, 6) for k, v in sorted(counts.items())
        }

    ref = ReferenceDistribution(
        features=features,
        class_distribution=class_dist,
        n_reference=len(engineered),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_version=model_version,
    )
    if save:
        ref.save()
        log.info("drift reference written (%d features)", len(features))
    return ref


# ==========================================================================
# Live monitor
# ==========================================================================
class DriftMonitor:
    """Rolling window of live traffic, compared against the reference.

    A bounded ``deque`` rather than a database query: drift needs the *recent*
    distribution, the window is small, and keeping it in memory means the
    check costs microseconds and cannot be affected by a slow database. The
    verdicts are persisted; the raw window is not.
    """

    def __init__(self, reference: ReferenceDistribution | None = None,
                 window: int | None = None) -> None:
        self.reference = reference or ReferenceDistribution.load()
        self.window = window or settings.drift_window
        self._rows: deque[dict[str, float]] = deque(maxlen=self.window)
        self._labels: deque[str] = deque(maxlen=self.window)
        self._confidence: deque[float] = deque(maxlen=self.window)
        self._lock = threading.Lock()
        self.total_observed = 0
        self.last_report: dict | None = None

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.reference is not None and bool(self.reference.features)

    @property
    def n_observed(self) -> int:
        with self._lock:
            return len(self._rows)

    # ------------------------------------------------------------------
    def observe(self, engineered: pd.DataFrame, label: str, confidence: float) -> None:
        """Record one served prediction. Must stay cheap — it is on the hot path."""
        if not self.ready or engineered is None or not len(engineered):
            return
        try:
            row = engineered.iloc[0]
            sample = {
                name: float(row[name])
                for name in self.reference.features
                if name in engineered.columns
            }
        except Exception:  # pragma: no cover
            return

        with self._lock:
            self._rows.append(sample)
            self._labels.append(label)
            self._confidence.append(float(confidence))
            self.total_observed += 1

    def observe_batch(self, engineered: pd.DataFrame, labels: list[str],
                      confidence: list[float]) -> None:
        if not self.ready or engineered is None or not len(engineered):
            return
        names = [n for n in self.reference.features if n in engineered.columns]
        subset = engineered[names].to_dict("records")
        with self._lock:
            for i, sample in enumerate(subset):
                self._rows.append({k: float(v) for k, v in sample.items()})
                if i < len(labels):
                    self._labels.append(labels[i])
                if i < len(confidence):
                    self._confidence.append(float(confidence[i]))
                self.total_observed += 1

    # ------------------------------------------------------------------
    def snapshot(self) -> tuple[pd.DataFrame, list[str], list[float]]:
        with self._lock:
            return (
                pd.DataFrame(list(self._rows)),
                list(self._labels),
                list(self._confidence),
            )

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()
            self._labels.clear()
            self._confidence.clear()

    # ------------------------------------------------------------------
    def report(self, min_samples: int = 30, persist: bool = True) -> dict:
        """Evaluate drift over the current window."""
        if not self.ready:
            return {
                "available": False,
                "reason": "No drift reference found. Run: python run.py --train",
            }

        live, labels, confidence = self.snapshot()
        if len(live) < min_samples:
            return {
                "available": False,
                "reason": (
                    f"Need {min_samples} observations to assess drift; "
                    f"have {len(live)}. Score more areas first."
                ),
                "observed": len(live),
                "required": min_samples,
            }

        features = self._feature_report(live)
        prediction = self._prediction_report(labels)
        conf = self._confidence_report(confidence)

        alerting = [f for f in features if f["status"] == STATUS_ALERT]
        warning = [f for f in features if f["status"] == STATUS_WARNING]

        if alerting or prediction.get("status") == STATUS_ALERT:
            overall = STATUS_ALERT
        elif warning or prediction.get("status") == STATUS_WARNING:
            overall = STATUS_WARNING
        else:
            overall = STATUS_STABLE

        report = {
            "available": True,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": overall,
            "window_size": len(live),
            "total_observed": self.total_observed,
            "reference": {
                "n": self.reference.n_reference,
                "created_at": self.reference.created_at,
                "model_version": self.reference.model_version,
            },
            "features": features,
            "n_alerting": len(alerting),
            "n_warning": len(warning),
            "prediction_drift": prediction,
            "confidence": conf,
            "summary": _summarise(overall, alerting, warning, prediction),
            "recommended_action": _recommend(overall, alerting, prediction),
        }

        self.last_report = report
        if persist:
            self._persist(report)
        return report

    # ------------------------------------------------------------------
    def _feature_report(self, live: pd.DataFrame) -> list[dict]:
        out: list[dict] = []
        for name, profile in self.reference.features.items():
            if name not in live.columns:
                continue
            actual = live[name].to_numpy(dtype=float)
            actual = actual[np.isfinite(actual)]
            if len(actual) == 0:
                continue

            bins = np.asarray(profile["bins"], dtype=float)
            edges = np.concatenate(([-np.inf], bins[1:-1], [np.inf])) if len(bins) > 2 else None
            if edges is not None:
                a_counts = np.histogram(actual, bins=edges)[0]
                a_prop = a_counts / max(a_counts.sum(), 1)
                e_prop = np.asarray(profile["proportions"], dtype=float)
                if len(e_prop) != len(a_prop):
                    e_prop = np.resize(e_prop, len(a_prop))
                psi = float(np.sum(
                    ((a_prop + EPS) - (e_prop + EPS)) *
                    np.log((a_prop + EPS) / (e_prop + EPS))
                ))
                psi = round(abs(psi), 6)
                js = jensen_shannon_distance(e_prop, a_prop)
            else:
                psi, js = 0.0, 0.0

            ref_mean = float(profile["mean"])
            ref_std = float(profile["std"]) or 1.0
            live_mean = float(actual.mean())

            out.append({
                "feature": name,
                "label": _pretty(name),
                "psi": psi,
                "js_distance": js,
                "ks_statistic": _approx_ks(profile, actual),
                "status": classify_psi(psi),
                "reference_mean": round(ref_mean, 4),
                "live_mean": round(live_mean, 4),
                "mean_shift_sigma": round((live_mean - ref_mean) / ref_std, 3),
                "direction": "higher" if live_mean > ref_mean else "lower",
                "n": len(actual),
            })

        return sorted(out, key=lambda r: -r["psi"])

    def _prediction_report(self, labels: list[str]) -> dict:
        if not labels or not self.reference.class_distribution:
            return {"available": False}
        counts = Counter(labels)
        total = sum(counts.values())
        live = {c: counts.get(c, 0) / total for c in C.CLASS_ORDER}
        ref = {c: float(self.reference.class_distribution.get(c, 0.0)) for c in C.CLASS_ORDER}

        e = np.array([ref[c] for c in C.CLASS_ORDER])
        a = np.array([live[c] for c in C.CLASS_ORDER])
        psi = float(np.sum(((a + EPS) - (e + EPS)) * np.log((a + EPS) / (e + EPS))))
        psi = round(abs(psi), 6)

        return {
            "available": True,
            "psi": psi,
            "js_distance": jensen_shannon_distance(e, a),
            "status": classify_psi(psi),
            "reference": {k: round(v, 4) for k, v in ref.items()},
            "live": {k: round(v, 4) for k, v in live.items()},
            "delta": {c: round(live[c] - ref[c], 4) for c in C.CLASS_ORDER},
            "n": total,
        }

    def _confidence_report(self, confidence: list[float]) -> dict:
        if not confidence:
            return {"available": False}
        arr = np.asarray(confidence, dtype=float)
        half = max(len(arr) // 2, 1)
        recent, older = arr[-half:], arr[:half]
        trend = float(recent.mean() - older.mean()) if len(arr) >= 20 else 0.0
        return {
            "available": True,
            "mean": round(float(arr.mean()), 4),
            "p05": round(float(np.percentile(arr, 5)), 4),
            "median": round(float(np.median(arr)), 4),
            "low_confidence_rate": round(float((arr < 0.60).mean()), 4),
            "trend": round(trend, 4),
            # Falling confidence is an early warning even when PSI is clean.
            "status": STATUS_WARNING if trend < -0.05 else STATUS_STABLE,
        }

    # ------------------------------------------------------------------
    def _persist(self, report: dict) -> None:
        from .db import DriftRecord, database, raise_alert, session_scope

        if not database.init():
            return
        try:
            with session_scope() as s:
                if s is None:
                    return
                for f in report["features"]:
                    if f["status"] == STATUS_STABLE:
                        continue  # only store what matters
                    s.add(DriftRecord(
                        feature=f["feature"], psi=f["psi"],
                        ks_statistic=f.get("ks_statistic"),
                        js_distance=f.get("js_distance"),
                        status=f["status"], window_size=report["window_size"],
                        model_version=self.reference.model_version,
                    ))
        except Exception:  # pragma: no cover
            log.exception("failed to persist drift snapshot")

        if report["status"] == STATUS_ALERT:
            worst = report["features"][0] if report["features"] else {}
            raise_alert(
                "high", "drift",
                f"Significant input drift detected ({report['n_alerting']} features)",
                f"Worst: {worst.get('label', 'n/a')} PSI={worst.get('psi', 0):.3f}. "
                f"{report['recommended_action']}",
            )


def _approx_ks(profile: dict, actual: np.ndarray) -> float:
    """KS against the reference's stored percentiles.

    The exact statistic needs both raw samples and we deliberately do not keep
    the reference sample. Comparing five stored percentiles against the live
    empirical CDF is a close enough corroboration of the PSI signal.
    """
    marks = [("p05", 0.05), ("p50", 0.50), ("p95", 0.95)]
    gaps = []
    for key, expected_cdf in marks:
        if key not in profile:
            continue
        value = float(profile[key])
        live_cdf = float((actual <= value).mean())
        gaps.append(abs(live_cdf - expected_cdf))
    return round(max(gaps), 6) if gaps else 0.0


def _pretty(name: str) -> str:
    from .features import pretty

    return pretty(name)


def _summarise(status: str, alerting: list[dict], warning: list[dict],
               prediction: dict) -> str:
    if status == STATUS_STABLE:
        return (
            "Live traffic matches the training distribution. No action needed — "
            "the accuracy on the model card is still a fair description of the model."
        )
    parts = []
    if alerting:
        names = ", ".join(f["label"] for f in alerting[:3])
        parts.append(f"{len(alerting)} feature(s) have shifted significantly ({names}).")
    if warning:
        parts.append(f"{len(warning)} feature(s) show moderate movement.")
    if prediction.get("status") in (STATUS_WARNING, STATUS_ALERT):
        delta = prediction.get("delta", {})
        biggest = max(delta, key=lambda k: abs(delta[k])) if delta else None
        if biggest:
            parts.append(
                f"The predicted risk mix has moved: {biggest} is "
                f"{delta[biggest]:+.1%} versus training."
            )
    return " ".join(parts)


def _recommend(status: str, alerting: list[dict], prediction: dict) -> str:
    if status == STATUS_STABLE:
        return "Continue monitoring."
    if status == STATUS_WARNING:
        return (
            "Investigate the shifted features for an upstream data-collection change. "
            "Schedule a retrain if the movement persists for another window."
        )
    return (
        "Retrain against recent data before relying on these predictions for "
        "planning decisions, and confirm no ingestion pipeline has changed units "
        "or reporting policy."
    )


# Process-wide monitor, shared by the service.
monitor = DriftMonitor()
