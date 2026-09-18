"""Inference service — the single place where a model touches a request.

Loaded once at process start and reused, so a prediction costs one forward
pass rather than a disk read plus deserialisation.

Beyond the forward pass
-----------------------
A single ``predict`` call now also, in this order:

1. checks the :mod:`cache` for an identical recent request,
2. scores the model and builds the SHAP explanation,
3. wraps the answer in a :mod:`conformal` prediction set, so the response
   carries a coverage *guarantee* rather than only a softmax number,
4. runs the counterfactual lever scan,
5. feeds the engineered row to the :mod:`drift` monitor,
6. records Prometheus metrics and persists the row for audit.

Steps 5 and 6 are strictly off the critical path — they observe, they never
block, and every one of them is individually allowed to fail without
affecting the answer returned to the caller. A safety service that goes down
because its metrics backend went down is not a safety service.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .cache import payload_key, prediction_cache
from .conformal import ConformalPredictor
from .explain import RiskExplainer, counterfactual_scan
from .settings import settings

log = logging.getLogger("riskradar.service")


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before training has produced artefacts."""


class RiskService:
    """Thread-safe singleton wrapper around the trained pipeline."""

    _instance: RiskService | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.model = None
        self.metadata: dict = {}
        self.leaderboard: list[dict] = []
        self.analytics: dict = {}
        self.geo: list[dict] = []
        self.explainer: RiskExplainer | None = None
        self.conformal: ConformalPredictor | None = None
        self.started_at = time.time()
        self.predictions_served = 0
        self.errors_served = 0
        self.load_error: str | None = None
        self.load()

    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> RiskService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load every artefact the API serves. Missing files degrade, not crash.

        A pickled scikit-learn model is only guaranteed to load under the
        library versions it was written with. Rather than let an
        ``AttributeError`` from deep inside joblib reach the user as a stack
        trace, we catch it and turn it into an actionable instruction.
        """
        self.load_error = None
        if C.MODEL_PATH.exists():
            try:
                self.model = joblib.load(C.MODEL_PATH)
                background = joblib.load(C.EXPLAINER_PATH) if C.EXPLAINER_PATH.exists() else None
                self.explainer = RiskExplainer(self.model, background)
            except Exception as exc:
                self.model = None
                self.explainer = None
                self.load_error = (
                    f"The saved model could not be loaded ({exc.__class__.__name__}: {exc}). "
                    "This almost always means your scikit-learn / numpy / LightGBM versions "
                    "differ from the ones the artefact was built with. "
                    "Fix it by retraining locally:  python run.py --train"
                )

        for attr, path, default in (
            ("metadata", C.METADATA_PATH, {}),
            ("leaderboard", C.LEADERBOARD_PATH, []),
            ("analytics", C.ANALYTICS_PATH, {}),
            ("geo", C.GEO_PATH, []),
        ):
            try:
                setattr(self, attr, json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                setattr(self, attr, default)

        # Conformal calibration is optional: without it the API still answers,
        # it just cannot offer a coverage guarantee, and says so in the payload.
        self.conformal = ConformalPredictor()

        # A new artefact invalidates every cached answer and every risk grid.
        prediction_cache.invalidate()
        try:
            from .routing import engine as route_engine

            route_engine.invalidate()
        except Exception:  # pragma: no cover
            pass

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _require(self):
        if self.model is None:
            raise ModelNotLoadedError(
                self.load_error
                or "No trained model found. Run:  python run.py --train"
            )
        return self.model

    # ------------------------------------------------------------------
    @staticmethod
    def to_frame(payload: dict | list[dict]) -> pd.DataFrame:
        """Coerce request payload(s) into the raw column order the pipeline expects."""
        rows = payload if isinstance(payload, list) else [payload]
        df = pd.DataFrame(rows)
        for col in C.RAW_FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df[C.RAW_FEATURE_COLUMNS]

    @staticmethod
    def safety_score(proba: np.ndarray) -> float:
        """Map a class distribution to an intuitive 0–100 safety score.

        A single number is what a citizen actually wants; the expected risk
        band (0=Low … 2=High) is inverted and rescaled so 100 means "all
        probability mass on Low".
        """
        expected = float(np.dot(proba, [0.0, 1.0, 2.0]))
        return round(100.0 * (1.0 - expected / 2.0), 1)

    # ------------------------------------------------------------------
    def predict(self, payload: dict, explain: bool = True,
                recommend: bool = True, conformal: bool = True,
                alpha: float | None = None, observe: bool = True) -> dict:
        """Score a single area and, optionally, explain, quantify and prescribe.

        Parameters
        ----------
        explain     include SHAP attribution
        recommend   include the counterfactual intervention scan
        conformal   include the calibrated prediction set
        alpha       per-request coverage override (0.01 = 99% coverage)
        observe     feed monitors and persistence; turn off for internal calls
                    so a what-if sweep does not pollute the drift window
        """
        model = self._require()
        t0 = time.perf_counter()

        cache_key = payload_key(
            payload,
            model_version=str(self.metadata.get("version", "0"))
            + ":" + str(self.metadata.get("trained_at", "")),
            explain=explain, recommend=recommend, conformal=conformal, alpha=alpha,
        )
        cached = prediction_cache.get(cache_key)
        if cached is not None:
            hit = dict(cached)
            hit["cached"] = True
            hit["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            self.predictions_served += 1
            if observe:
                self._observe(hit, payload, frame=None, engineered=None, cache_hit=True)
            return hit

        frame = self.to_frame(payload)
        # Engineer once and reuse for scoring *and* explanation — rebuilding
        # the 79-column matrix is the single most expensive step of a request.
        engineered = model.named_steps["features"].transform(frame) \
            if hasattr(model, "named_steps") else None
        proba = np.asarray(model.predict_proba(frame))[0]
        idx = int(np.argmax(proba))
        label = C.INT_TO_CLASS[idx]

        result = {
            "risk_level": label,
            "risk_index": idx,
            "confidence": round(float(proba[idx]), 4),
            "probabilities": [
                {"label": C.INT_TO_CLASS[i], "probability": round(float(p), 4)}
                for i, p in enumerate(proba)
            ],
            "safety_score": self.safety_score(proba),
            "color": C.CLASS_COLOR[label],
            "advice": C.CLASS_ADVICE[label],
            "model_name": self.metadata.get("model_name", "RiskRadar"),
            "model_version": self.metadata.get("version", "2.0.0"),
            "explanation": None,
            "recommendations": [],
        }

        if explain and self.explainer is not None:
            try:
                result["explanation"] = self.explainer.explain(
                    frame, idx, features=engineered
                )
            except Exception as exc:  # pragma: no cover
                result["explanation"] = {
                    "method": f"unavailable ({exc.__class__.__name__})",
                    "predicted_class": label, "baseline": 0.0,
                    "total_contribution": 0.0, "drivers": [],
                    "risk_factors": [], "protective_factors": [],
                    "narrative": f"Assessed as {label} risk.",
                }

        if conformal and self.conformal is not None:
            try:
                result["conformal"] = self.conformal.predict_set(proba, alpha=alpha)
            except Exception as exc:  # pragma: no cover
                result["conformal"] = {"available": False, "reason": str(exc)}

        if recommend:
            try:
                result["recommendations"] = counterfactual_scan(model, frame, idx)
            except Exception:  # pragma: no cover
                result["recommendations"] = []

        self.predictions_served += 1
        result["cached"] = False
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        prediction_cache.set(cache_key, result)
        if observe:
            self._observe(result, payload, frame=frame, engineered=engineered,
                          cache_hit=False)
        return result

    # ------------------------------------------------------------------
    def _observe(self, result: dict, payload: dict, *, frame, engineered,
                 cache_hit: bool) -> None:
        """Side-channel work: monitors, metrics, shadow, persistence.

        Every block is independently guarded. This runs after the answer is
        computed, and nothing in here is permitted to change or block it.
        """
        try:
            from .observability import metrics, request_id_var

            metrics.observe_prediction(result, cache_hit=cache_hit)
        except Exception:  # pragma: no cover
            pass

        if engineered is not None:
            try:
                from .drift import monitor

                monitor.observe(engineered, result["risk_level"], result["confidence"])
            except Exception:  # pragma: no cover
                pass

        if frame is not None:
            try:
                from .registry import shadow

                shadow.observe(frame, result["risk_level"])
            except Exception:  # pragma: no cover
                pass

        if settings.persist_predictions and not cache_hit:
            try:
                from .db import log_prediction
                from .observability import request_id_var

                log_prediction(result, payload=payload,
                               request_id=request_id_var.get(), cache_hit=cache_hit)
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------
    def predict_batch(self, df: pd.DataFrame) -> dict:
        """Vectorised scoring for an uploaded CSV."""
        model = self._require()
        t0 = time.perf_counter()

        frame = df.copy()
        for col in C.RAW_FEATURE_COLUMNS:
            if col not in frame.columns:
                frame[col] = np.nan
        proba = np.asarray(model.predict_proba(frame[C.RAW_FEATURE_COLUMNS]))
        idx = proba.argmax(axis=1)
        labels = [C.INT_TO_CLASS[int(i)] for i in idx]
        scores = [self.safety_score(p) for p in proba]
        confidence = proba.max(axis=1).round(4).tolist()

        # A bulk upload is the richest drift signal available — it is a real
        # sample of current conditions, not one hand-tuned dashboard record.
        try:
            from .drift import monitor

            if monitor.ready:
                engineered = model.named_steps["features"].transform(
                    frame[C.RAW_FEATURE_COLUMNS]
                )
                monitor.observe_batch(engineered, labels, confidence)
        except Exception:  # pragma: no cover
            pass

        conformal_sets: list[list[str]] = []
        if self.conformal is not None and self.conformal.ready:
            try:
                conformal_sets = [
                    self.conformal.predict_set(p)["prediction_set"] for p in proba
                ]
            except Exception:  # pragma: no cover
                conformal_sets = []

        self.predictions_served += len(labels)
        return {
            "labels": labels,
            "confidence": confidence,
            "safety_score": scores,
            "probabilities": proba.round(4).tolist(),
            "conformal_sets": conformal_sets,
            "n_abstained": sum(1 for s in conformal_sets if len(s) != 1),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    # ------------------------------------------------------------------
    def what_if(self, base: dict, field: str, values: list[float]) -> dict:
        """Sweep one input across a range and trace the risk response curve."""
        model = self._require()
        rows = []
        for v in values:
            row = dict(base)
            row[field] = v
            rows.append(row)

        frame = self.to_frame(rows)
        proba = np.asarray(model.predict_proba(frame))
        return {
            "field": field,
            "values": values,
            "risk_levels": [C.INT_TO_CLASS[int(i)] for i in proba.argmax(axis=1)],
            "safety_scores": [self.safety_score(p) for p in proba],
            "high_risk_probability": proba[:, 2].round(4).tolist(),
        }

    # ------------------------------------------------------------------
    def health(self) -> dict:
        return {
            "status": "operational" if self.ready else "degraded",
            "detail": self.load_error,
            "model_loaded": self.ready,
            "model_name": self.metadata.get("model_name"),
            "accuracy": self.metadata.get("metrics", {}).get("accuracy"),
            "version": self.metadata.get("version", "2.0.0"),
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "predictions_served": self.predictions_served,
        }

    # ------------------------------------------------------------------
    def system_status(self) -> dict:
        """Consolidated platform status — one call for the whole control room."""
        from .cache import prediction_cache
        from .db import database
        from .drift import monitor
        from .observability import health_score, metrics
        from .registry import registry, shadow
        from .security import security_status

        cache_stats = prediction_cache.stats()
        drift_status = (monitor.last_report or {}).get("status")
        total = max(self.predictions_served, 1)

        score = health_score(
            model_ready=self.ready,
            drift_status=drift_status,
            error_rate=self.errors_served / total,
            cache_hit_rate=cache_stats["hit_rate"],
            db_ready=database.ready,
        )

        return {
            "health": score,
            "service": self.health(),
            "model": {
                "name": self.metadata.get("model_name"),
                "version": self.metadata.get("version"),
                "trained_at": self.metadata.get("trained_at"),
                "accuracy": self.metadata.get("metrics", {}).get("accuracy"),
                "n_features": self.metadata.get("n_engineered_features"),
            },
            "conformal": {
                "available": bool(self.conformal and self.conformal.ready),
                "calibration": (
                    self.conformal.calibration.to_dict()
                    if self.conformal and self.conformal.calibration else None
                ),
            },
            "drift": {
                "ready": monitor.ready,
                "observed": monitor.n_observed,
                "window": monitor.window,
                "status": drift_status or "not-evaluated",
            },
            "cache": cache_stats,
            "database": database.health(),
            "registry": registry.summary(),
            "shadow": shadow.stats(),
            "security": security_status(),
            "metrics_enabled": metrics.enabled,
            "settings": settings.public_dict(),
        }


def get_service() -> RiskService:
    """FastAPI dependency provider."""
    return RiskService.instance()
