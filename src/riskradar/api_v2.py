"""The v2 API surface — the platform layer.

Kept in its own module and mounted as a router rather than being merged into
``api.py``, for one reason: **the v1 contract must not move.** Anything already
calling ``/api/predict`` keeps working, byte for byte, while everything new
lands under ``/api/v2/``. That is the whole discipline of API versioning, and
it is much easier to hold if the two surfaces are physically separate files.

Route groups
------------
``/api/v2/predict``     scoring with conformal sets and uncertainty
``/api/v2/route``       risk-aware route planning
``/api/v2/optimise``    budget-constrained intervention allocation
``/api/v2/monitor``     drift, live feed, health score, alerts
``/api/v2/fairness``    bias audit across operational strata
``/api/v2/registry``    model versions, shadow evaluation, promotion, rollback
``/api/v2/admin``       API keys, retention, cache control
``/api/v2/system``      status, readiness audit, settings
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import config as C
from .cache import prediction_cache
from .schemas import AreaFeatures
from .security import Principal, require_admin, require_analyst, require_viewer
from .service import RiskService, get_service
from .settings import settings

log = logging.getLogger("riskradar.api.v2")

router = APIRouter(prefix="/api/v2")


# ==========================================================================
# Request models
# ==========================================================================
class AssessRequest(BaseModel):
    area: AreaFeatures
    alpha: float = Field(
        0.10, ge=0.001, le=0.5,
        description="Conformal error rate. 0.10 = 90% guaranteed coverage.",
    )
    explain: bool = True
    recommend: bool = True


class RouteRequest(BaseModel):
    origin_lat: float = Field(..., ge=-90, le=90)
    origin_lon: float = Field(..., ge=-180, le=180)
    dest_lat: float = Field(..., ge=-90, le=90)
    dest_lon: float = Field(..., ge=-180, le=180)
    risk_aversion: float = Field(
        1.5, ge=0.0, le=8.0,
        description="0 = shortest path. Higher = accept more distance to avoid risk.",
    )
    hour: int = Field(22, ge=0, le=23)
    resolution: int = Field(48, ge=12, le=96)


class OptimiseRequest(BaseModel):
    budget: float = Field(
        4_000_000, gt=0, le=10_000_000_000,
        description="Total budget in rupees.",
    )
    n_areas: int = Field(40, ge=2, le=300)
    risk_filter: Literal["Low", "Medium", "High"] | None = "High"
    levers: list[str] | None = Field(
        None, description="Subset of the intervention catalogue. None = all."
    )
    population_weighted: bool = True
    seed: int = C.RANDOM_STATE


class FeedbackRequest(BaseModel):
    prediction_id: str | None = None
    actual_risk: Literal["Low", "Medium", "High"]
    predicted_risk: Literal["Low", "Medium", "High"] | None = None
    reporter: str | None = Field(None, max_length=80)
    notes: str | None = Field(None, max_length=1000)


class KeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    role: Literal["viewer", "analyst", "admin"] = "viewer"
    expires_in_days: int | None = Field(None, ge=1, le=3650)
    rate_limit_per_minute: int | None = Field(None, ge=1, le=100_000)


# ==========================================================================
# Inference
# ==========================================================================
@router.post("/predict", tags=["v2 · Inference"])
def assess(req: AssessRequest, principal: Principal = Depends(require_viewer),
           svc: RiskService = Depends(get_service)) -> dict:
    """Score an area with a conformal prediction set alongside the point estimate.

    The difference from ``/api/predict`` is the guarantee. A softmax confidence
    is the model's opinion; the conformal set is a statement that holds
    ``1 − alpha`` of the time regardless of whether the model is any good.
    """
    return svc.predict(
        req.area.model_dump(), explain=req.explain, recommend=req.recommend,
        conformal=True, alpha=req.alpha,
    )


@router.post("/predict/uncertainty", tags=["v2 · Inference"])
def uncertainty_curve(area: AreaFeatures,
                      principal: Principal = Depends(require_viewer),
                      svc: RiskService = Depends(get_service)) -> dict:
    """How the prediction set grows as the coverage requirement tightens.

    This is the most honest single view of a model's confidence: at 80%
    coverage the answer may be a clean singleton, and at 99% it may need all
    three bands. Where the set widens is where the model actually stops knowing.
    """
    if svc.conformal is None or not svc.conformal.ready:
        raise HTTPException(
            503,
            "No conformal calibration available. Run:  python run.py --calibrate",
        )

    payload = area.model_dump()
    base = svc.predict(payload, explain=False, recommend=False, conformal=False,
                       observe=False)
    proba = np.array([p["probability"] for p in base["probabilities"]])

    curve = []
    for alpha in (0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01):
        result = svc.conformal.predict_set(proba, alpha=alpha)
        curve.append({
            "alpha": alpha,
            "coverage": round(1 - alpha, 3),
            "prediction_set": result["prediction_set"],
            "set_size": result["set_size"],
            "certain": result["certain"],
        })

    singleton_until = next(
        (c["coverage"] for c in reversed(curve) if c["set_size"] == 1), None
    )
    return {
        "point_prediction": base["risk_level"],
        "softmax_confidence": base["confidence"],
        "probabilities": base["probabilities"],
        "curve": curve,
        "certain_up_to_coverage": singleton_until,
        "calibration": svc.conformal.calibration.to_dict(),
        "interpretation": (
            f"The model commits to a single band up to {singleton_until:.0%} required "
            "coverage; beyond that it must hedge."
            if singleton_until else
            "The model cannot commit to a single band at any tested coverage level — "
            "this input is genuinely ambiguous."
        ),
    }


@router.post("/feedback", tags=["v2 · Inference"])
def submit_feedback(req: FeedbackRequest, request: Request,
                    principal: Principal = Depends(require_analyst)) -> dict:
    """Report what actually happened — the only way concept drift is detectable.

    Covariate drift can be measured without labels. Whether the model is still
    *right* cannot. This endpoint is the loop that closes.
    """
    from .db import FeedbackRecord, database, session_scope

    if not database.init():
        raise HTTPException(503, "Feedback needs the database. Install SQLAlchemy.")

    correct = None
    if req.predicted_risk:
        correct = req.predicted_risk == req.actual_risk

    with session_scope() as s:
        if s is None:
            raise HTTPException(503, "Database unavailable.")
        record = FeedbackRecord(
            prediction_id=req.prediction_id,
            actual_risk=req.actual_risk,
            predicted_risk=req.predicted_risk,
            correct=correct,
            reporter=req.reporter or principal.name,
            notes=req.notes,
        )
        s.add(record)
        s.flush()
        record_id = record.id

    return {"recorded": True, "id": record_id,
            "message": "Thank you — this feeds the next retraining cycle."}


# ==========================================================================
# Route planning
# ==========================================================================
@router.post("/route", tags=["v2 · Planning"])
def plan_route(req: RouteRequest, principal: Principal = Depends(require_viewer),
               svc: RiskService = Depends(get_service)) -> dict:
    """Safest walking route versus shortest, with the trade-off quantified."""
    from .routing import engine

    model = svc._require()
    result = engine.route(
        model,
        origin=(req.origin_lat, req.origin_lon),
        destination=(req.dest_lat, req.dest_lon),
        risk_aversion=req.risk_aversion,
        hour=req.hour,
        resolution=req.resolution,
    )
    if not result.get("available"):
        raise HTTPException(422, result.get("reason", "Route could not be computed."))
    return result


@router.get("/route/grid", tags=["v2 · Planning"])
def risk_grid(hour: int = Query(22, ge=0, le=23),
              resolution: int = Query(40, ge=8, le=96),
              principal: Principal = Depends(require_viewer),
              svc: RiskService = Depends(get_service)) -> dict:
    """The scored risk field itself — the heat layer under the map."""
    from .routing import engine

    grid = engine.build_grid(svc._require(), resolution=resolution, hour=hour)
    return grid.to_payload()


@router.get("/route/timelapse", tags=["v2 · Planning"])
def risk_timelapse(resolution: int = Query(24, ge=8, le=48),
                   step: int = Query(3, ge=1, le=6),
                   principal: Principal = Depends(require_viewer),
                   svc: RiskService = Depends(get_service)) -> dict:
    """The whole city's risk field across 24 hours — an animatable sequence.

    Deliberately coarse by default: the payload is ``resolution² × frames``
    cells, which grows fast. 24×24 at 3-hour steps is ~4,600 cells and renders
    smoothly in a browser.
    """
    from .routing import engine

    model = svc._require()
    frames = []
    for hour in range(0, 24, step):
        grid = engine.build_grid(model, resolution=resolution, hour=hour)
        frames.append({
            "hour": hour,
            "mean_risk": round(float(grid.risk.mean()), 4),
            "max_risk": round(float(grid.risk.max()), 4),
            "high_cells": int((grid.labels == "High").sum()),
            "medium_cells": int((grid.labels == "Medium").sum()),
            "low_cells": int((grid.labels == "Low").sum()),
            "risk": [[round(float(v), 3) for v in row] for row in grid.risk],
        })

    peak = max(frames, key=lambda f: f["mean_risk"])
    calm = min(frames, key=lambda f: f["mean_risk"])
    return {
        "resolution": resolution,
        "step_hours": step,
        "bounds": engine.build_grid(model, resolution=resolution, hour=0).bounds(),
        "frames": frames,
        "peak_hour": peak["hour"],
        "safest_hour": calm["hour"],
        "narrative": (
            f"City-wide risk peaks at {peak['hour']:02d}:00 "
            f"({peak['high_cells']} high-risk cells) and is lowest at "
            f"{calm['hour']:02d}:00 ({calm['high_cells']} high-risk cells)."
        ),
    }


# ==========================================================================
# Optimisation
# ==========================================================================
@router.post("/optimise", tags=["v2 · Planning"])
def optimise(req: OptimiseRequest, principal: Principal = Depends(require_analyst),
             svc: RiskService = Depends(get_service)) -> dict:
    """Allocate a safety budget across areas to maximise risk reduction."""
    from .optimizer import CATALOGUE, InterventionOptimiser, sample_areas

    model = svc._require()
    if req.levers:
        unknown = [k for k in req.levers if k not in CATALOGUE]
        if unknown:
            raise HTTPException(
                422, f"Unknown interventions: {unknown}. Available: {list(CATALOGUE)}"
            )

    areas = sample_areas(
        n=req.n_areas, risk_filter=req.risk_filter, model=model, seed=req.seed
    )
    t0 = time.perf_counter()
    result = InterventionOptimiser(model).optimise(
        areas, budget=req.budget, levers=req.levers,
        population_weighted=req.population_weighted,
    )
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


@router.get("/optimise/catalogue", tags=["v2 · Planning"])
def catalogue(principal: Principal = Depends(require_viewer)) -> dict:
    """The intervention menu, with unit costs and capacity limits."""
    from .optimizer import CATALOGUE

    return {
        "currency": "INR",
        "interventions": [
            {
                "key": i.key, "label": i.label, "unit": i.unit,
                "unit_cost": i.unit_cost, "max_units": i.max_units,
                "category": i.category, "description": i.description,
            }
            for i in CATALOGUE.values()
        ],
    }


@router.post("/optimise/curve", tags=["v2 · Planning"])
def budget_curve(req: OptimiseRequest, principal: Principal = Depends(require_analyst),
                 svc: RiskService = Depends(get_service)) -> dict:
    """Risk reduction as a function of budget — where diminishing returns start.

    This is the chart a city finance committee actually needs: not "what does
    ₹40 lakh buy" but "at what point does the next rupee stop helping".
    """
    from .optimizer import InterventionOptimiser, sample_areas

    model = svc._require()
    areas = sample_areas(n=min(req.n_areas, 60), risk_filter=req.risk_filter,
                         model=model, seed=req.seed)
    optimiser = InterventionOptimiser(model)

    points = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        budget = req.budget * fraction
        result = optimiser.optimise(areas, budget=budget, levers=req.levers,
                                    population_weighted=req.population_weighted)
        points.append({
            "budget": round(budget, 2),
            "budget_lakh": round(budget / 100_000, 2),
            "spent": result["spent"],
            "reduction_percent": result["objective"]["reduction_percent"],
            "areas_improved": result["n_areas_improved"],
            "bands_moved": result["bands_moved"],
            "safety_after": result["safety_score"]["after"],
        })

    # Knee = last point where marginal return per lakh is still above half the
    # best marginal return seen so far.
    best_marginal = 0.0
    knee = points[-1]["budget_lakh"]
    previous = {"budget_lakh": 0.0, "reduction_percent": 0.0}
    for point in points:
        d_budget = point["budget_lakh"] - previous["budget_lakh"]
        marginal = (
            (point["reduction_percent"] - previous["reduction_percent"]) / d_budget
            if d_budget else 0.0
        )
        point["marginal_return_per_lakh"] = round(marginal, 4)
        best_marginal = max(best_marginal, marginal)
        if best_marginal and marginal >= 0.5 * best_marginal:
            knee = point["budget_lakh"]
        previous = point

    return {
        "n_areas": len(areas),
        "points": points,
        "diminishing_returns_at_lakh": knee,
        "narrative": (
            f"Returns stay strong up to about ₹{knee:,.1f} lakh across "
            f"{len(areas)} areas; beyond that each additional lakh buys "
            "materially less risk reduction."
        ),
    }


# ==========================================================================
# Monitoring
# ==========================================================================
@router.get("/monitor/drift", tags=["v2 · Monitoring"])
def drift_report(min_samples: int = Query(30, ge=5, le=5000),
                 principal: Principal = Depends(require_viewer)) -> dict:
    """PSI / KS / JS drift across the live traffic window."""
    from .drift import monitor

    return monitor.report(min_samples=min_samples)


@router.post("/monitor/drift/simulate", tags=["v2 · Monitoring"])
def simulate_drift(shift: float = Query(1.0, ge=0.0, le=5.0,
                                        description="Shift magnitude in σ"),
                   n: int = Query(300, ge=50, le=5000),
                   principal: Principal = Depends(require_analyst),
                   svc: RiskService = Depends(get_service)) -> dict:
    """Inject artificially shifted traffic to prove the detector actually fires.

    A monitor that has never been seen to alarm is indistinguishable from a
    monitor that does not work. This makes the detector demonstrable: push
    synthetic drift through it and watch PSI climb.
    """
    from .data import load_dataset
    from .drift import monitor

    if not monitor.ready:
        raise HTTPException(503, "No drift reference available. Train the model first.")

    svc._require()   # fail fast with 503 if no model is loaded
    df = load_dataset().sample(n=n, random_state=C.RANDOM_STATE)[C.RAW_FEATURE_COLUMNS]
    df = df.reset_index(drop=True).copy()

    if shift > 0:
        # A realistic degradation story: crime reporting up, lighting failing,
        # activity down — exactly the pattern of a neighbourhood deteriorating.
        df["Crime_Count"] = df["Crime_Count"] * (1 + 0.35 * shift)
        df["Harassment_Count"] = df["Harassment_Count"] * (1 + 0.5 * shift)
        df["Broken_Streetlights"] = df["Broken_Streetlights"] * (1 + 0.6 * shift)
        df["Working_Streetlights"] = (df["Working_Streetlights"] * (1 - 0.2 * shift)).clip(lower=0)
        df["Footfall"] = (df["Footfall"] * (1 - 0.25 * shift)).clip(lower=0)
        df["CCTV_Count"] = (df["CCTV_Count"] * (1 - 0.3 * shift)).clip(lower=0)

    out = svc.predict_batch(df)
    report = monitor.report(min_samples=10)
    return {
        "injected": len(df),
        "shift_sigma": shift,
        "resulting_distribution": {
            c: out["labels"].count(c) for c in C.CLASS_ORDER
        },
        "drift": report,
    }


@router.post("/monitor/drift/reset", tags=["v2 · Monitoring"])
def reset_drift(principal: Principal = Depends(require_analyst)) -> dict:
    from .drift import monitor

    monitor.reset()
    return {"reset": True, "message": "Drift window cleared."}


@router.get("/monitor/live", tags=["v2 · Monitoring"])
def live_feed(limit: int = Query(100, ge=1, le=1000),
              hours: int | None = Query(None, ge=1, le=8760),
              principal: Principal = Depends(require_viewer)) -> dict:
    """Recent predictions from the audit store."""
    from .db import database, recent_predictions

    rows = recent_predictions(limit=limit, since_hours=hours)
    if not rows and not database.ready:
        return {
            "available": False,
            "reason": "Persistence is disabled, so there is no history to show.",
            "predictions": [],
        }

    distribution = {c: sum(1 for r in rows if r["risk_level"] == c) for c in C.CLASS_ORDER}
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"]]
    return {
        "available": True,
        "count": len(rows),
        "distribution": distribution,
        "abstention_rate": (
            round(sum(1 for r in rows if r["abstained"]) / len(rows), 4) if rows else 0.0
        ),
        "latency": {
            "mean_ms": round(float(np.mean(latencies)), 2) if latencies else None,
            "p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
        },
        "predictions": rows,
    }


@router.get("/monitor/alerts", tags=["v2 · Monitoring"])
def alerts(limit: int = Query(50, ge=1, le=500),
           unacknowledged_only: bool = True,
           principal: Principal = Depends(require_viewer)) -> dict:
    from sqlalchemy import select

    from .db import AlertRecord, database, session_scope

    if not database.init():
        return {"available": False, "alerts": []}

    with session_scope() as s:
        if s is None:
            return {"available": False, "alerts": []}
        stmt = select(AlertRecord).order_by(AlertRecord.created_at.desc())
        if unacknowledged_only:
            stmt = stmt.where(AlertRecord.acknowledged.is_(False))
        rows = s.execute(stmt.limit(limit)).scalars().all()
        return {
            "available": True,
            "count": len(rows),
            "alerts": [{
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "severity": r.severity, "category": r.category,
                "title": r.title, "detail": r.detail,
                "acknowledged": r.acknowledged,
            } for r in rows],
        }


@router.post("/monitor/alerts/{alert_id}/ack", tags=["v2 · Monitoring"])
def acknowledge(alert_id: str, principal: Principal = Depends(require_analyst)) -> dict:
    from sqlalchemy import select

    from .db import AlertRecord, database, session_scope

    if not database.init():
        raise HTTPException(503, "Database unavailable.")
    with session_scope() as s:
        if s is None:
            raise HTTPException(503, "Database unavailable.")
        row = s.execute(
            select(AlertRecord).where(AlertRecord.id == alert_id)
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(404, "No such alert.")
        row.acknowledged = True
        row.acknowledged_by = principal.name
    return {"acknowledged": True, "id": alert_id}


@router.websocket("/monitor/stream")
async def stream(websocket: WebSocket) -> None:
    """Live telemetry over a WebSocket, for the control-room view.

    Push rather than poll: a dashboard polling every second is a request per
    second per open tab, and the payload is mostly unchanged. One socket
    carries the delta and costs nothing when idle.
    """
    await websocket.accept()
    svc = RiskService.instance()
    try:
        while True:
            from .drift import monitor

            payload = {
                "ts": time.time(),
                "predictions_served": svc.predictions_served,
                "uptime_seconds": round(time.time() - svc.started_at, 1),
                "cache": prediction_cache.stats(),
                "drift": {
                    "observed": monitor.n_observed,
                    "window": monitor.window,
                    "status": (monitor.last_report or {}).get("status", "unknown"),
                },
                "model_ready": svc.ready,
            }
            await websocket.send_json(payload)
            await asyncio.sleep(2.0)
    except Exception:
        # Any disconnect ends the loop; nothing to clean up beyond the socket.
        return


# ==========================================================================
# Fairness
# ==========================================================================
@router.get("/fairness", tags=["v2 · Governance"])
def fairness_audit(sample: int = Query(6000, ge=500, le=20_000),
                   positive_class: Literal["Low", "Medium", "High"] = "High",
                   principal: Principal = Depends(require_viewer),
                   svc: RiskService = Depends(get_service)) -> dict:
    """Bias audit across density, infrastructure, time, land use and policing."""
    from .fairness import audit_from_dataset

    try:
        return audit_from_dataset(
            svc._require(), sample=sample, positive_class=positive_class
        )
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/fairness/dimensions", tags=["v2 · Governance"])
def fairness_dimensions(principal: Principal = Depends(require_viewer)) -> dict:
    from .fairness import GROUP_DEFINITIONS

    return {
        "dimensions": [
            {"key": k, "label": v["label"], "rationale": v["rationale"]}
            for k, v in GROUP_DEFINITIONS.items()
        ]
    }


@router.get("/conformal", tags=["v2 · Governance"])
def conformal_status(principal: Principal = Depends(require_viewer),
                     svc: RiskService = Depends(get_service)) -> dict:
    """The calibration artefact and its verified coverage."""
    if svc.conformal is None or svc.conformal.calibration is None:
        return {
            "available": False,
            "reason": "Not calibrated. Run:  python run.py --calibrate",
        }
    return {"available": True, **svc.conformal.calibration.to_dict()}


# ==========================================================================
# Registry
# ==========================================================================
@router.get("/registry", tags=["v2 · Registry"])
def list_registry(principal: Principal = Depends(require_viewer)) -> dict:
    from .registry import registry, shadow

    return {
        "summary": registry.summary(),
        "versions": registry.list_versions(),
        "shadow": shadow.stats(),
    }


@router.get("/registry/verify", tags=["v2 · Registry"])
def verify_registry(principal: Principal = Depends(require_viewer)) -> dict:
    """Confirm every registered artefact still hashes to what was recorded."""
    from .registry import registry

    results = registry.verify_all()
    failed = [r for r in results if not r["verified"]]
    return {
        "checked": len(results),
        "failed": len(failed),
        "integrity": "intact" if not failed else "compromised",
        "results": results,
    }


@router.post("/registry/{version}/stage", tags=["v2 · Registry"])
def set_stage(version: str,
              stage: Literal["staging", "challenger", "champion", "archived"],
              principal: Principal = Depends(require_admin)) -> dict:
    from .registry import registry, shadow

    try:
        mv = registry.set_stage(version, stage, actor=principal.name)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc

    if stage == "challenger":
        shadow.load(mv)
    elif shadow.version == version:
        shadow.unload()

    return {"version": version, "stage": stage, "detail": mv.to_dict()}


@router.get("/registry/{version}/gate", tags=["v2 · Registry"])
def promotion_gate(version: str, principal: Principal = Depends(require_viewer)) -> dict:
    """Would this version be allowed to become champion, and if not, why not?"""
    from .registry import registry, shadow

    return registry.promotion_gate(version, shadow=shadow)


@router.post("/registry/{version}/promote", tags=["v2 · Registry"])
def promote(version: str, force: bool = Query(False),
            principal: Principal = Depends(require_admin)) -> dict:
    from .registry import registry, shadow

    try:
        result = registry.promote(version, actor=principal.name, force=force, shadow=shadow)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    if result["promoted"]:
        prediction_cache.invalidate()
    return result


@router.post("/registry/prune", tags=["v2 · Registry"])
def prune_registry(keep_archived: int = Query(3, ge=0, le=50),
                   principal: Principal = Depends(require_admin)) -> dict:
    """Reclaim disk from superseded artefacts, keeping rollback targets."""
    from .registry import registry

    return registry.prune(keep_archived=keep_archived, actor=principal.name)


@router.post("/registry/rollback", tags=["v2 · Registry"])
def rollback(principal: Principal = Depends(require_admin)) -> dict:
    from .registry import registry

    result = registry.rollback(actor=principal.name)
    if result["rolled_back"]:
        prediction_cache.invalidate()
    return result


# ==========================================================================
# Admin
# ==========================================================================
@router.get("/admin/keys", tags=["v2 · Admin"])
def list_keys(principal: Principal = Depends(require_admin)) -> dict:
    from .security import keystore

    return {"keys": keystore.list_keys()}


@router.post("/admin/keys", tags=["v2 · Admin"])
def create_key(req: KeyRequest, principal: Principal = Depends(require_admin)) -> dict:
    from .db import record_audit
    from .security import keystore

    try:
        created = keystore.create(
            req.name, req.role, expires_in_days=req.expires_in_days,
            rate_limit_per_minute=req.rate_limit_per_minute,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc

    record_audit(principal.name, "key.create", target=created["id"],
                 detail=f"role={req.role}")
    return created


@router.delete("/admin/keys/{key_id}", tags=["v2 · Admin"])
def revoke_key(key_id: str, principal: Principal = Depends(require_admin)) -> dict:
    from .db import record_audit
    from .security import keystore

    if not keystore.revoke(key_id):
        raise HTTPException(404, "No such key.")
    record_audit(principal.name, "key.revoke", target=key_id)
    return {"revoked": True, "id": key_id}


@router.get("/admin/audit", tags=["v2 · Admin"])
def audit_log(limit: int = Query(100, ge=1, le=1000),
              principal: Principal = Depends(require_admin)) -> dict:
    from sqlalchemy import select

    from .db import AuditRecord, database, session_scope

    if not database.init():
        return {"available": False, "entries": []}
    with session_scope() as s:
        if s is None:
            return {"available": False, "entries": []}
        rows = s.execute(
            select(AuditRecord).order_by(AuditRecord.created_at.desc()).limit(limit)
        ).scalars().all()
        return {
            "available": True,
            "entries": [{
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "actor": r.actor, "action": r.action, "target": r.target,
                "outcome": r.outcome, "detail": r.detail, "ip": r.ip,
            } for r in rows],
        }


@router.post("/admin/cache/clear", tags=["v2 · Admin"])
def clear_cache(principal: Principal = Depends(require_admin)) -> dict:
    stats = prediction_cache.stats()
    prediction_cache.invalidate()
    return {"cleared": True, "entries_removed": stats["size"], "stats_before": stats}


@router.post("/admin/retention/purge", tags=["v2 · Admin"])
def purge(principal: Principal = Depends(require_admin)) -> dict:
    from .db import purge_expired, record_audit

    removed = purge_expired()
    record_audit(principal.name, "retention.purge", detail=f"{removed} rows")
    return {
        "purged": removed,
        "retention_days": settings.retention_days,
        "message": f"Removed {removed} record(s) older than {settings.retention_days} days.",
    }


@router.post("/admin/token", tags=["v2 · Admin"])
def exchange_token(principal: Principal = Depends(require_viewer)) -> dict:
    """Exchange the presented API key for a short-lived bearer token."""
    from .security import issue_token

    try:
        return issue_token(principal)
    except RuntimeError as exc:
        raise HTTPException(501, str(exc)) from exc


# ==========================================================================
# System
# ==========================================================================
@router.get("/system/status", tags=["v2 · System"])
def system_status(principal: Principal = Depends(require_viewer),
                  svc: RiskService = Depends(get_service)) -> dict:
    """Everything the control room needs, in one call."""
    return svc.system_status()


@router.get("/system/readiness", tags=["v2 · System"])
def readiness(principal: Principal = Depends(require_viewer),
              svc: RiskService = Depends(get_service)) -> dict:
    """Production-readiness audit: would it be safe to expose this publicly?"""
    from .db import database
    from .registry import registry

    findings = list(settings.audit())

    if not svc.ready:
        findings.append({
            "level": "critical", "setting": "MODEL",
            "message": "No model artefact is loaded.",
        })
    if svc.conformal is None or not svc.conformal.ready:
        findings.append({
            "level": "medium", "setting": "CONFORMAL",
            "message": "No conformal calibration — responses carry no coverage guarantee. "
                       "Run: python run.py --calibrate",
        })
    if not database.ready:
        findings.append({
            "level": "medium", "setting": "DATABASE",
            "message": "Persistence unavailable — no audit trail is being recorded.",
        })
    if registry.champion() is None:
        findings.append({
            "level": "low", "setting": "REGISTRY",
            "message": "No champion registered; rollback is unavailable. "
                       "Run: python run.py --register",
        })

    weights = {"critical": 25, "high": 12, "medium": 6, "low": 2, "info": 0}
    score = max(0, 100 - sum(weights.get(f["level"], 0) for f in findings))
    blocking = [f for f in findings if f["level"] in ("critical", "high")]

    return {
        "ready_for_production": not blocking,
        "score": score,
        "environment": settings.environment,
        "findings": findings,
        "n_blocking": len(blocking),
        "summary": (
            "No blocking issues — safe to expose with the current configuration."
            if not blocking else
            f"{len(blocking)} blocking issue(s) must be resolved before exposing "
            "this deployment publicly."
        ),
    }


@router.get("/system/settings", tags=["v2 · System"])
def show_settings(principal: Principal = Depends(require_admin)) -> dict:
    """Effective configuration, with every secret redacted."""
    return {"settings": settings.public_dict(), "findings": settings.audit()}


@router.get("/metrics", tags=["v2 · System"], include_in_schema=True)
def prometheus_metrics(svc: RiskService = Depends(get_service)) -> Response:
    """Prometheus exposition. Deliberately unauthenticated — scrapers rarely
    carry credentials, and the payload contains only aggregate counters."""
    from .drift import monitor
    from .observability import metrics

    if metrics.enabled:
        metrics.model_ready.set(1 if svc.ready else 0)
        metrics.uptime.set(time.time() - svc.started_at)
        accuracy = svc.metadata.get("metrics", {}).get("accuracy")
        if accuracy:
            metrics.model_accuracy.set(float(accuracy))
        report = monitor.last_report or {}
        if report.get("features"):
            metrics.drift_psi.set(float(report["features"][0]["psi"]))
        metrics.cache_hit_rate.set(prediction_cache.stats()["hit_rate"])

    from .observability import CONTENT_TYPE_LATEST

    return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)
