"""FastAPI application — the RiskRadar serving layer.

Start it with::

    uvicorn riskradar.api:app --reload      (from the src/ directory)
    python run.py                            (from the project root)

Interactive OpenAPI docs are served at ``/docs``.
"""

from __future__ import annotations

import base64
import contextlib
import io
import logging
import time

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from . import __version__
from . import config as C
from .api_v2 import router as v2_router
from .cache import prediction_cache
from .db import database
from .observability import configure_logging, metrics, new_request_id, request_id_var
from .registry import registry, shadow
from .schemas import (
    AreaFeatures,
    BatchResponse,
    HealthResponse,
    PredictionResponse,
    WhatIfRequest,
)
from .service import ModelNotLoadedError, RiskService, get_service
from .settings import settings

DESCRIPTION = """
**RiskRadar** predicts women's-safety risk for a location-hour from crime,
infrastructure, population and environmental signals.

* `POST /api/predict` — score one area, with SHAP explanation and intervention advice
* `POST /api/predict/batch` — score an uploaded CSV
* `POST /api/what-if` — sweep one input and trace the risk response curve
* `GET  /api/analytics` — pre-aggregated dataset insight for the dashboard
* `GET  /api/model` — model card, leaderboard and feature importance

Every prediction is explainable: the response carries the signed SHAP
contribution of each engineered feature.
"""

@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm every subsystem before the first request lands.

    Building a TreeExplainer takes ~7 seconds. Doing it lazily would make one
    unlucky user pay that cost; doing it at boot makes every request fast. The
    same argument applies to the database schema and the shadow challenger, so
    all of it happens here.
    """
    configure_logging()
    logger = logging.getLogger("riskradar")

    database.init()

    svc = RiskService.instance()
    if svc.ready:
        try:
            svc.predict(AreaFeatures().model_dump(), observe=False)
            svc.predictions_served = 0
            prediction_cache.invalidate()
            logger.info("warm - %s ready", svc.metadata.get("model_name", "model"))
        except Exception as exc:  # pragma: no cover
            logger.warning("warm-up skipped: %s", exc)
    else:
        logger.warning("no model artefact found - run: python run.py --train")

    # Auto-register the serving artefact so the registry is never empty on a
    # fresh checkout — rollback and integrity checks need a baseline entry.
    try:
        if svc.ready and registry.champion() is None and C.MODEL_PATH.exists():
            mv = registry.register(
                C.MODEL_PATH,
                model_name=svc.metadata.get("model_name", "RiskRadar"),
                metrics=svc.metadata.get("metrics", {}),
                notes="Auto-registered from the shipped artefact at first boot.",
            )
            registry.set_stage(mv.version, "champion", actor="system")
            logger.info("registered shipped artefact as champion %s", mv.version)
    except Exception as exc:  # pragma: no cover
        logger.debug("registry bootstrap skipped: %s", exc)

    # Reattach a challenger across restarts so shadow evaluation survives a deploy.
    try:
        challenger = registry.challenger()
        if challenger is not None:
            shadow.load(challenger)
    except Exception:  # pragma: no cover
        pass

    if svc.conformal is None or not svc.conformal.ready:
        logger.info(
            "no conformal calibration - responses carry no coverage guarantee "
            "(fix with: python run.py --calibrate)"
        )

    yield

    database.dispose()


app = FastAPI(
    lifespan=lifespan,
    title="RiskRadar API",
    description=DESCRIPTION,
    version=__version__,
    contact={"name": "N. Venkatesan and Neethivendhan T."},
    license_info={"name": "MIT"},
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # Credentials cannot be combined with a wildcard origin — browsers reject
    # that pairing outright, so the flag has to track the configured origins.
    allow_credentials="*" not in settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-ms", "X-Request-ID"],
)


@app.middleware("http")
async def observe_request(request: Request, call_next):
    """Assign a request ID, time the call, and record it.

    The ID is honoured from an inbound ``X-Request-ID`` when a gateway already
    issued one, so a trace survives across service boundaries instead of being
    renamed at every hop.
    """
    request_id = request.headers.get("x-request-id") or new_request_id()
    token = request_id_var.set(request_id)
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        elapsed = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-ms"] = f"{elapsed:.2f}"
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        metrics.observe_request(
            request.method, request.url.path, status,
            (time.perf_counter() - t0) * 1000,
        )
        request_id_var.reset(token)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Baseline hardening headers.

    The CSP allows inline script and style because the dashboard is one
    self-contained file by design (see :func:`index`); it still blocks
    third-party script origins, which is the part that matters for a page that
    loads no external code.
    """
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "connect-src 'self' ws: wss:; font-src 'self' data:",
    )
    if settings.is_production:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    return response


@app.exception_handler(ModelNotLoadedError)
async def _model_missing(_: Request, exc: ModelNotLoadedError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ==========================================================================
# System
# ==========================================================================
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health(svc: RiskService = Depends(get_service)):
    """Liveness + readiness probe."""
    return svc.health()


@app.get("/api/model", tags=["System"])
def model_card(svc: RiskService = Depends(get_service)):
    """Model card: metrics, leaderboard, feature importance, data quality."""
    if not svc.metadata:
        raise HTTPException(503, "Model metadata not found — run the training pipeline.")
    return {
        "metadata": svc.metadata,
        "leaderboard": svc.leaderboard,
        "schema": {
            "numeric": C.NUMERIC_COLUMNS,
            "categorical": C.CATEGORICAL_CHOICES,
            "classes": C.CLASS_ORDER,
        },
    }


@app.get("/api/analytics", tags=["Insight"])
def analytics(svc: RiskService = Depends(get_service)):
    """Pre-aggregated dataset statistics powering the analytics dashboard."""
    if not svc.analytics:
        raise HTTPException(503, "Analytics not generated — run the training pipeline.")
    return svc.analytics


@app.get("/api/geo", tags=["Insight"])
def geo(
    svc: RiskService = Depends(get_service),
    risk: str | None = Query(None, description="Filter by Low | Medium | High"),
    limit: int = Query(2500, ge=1, le=10_000),
):
    """Class-stratified geo sample for the interactive risk map."""
    points = svc.geo
    if risk:
        points = [p for p in points if p["risk"].lower() == risk.lower()]
    return {"count": len(points[:limit]), "points": points[:limit]}


# ==========================================================================
# Inference
# ==========================================================================
@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(
    payload: AreaFeatures,
    explain: bool = Query(True, description="Include SHAP attribution"),
    recommend: bool = Query(True, description="Include intervention suggestions"),
    svc: RiskService = Depends(get_service),
):
    """Score a single area-hour and explain the verdict."""
    return svc.predict(payload.model_dump(), explain=explain, recommend=recommend)


@app.post("/api/predict/batch", response_model=BatchResponse, tags=["Inference"])
async def predict_batch(
    file: UploadFile = File(..., description="CSV with the RiskRadar feature columns"),
    svc: RiskService = Depends(get_service),
):
    """Score an uploaded CSV and return results plus a downloadable annotated file."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")

    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (50 MB limit).")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(400, "The uploaded file contains no rows.")

    missing = [c for c in C.RAW_FEATURE_COLUMNS if c not in df.columns]
    if len(missing) > len(C.RAW_FEATURE_COLUMNS) // 2:
        raise HTTPException(
            422,
            f"CSV is missing {len(missing)} required columns, e.g. {missing[:6]}. "
            "Download the template from the Batch page.",
        )

    df = df.head(50_000)
    out = svc.predict_batch(df)

    annotated = df.copy()
    annotated["Predicted_Risk"] = out["labels"]
    annotated["Confidence"] = out["confidence"]
    annotated["Safety_Score"] = out["safety_score"]
    buf = io.StringIO()
    annotated.to_csv(buf, index=False)

    distribution = {c: int(out["labels"].count(c)) for c in C.CLASS_ORDER}
    preview = [
        {
            "row": i + 1,
            "risk_level": out["labels"][i],
            "confidence": out["confidence"][i],
            "safety_score": out["safety_score"][i],
        }
        for i in range(min(200, len(out["labels"])))
    ]

    return {
        "rows_processed": len(out["labels"]),
        "latency_ms": out["latency_ms"],
        "distribution": distribution,
        "average_safety_score": round(
            sum(out["safety_score"]) / len(out["safety_score"]), 2
        ),
        "results": preview,
        "csv_base64": base64.b64encode(buf.getvalue().encode()).decode(),
    }


@app.post("/api/what-if", tags=["Inference"])
def what_if(req: WhatIfRequest, svc: RiskService = Depends(get_service)):
    """Sweep a single input across values and trace how risk responds."""
    return svc.what_if(req.base.model_dump(), req.field, req.values)


@app.get("/api/template.csv", tags=["Inference"])
def template():
    """Download a correctly-shaped CSV template for batch scoring."""
    example = AreaFeatures().model_dump()
    buf = io.StringIO()
    pd.DataFrame([example] * 3)[C.RAW_FEATURE_COLUMNS].to_csv(buf, index=False)
    return JSONResponse(
        content={"filename": "riskradar_template.csv", "content": buf.getvalue()}
    )


# ==========================================================================
# v2 platform surface
# ==========================================================================
# Registered here, *before* the SPA catch-all below. Starlette matches routes
# in registration order, so anything added after `/{path:path}` would be
# unreachable — a genuinely nasty bug to diagnose, because the routes appear
# correctly in /docs while returning 404 in practice.
app.include_router(v2_router)


# ==========================================================================
# Frontend (mounted last so /api/* always wins)
# ==========================================================================
NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

INDEX = C.FRONTEND_DIR / "index.html"


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    """Serve the dashboard.

    The whole interface is ONE self-contained file: the stylesheet and the
    controller are inlined into the HTML. That is deliberate. A split frontend
    needs the browser to successfully fetch two more URLs, and when either one
    fails — wrong base path, stale cache, a mount shadowing the route — the
    page renders as raw unstyled HTML and looks catastrophically broken while
    the server is perfectly healthy. With a single file there is nothing left
    to fail: if you can read this page at all, you have the whole interface.

    It also means no StaticFiles mount, so no static handler can ever shadow
    an /api/ route.
    """
    if not INDEX.exists():
        raise HTTPException(500, f"Dashboard file is missing: {INDEX}")
    return HTMLResponse(INDEX.read_text(encoding="utf-8"), headers=NO_CACHE)


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str) -> HTMLResponse:
    """Any unknown path returns the dashboard rather than a bare 404.

    Registered last, so every real /api/ route above still wins.
    """
    if path.startswith("api/"):
        raise HTTPException(404, f"No such API endpoint: /{path}")
    return index()
