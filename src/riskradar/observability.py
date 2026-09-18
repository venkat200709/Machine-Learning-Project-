"""Observability — metrics, structured logs and request correlation.

Three questions an operator asks at 2am
----------------------------------------
1. *Is it up and how fast is it?* → Prometheus metrics, scraped at ``/metrics``.
2. *What happened to this one request?* → a request ID, generated at the edge,
   returned in a response header and attached to every log line and database
   row it touches.
3. *Is the model behaving?* → prediction counters by class, confidence
   histograms, abstention rate, cache hit rate, shadow disagreement.

The third is the one most services omit, and it is the only one that catches a
model failing quietly. A service can be perfectly healthy by every
infrastructure metric while emitting nonsense.

Histogram buckets are chosen for this workload, not copied from a template:
latency in milliseconds around a 20–200 ms working range, and confidence
bucketed finely near 1.0 where a multiclass model actually lives.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

from .settings import settings

# Correlates every log line and metric emitted while serving one request.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    HAS_PROMETHEUS = True
except Exception:  # pragma: no cover
    HAS_PROMETHEUS = False
    CONTENT_TYPE_LATEST = "text/plain"


# ==========================================================================
# Metrics
# ==========================================================================
class Metrics:
    """Prometheus collectors, with a no-op fallback.

    A dedicated registry rather than the global default: the global registry
    raises on duplicate registration, which breaks every test that constructs
    a second app instance.
    """

    def __init__(self) -> None:
        self.enabled = HAS_PROMETHEUS and settings.metrics_enabled
        if not self.enabled:
            return

        self.registry = CollectorRegistry()
        ns = "riskradar"

        self.requests = Counter(
            f"{ns}_http_requests_total", "HTTP requests processed",
            ["method", "path", "status"], registry=self.registry,
        )
        self.request_latency = Histogram(
            f"{ns}_http_request_duration_ms", "HTTP request latency (ms)",
            ["method", "path"],
            buckets=(1, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 5000),
            registry=self.registry,
        )
        self.predictions = Counter(
            f"{ns}_predictions_total", "Predictions served",
            ["risk_level", "source"], registry=self.registry,
        )
        self.prediction_latency = Histogram(
            f"{ns}_prediction_duration_ms", "Model inference latency (ms)",
            buckets=(1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280),
            registry=self.registry,
        )
        self.confidence = Histogram(
            f"{ns}_prediction_confidence", "Predicted-class probability",
            # Dense near 1.0: that is where a well-calibrated multiclass
            # model spends almost all of its mass, and coarse buckets there
            # would hide exactly the degradation we care about.
            buckets=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0),
            registry=self.registry,
        )
        self.abstentions = Counter(
            f"{ns}_conformal_abstentions_total",
            "Predictions where the conformal set was not a singleton",
            ["kind"], registry=self.registry,
        )
        self.cache_events = Counter(
            f"{ns}_cache_events_total", "Inference cache events",
            ["event"], registry=self.registry,
        )
        self.errors = Counter(
            f"{ns}_errors_total", "Handled errors", ["kind"], registry=self.registry,
        )
        self.rate_limited = Counter(
            f"{ns}_rate_limited_total", "Requests rejected by the rate limiter",
            registry=self.registry,
        )
        self.shadow_disagreements = Counter(
            f"{ns}_shadow_disagreements_total",
            "Challenger disagreed with champion", ["severity"], registry=self.registry,
        )

        self.model_accuracy = Gauge(
            f"{ns}_model_accuracy", "Hold-out accuracy of the serving model",
            registry=self.registry,
        )
        self.model_ready = Gauge(
            f"{ns}_model_ready", "1 when a model is loaded", registry=self.registry,
        )
        self.drift_psi = Gauge(
            f"{ns}_drift_psi_max", "Highest feature PSI in the live window",
            registry=self.registry,
        )
        self.uptime = Gauge(
            f"{ns}_uptime_seconds", "Process uptime", registry=self.registry,
        )
        self.cache_hit_rate = Gauge(
            f"{ns}_cache_hit_rate", "Rolling inference cache hit rate",
            registry=self.registry,
        )

    # ------------------------------------------------------------------
    def observe_request(self, method: str, path: str, status: int, ms: float) -> None:
        if not self.enabled:
            return
        route = _normalise_path(path)
        self.requests.labels(method=method, path=route, status=str(status)).inc()
        self.request_latency.labels(method=method, path=route).observe(ms)
        if status == 429:
            self.rate_limited.inc()
        elif status >= 500:
            self.errors.labels(kind="server").inc()
        elif status >= 400:
            self.errors.labels(kind="client").inc()

    def observe_prediction(self, result: dict, *, source: str = "api",
                           cache_hit: bool = False) -> None:
        if not self.enabled:
            return
        self.predictions.labels(
            risk_level=result.get("risk_level", "unknown"), source=source
        ).inc()
        self.prediction_latency.observe(float(result.get("latency_ms", 0.0)))
        self.confidence.observe(float(result.get("confidence", 0.0)))
        self.cache_events.labels(event="hit" if cache_hit else "miss").inc()

        conformal = result.get("conformal") or {}
        if conformal.get("out_of_distribution"):
            self.abstentions.labels(kind="empty_set").inc()
        elif conformal.get("abstain"):
            self.abstentions.labels(kind="ambiguous").inc()

    def render(self) -> bytes:
        if not self.enabled:
            return b"# prometheus_client is not installed\n"
        return generate_latest(self.registry)


def _normalise_path(path: str) -> str:
    """Collapse identifiers so the cardinality of the path label stays bounded.

    ``/api/v2/predictions/ab12cd34`` and a million siblings must not each
    become their own time series — that is the classic way to melt a
    Prometheus server.
    """
    parts = []
    for segment in path.split("/"):
        if len(segment) >= 16 and all(c in "0123456789abcdefABCDEF-" for c in segment):
            parts.append(":id")
        elif segment.isdigit():
            parts.append(":n")
        else:
            parts.append(segment)
    return "/".join(parts) or "/"


metrics = Metrics()


# ==========================================================================
# Structured logging
# ==========================================================================
class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the request ID attached automatically."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": settings.service_name,
            "env": settings.environment,
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload, default=str)


class PlainFormatter(logging.Formatter):
    """Human-readable, with the request ID only when there is one."""

    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get()
        suffix = f"  [{rid[:8]}]" if rid and rid != "-" else ""
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} " \
               f"{record.name:<22} {record.getMessage()}{suffix}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


_configured = False


def configure_logging(force: bool = False) -> None:
    """Install the formatter on the root logger. Idempotent."""
    global _configured
    if _configured and not force:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.json_logs else PlainFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # uvicorn installs its own handlers; let them propagate to ours instead so
    # there is exactly one log format in the output.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    # SQLAlchemy's INFO level is a full SQL echo — far too loud by default.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    _configured = True


def log_with(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emit a log line carrying structured fields."""
    logger.log(level, message, extra={"extra_fields": fields})


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


# ==========================================================================
# Health scoring
# ==========================================================================
def health_score(*, model_ready: bool, drift_status: str | None,
                 error_rate: float, cache_hit_rate: float,
                 db_ready: bool) -> dict:
    """Roll the signals into one 0–100 number an operator can act on.

    Weighted by consequence, not by convenience: a missing model is total
    failure, drift is serious, a cold cache is cosmetic.
    """
    score = 100.0
    reasons: list[str] = []

    if not model_ready:
        score -= 60
        reasons.append("No model is loaded — the service cannot score requests.")
    if drift_status == "alert":
        score -= 25
        reasons.append("Significant input drift; predictions may be unreliable.")
    elif drift_status == "warning":
        score -= 10
        reasons.append("Moderate input drift detected.")
    if error_rate > 0.05:
        score -= 20
        reasons.append(f"Elevated error rate ({error_rate:.1%}).")
    elif error_rate > 0.01:
        score -= 5
        reasons.append(f"Error rate slightly raised ({error_rate:.1%}).")
    if not db_ready:
        score -= 8
        # Actionable, not just descriptive: the two realistic causes are a
        # missing package and a filesystem that cannot do SQLite locking, and
        # the operator cannot tell which from "unavailable" alone.
        reasons.append(
            "Persistence unavailable — history and audit are not being recorded. "
            "Fix with: pip install 'sqlalchemy>=2.0', or set RISKRADAR_DATABASE_URL "
            "to a local (non-synced) path."
        )
    if cache_hit_rate < 0.05:
        score -= 2
        reasons.append("Cache is cold.")

    score = max(0.0, min(100.0, score))
    grade = (
        "healthy" if score >= 90 else
        "degraded" if score >= 70 else
        "impaired" if score >= 40 else "critical"
    )
    return {
        "score": round(score, 1),
        "grade": grade,
        "reasons": reasons or ["All systems nominal."],
    }
