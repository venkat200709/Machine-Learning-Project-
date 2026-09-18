"""Persistence layer — the platform's memory.

Why a database at all?
----------------------
A classifier that forgets every prediction the moment it returns cannot be
operated. You cannot answer "was the model behaving strangely last Tuesday?",
you cannot close the feedback loop, you cannot detect drift, and you cannot
prove to an auditor what the system told a citizen on a given night. Those
four capabilities are the difference between a model and a *product*.

Design
------
* **SQLAlchemy 2.0 ORM**, typed with ``Mapped[...]`` annotations.
* **SQLite by default**, PostgreSQL/TimescaleDB in production, selected purely
  by ``RISKRADAR_DATABASE_URL`` — no code changes between the two.
* **Degrades, never crashes.** If SQLAlchemy is not installed or the database
  is unreachable, :func:`session_scope` yields ``None`` and every caller
  no-ops. Losing the audit trail must never take down the safety service; the
  outage is reported through ``/api/v2/system/readiness`` instead.
* **Write path is fire-and-forget.** Logging a prediction happens on a
  background task so a slow disk can never add latency to a request.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

from .settings import settings, sqlite_path

log = logging.getLogger("riskradar.db")

# --------------------------------------------------------------------------
# Optional dependency
# --------------------------------------------------------------------------
try:
    from sqlalchemy import (
        Boolean,
        DateTime,
        Float,
        Index,
        Integer,
        String,
        Text,
        create_engine,
        delete,
        func,
        select,
    )
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
    from sqlalchemy.pool import StaticPool

    HAS_SQLALCHEMY = True
except Exception:  # pragma: no cover - exercised only on minimal installs
    HAS_SQLALCHEMY = False

    class DeclarativeBase:  # type: ignore[no-redef]
        """Stub so the module still imports without SQLAlchemy."""


def utcnow() -> datetime:
    """Timezone-aware UTC. Naive datetimes in a database are a future bug."""
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


# ==========================================================================
# Schema
# ==========================================================================
if HAS_SQLALCHEMY:

    class Base(DeclarativeBase):
        pass

    class PredictionRecord(Base):
        """One scored request. The backbone of monitoring, drift and audit."""

        __tablename__ = "predictions"

        id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), default=utcnow, index=True
        )
        request_id: Mapped[str | None] = mapped_column(String(36), index=True)
        api_key_id: Mapped[str | None] = mapped_column(String(32), index=True)

        model_name: Mapped[str] = mapped_column(String(80))
        model_version: Mapped[str] = mapped_column(String(40), index=True)

        risk_level: Mapped[str] = mapped_column(String(10), index=True)
        risk_index: Mapped[int] = mapped_column(Integer)
        confidence: Mapped[float] = mapped_column(Float)
        safety_score: Mapped[float] = mapped_column(Float)
        prob_low: Mapped[float] = mapped_column(Float, default=0.0)
        prob_medium: Mapped[float] = mapped_column(Float, default=0.0)
        prob_high: Mapped[float] = mapped_column(Float, default=0.0)

        # Conformal prediction set — how *honest* the model was being.
        conformal_set: Mapped[str | None] = mapped_column(String(40))
        conformal_size: Mapped[int | None] = mapped_column(Integer)
        abstained: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

        # Denormalised hot columns: drift and geo queries hit these constantly
        # and should never have to parse a JSON blob to read one number.
        latitude: Mapped[float | None] = mapped_column(Float)
        longitude: Mapped[float | None] = mapped_column(Float)
        hour: Mapped[int | None] = mapped_column(Integer, index=True)
        crime_count: Mapped[float | None] = mapped_column(Float)
        threat_score: Mapped[float | None] = mapped_column(Float)

        latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
        cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
        source: Mapped[str] = mapped_column(String(20), default="api", index=True)

        # Only populated when RISKRADAR_PERSIST_PAYLOADS is on.
        payload_json: Mapped[str | None] = mapped_column(Text)
        top_drivers: Mapped[str | None] = mapped_column(Text)

        __table_args__ = (
            Index("ix_pred_created_risk", "created_at", "risk_level"),
            Index("ix_pred_version_created", "model_version", "created_at"),
        )

    class FeedbackRecord(Base):
        """Ground truth reported after the fact — closes the learning loop."""

        __tablename__ = "feedback"

        id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), default=utcnow, index=True
        )
        prediction_id: Mapped[str | None] = mapped_column(String(32), index=True)
        actual_risk: Mapped[str] = mapped_column(String(10), index=True)
        predicted_risk: Mapped[str | None] = mapped_column(String(10))
        correct: Mapped[bool | None] = mapped_column(Boolean, index=True)
        reporter: Mapped[str | None] = mapped_column(String(80))
        notes: Mapped[str | None] = mapped_column(Text)

    class ApiKeyRecord(Base):
        """Hashed API credentials with a role and a revocation flag."""

        __tablename__ = "api_keys"

        id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
        name: Mapped[str] = mapped_column(String(80))
        # The plaintext key is shown once at creation and never stored.
        key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
        prefix: Mapped[str] = mapped_column(String(12), index=True)
        role: Mapped[str] = mapped_column(String(20), default="viewer", index=True)
        rate_limit_per_minute: Mapped[int | None] = mapped_column(Integer)
        active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
        expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
        last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
        call_count: Mapped[int] = mapped_column(Integer, default=0)

    class AuditRecord(Base):
        """Append-only trail of every state-changing action."""

        __tablename__ = "audit_log"

        id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), default=utcnow, index=True
        )
        actor: Mapped[str] = mapped_column(String(80), index=True)
        action: Mapped[str] = mapped_column(String(60), index=True)
        target: Mapped[str | None] = mapped_column(String(120))
        outcome: Mapped[str] = mapped_column(String(20), default="success")
        detail: Mapped[str | None] = mapped_column(Text)
        ip: Mapped[str | None] = mapped_column(String(64))

    class DriftRecord(Base):
        """A drift evaluation at a point in time, per feature."""

        __tablename__ = "drift_snapshots"

        id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), default=utcnow, index=True
        )
        feature: Mapped[str] = mapped_column(String(60), index=True)
        psi: Mapped[float] = mapped_column(Float)
        ks_statistic: Mapped[float | None] = mapped_column(Float)
        js_distance: Mapped[float | None] = mapped_column(Float)
        status: Mapped[str] = mapped_column(String(12), index=True)
        window_size: Mapped[int] = mapped_column(Integer)
        model_version: Mapped[str | None] = mapped_column(String(40))

    class ModelVersionRecord(Base):
        """Model registry entry — what exists, and what is serving traffic."""

        __tablename__ = "model_versions"

        id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
        version: Mapped[str] = mapped_column(String(40), unique=True, index=True)
        model_name: Mapped[str] = mapped_column(String(80))
        stage: Mapped[str] = mapped_column(String(20), default="staging", index=True)
        accuracy: Mapped[float | None] = mapped_column(Float)
        f1_macro: Mapped[float | None] = mapped_column(Float)
        roc_auc: Mapped[float | None] = mapped_column(Float)
        ece: Mapped[float | None] = mapped_column(Float)
        artefact_path: Mapped[str | None] = mapped_column(String(300))
        artefact_sha256: Mapped[str | None] = mapped_column(String(64))
        notes: Mapped[str | None] = mapped_column(Text)
        promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
        promoted_by: Mapped[str | None] = mapped_column(String(80))

    class AlertRecord(Base):
        """Operational alerts raised by the monitors."""

        __tablename__ = "alerts"

        id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), default=utcnow, index=True
        )
        severity: Mapped[str] = mapped_column(String(12), index=True)
        category: Mapped[str] = mapped_column(String(30), index=True)
        title: Mapped[str] = mapped_column(String(160))
        detail: Mapped[str | None] = mapped_column(Text)
        acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
        acknowledged_by: Mapped[str | None] = mapped_column(String(80))

else:  # pragma: no cover
    Base = None  # type: ignore[assignment]
    PredictionRecord = FeedbackRecord = ApiKeyRecord = None  # type: ignore[assignment]
    AuditRecord = DriftRecord = ModelVersionRecord = AlertRecord = None  # type: ignore[assignment]


# ==========================================================================
# Engine management
# ==========================================================================
class Database:
    """Lazily-initialised engine holder with an explicit health story."""

    def __init__(self) -> None:
        self._engine = None
        self._sessionmaker = None
        self._lock = threading.Lock()
        self._ready = False
        self.error: str | None = None
        self.url = settings.database_url

    # ------------------------------------------------------------------
    def init(self) -> bool:
        """Create the engine and the schema. Idempotent, thread-safe."""
        if self._ready:
            return True
        if not HAS_SQLALCHEMY:
            self.error = (
                "SQLAlchemy is not installed, so prediction history, drift "
                "tracking and the audit trail are disabled. "
                "Enable them with:  pip install 'sqlalchemy>=2.0'"
            )
            return False

        with self._lock:
            if self._ready:
                return True

            attempts = [self.url]
            # SQLite needs POSIX byte-range locks, which OneDrive, Dropbox,
            # network shares and some container mounts do not implement — the
            # symptom is a bare "disk I/O error" on the very first write. That
            # is a configuration problem, not a reason to lose the audit trail,
            # so fall back to local temp storage and say so loudly.
            if self.url.startswith("sqlite") and ":memory:" not in self.url:
                attempts.append(_fallback_sqlite_url())

            for attempt, url in enumerate(attempts):
                try:
                    self._connect(url)
                    self.url = url
                    self._ready = True
                    self.error = None
                    if attempt:
                        log.warning(
                            "the project folder does not support SQLite locking "
                            "(likely a synced or network drive) — the database now "
                            "lives at %s. Set RISKRADAR_DATABASE_URL to choose your own "
                            "location, or use PostgreSQL.", url,
                        )
                    else:
                        log.info("database ready (%s)", settings.database_backend)
                    return True
                except Exception as exc:
                    self.error = f"{exc.__class__.__name__}: {exc}"
                    self._engine = None

            log.warning("database unavailable — running stateless (%s)", self.error)
            self._ready = False
            return False

    def _connect(self, url: str) -> None:
        """Build the engine, create the schema, verify a write actually lands."""
        kwargs: dict[str, Any] = {"echo": settings.db_echo, "future": True}
        if url.startswith("sqlite"):
            # FastAPI serves requests on a threadpool, so the connection is
            # legitimately used from several threads.
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 10}
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
            path = sqlite_path(url)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
        else:
            kwargs["pool_size"] = settings.db_pool_size
            kwargs["pool_pre_ping"] = True

        self._engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            self._tune_sqlite()
        Base.metadata.create_all(self._engine)
        self._sessionmaker = sessionmaker(
            bind=self._engine, expire_on_commit=False, class_=Session
        )

    def _tune_sqlite(self) -> None:
        """WAL + relaxed sync: concurrent readers during writes, ~10x faster.

        Safe here because the durability we would trade away is one lost
        prediction *log entry* on power failure, not a lost transaction.

        Runs before ``create_all`` on purpose: a filesystem that cannot do
        SQLite locking fails here, on a cheap PRAGMA, rather than part-way
        through creating eight tables.
        """
        from sqlalchemy import text

        with self._engine.begin() as conn:
            for pragma in (
                "PRAGMA journal_mode=WAL",
                "PRAGMA synchronous=NORMAL",
                "PRAGMA busy_timeout=5000",
                "PRAGMA foreign_keys=ON",
            ):
                conn.execute(text(pragma))

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready

    def session(self):
        if not self.init():
            return None
        return self._sessionmaker()

    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "enabled": HAS_SQLALCHEMY,
            "ready": self._ready,
            "backend": settings.database_backend,
            "url": _redacted(self.url),
            "error": self.error,
        }
        if self._ready:
            try:
                with self._engine.connect() as conn:
                    from sqlalchemy import text

                    conn.execute(text("SELECT 1"))
                info["reachable"] = True
                info["tables"] = sorted(Base.metadata.tables)
            except Exception as exc:
                info["reachable"] = False
                info["error"] = f"{exc.__class__.__name__}: {exc}"
        return info

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
        self._engine = None
        self._sessionmaker = None
        self._ready = False


def _fallback_sqlite_url() -> str:
    """A database location that is guaranteed to be a real local filesystem."""
    import tempfile
    from pathlib import Path

    target = Path(tempfile.gettempdir()) / "riskradar" / "riskradar.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{target.as_posix()}"


def _redacted(url: str) -> str:
    from .settings import redact_url

    return redact_url(url)


database = Database()


@contextlib.contextmanager
def session_scope() -> Iterator[Any]:
    """Transactional session, or ``None`` when persistence is unavailable.

    Callers write ``with session_scope() as s: if s is None: return`` — one
    branch, and the feature silently degrades instead of raising.
    """
    session = database.session()
    if session is None:
        yield None
        return
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        log.exception("database transaction rolled back")
        raise
    finally:
        session.close()


# ==========================================================================
# Repository helpers
# ==========================================================================
def log_prediction(result: dict, *, payload: dict | None = None, request_id: str | None = None,
                   api_key_id: str | None = None, source: str = "api",
                   cache_hit: bool = False) -> str | None:
    """Persist one prediction. Returns the row id, or None if disabled."""
    if not settings.persist_predictions or not database.init():
        return None

    probs = {p["label"]: p["probability"] for p in result.get("probabilities", [])}
    conformal = result.get("conformal") or {}
    drivers = (result.get("explanation") or {}).get("drivers") or []

    record_id = new_id()
    try:
        with session_scope() as s:
            if s is None:
                return None
            s.add(PredictionRecord(
                id=record_id,
                request_id=request_id,
                api_key_id=api_key_id,
                model_name=str(result.get("model_name", "unknown"))[:80],
                model_version=str(result.get("model_version", "0"))[:40],
                risk_level=result.get("risk_level", "Unknown")[:10],
                risk_index=int(result.get("risk_index", 0)),
                confidence=float(result.get("confidence", 0.0)),
                safety_score=float(result.get("safety_score", 0.0)),
                prob_low=float(probs.get("Low", 0.0)),
                prob_medium=float(probs.get("Medium", 0.0)),
                prob_high=float(probs.get("High", 0.0)),
                conformal_set=",".join(conformal.get("prediction_set", []))[:40] or None,
                conformal_size=conformal.get("set_size"),
                abstained=bool(conformal.get("abstain", False)),
                latitude=_num(payload, "Latitude"),
                longitude=_num(payload, "Longitude"),
                hour=int(payload["Hour"]) if payload and "Hour" in payload else None,
                crime_count=_num(payload, "Crime_Count"),
                threat_score=_driver_value(drivers, "Threat_Score"),
                latency_ms=float(result.get("latency_ms", 0.0)),
                cache_hit=cache_hit,
                source=source[:20],
                payload_json=json.dumps(payload) if (payload and settings.persist_payloads) else None,
                top_drivers=json.dumps([
                    {"f": d["feature"], "i": d["impact"]} for d in drivers[:5]
                ]) if drivers else None,
            ))
        return record_id
    except Exception:  # pragma: no cover - never let logging break inference
        log.exception("failed to persist prediction")
        return None


def _num(payload: dict | None, key: str) -> float | None:
    if not payload or key not in payload:
        return None
    try:
        return float(payload[key])
    except (TypeError, ValueError):
        return None


def _driver_value(drivers: list[dict], feature: str) -> float | None:
    for d in drivers:
        if d.get("feature") == feature:
            return float(d.get("value", 0.0))
    return None


def record_audit(actor: str, action: str, *, target: str | None = None,
                 outcome: str = "success", detail: str | None = None,
                 ip: str | None = None) -> None:
    """Append to the audit trail. Silent on failure by design."""
    if not database.init():
        return
    try:
        with session_scope() as s:
            if s is None:
                return
            s.add(AuditRecord(
                actor=actor[:80], action=action[:60],
                target=(target or None), outcome=outcome[:20],
                detail=detail, ip=(ip or None),
            ))
    except Exception:  # pragma: no cover
        log.exception("failed to write audit record")


def raise_alert(severity: str, category: str, title: str, detail: str | None = None) -> None:
    """Record an operational alert, de-duplicating recent identical titles."""
    if not database.init():
        return
    try:
        with session_scope() as s:
            if s is None:
                return
            cutoff = utcnow() - timedelta(hours=1)
            existing = s.execute(
                select(func.count()).select_from(AlertRecord)
                .where(AlertRecord.title == title[:160])
                .where(AlertRecord.created_at >= cutoff)
                .where(AlertRecord.acknowledged.is_(False))
            ).scalar_one()
            if existing:
                return
            s.add(AlertRecord(
                severity=severity[:12], category=category[:30],
                title=title[:160], detail=detail,
            ))
    except Exception:  # pragma: no cover
        log.exception("failed to raise alert")


def recent_predictions(limit: int = 500, *, since_hours: int | None = None) -> list[dict]:
    """Most recent predictions as plain dicts, newest first."""
    if not database.init():
        return []
    try:
        with session_scope() as s:
            if s is None:
                return []
            stmt = select(PredictionRecord).order_by(PredictionRecord.created_at.desc())
            if since_hours:
                stmt = stmt.where(
                    PredictionRecord.created_at >= utcnow() - timedelta(hours=since_hours)
                )
            rows = s.execute(stmt.limit(limit)).scalars().all()
            return [_prediction_dict(r) for r in rows]
    except Exception:  # pragma: no cover
        log.exception("failed to read predictions")
        return []


def _prediction_dict(r: Any) -> dict:
    return {
        "id": r.id,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "risk_level": r.risk_level,
        "confidence": round(r.confidence, 4),
        "safety_score": r.safety_score,
        "probabilities": {"Low": r.prob_low, "Medium": r.prob_medium, "High": r.prob_high},
        "conformal_set": r.conformal_set,
        "abstained": r.abstained,
        "latitude": r.latitude,
        "longitude": r.longitude,
        "hour": r.hour,
        "latency_ms": r.latency_ms,
        "cache_hit": r.cache_hit,
        "model_version": r.model_version,
        "source": r.source,
    }


def purge_expired() -> int:
    """Delete records past the retention window. Returns rows removed."""
    if not database.init():
        return 0
    cutoff = utcnow() - timedelta(days=settings.retention_days)
    removed = 0
    try:
        with session_scope() as s:
            if s is None:
                return 0
            for table in (PredictionRecord, DriftRecord, AlertRecord):
                res = s.execute(delete(table).where(table.created_at < cutoff))
                removed += int(res.rowcount or 0)
    except Exception:  # pragma: no cover
        log.exception("retention purge failed")
    return removed
