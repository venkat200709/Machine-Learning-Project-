"""Runtime settings — twelve-factor configuration for the platform layer.

Why a second config module?
---------------------------
``config.py`` holds *scientific* constants: the label schema, the feature
contract, the random seed. Those are properties of the **model** and must be
identical on every machine or the artefact is invalid.

This module holds *operational* settings: database URLs, rate limits, secrets,
feature flags. Those are properties of the **deployment** and must differ
between a laptop and production. Mixing the two is how a training seed ends up
in an environment variable and a database password ends up in git.

Deliberately zero-dependency
----------------------------
This is read at import time by every other module, including the launcher's
pre-flight check. If it depended on ``pydantic-settings`` then a missing
package would turn into an ImportError *before* the friendly diagnostics could
run. Standard library only, so it cannot fail.

Every setting is overridable by an environment variable prefixed ``RISKRADAR_``::

    RISKRADAR_DATABASE_URL=postgresql+psycopg://user:pw@db:5432/riskradar
    RISKRADAR_AUTH_ENABLED=true
    RISKRADAR_RATE_LIMIT_PER_MINUTE=600
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from . import config as C

ENV_PREFIX = "RISKRADAR_"


# --------------------------------------------------------------------------
# Typed environment readers
# --------------------------------------------------------------------------
_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f"}


def _env(name: str) -> str | None:
    return os.environ.get(ENV_PREFIX + name.upper())


def env_str(name: str, default: str) -> str:
    raw = _env(name)
    return default if raw is None else raw


def env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    low = raw.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(
        f"{ENV_PREFIX}{name.upper()}={raw!r} is not a boolean. "
        f"Use one of {sorted(_TRUE | _FALSE)}."
    )


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name.upper()}={raw!r} is not an integer.") from exc
    if minimum is not None and value < minimum:
        raise ValueError(
            f"{ENV_PREFIX}{name.upper()}={value} is below the minimum of {minimum}."
        )
    return value


def env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name.upper()}={raw!r} is not a number.") from exc


def env_list(name: str, default: list[str]) -> list[str]:
    raw = _env(name)
    if raw is None:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the deployment configuration.

    Frozen on purpose: a setting that can be mutated at runtime is a setting
    that will be mutated at runtime, and then no two requests share the same
    configuration. Reload by constructing a new instance.
    """

    # -- identity -------------------------------------------------------
    environment: str = field(default_factory=lambda: env_str("ENV", "development"))
    service_name: str = field(default_factory=lambda: env_str("SERVICE_NAME", "riskradar"))

    # -- persistence ----------------------------------------------------
    # SQLite by default so a clean checkout has a working database with no
    # container to start. Point at Postgres/Timescale in production.
    database_url: str = field(
        default_factory=lambda: env_str(
            "DATABASE_URL", f"sqlite:///{(C.PROJECT_ROOT / 'data' / 'riskradar.db').as_posix()}"
        )
    )
    db_echo: bool = field(default_factory=lambda: env_bool("DB_ECHO", False))
    db_pool_size: int = field(default_factory=lambda: env_int("DB_POOL_SIZE", 5, minimum=1))
    persist_predictions: bool = field(
        default_factory=lambda: env_bool("PERSIST_PREDICTIONS", True)
    )
    # Never store the raw feature payload unless explicitly enabled: it is a
    # location plus a timestamp, which is personal data in most jurisdictions.
    persist_payloads: bool = field(default_factory=lambda: env_bool("PERSIST_PAYLOADS", False))
    retention_days: int = field(default_factory=lambda: env_int("RETENTION_DAYS", 90, minimum=1))

    # -- security -------------------------------------------------------
    auth_enabled: bool = field(default_factory=lambda: env_bool("AUTH_ENABLED", False))
    admin_key: str = field(default_factory=lambda: env_str("ADMIN_KEY", ""))
    jwt_secret: str = field(default_factory=lambda: env_str("JWT_SECRET", ""))
    jwt_ttl_minutes: int = field(default_factory=lambda: env_int("JWT_TTL_MINUTES", 720))
    cors_origins: list[str] = field(default_factory=lambda: env_list("CORS_ORIGINS", ["*"]))
    rate_limit_per_minute: int = field(
        default_factory=lambda: env_int("RATE_LIMIT_PER_MINUTE", 240, minimum=1)
    )
    rate_limit_burst: int = field(default_factory=lambda: env_int("RATE_LIMIT_BURST", 60, minimum=1))
    max_upload_mb: int = field(default_factory=lambda: env_int("MAX_UPLOAD_MB", 50, minimum=1))
    max_batch_rows: int = field(
        default_factory=lambda: env_int("MAX_BATCH_ROWS", 50_000, minimum=1)
    )

    # -- inference ------------------------------------------------------
    cache_enabled: bool = field(default_factory=lambda: env_bool("CACHE_ENABLED", True))
    cache_size: int = field(default_factory=lambda: env_int("CACHE_SIZE", 2048, minimum=1))
    cache_ttl_seconds: int = field(default_factory=lambda: env_int("CACHE_TTL_SECONDS", 300))
    conformal_alpha: float = field(default_factory=lambda: env_float("CONFORMAL_ALPHA", 0.10))
    shadow_enabled: bool = field(default_factory=lambda: env_bool("SHADOW_ENABLED", True))

    # -- monitoring -----------------------------------------------------
    metrics_enabled: bool = field(default_factory=lambda: env_bool("METRICS_ENABLED", True))
    json_logs: bool = field(default_factory=lambda: env_bool("JSON_LOGS", False))
    log_level: str = field(default_factory=lambda: env_str("LOG_LEVEL", "INFO"))
    drift_window: int = field(default_factory=lambda: env_int("DRIFT_WINDOW", 500, minimum=30))
    drift_psi_warn: float = field(default_factory=lambda: env_float("DRIFT_PSI_WARN", 0.10))
    drift_psi_alert: float = field(default_factory=lambda: env_float("DRIFT_PSI_ALERT", 0.25))

    # ------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod", "live"}

    @property
    def database_backend(self) -> str:
        """'sqlite' | 'postgresql' | ... — the scheme without the driver suffix."""
        return self.database_url.split(":", 1)[0].split("+", 1)[0]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    # ------------------------------------------------------------------
    def resolved_admin_key(self) -> str:
        """The admin key, generating an ephemeral one if none was configured.

        Returning a random key rather than an empty string means the admin
        surface is *never* accidentally unauthenticated. If the operator did
        not set one, nobody holds it, and the endpoints are simply closed.
        """
        return self.admin_key or _EPHEMERAL_ADMIN_KEY

    def resolved_jwt_secret(self) -> str:
        return self.jwt_secret or _EPHEMERAL_JWT_SECRET

    def audit(self) -> list[dict[str, Any]]:
        """Production-readiness findings, surfaced by ``/api/v2/system/readiness``.

        A flagship deployment should be able to answer "is this safe to expose
        to the internet?" without a human reading the config by eye.
        """
        issues: list[dict[str, Any]] = []

        def add(level: str, setting: str, message: str) -> None:
            issues.append({"level": level, "setting": setting, "message": message})

        if self.is_production:
            if not self.auth_enabled:
                add("critical", "AUTH_ENABLED",
                    "Authentication is disabled in a production environment.")
            if "*" in self.cors_origins:
                add("critical", "CORS_ORIGINS",
                    "CORS allows every origin. Pin this to your dashboard's domain.")
            if not self.admin_key:
                add("critical", "ADMIN_KEY",
                    "No admin key configured; admin routes are closed but unusable.")
            if not self.jwt_secret:
                add("high", "JWT_SECRET",
                    "JWT secret is ephemeral — tokens are invalidated on every restart.")
            if self.database_backend == "sqlite":
                add("high", "DATABASE_URL",
                    "SQLite is single-writer. Use PostgreSQL for concurrent production traffic.")
            if self.persist_payloads:
                add("medium", "PERSIST_PAYLOADS",
                    "Raw request payloads are being stored; confirm this is lawful for your data.")
            if not self.json_logs:
                add("low", "JSON_LOGS",
                    "Plain-text logs are hard to aggregate. Enable JSON logging.")
        else:
            if not self.auth_enabled:
                add("info", "AUTH_ENABLED",
                    "Authentication is off — expected in development.")

        return issues

    def public_dict(self) -> dict[str, Any]:
        """Serialisable view with every secret removed."""
        secret_names = {"admin_key", "jwt_secret"}
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = ("set" if value else "unset") if f.name in secret_names else value
        out["database_backend"] = self.database_backend
        # A URL can carry a password. Never emit the credential portion.
        out["database_url"] = redact_url(self.database_url)
        return out


def redact_url(url: str) -> str:
    """Strip credentials from a connection string before it is logged."""
    if "@" not in url or "//" not in url:
        return url
    scheme, rest = url.split("//", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}//{user}:***@{host}"


# Generated once per process. See ``resolved_admin_key`` for the rationale.
_EPHEMERAL_ADMIN_KEY = "eph_" + secrets.token_urlsafe(32)
_EPHEMERAL_JWT_SECRET = secrets.token_urlsafe(48)

settings = Settings()


def reload_settings() -> Settings:
    """Re-read the environment. Used by tests that patch ``os.environ``."""
    global settings
    settings = Settings()
    return settings


def sqlite_path(url: str) -> Path | None:
    """Filesystem path behind a SQLite URL, or None for other backends."""
    if not url.startswith("sqlite"):
        return None
    tail = url.split("///", 1)[-1]
    return Path(tail) if tail and tail != ":memory:" else None
