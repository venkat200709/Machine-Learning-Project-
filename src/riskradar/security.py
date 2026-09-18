"""Authentication, authorisation and abuse control.

Threat model
------------
This service answers "is this place dangerous for a woman right now?". That
makes two things valuable to an attacker: the *answers* (a scraped risk map is
a target list) and the *availability* (an unavailable safety service fails
exactly when it matters). So the controls here are:

* **Identity** — hashed API keys with a role, expiry and revocation, or short
  lived JWTs issued against them.
* **Authorisation** — a three-tier role ladder checked at the route.
* **Abuse control** — per-identity token-bucket rate limiting, so one noisy
  client cannot starve the rest.
* **Audit** — every admin action lands in an append-only log.

Key handling
------------
Keys are stored as SHA-256 digests, never plaintext, and compared with
:func:`hmac.compare_digest` so a timing side-channel cannot be used to walk the
digest. The plaintext is returned exactly once, at creation.

Everything is **off by default** (``RISKRADAR_AUTH_ENABLED=false``) so a
student cloning the repo gets a working dashboard with no setup. Turning it on
is a single environment variable.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from fastapi import Header, HTTPException, Request

from .db import ApiKeyRecord, database, record_audit, session_scope, utcnow
from .settings import settings

log = logging.getLogger("riskradar.security")

KEY_PREFIX = "rr_"

# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------
# Ordered ladder: a role satisfies any requirement at or below its own rank.
ROLE_RANK: dict[str, int] = {"viewer": 10, "analyst": 20, "admin": 30}
ROLES = tuple(ROLE_RANK)

ROLE_DESCRIPTION = {
    "viewer": "Read-only: scoring, analytics, model card.",
    "analyst": "Adds batch scoring, what-if, optimisation and feedback submission.",
    "admin": "Full control: key management, model promotion, retention, alerts.",
}


def role_satisfies(actual: str, required: str) -> bool:
    return ROLE_RANK.get(actual, 0) >= ROLE_RANK.get(required, 99)


# --------------------------------------------------------------------------
# Principals
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Principal:
    """Who is making this request."""

    id: str
    name: str
    role: str
    anonymous: bool = False
    rate_limit: int | None = None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "role": self.role,
            "anonymous": self.anonymous,
            "permissions": ROLE_DESCRIPTION.get(self.role, ""),
        }


# When auth is disabled every caller is this. Given the *admin* role, because
# a local developer with no key should not be locked out of their own dashboard.
ANONYMOUS = Principal(id="anonymous", name="anonymous", role="admin", anonymous=True)


# --------------------------------------------------------------------------
# Key derivation
# --------------------------------------------------------------------------
def hash_key(raw: str) -> str:
    """SHA-256 of the presented key. Stored value; never reversible."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_key() -> tuple[str, str, str]:
    """Return ``(plaintext, digest, prefix)`` for a fresh credential."""
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, hash_key(raw), raw[:12]


def constant_time_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    """In-process token bucket, keyed by principal.

    Chosen over a fixed window because a fixed window lets a client fire two
    full quotas back-to-back across the boundary. A bucket smooths that out
    while still permitting a genuine burst up to its capacity.

    In-process is the right scope here: each replica limits its own share, and
    a shared Redis counter would put a network hop on the hot path of a
    latency-sensitive safety endpoint. Swap in Redis at the ingress if you run
    many replicas.
    """

    def __init__(self, rate_per_minute: int, burst: int) -> None:
        self.rate = max(rate_per_minute, 1) / 60.0
        self.burst = max(burst, 1)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self.rejected = 0

    def check(self, key: str, *, rate_per_minute: int | None = None) -> tuple[bool, float]:
        """Consume one token. Returns ``(allowed, retry_after_seconds)``."""
        rate = (max(rate_per_minute, 1) / 60.0) if rate_per_minute else self.rate
        capacity = max(self.burst, int(rate * 60) if rate_per_minute else self.burst)
        now = time.monotonic()

        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(capacity), updated=now)
                self._buckets[key] = bucket
                # Opportunistic eviction so an attacker rotating identities
                # cannot grow this dict without bound.
                if len(self._buckets) > 20_000:
                    self._evict(now)

            bucket.tokens = min(capacity, bucket.tokens + (now - bucket.updated) * rate)
            bucket.updated = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0

            self.rejected += 1
            return False, round((1.0 - bucket.tokens) / rate, 2)

    def _evict(self, now: float) -> None:
        stale = [k for k, b in self._buckets.items() if now - b.updated > 300]
        for k in stale:
            self._buckets.pop(k, None)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
            self.rejected = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "tracked_identities": len(self._buckets),
                "rejected_total": self.rejected,
                "rate_per_minute": round(self.rate * 60),
                "burst": self.burst,
            }


limiter = TokenBucketLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)


# --------------------------------------------------------------------------
# Key store
# --------------------------------------------------------------------------
@dataclass
class _CachedKey:
    principal: Principal
    expires: float


class KeyStore:
    """Database-backed API keys with a short in-memory cache.

    Every authenticated request would otherwise be a database round-trip. A
    30-second cache removes that from the hot path; the cost is that a
    revocation takes up to 30 seconds to propagate, which :meth:`invalidate`
    short-circuits for the local process.
    """

    CACHE_TTL = 30.0

    def __init__(self) -> None:
        self._cache: dict[str, _CachedKey] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def create(self, name: str, role: str = "viewer", *,
               expires_in_days: int | None = None,
               rate_limit_per_minute: int | None = None) -> dict:
        """Mint a key. The plaintext in the response is shown exactly once."""
        if role not in ROLE_RANK:
            raise ValueError(f"Unknown role {role!r}. Choose from {list(ROLE_RANK)}.")
        if not database.init():
            raise RuntimeError(
                "API key management needs the database. "
                "Install SQLAlchemy or set RISKRADAR_DATABASE_URL."
            )

        raw, digest, prefix = generate_key()
        expires_at = (
            utcnow() + timedelta(days=expires_in_days) if expires_in_days else None
        )
        with session_scope() as s:
            if s is None:
                raise RuntimeError("Database unavailable.")
            record = ApiKeyRecord(
                name=name[:80], key_hash=digest, prefix=prefix, role=role,
                rate_limit_per_minute=rate_limit_per_minute,
                expires_at=expires_at,
            )
            s.add(record)
            s.flush()
            created = {
                "id": record.id, "name": record.name, "role": record.role,
                "prefix": record.prefix,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "api_key": raw,
                "warning": "Store this key now — it cannot be retrieved again.",
            }
        return created

    # ------------------------------------------------------------------
    def resolve(self, raw: str) -> Principal | None:
        """Look up a presented key, honouring cache, expiry and revocation."""
        digest = hash_key(raw)

        with self._lock:
            hit = self._cache.get(digest)
            if hit and hit.expires > time.monotonic():
                return hit.principal

        if not database.init():
            return None

        try:
            from sqlalchemy import select

            with session_scope() as s:
                if s is None:
                    return None
                record = s.execute(
                    select(ApiKeyRecord).where(ApiKeyRecord.key_hash == digest)
                ).scalar_one_or_none()

                if record is None or not record.active:
                    return None
                if record.expires_at and _aware(record.expires_at) < utcnow():
                    return None
                if not constant_time_equal(record.key_hash, digest):
                    return None  # pragma: no cover - defence in depth

                record.last_used_at = utcnow()
                record.call_count = (record.call_count or 0) + 1
                principal = Principal(
                    id=record.id, name=record.name, role=record.role,
                    rate_limit=record.rate_limit_per_minute,
                )
        except Exception:  # pragma: no cover
            log.exception("api key lookup failed")
            return None

        with self._lock:
            self._cache[digest] = _CachedKey(principal, time.monotonic() + self.CACHE_TTL)
        return principal

    # ------------------------------------------------------------------
    def revoke(self, key_id: str) -> bool:
        if not database.init():
            return False
        from sqlalchemy import select

        with session_scope() as s:
            if s is None:
                return False
            record = s.execute(
                select(ApiKeyRecord).where(ApiKeyRecord.id == key_id)
            ).scalar_one_or_none()
            if record is None:
                return False
            record.active = False
            self.invalidate(record.key_hash)
        return True

    def list_keys(self) -> list[dict]:
        if not database.init():
            return []
        from sqlalchemy import select

        with session_scope() as s:
            if s is None:
                return []
            rows = s.execute(
                select(ApiKeyRecord).order_by(ApiKeyRecord.created_at.desc())
            ).scalars().all()
            return [{
                "id": r.id, "name": r.name, "role": r.role, "prefix": r.prefix,
                "active": r.active, "call_count": r.call_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "last_used_at": r.last_used_at.isoformat() if r.last_used_at else None,
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
            } for r in rows]

    def invalidate(self, digest: str | None = None) -> None:
        with self._lock:
            if digest is None:
                self._cache.clear()
            else:
                self._cache.pop(digest, None)


keystore = KeyStore()


def _aware(dt: datetime) -> datetime:
    """SQLite hands back naive datetimes; treat them as UTC."""
    from datetime import timezone

    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# JWT (optional convenience for browser sessions)
# --------------------------------------------------------------------------
try:
    from jose import JWTError, jwt

    HAS_JWT = True
except Exception:  # pragma: no cover
    HAS_JWT = False


def issue_token(principal: Principal) -> dict:
    """Exchange a long-lived API key for a short-lived bearer token."""
    if not HAS_JWT:
        raise RuntimeError("JWT support needs:  pip install 'python-jose[cryptography]'")
    expires = utcnow() + timedelta(minutes=settings.jwt_ttl_minutes)
    token = jwt.encode(
        {
            "sub": principal.id, "name": principal.name, "role": principal.role,
            "exp": expires, "iat": utcnow(), "iss": settings.service_name,
        },
        settings.resolved_jwt_secret(),
        algorithm="HS256",
    )
    return {
        "access_token": token, "token_type": "bearer",
        "expires_at": expires.isoformat(),
        "expires_in": settings.jwt_ttl_minutes * 60,
        "role": principal.role,
    }


def decode_token(token: str) -> Principal | None:
    if not HAS_JWT:
        return None
    try:
        claims = jwt.decode(
            token, settings.resolved_jwt_secret(), algorithms=["HS256"],
            issuer=settings.service_name,
        )
    except JWTError:
        return None
    return Principal(
        id=str(claims.get("sub", "token")),
        name=str(claims.get("name", "token")),
        role=str(claims.get("role", "viewer")),
    )


# --------------------------------------------------------------------------
# FastAPI dependencies
# --------------------------------------------------------------------------
def _client_ip(request: Request) -> str:
    """Best-effort client address, trusting a single proxy hop."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def identify(request: Request,
             x_api_key: str | None = Header(default=None, alias="X-API-Key"),
             authorization: str | None = Header(default=None)) -> Principal:
    """Resolve the caller. Raises 401 only when auth is enabled.

    Accepts, in order: ``X-API-Key``, ``Authorization: Bearer <jwt>``,
    ``Authorization: Bearer <api key>``. The admin key from the environment is
    always honoured so an operator can never lock themselves out of a running
    instance.
    """
    raw_key = x_api_key
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()

    candidate = raw_key or bearer

    if candidate:
        # Environment admin key: the break-glass credential.
        if settings.admin_key and constant_time_equal(candidate, settings.resolved_admin_key()):
            return Principal(id="root", name="environment-admin", role="admin")
        principal = keystore.resolve(candidate)
        if principal is not None:
            request.state.principal = principal
            return principal
        if bearer and candidate == bearer:
            token_principal = decode_token(bearer)
            if token_principal is not None:
                request.state.principal = token_principal
                return token_principal

    if not settings.auth_enabled:
        request.state.principal = ANONYMOUS
        return ANONYMOUS

    record_audit("unknown", "auth.failed", outcome="denied", ip=_client_ip(request),
                 detail=f"{request.method} {request.url.path}")
    raise HTTPException(
        status_code=401,
        detail="Missing or invalid credentials. Send X-API-Key: <your key>.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def enforce_rate_limit(request: Request, principal: Principal) -> None:
    """Apply the token bucket, keyed by principal (or IP when anonymous)."""
    key = principal.id if not principal.anonymous else f"ip:{_client_ip(request)}"
    allowed, retry_after = limiter.check(key, rate_per_minute=principal.rate_limit)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry in {retry_after:.0f}s.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


def require(role: str):
    """Dependency factory: authenticate, rate-limit, then check the role.

    Used as ``principal: Principal = Depends(require("analyst"))``.
    """

    def dependency(request: Request,
                   x_api_key: str | None = Header(default=None, alias="X-API-Key"),
                   authorization: str | None = Header(default=None)) -> Principal:
        principal = identify(request, x_api_key, authorization)
        enforce_rate_limit(request, principal)

        if settings.auth_enabled and not role_satisfies(principal.role, role):
            record_audit(principal.name, "authz.denied", outcome="denied",
                         target=request.url.path, ip=_client_ip(request),
                         detail=f"role={principal.role} required={role}")
            raise HTTPException(
                status_code=403,
                detail=f"This endpoint requires the '{role}' role; you have '{principal.role}'.",
            )
        return principal

    return dependency


require_viewer = require("viewer")
require_analyst = require("analyst")
require_admin = require("admin")


def security_status() -> dict:
    """Summary rendered on the dashboard's security panel."""
    return {
        "auth_enabled": settings.auth_enabled,
        "environment": settings.environment,
        "admin_key_configured": bool(settings.admin_key),
        "jwt_available": HAS_JWT,
        "jwt_secret_configured": bool(settings.jwt_secret),
        "cors_origins": settings.cors_origins,
        "rate_limit": limiter.stats(),
        "roles": {r: ROLE_DESCRIPTION[r] for r in ROLES},
        "findings": settings.audit(),
    }
