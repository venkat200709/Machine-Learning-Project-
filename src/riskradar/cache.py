"""Inference cache — an LRU with a TTL, sized for the actual access pattern.

Why cache a model that answers in 30 ms?
-----------------------------------------
Because the expensive part is not the forward pass. A full ``/predict`` builds
the 79-column feature matrix, runs SHAP over it and then executes a
counterfactual scan of ~13 additional forward passes. That is two orders of
magnitude more work than the classification itself.

And the traffic is extremely repetitive. A dashboard user nudges one slider and
re-scores; a monitoring screen re-polls the same watchlist every few seconds; a
route request scores the same junction from several candidate paths. Identical
inputs, repeatedly, within seconds.

Correctness
-----------
The key is a SHA-256 over the canonicalised payload *plus the model version and
the flags that change the response shape*. Omitting the version would serve a
retired model's answers after a promotion — a subtle and serious bug, since the
response would carry the new model's name over the old model's numbers.

Entries expire on a TTL as well as on capacity. A pure LRU would happily serve
a five-hour-old assessment of a street, and "was it safe five hours ago" is not
the question anyone asked.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from .settings import settings


def payload_key(payload: dict, *, model_version: str, **flags: Any) -> str:
    """Stable hash of a request. Order-independent, float-safe.

    ``sort_keys`` makes the key independent of JSON field order, and floats are
    rounded to six places so that 3.0000000001 and 3.0 — which the model cannot
    distinguish — do not become separate cache entries.
    """
    normalised = {
        k: (round(v, 6) if isinstance(v, float) else v)
        for k, v in sorted(payload.items())
    }
    blob = json.dumps(
        {"p": normalised, "v": model_version, "f": sorted(flags.items())},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class TTLCache:
    """Thread-safe LRU with per-entry expiry and hit/miss accounting."""

    def __init__(self, maxsize: int | None = None, ttl: float | None = None) -> None:
        self.maxsize = maxsize if maxsize is not None else settings.cache_size
        self.ttl = ttl if ttl is not None else settings.cache_ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    # ------------------------------------------------------------------
    def get(self, key: str) -> Any | None:
        if not settings.cache_enabled:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires, value = entry
            if expires < now:
                del self._store[key]
                self.expirations += 1
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        if not settings.cache_enabled:
            return
        with self._lock:
            self._store[key] = (time.monotonic() + self.ttl, value)
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)
                self.evictions += 1

    def invalidate(self, key: str | None = None) -> None:
        """Drop one entry, or the whole cache after a model promotion."""
        with self._lock:
            if key is None:
                self._store.clear()
            else:
                self._store.pop(key, None)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            size = len(self._store)
        total = self.hits + self.misses
        return {
            "enabled": settings.cache_enabled,
            "size": size,
            "maxsize": self.maxsize,
            "ttl_seconds": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "evictions": self.evictions,
            "expirations": self.expirations,
            # Every hit skips SHAP plus a ~13-pass counterfactual scan.
            "estimated_ms_saved": round(self.hits * 35.0, 1),
        }

    def reset_stats(self) -> None:
        with self._lock:
            self.hits = self.misses = self.evictions = self.expirations = 0


prediction_cache = TTLCache()
