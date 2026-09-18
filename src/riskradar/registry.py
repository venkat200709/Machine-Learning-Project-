"""Model registry — versioning, promotion, shadow evaluation and rollback.

The problem with one file called ``riskradar_model.joblib``
------------------------------------------------------------
Retraining overwrites it. That single fact causes every deployment problem
this module exists to solve:

* You cannot roll back, because the previous model no longer exists.
* You cannot reproduce a past decision, because you cannot reconstruct the
  artefact that made it.
* You cannot test a new model on live traffic without risking live traffic.
* You cannot prove the file on the server is the file you evaluated.

The registry
------------
Every trained artefact is content-addressed by SHA-256 and filed under
``models/registry/<version>/``. Versions carry a **stage**:

``staging``    registered, not serving
``challenger`` scored on live traffic in shadow, results compared, no user impact
``champion``   serving real requests — exactly one at a time
``archived``   superseded, retained for rollback and audit

Shadow evaluation
-----------------
A challenger sees every request the champion sees, and its answer is recorded
and compared — but never returned. That is the only honest way to evaluate a
model on production traffic, because offline test data is by definition the
past. Shadow scoring runs *after* the response is sent, so it cannot add
latency, and a challenger that throws is disabled rather than allowed to
affect the request.

Promotion is gated
------------------
:func:`promote` refuses to make a challenger champion unless it has been
scored on a minimum volume of live traffic and its agreement and confidence
profile clear the configured floors. A registry that lets anyone promote
anything is just a directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import config as C
from .settings import settings

log = logging.getLogger("riskradar.registry")

REGISTRY_DIR = C.MODELS_DIR / "registry"
INDEX_PATH = REGISTRY_DIR / "index.json"

STAGES = ("staging", "challenger", "champion", "archived")

# Promotion gates. Deliberately conservative — the cost of a bad champion in a
# safety system is not symmetric with the benefit of a slightly better one.
MIN_SHADOW_SAMPLES = 200
MAX_DISAGREEMENT_RATE = 0.15
MIN_MEAN_CONFIDENCE = 0.70


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash of an artefact. This is the version's real identity."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# ==========================================================================
# Version records
# ==========================================================================
@dataclass
class ModelVersion:
    version: str
    model_name: str
    stage: str = "staging"
    created_at: str = field(default_factory=utcnow_iso)
    artefact_path: str = ""
    sha256: str = ""
    size_bytes: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    promoted_at: str | None = None
    promoted_by: str | None = None
    parent_version: str | None = None

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model_name": self.model_name,
            "stage": self.stage,
            "created_at": self.created_at,
            "artefact_path": self.artefact_path,
            "sha256": self.sha256,
            "sha256_short": self.sha256[:12],
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / 1_048_576, 2),
            "metrics": self.metrics,
            "notes": self.notes,
            "promoted_at": self.promoted_at,
            "promoted_by": self.promoted_by,
            "parent_version": self.parent_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ModelVersion:
        return cls(
            version=data["version"],
            model_name=data.get("model_name", "unknown"),
            stage=data.get("stage", "staging"),
            created_at=data.get("created_at", utcnow_iso()),
            artefact_path=data.get("artefact_path", ""),
            sha256=data.get("sha256", ""),
            size_bytes=int(data.get("size_bytes", 0)),
            metrics=data.get("metrics", {}),
            notes=data.get("notes", ""),
            promoted_at=data.get("promoted_at"),
            promoted_by=data.get("promoted_by"),
            parent_version=data.get("parent_version"),
        )

    def verify(self) -> tuple[bool, str]:
        """Confirm the artefact on disk is the artefact that was registered."""
        path = Path(self.artefact_path)
        if not path.exists():
            return False, f"Artefact missing: {path}"
        if not self.sha256:
            return True, "No hash recorded (registered before integrity checks)."
        actual = sha256_file(path)
        if actual != self.sha256:
            return False, (
                f"Integrity failure: on-disk hash {actual[:12]} does not match "
                f"registered {self.sha256[:12]}. The artefact has been modified."
            )
        return True, "Verified."


# ==========================================================================
# Registry
# ==========================================================================
class ModelRegistry:
    """File-backed registry with a JSON index and an optional database mirror.

    JSON is the source of truth on purpose: the registry has to work on a
    laptop with no database, and a model server that cannot start because a
    database is down is a worse outcome than one without version history.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or REGISTRY_DIR
        self.index_path = self.root / "index.json"
        self._lock = threading.RLock()
        self._versions: dict[str, ModelVersion] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        with self._lock:
            self._versions = {}
            if not self.index_path.exists():
                return
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                for item in data.get("versions", []):
                    mv = ModelVersion.from_dict(item)
                    self._versions[mv.version] = mv
            except Exception:  # pragma: no cover
                log.warning("registry index unreadable; starting empty")

    def _save(self) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": utcnow_iso(),
                "versions": [v.to_dict() for v in self._versions.values()],
            }
            tmp = self.index_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            # Atomic replace: a crash mid-write must not leave a truncated index.
            tmp.replace(self.index_path)

    # ------------------------------------------------------------------
    def register(self, artefact: Path, *, model_name: str, metrics: dict | None = None,
                 version: str | None = None, notes: str = "",
                 stage: str = "staging", copy: bool = True) -> ModelVersion:
        """File an artefact into the registry under a new version."""
        artefact = Path(artefact)
        if not artefact.exists():
            raise FileNotFoundError(f"No artefact at {artefact}")

        version = version or self._next_version()
        with self._lock:
            if version in self._versions:
                raise ValueError(f"Version {version} already exists.")

            target_dir = self.root / version
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / artefact.name

            if copy and artefact.resolve() != target.resolve():
                shutil.copy2(artefact, target)
            else:
                target = artefact

            mv = ModelVersion(
                version=version,
                model_name=model_name,
                stage=stage,
                artefact_path=str(target),
                sha256=sha256_file(target),
                size_bytes=target.stat().st_size,
                metrics=metrics or {},
                notes=notes,
                parent_version=self.champion().version if self.champion() else None,
            )
            self._versions[version] = mv
            self._save()

        self._mirror(mv)
        log.info("registered model version %s (%s)", version, model_name)
        return mv

    def _next_version(self) -> str:
        """Monotonic ``vN`` label. Simple, sortable, and never reused."""
        existing = [
            int(v[1:]) for v in self._versions
            if v.startswith("v") and v[1:].isdigit()
        ]
        return f"v{max(existing, default=0) + 1}"

    # ------------------------------------------------------------------
    def get(self, version: str) -> ModelVersion | None:
        return self._versions.get(version)

    def list_versions(self) -> list[dict]:
        with self._lock:
            return sorted(
                (v.to_dict() for v in self._versions.values()),
                key=lambda d: d["created_at"], reverse=True,
            )

    def by_stage(self, stage: str) -> ModelVersion | None:
        with self._lock:
            for v in self._versions.values():
                if v.stage == stage:
                    return v
        return None

    def champion(self) -> ModelVersion | None:
        return self.by_stage("champion")

    def challenger(self) -> ModelVersion | None:
        return self.by_stage("challenger")

    # ------------------------------------------------------------------
    def set_stage(self, version: str, stage: str, *, actor: str = "system") -> ModelVersion:
        """Move a version between stages, keeping the invariants."""
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Choose from {STAGES}.")

        with self._lock:
            mv = self._versions.get(version)
            if mv is None:
                raise KeyError(f"No such version: {version}")

            ok, message = mv.verify()
            if not ok and stage in ("champion", "challenger"):
                raise RuntimeError(message)

            # Exactly one champion, exactly one challenger.
            if stage in ("champion", "challenger"):
                for other in self._versions.values():
                    if other.version != version and other.stage == stage:
                        other.stage = "archived" if stage == "champion" else "staging"

            previous = mv.stage
            mv.stage = stage
            if stage == "champion":
                mv.promoted_at = utcnow_iso()
                mv.promoted_by = actor
            self._save()

        self._mirror(mv)
        self._audit(actor, f"registry.stage.{stage}", version,
                    f"{previous} -> {stage}")
        return mv

    def promote(self, version: str, *, actor: str = "system", force: bool = False,
                shadow: ShadowEvaluator | None = None) -> dict:
        """Promote to champion, subject to the shadow-traffic gates."""
        mv = self._versions.get(version)
        if mv is None:
            raise KeyError(f"No such version: {version}")

        gate = self.promotion_gate(version, shadow=shadow)
        if not gate["passed"] and not force:
            return {"promoted": False, "gate": gate,
                    "message": "Promotion blocked. " + " ".join(gate["blockers"])}

        previous = self.champion()
        self.set_stage(version, "champion", actor=actor)
        return {
            "promoted": True,
            "version": version,
            "previous_champion": previous.version if previous else None,
            "forced": bool(force and not gate["passed"]),
            "gate": gate,
            "message": f"{version} is now champion."
                       + (" Gates were overridden." if force and not gate["passed"] else ""),
        }

    def promotion_gate(self, version: str,
                       shadow: ShadowEvaluator | None = None) -> dict:
        """Evaluate whether a version is allowed to become champion."""
        mv = self._versions.get(version)
        blockers: list[str] = []
        checks: list[dict] = []

        if mv is None:
            return {"passed": False, "blockers": [f"No such version: {version}"], "checks": []}

        ok, message = mv.verify()
        checks.append({"check": "artefact_integrity", "passed": ok, "detail": message})
        if not ok:
            blockers.append(message)

        if mv.stage == "champion":
            return {"passed": False, "blockers": [f"{version} is already champion."],
                    "checks": checks}

        if mv.stage == "challenger" and shadow is not None:
            stats = shadow.stats()
            enough = stats["n"] >= MIN_SHADOW_SAMPLES
            checks.append({
                "check": "shadow_volume", "passed": enough,
                "detail": f"{stats['n']} shadow predictions (need {MIN_SHADOW_SAMPLES}).",
            })
            if not enough:
                blockers.append(
                    f"Only {stats['n']} shadow predictions; {MIN_SHADOW_SAMPLES} required."
                )

            if stats["n"]:
                agree_ok = stats["disagreement_rate"] <= MAX_DISAGREEMENT_RATE
                checks.append({
                    "check": "agreement", "passed": agree_ok,
                    "detail": f"Disagrees with champion on {stats['disagreement_rate']:.1%} "
                              f"(limit {MAX_DISAGREEMENT_RATE:.0%}).",
                })
                if not agree_ok:
                    blockers.append(
                        f"Challenger disagrees with the champion on "
                        f"{stats['disagreement_rate']:.1%} of live traffic — "
                        "investigate before promoting."
                    )

                conf_ok = (stats["mean_confidence"] or 0) >= MIN_MEAN_CONFIDENCE
                checks.append({
                    "check": "confidence", "passed": conf_ok,
                    "detail": f"Mean confidence {stats['mean_confidence']:.3f} "
                              f"(floor {MIN_MEAN_CONFIDENCE}).",
                })
                if not conf_ok:
                    blockers.append("Challenger's mean confidence is below the floor.")

                err_ok = stats["error_rate"] == 0.0
                checks.append({
                    "check": "stability", "passed": err_ok,
                    "detail": f"{stats['errors']} scoring error(s) in shadow.",
                })
                if not err_ok:
                    blockers.append(f"Challenger raised {stats['errors']} errors in shadow.")

        elif mv.stage != "challenger":
            checks.append({
                "check": "shadow_evaluated", "passed": False,
                "detail": f"Version is in '{mv.stage}', not 'challenger'. "
                          "Shadow-test on live traffic before promoting.",
            })
            blockers.append(
                f"{version} has not been shadow-tested. Set it to 'challenger' first."
            )

        champion = self.champion()
        if champion and mv.metrics and champion.metrics:
            new_acc = mv.metrics.get("accuracy")
            old_acc = champion.metrics.get("accuracy")
            if new_acc is not None and old_acc is not None:
                # A small regression can be acceptable if it buys robustness,
                # so this warns rather than blocks, and records the trade.
                better = new_acc >= old_acc - 0.005
                checks.append({
                    "check": "offline_accuracy", "passed": better,
                    "detail": f"{new_acc:.4f} vs champion {old_acc:.4f}.",
                })
                if not better:
                    blockers.append(
                        f"Offline accuracy regressed materially: {new_acc:.4f} < {old_acc:.4f}."
                    )

        return {"passed": not blockers, "blockers": blockers, "checks": checks}

    def rollback(self, *, actor: str = "system") -> dict:
        """Return to the most recently archived champion."""
        with self._lock:
            archived = [
                v for v in self._versions.values()
                if v.stage == "archived" and v.promoted_at
            ]
        if not archived:
            return {"rolled_back": False, "message": "No previous champion to roll back to."}

        previous = max(archived, key=lambda v: v.promoted_at or "")
        current = self.champion()
        self.set_stage(previous.version, "champion", actor=actor)
        return {
            "rolled_back": True,
            "version": previous.version,
            "replaced": current.version if current else None,
            "message": f"Rolled back to {previous.version}.",
        }

    # ------------------------------------------------------------------
    def prune(self, keep_archived: int = 3, *, actor: str = "system") -> dict:
        """Delete old archived artefacts, and any directory the index forgot.

        Each version is a full copy of a multi-megabyte pipeline, so an
        unattended registry grows without bound. Champion, challenger and
        staging entries are never touched; only archived ones beyond
        ``keep_archived`` are removed, so rollback stays possible.
        """
        removed: list[str] = []
        freed = 0

        with self._lock:
            archived = sorted(
                (v for v in self._versions.values() if v.stage == "archived"),
                key=lambda v: v.created_at, reverse=True,
            )
            for mv in archived[keep_archived:]:
                directory = Path(mv.artefact_path).parent
                freed += mv.size_bytes
                try:
                    if directory.is_dir() and directory.parent == self.root:
                        shutil.rmtree(directory, ignore_errors=True)
                except Exception:  # pragma: no cover
                    log.warning("could not remove %s", directory)
                self._versions.pop(mv.version, None)
                removed.append(mv.version)

            # Orphans: directories with no index entry, left behind by an
            # interrupted registration or a hand-edited index.
            known = {Path(v.artefact_path).parent.name for v in self._versions.values()}
            orphans = []
            if self.root.is_dir():
                for child in self.root.iterdir():
                    if child.is_dir() and child.name not in known:
                        orphans.append(child.name)
                        shutil.rmtree(child, ignore_errors=True)

            self._save()

        self._audit(actor, "registry.prune", ",".join(removed + orphans) or "none",
                    f"{len(removed)} archived, {len(orphans)} orphaned")
        return {
            "removed_versions": removed,
            "removed_orphans": orphans,
            "freed_mb": round(freed / 1_048_576, 2),
            "remaining": len(self._versions),
        }

    def verify_all(self) -> list[dict]:
        results = []
        for v in self._versions.values():
            ok, message = v.verify()
            results.append({
                "version": v.version, "stage": v.stage,
                "verified": ok, "detail": message,
            })
        return results

    def summary(self) -> dict:
        champion = self.champion()
        challenger = self.challenger()
        return {
            "n_versions": len(self._versions),
            "champion": champion.to_dict() if champion else None,
            "challenger": challenger.to_dict() if challenger else None,
            "stages": dict(Counter(v.stage for v in self._versions.values())),
            "registry_path": str(self.root),
            "gates": {
                "min_shadow_samples": MIN_SHADOW_SAMPLES,
                "max_disagreement_rate": MAX_DISAGREEMENT_RATE,
                "min_mean_confidence": MIN_MEAN_CONFIDENCE,
            },
        }

    # ------------------------------------------------------------------
    def _mirror(self, mv: ModelVersion) -> None:
        """Best-effort copy into the database, for cross-replica visibility."""
        try:
            from sqlalchemy import select

            from .db import ModelVersionRecord, database, session_scope

            if not database.init():
                return
            with session_scope() as s:
                if s is None:
                    return
                row = s.execute(
                    select(ModelVersionRecord)
                    .where(ModelVersionRecord.version == mv.version)
                ).scalar_one_or_none()
                if row is None:
                    row = ModelVersionRecord(version=mv.version, model_name=mv.model_name)
                    s.add(row)
                row.model_name = mv.model_name
                row.stage = mv.stage
                row.accuracy = mv.metrics.get("accuracy")
                row.f1_macro = mv.metrics.get("f1_macro")
                row.roc_auc = mv.metrics.get("roc_auc_ovr")
                row.ece = mv.metrics.get("expected_calibration_error")
                row.artefact_path = mv.artefact_path[:300]
                row.artefact_sha256 = mv.sha256
                row.notes = mv.notes
        except Exception:  # pragma: no cover
            log.debug("registry database mirror skipped", exc_info=True)

    @staticmethod
    def _audit(actor: str, action: str, target: str, detail: str) -> None:
        try:
            from .db import record_audit

            record_audit(actor, action, target=target, detail=detail)
        except Exception:  # pragma: no cover
            pass


# ==========================================================================
# Shadow evaluation
# ==========================================================================
class ShadowEvaluator:
    """Scores live traffic with the challenger without affecting responses.

    Runs off the request path. If the challenger raises, it is counted and
    disabled after a threshold rather than being retried forever — a broken
    challenger must be loud in the metrics and invisible to users.
    """

    MAX_ERRORS = 25

    def __init__(self, window: int = 2000) -> None:
        self.model = None
        self.version: str | None = None
        self._lock = threading.Lock()
        self._agree: deque[bool] = deque(maxlen=window)
        self._confidence: deque[float] = deque(maxlen=window)
        self._champion_labels: deque[str] = deque(maxlen=window)
        self._challenger_labels: deque[str] = deque(maxlen=window)
        self.errors = 0
        self.total = 0
        self.disabled_reason: str | None = None

    # ------------------------------------------------------------------
    def load(self, version: ModelVersion) -> bool:
        """Attach a challenger artefact. Returns False if it cannot be loaded."""
        try:
            import joblib

            model = joblib.load(version.artefact_path)
        except Exception as exc:
            self.disabled_reason = f"Could not load {version.version}: {exc}"
            log.warning("shadow load failed: %s", self.disabled_reason)
            return False

        with self._lock:
            self.model = model
            self.version = version.version
            self._agree.clear()
            self._confidence.clear()
            self._champion_labels.clear()
            self._challenger_labels.clear()
            self.errors = 0
            self.total = 0
            self.disabled_reason = None
        log.info("shadow challenger %s loaded", version.version)
        return True

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.version = None
            self.disabled_reason = None

    @property
    def active(self) -> bool:
        return (
            settings.shadow_enabled
            and self.model is not None
            and self.disabled_reason is None
        )

    # ------------------------------------------------------------------
    def observe(self, frame, champion_label: str) -> None:
        """Score one request in the background and record the comparison."""
        if not self.active:
            return
        try:
            proba = np.asarray(self.model.predict_proba(frame))[0]
            idx = int(np.argmax(proba))
            label = C.INT_TO_CLASS[idx]
            with self._lock:
                self._agree.append(label == champion_label)
                self._confidence.append(float(proba[idx]))
                self._champion_labels.append(champion_label)
                self._challenger_labels.append(label)
                self.total += 1
        except Exception:
            with self._lock:
                self.errors += 1
                if self.errors >= self.MAX_ERRORS:
                    self.disabled_reason = (
                        f"Disabled after {self.errors} scoring errors. "
                        "The challenger artefact is incompatible with live inputs."
                    )
                    log.warning("shadow evaluator disabled: %s", self.disabled_reason)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            n = len(self._agree)
            agree = list(self._agree)
            confidence = list(self._confidence)
            champ = list(self._champion_labels)
            chall = list(self._challenger_labels)
            errors = self.errors
            total = self.total

        if not n:
            return {
                "active": self.active, "version": self.version, "n": 0,
                "errors": errors, "error_rate": 0.0,
                "agreement_rate": None, "disagreement_rate": 0.0,
                "mean_confidence": None, "confusion": {},
                "disabled_reason": self.disabled_reason,
                "verdict": "No shadow traffic yet.",
            }

        agreement = float(np.mean(agree))
        confusion: dict[str, int] = {}
        for a, b in zip(champ, chall, strict=True):
            if a != b:
                confusion[f"{a} -> {b}"] = confusion.get(f"{a} -> {b}", 0) + 1

        # A disagreement that crosses two bands is qualitatively worse than a
        # neighbouring-band one, so it is counted separately.
        severe = sum(
            count for key, count in confusion.items()
            if abs(C.CLASS_TO_INT[key.split(" -> ")[0]]
                   - C.CLASS_TO_INT[key.split(" -> ")[1]]) >= 2
        )

        return {
            "active": self.active,
            "version": self.version,
            "n": n,
            "total_scored": total,
            "errors": errors,
            "error_rate": round(errors / max(total + errors, 1), 5),
            "agreement_rate": round(agreement, 5),
            "disagreement_rate": round(1.0 - agreement, 5),
            "severe_disagreements": severe,
            "mean_confidence": round(float(np.mean(confidence)), 5),
            "confusion": dict(sorted(confusion.items(), key=lambda kv: -kv[1])),
            "disabled_reason": self.disabled_reason,
            "verdict": _shadow_verdict(agreement, severe, n, errors),
        }


def _shadow_verdict(agreement: float, severe: int, n: int, errors: int) -> str:
    if errors:
        return f"Unstable: {errors} scoring errors. Do not promote."
    # Severe splits are checked *before* the volume gate on purpose. A single
    # Low↔High disagreement means one of the two models would call a dangerous
    # street safe — that is disqualifying evidence on its own, and waiting for
    # 200 samples to mention it would bury the most important signal there is.
    if severe:
        return (
            f"{severe} Low↔High disagreement(s) with the champion"
            + (f" in only {n} predictions" if n < MIN_SHADOW_SAMPLES else "")
            + ". Investigate before promoting; these are the errors that matter."
        )
    if n < MIN_SHADOW_SAMPLES:
        return f"Collecting evidence — {n}/{MIN_SHADOW_SAMPLES} predictions so far."
    if agreement >= 1 - MAX_DISAGREEMENT_RATE:
        return (
            f"Behaving consistently with the champion ({agreement:.1%} agreement) "
            "and stable. Eligible for promotion."
        )
    return (
        f"Diverges from the champion on {1 - agreement:.1%} of traffic. "
        "That may be an improvement, but it needs labelled evidence first."
    )


# Process-wide singletons.
registry = ModelRegistry()
shadow = ShadowEvaluator()
