"""Post-training artefacts — everything the platform needs that the model isn't.

A trained pipeline is not a deployable system. Three more artefacts have to
exist before the serving layer can honour its promises:

* **Conformal calibration** — without it ``/predict`` can report a softmax
  number but not a coverage guarantee.
* **Drift reference** — without it the monitor has nothing to compare live
  traffic against, and drift detection silently does nothing.
* **Registry entry** — without it there is no rollback target and no integrity
  hash for the artefact being served.

All three are derived from the model plus the dataset, take seconds rather
than minutes, and are therefore regenerated automatically at the end of every
training run. They can also be rebuilt on their own against an already-trained
model::

    python run.py --calibrate

which is the path a user takes after cloning a repo that ships a model but not
these derived files.
"""

from __future__ import annotations

import logging
from typing import Any

from . import config as C

log = logging.getLogger("riskradar.bootstrap")


def build_conformal(model, *, alpha: float = 0.10, method: str = "auto",
                    model_version: str = "") -> dict:
    """Fit and verify conformal quantiles."""
    from .conformal import calibrate_from_dataset

    cal = calibrate_from_dataset(
        model, alpha=alpha, method=method, mondrian=True,
        model_version=model_version, save=True,
    )
    coverage = cal.coverage or {}
    return {
        "written": str(_conformal_path()),
        "n_calibration": cal.n_calibration,
        "target_coverage": coverage.get("target_coverage"),
        "empirical_coverage": coverage.get("empirical_coverage"),
        "guarantee_met": coverage.get("guarantee_met"),
        "avg_set_size": coverage.get("avg_set_size"),
        "singleton_rate": coverage.get("singleton_rate"),
    }


def _conformal_path():
    from .conformal import CALIBRATION_PATH

    return CALIBRATION_PATH


def build_drift_reference(model, *, sample: int = 30_000,
                          model_version: str = "") -> dict:
    """Snapshot the training distribution the drift monitor compares against."""
    import numpy as np

    from .data import load_dataset, split_xy, stratified_split
    from .drift import build_reference

    df = load_dataset()
    X, y = split_xy(df)
    X_train, _, y_train, _ = stratified_split(X, y)

    if sample and len(X_train) > sample:
        rng = np.random.default_rng(C.RANDOM_STATE)
        idx = rng.choice(len(X_train), size=sample, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train[idx]

    # Reference the *engineered* matrix, not the raw one: drift on
    # ``Threat_Score`` is a far more actionable signal than drift on the six
    # raw columns that feed it, and it is what the model actually consumes.
    engineered = (
        model.named_steps["features"].transform(X_train)
        if hasattr(model, "named_steps") else X_train
    )
    ref = build_reference(engineered, y_train, model_version=model_version, save=True)

    from .drift import REFERENCE_PATH, monitor

    monitor.reference = ref  # adopt it immediately, no restart needed
    return {
        "written": str(REFERENCE_PATH),
        "n_reference": ref.n_reference,
        "n_features_monitored": len(ref.features),
        "class_distribution": ref.class_distribution,
    }


def register_champion(metadata: dict, *, actor: str = "training") -> dict:
    """File the freshly-trained artefact in the registry and promote it."""
    from .registry import registry

    if not C.MODEL_PATH.exists():
        return {"registered": False, "reason": "No model artefact on disk."}

    # Versions are content-addressed, so re-running calibration against an
    # unchanged model must not mint v5, v6, v7... Identical bytes are the same
    # version by definition.
    from .registry import sha256_file

    digest = sha256_file(C.MODEL_PATH)
    for existing in registry.list_versions():
        if existing["sha256"] == digest:
            if existing["stage"] != "champion" and registry.champion() is None:
                registry.set_stage(existing["version"], "champion", actor=actor)
            return {
                "registered": False, "version": existing["version"],
                "sha256": digest[:12], "reason": "Identical artefact already registered.",
            }

    try:
        mv = registry.register(
            C.MODEL_PATH,
            model_name=metadata.get("model_name", "RiskRadar"),
            metrics=metadata.get("metrics", {}),
            notes=f"Trained {metadata.get('trained_at', 'unknown')} — "
                  f"{metadata.get('model_name', 'model')}.",
        )
        # A newly trained model is the champion by definition: it was produced
        # by the pipeline that just measured it, and there is nothing to shadow
        # it against. Gated promotion applies to *candidate* artefacts arriving
        # from elsewhere, which is what the challenger stage is for.
        registry.set_stage(mv.version, "champion", actor=actor)
        return {"registered": True, "version": mv.version, "sha256": mv.sha256[:12]}
    except Exception as exc:  # pragma: no cover
        log.warning("registry write failed: %s", exc)
        return {"registered": False, "reason": str(exc)}


def bootstrap_all(model=None, metadata: dict | None = None, *,
                  alpha: float = 0.10, verbose: bool = True) -> dict[str, Any]:
    """Generate every post-training artefact. Safe to re-run at any time."""
    import joblib

    if model is None:
        if not C.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model at {C.MODEL_PATH}. Train one first:  python run.py --train"
            )
        model = joblib.load(C.MODEL_PATH)

    if metadata is None:
        import json

        try:
            metadata = json.loads(C.METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    version = str(metadata.get("version", "")) + "@" + str(metadata.get("trained_at", ""))
    results: dict[str, Any] = {}

    def step(name: str, fn):
        if verbose:
            print(f"  {name} ...", end="", flush=True)
        try:
            results[name] = fn()
            if verbose:
                print(" done")
        except Exception as exc:
            results[name] = {"error": f"{exc.__class__.__name__}: {exc}"}
            if verbose:
                print(f" FAILED ({exc})")

    if verbose:
        print("\nBuilding platform artefacts")
        print("-" * 72)

    step("conformal", lambda: build_conformal(model, alpha=alpha, model_version=version))
    step("drift_reference", lambda: build_drift_reference(model, model_version=version))
    step("registry", lambda: register_champion(metadata))

    if verbose:
        cal = results.get("conformal", {})
        if cal.get("empirical_coverage") is not None:
            print(
                f"\n  conformal : {cal['empirical_coverage']:.1%} empirical coverage "
                f"against a {cal['target_coverage']:.0%} target "
                f"({'guarantee holds' if cal.get('guarantee_met') else 'GUARANTEE MISSED'}), "
                f"mean set size {cal['avg_set_size']}"
            )
        ref = results.get("drift_reference", {})
        if ref.get("n_features_monitored"):
            print(f"  drift     : {ref['n_features_monitored']} features profiled "
                  f"from {ref['n_reference']:,} training rows")
        reg = results.get("registry", {})
        if reg.get("registered"):
            print(f"  registry  : champion {reg['version']} (sha {reg['sha256']})")
        print("-" * 72)

    return results
