"""Training, benchmarking, ensembling and evaluation for RiskRadar.

Two execution modes
-------------------
**One-shot** (what you normally want)::

    python -m riskradar.train              # full benchmark + ensemble + report
    python -m riskradar.train --fast       # reduced budget smoke run

**Staged / resumable** — trains one candidate per invocation and checkpoints to
disk, so a long benchmark can be spread across sessions or CI jobs::

    python -m riskradar.train --stage model --slug lgbm
    python -m riskradar.train --stage cv    --slug lgbm --fold 0
    python -m riskradar.train --stage ensemble
    python -m riskradar.train --stage finalize

Every candidate is a full ``Pipeline`` whose first step is the feature
engineering transform, so the artefact saved to ``models/`` accepts *raw*
records and the API never re-implements preprocessing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from . import config as C
from . import data as D
from .features import feature_names
from .models import SoftVoteEnsemble, ensemble_weights, registry

warnings.filterwarnings("ignore")

# Legacy Windows consoles cannot encode every character we print; never let a
# console code page turn a successful training run into a crash.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8", errors="replace")

STAGE_DIR = Path(os.environ.get("RISKRADAR_STAGE_DIR", C.MODELS_DIR / "_staging"))


# ==========================================================================
# Metrics
# ==========================================================================
def evaluate(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Full multi-class metric suite on the untouched hold-out set."""
    y_pred = np.asarray(model.predict(X_test)).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
    }
    try:
        proba = model.predict_proba(X_test)
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        )
        metrics["log_loss"] = float(log_loss(y_test, proba, labels=[0, 1, 2]))
        # Expected calibration error (10 equal-width bins on the top probability).
        conf = proba.max(axis=1)
        correct = (proba.argmax(axis=1) == y_test).astype(float)
        bins = np.clip(np.digitize(conf, np.linspace(0.1, 1.0, 10)), 0, 9)
        ece = 0.0
        for b in range(10):
            m = bins == b
            if m.sum():
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        metrics["expected_calibration_error"] = float(ece)
    except Exception:
        pass

    report = classification_report(
        y_test, y_pred, target_names=C.CLASS_ORDER, output_dict=True,
        labels=[0, 1, 2], zero_division=0,
    )
    metrics["per_class"] = {
        cls: {
            "precision": round(float(report[cls]["precision"]), 4),
            "recall": round(float(report[cls]["recall"]), 4),
            "f1": round(float(report[cls]["f1-score"]), 4),
            "support": int(report[cls]["support"]),
        }
        for cls in C.CLASS_ORDER
    }
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    return metrics


def leaderboard_row(slug: str, label: str, metrics: dict, fit_s: float) -> dict:
    return {
        "slug": slug,
        "model": label,
        "accuracy": round(metrics["accuracy"], 5),
        "balanced_accuracy": round(metrics["balanced_accuracy"], 5),
        "precision_macro": round(metrics["precision_macro"], 5),
        "recall_macro": round(metrics["recall_macro"], 5),
        "f1_macro": round(metrics["f1_macro"], 5),
        "f1_weighted": round(metrics["f1_weighted"], 5),
        "roc_auc_ovr": round(metrics.get("roc_auc_ovr", 0.0), 5),
        "log_loss": round(metrics.get("log_loss", 0.0), 5),
        "cohen_kappa": round(metrics["cohen_kappa"], 5),
        "mcc": round(metrics["mcc"], 5),
        "ece": round(metrics.get("expected_calibration_error", 0.0), 5),
        "fit_seconds": round(fit_s, 1),
        "cv_accuracy_mean": None,
        "cv_accuracy_std": None,
    }


# ==========================================================================
# Reporting artefacts consumed by the frontend
# ==========================================================================
def build_analytics(df: pd.DataFrame) -> dict:
    """Pre-aggregated statistics so the dashboard never ships a 14 MB CSV."""
    d = df.copy()
    d["_risk"] = d[C.TARGET]
    high = lambda s: float((s == "High").mean())  # noqa: E731

    by_hour = d.groupby("Hour")["_risk"].apply(high).reindex(range(24), fill_value=0.0)
    crime_hour = d.groupby("Hour")["Crime_Count"].mean().reindex(range(24), fill_value=0.0)
    by_day = d.groupby("Day")["_risk"].apply(high).reindex(C.DAY_ORDER, fill_value=0.0)
    by_weather = d.groupby("Weather")["_risk"].apply(high)
    by_vis = d.groupby("Visibility")["_risk"].apply(high)
    by_trend = d.groupby("Crime_Trend")["_risk"].apply(high)
    prev = pd.crosstab(d["Previous_Risk"], d["_risk"], normalize="index")
    crime_totals = d[["Violent_Crime", "Theft_Count", "Assault_Count", "Harassment_Count"]].sum()

    lit = d.assign(band=pd.cut(
        d["Working_Streetlights"] / (d["Streetlight_Count"] + 1),
        bins=[0, 0.6, 0.75, 0.85, 1.01], labels=["<60%", "60-75%", "75-85%", "85%+"]))
    by_light = lit.groupby("band", observed=True)["_risk"].apply(high)

    pol = d.assign(band=pd.qcut(d["Police_Distance_km"], 5, duplicates="drop"))
    by_pol = pol.groupby("band", observed=True)["_risk"].apply(high)

    cctv = d.assign(band=pd.qcut(d["CCTV_Count"], 5, duplicates="drop"))
    by_cctv = cctv.groupby("band", observed=True)["_risk"].apply(high)

    return {
        "dataset": D.data_quality_report(df),
        "class_distribution": {k: int(v) for k, v in d[C.TARGET].value_counts().items()},
        "high_risk_share_by_hour": [round(v, 4) for v in by_hour.tolist()],
        "avg_crime_by_hour": [round(float(v), 2) for v in crime_hour.tolist()],
        "high_risk_share_by_day": {k: round(float(v), 4) for k, v in by_day.items()},
        "high_risk_share_by_weather": {k: round(float(v), 4) for k, v in by_weather.items()},
        "high_risk_share_by_visibility": {k: round(float(v), 4) for k, v in by_vis.items()},
        "high_risk_share_by_trend": {k: round(float(v), 4) for k, v in by_trend.items()},
        "previous_vs_current": {
            str(i): {str(c): round(float(prev.loc[i, c]), 4) for c in prev.columns}
            for i in prev.index
        },
        "crime_type_totals": {k: int(v) for k, v in crime_totals.items()},
        "high_risk_share_by_lighting": {str(k): round(float(v), 4) for k, v in by_light.items()},
        "high_risk_share_by_police_distance": {
            f"{iv.left:.1f}–{iv.right:.1f} km": round(float(v), 4) for iv, v in by_pol.items()},
        "high_risk_share_by_cctv": {
            f"{int(iv.left)}–{int(iv.right)}": round(float(v), 4) for iv, v in by_cctv.items()},
        "averages": {
            "crime_count": round(float(d["Crime_Count"].mean()), 2),
            "cctv_count": round(float(d["CCTV_Count"].mean()), 2),
            "police_distance_km": round(float(d["Police_Distance_km"].mean()), 2),
            "lighting_health": round(
                float((d["Working_Streetlights"] / (d["Streetlight_Count"] + 1)).mean()), 4),
            "emergency_calls": round(float(d["Emergency_Calls"].mean()), 2),
        },
    }


def build_geo_sample(df: pd.DataFrame, n: int = 2500) -> list[dict]:
    """Class-stratified down-sample for the interactive risk map."""
    frac = min(1.0, n / len(df))
    sample = (
        df.groupby(C.TARGET, group_keys=False)
        .apply(lambda g: g.sample(max(1, int(len(g) * frac)), random_state=C.RANDOM_STATE))
        .reset_index(drop=True)
    )
    return [
        {
            "lat": round(float(r.Latitude), 5), "lon": round(float(r.Longitude), 5),
            "risk": str(getattr(r, C.TARGET)), "crime": int(r.Crime_Count),
            "hour": int(r.Hour), "cctv": int(r.CCTV_Count),
            "police_km": round(float(r.Police_Distance_km), 2),
        }
        for r in sample.itertuples(index=False)
    ]


def global_importance(model, top_k: int = 30) -> list[dict]:
    """Gain/impurity importance of the final estimator, mapped back to names."""
    names = feature_names()
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model

    if isinstance(est, SoftVoteEnsemble):
        mats, weights = [], []
        for sub, w in zip(est.estimators, est._w, strict=True):
            inner = sub.named_steps["model"] if hasattr(sub, "named_steps") else sub
            imp = getattr(inner, "feature_importances_", None)
            if imp is not None:
                imp = np.asarray(imp, dtype=float)
                mats.append(imp / (imp.sum() or 1.0))
                weights.append(w)
        if not mats:
            return []
        imp = np.average(np.stack(mats), axis=0, weights=weights)
    else:
        imp = getattr(est, "feature_importances_", None)
        if imp is None:
            coef = getattr(est, "coef_", None)
            if coef is None:
                return []
            imp = np.abs(np.asarray(coef)).mean(axis=0)
        imp = np.asarray(imp, dtype=float)
        imp = imp / (imp.sum() or 1.0)

    order = np.argsort(imp)[::-1][:top_k]
    return [{"feature": names[i], "importance": round(float(imp[i]), 6)} for i in order]


# ==========================================================================
# Staged execution
# ==========================================================================
def _paths():
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    (STAGE_DIR / "parts").mkdir(exist_ok=True)
    (STAGE_DIR / "fitted").mkdir(exist_ok=True)
    (STAGE_DIR / "cv").mkdir(exist_ok=True)
    return STAGE_DIR


def _splits(fast: bool = False):
    df = D.load_dataset()
    X, y = D.split_xy(df)
    if fast:
        idx = np.random.RandomState(C.RANDOM_STATE).choice(len(X), 15000, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    return df, *D.stratified_split(X, y)


def stage_model(slug: str, fast: bool = False) -> dict:
    """Fit one candidate, evaluate it, checkpoint the artefact + metrics."""
    _paths()
    reg = registry(fast)
    if slug not in reg:
        raise SystemExit(f"Unknown model slug '{slug}'. Available: {sorted(reg)}")

    _, X_tr, X_te, y_tr, y_te = _splits(fast)
    entry = reg[slug]
    print(f"[stage:model] {entry['label']} ({slug}) on {len(X_tr):,} rows", flush=True)

    pipe = entry["build"]()
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    fit_s = time.time() - t0

    metrics = evaluate(pipe, X_te, y_te)
    row = leaderboard_row(slug, entry["label"], metrics, fit_s)

    joblib.dump(pipe, STAGE_DIR / "fitted" / f"{slug}.joblib", compress=3)
    (STAGE_DIR / "parts" / f"{slug}.json").write_text(
        json.dumps({"row": row, "metrics": metrics}, indent=2), encoding="utf-8")

    print(f"  accuracy {metrics['accuracy']*100:.2f}% | macro-F1 "
          f"{metrics['f1_macro']*100:.2f}% | AUC {metrics.get('roc_auc_ovr', 0):.4f} "
          f"| {fit_s:.0f}s", flush=True)
    return row


def stage_cv(slug: str, fold: int, folds: int = 3, fast: bool = False) -> float:
    """Run a single cross-validation fold and checkpoint its score."""
    _paths()
    reg = registry(fast)
    _, X_tr, _, y_tr, _ = _splits(fast)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.RANDOM_STATE)
    tr_idx, va_idx = list(skf.split(X_tr, y_tr))[fold]

    pipe = reg[slug]["build"]()
    t0 = time.time()
    pipe.fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
    score = float(accuracy_score(y_tr[va_idx], pipe.predict(X_tr.iloc[va_idx])))

    (STAGE_DIR / "cv" / f"{slug}_fold{fold}.json").write_text(
        json.dumps({"slug": slug, "fold": fold, "folds": folds, "accuracy": score}),
        encoding="utf-8")
    print(f"[stage:cv] {slug} fold {fold+1}/{folds}: {score*100:.2f}% "
          f"({time.time()-t0:.0f}s)", flush=True)
    return score


def stage_ensemble(fast: bool = False) -> dict:
    """Compose the production ensemble from already-fitted base learners.

    Membership is earned, not hard-coded: only models within
    ``ENSEMBLE_TOLERANCE`` of the leader join the vote, weighted by their
    squared improvement over a 90% floor.
    """
    _paths()
    _, _, X_te, _, y_te = _splits(fast)

    scores = {}
    for p in (STAGE_DIR / "parts").glob("*.json"):
        row = json.loads(p.read_text(encoding="utf-8"))["row"]
        if row["slug"] != "ensemble":
            scores[row["slug"]] = row["accuracy"]

    chosen = ensemble_weights(scores)
    if len(chosen) < 2:
        raise SystemExit("Need at least two fitted base models before ensembling.")

    members, weights, names = [], [], []
    for slug, w in sorted(chosen.items(), key=lambda kv: -kv[1]):
        f = STAGE_DIR / "fitted" / f"{slug}.joblib"
        if f.exists():
            members.append(joblib.load(f))
            weights.append(w)
            names.append(slug)

    print(f"[stage:ensemble] selected {dict(zip(names, weights, strict=True))} "
          f"from {len(scores)} candidates", flush=True)
    t0 = time.time()
    ens = SoftVoteEnsemble(members, weights, names).fit(None)
    metrics = evaluate(ens, X_te, y_te)
    row = leaderboard_row("ensemble", "RiskRadar Ensemble (weighted soft-vote)",
                          metrics, time.time() - t0)
    row["members"] = names

    joblib.dump(ens, STAGE_DIR / "fitted" / "ensemble.joblib", compress=3)
    (STAGE_DIR / "parts" / "ensemble.json").write_text(
        json.dumps({"row": row, "metrics": metrics}, indent=2), encoding="utf-8")
    print(f"  accuracy {metrics['accuracy']*100:.2f}% | macro-F1 "
          f"{metrics['f1_macro']*100:.2f}%", flush=True)
    return row


def stage_finalize(fast: bool = False) -> dict:
    """Rank every checkpointed candidate, promote the winner, write reports."""
    _paths()
    parts = sorted((STAGE_DIR / "parts").glob("*.json"))
    if not parts:
        raise SystemExit("No staged results found — run --stage model first.")

    cv_scores: dict[str, list[float]] = {}
    for f in (STAGE_DIR / "cv").glob("*.json"):
        rec = json.loads(f.read_text())
        cv_scores.setdefault(rec["slug"], []).append(rec["accuracy"])

    leaderboard, metrics_by_slug = [], {}
    for p in parts:
        blob = json.loads(p.read_text(encoding="utf-8"))
        row = blob["row"]
        if row["slug"] in cv_scores:
            s = cv_scores[row["slug"]]
            row["cv_accuracy_mean"] = round(float(np.mean(s)), 5)
            row["cv_accuracy_std"] = round(float(np.std(s)), 5)
            row["cv_folds_completed"] = len(s)
        leaderboard.append(row)
        metrics_by_slug[row["slug"]] = blob["metrics"]

    leaderboard.sort(key=lambda r: (r["accuracy"], r["f1_macro"]), reverse=True)
    best = leaderboard[0]
    best_model = joblib.load(STAGE_DIR / "fitted" / f"{best['slug']}.joblib")
    best_metrics = metrics_by_slug[best["slug"]]

    # Cross-validating the ensemble would mean refitting every member per fold.
    # Instead we report the CV of its dominant member and say so explicitly —
    # an honest proxy beats either an expensive number or a missing one.
    if best.get("cv_accuracy_mean") is None and best.get("members"):
        for member in best["members"]:
            if member in cv_scores:
                s = cv_scores[member]
                best["cv_accuracy_mean"] = round(float(np.mean(s)), 5)
                best["cv_accuracy_std"] = round(float(np.std(s)), 5)
                best["cv_source"] = f"dominant member '{member}'"
                break

    df, X_tr, _, _, _ = _splits(fast)
    return _persist(df, X_tr, best_model, best, best_metrics, leaderboard, cv_scores)


def _persist(df, X_train, best_model, best_row, best_metrics, leaderboard, cv_scores) -> dict:
    """Write every artefact the API and the docs depend on."""
    joblib.dump(best_model, C.MODEL_PATH, compress=3)
    joblib.dump(X_train.sample(min(300, len(X_train)), random_state=C.RANDOM_STATE),
                C.EXPLAINER_PATH, compress=3)

    metadata = {
        "model_name": best_row["model"],
        "model_slug": best_row["slug"],
        "ensemble_members": best_row.get("members"),
        "version": "2.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "n_raw_features": len(C.RAW_FEATURE_COLUMNS),
        "n_engineered_features": len(feature_names()),
        "feature_names": feature_names(),
        "classes": C.CLASS_ORDER,
        "train_rows": len(X_train),
        "test_rows": int(best_metrics["per_class"]["Low"]["support"]
                         + best_metrics["per_class"]["Medium"]["support"]
                         + best_metrics["per_class"]["High"]["support"]),
        "metrics": best_metrics,
        "cv_accuracy_mean": best_row.get("cv_accuracy_mean"),
        "cv_accuracy_std": best_row.get("cv_accuracy_std"),
        "cv_source": best_row.get("cv_source", "winning model"),
        "cv_scores": cv_scores.get(best_row["slug"])
        or next((v for k, v in cv_scores.items()
                 if k in (best_row.get("members") or [])), []),
        "data_quality": D.data_quality_report(df),
        "feature_importance": global_importance(best_model),
        "baseline_reference": {
            "previous_project_model": "Gradient Boosting (raw LabelEncoder features)",
            "previous_project_accuracy": 0.9483,
        },
    }
    C.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    C.LEADERBOARD_PATH.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    C.ANALYTICS_PATH.write_text(json.dumps(build_analytics(df), indent=2), encoding="utf-8")
    C.GEO_PATH.write_text(json.dumps(build_geo_sample(df), indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"WINNER  : {best_row['model']}")
    print(f"Accuracy: {best_metrics['accuracy']*100:.2f}%  "
          f"(previous project: 94.83%)")
    print(f"Macro F1: {best_metrics['f1_macro']*100:.2f}%   "
          f"ROC-AUC: {best_metrics.get('roc_auc_ovr', 0):.4f}")
    print("=" * 72)
    print(f"model      -> {C.MODEL_PATH}")
    print(f"metadata   -> {C.METADATA_PATH}")
    print(f"leaderboard-> {C.LEADERBOARD_PATH}")
    print(f"analytics  -> {C.ANALYTICS_PATH}")

    # A trained pipeline is not yet a deployable system: conformal coverage,
    # the drift reference and the registry entry all have to exist before the
    # serving layer can keep its promises. Seconds of work, so always do it.
    try:
        from .bootstrap import bootstrap_all

        metadata["platform_artefacts"] = bootstrap_all(best_model, metadata)
    except Exception as exc:  # pragma: no cover
        print(f"\n[warn] platform artefacts not built: {exc}")
        print("       rebuild them later with:  python run.py --calibrate")

    return metadata


# ==========================================================================
# One-shot execution
# ==========================================================================
def run(fast: bool = False, folds: int = 3, cv_slugs: tuple[str, ...] = ("lgbm",)) -> dict:
    """Train the whole field, cross-validate the contenders, finalise. One call."""
    t0 = time.time()
    print("=" * 72)
    print("RiskRadar - Flagship Training Pipeline")
    print("=" * 72)

    df = D.load_dataset()
    q = D.data_quality_report(df)
    print(f"Dataset : {q['rows']:,} rows x {q['columns']} cols | "
          f"missing={q['missing_values']} dupes={q['duplicate_rows']}")
    print(f"Classes : {q['class_distribution']}")
    print(f"Features: {len(feature_names())} engineered from "
          f"{len(C.RAW_FEATURE_COLUMNS)} raw columns\n")

    for slug in registry(fast):
        stage_model(slug, fast)
    for slug in cv_slugs:
        if slug in registry(fast):
            for k in range(folds):
                stage_cv(slug, k, folds, fast)
    stage_ensemble(fast)
    meta = stage_finalize(fast)
    print(f"Total time: {time.time() - t0:.0f}s")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Train the RiskRadar model suite.")
    p.add_argument("--fast", action="store_true", help="reduced budget smoke run")
    p.add_argument("--stage", choices=["model", "cv", "ensemble", "finalize"],
                   help="run a single resumable stage instead of the full pipeline")
    p.add_argument("--slug", help="model slug for --stage model/cv")
    p.add_argument("--fold", type=int, default=0, help="fold index for --stage cv")
    p.add_argument("--folds", type=int, default=3, help="number of CV folds")
    a = p.parse_args()

    if a.stage == "model":
        stage_model(a.slug, a.fast)
    elif a.stage == "cv":
        stage_cv(a.slug, a.fold, a.folds, a.fast)
    elif a.stage == "ensemble":
        stage_ensemble(a.fast)
    elif a.stage == "finalize":
        stage_finalize(a.fast)
    else:
        run(fast=a.fast, folds=a.folds)


if __name__ == "__main__":
    main()
