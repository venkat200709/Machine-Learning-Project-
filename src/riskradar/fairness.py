"""Algorithmic fairness audit.

Why a safety model needs this more than most
--------------------------------------------
RiskRadar's output is not a passive score. If a city routes patrols, lighting
budgets and CCTV toward "High risk" areas, the model decides where public money
goes. Two failure modes follow directly:

1. **Over-flagging poor, dense neighbourhoods.** Crime *reporting* rates
   correlate with policing intensity, which correlates with income. A model
   trained on reported crime can learn "heavily policed" and output
   "dangerous", justifying more policing — a feedback loop that looks like
   accuracy.
2. **Under-serving the areas that need help.** If recall on the High class is
   worse in low-infrastructure areas, the model is quietest exactly where the
   danger is greatest, and the intervention budget flows elsewhere.

Choosing the groups
-------------------
The dataset carries no demographic attributes, and inventing them would be
worse than useless. Instead we audit across **operational strata** that are
genuinely present in the data and are the recognised proxies for equity of
service in urban planning:

* ``density`` — population-density quartile. The standard geographic proxy for
  informal and low-income settlement.
* ``infrastructure`` — lighting/CCTV/transit provision quartile. Directly
  measures whether an area is already served.
* ``time`` — night versus day. Coverage should not collapse after dark.
* ``land_use`` — commercial, residential, mixed, or neither.
* ``policing`` — distance-to-police quartile: is the model worse where help is
  furthest away?

Metrics
-------
Both of the classical fairness families are reported, because they measure
different things and cannot both be satisfied unless the model is perfect:

* **Independence** — selection rate parity. Demographic-parity difference and
  the four-fifths disparate-impact ratio.
* **Separation** — equalised odds. TPR and FPR gaps across groups.
* **Sufficiency** — calibration parity. Per-group expected calibration error,
  i.e. does "80% confident" mean the same thing in every group?

A gap is *not* automatically a bug. Genuinely more dangerous areas should be
flagged more often. The report therefore separates **selection-rate** gaps
(which may be legitimate) from **error-rate** gaps (which are much harder to
defend), and says so in the verdict.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger("riskradar.fairness")

# Four-fifths rule, from US employment-discrimination practice; the most widely
# cited numeric threshold for disparate impact.
DISPARATE_IMPACT_FLOOR = 0.80
# Above this, an error-rate gap between groups stops being noise.
EQUALISED_ODDS_TOLERANCE = 0.05
CALIBRATION_TOLERANCE = 0.05
MIN_GROUP_SIZE = 30


# ==========================================================================
# Group definitions
# ==========================================================================
def _quartile_labels(series: pd.Series, names: tuple[str, ...]) -> pd.Series:
    """Quartile bucketing that survives ties and degenerate distributions."""
    try:
        binned = pd.qcut(series.rank(method="first"), q=4, labels=list(names))
        return binned.astype(str)
    except Exception:  # pragma: no cover
        return pd.Series([names[0]] * len(series), index=series.index)


def density_groups(raw: pd.DataFrame) -> pd.Series:
    return _quartile_labels(
        raw["Population_Density"].astype(float),
        ("Q1 sparsest", "Q2", "Q3", "Q4 densest"),
    )


def infrastructure_groups(raw: pd.DataFrame) -> pd.Series:
    """Composite provision score: lighting health, CCTV and transit access."""
    lights = raw["Working_Streetlights"].astype(float)
    total = (raw["Streetlight_Count"].astype(float)).replace(0, np.nan)
    health = (lights / total).fillna(0.0)
    cctv = np.log1p(raw["CCTV_Count"].astype(float))
    transit = np.log1p(raw["Bus_Stop_Count"].astype(float))
    score = health + 0.5 * (cctv / (cctv.max() or 1)) + 0.5 * (transit / (transit.max() or 1))
    return _quartile_labels(score, ("Q1 least served", "Q2", "Q3", "Q4 best served"))


def time_groups(raw: pd.DataFrame) -> pd.Series:
    hour = raw["Hour"].astype(int)
    return pd.Series(
        np.where(hour.isin(sorted(C.LATE_NIGHT_HOURS)), "Late night (00-04)",
                 np.where(hour.isin(sorted(C.NIGHT_HOURS)), "Night (21-05)", "Day (06-20)")),
        index=raw.index,
    )


def land_use_groups(raw: pd.DataFrame) -> pd.Series:
    commercial = raw["Commercial_Area"].astype(int)
    residential = raw["Residential_Area"].astype(int)
    return pd.Series(
        np.select(
            [(commercial == 1) & (residential == 1),
             (commercial == 1) & (residential == 0),
             (commercial == 0) & (residential == 1)],
            ["Mixed use", "Commercial", "Residential"],
            default="Undesignated",
        ),
        index=raw.index,
    )


def policing_groups(raw: pd.DataFrame) -> pd.Series:
    return _quartile_labels(
        raw["Police_Distance_km"].astype(float),
        ("Q1 closest", "Q2", "Q3", "Q4 furthest"),
    )


GROUP_DEFINITIONS: dict[str, dict[str, Any]] = {
    "density": {
        "builder": density_groups,
        "label": "Population density quartile",
        "rationale": "Standard geographic proxy for low-income and informal settlement.",
    },
    "infrastructure": {
        "builder": infrastructure_groups,
        "label": "Safety-infrastructure provision quartile",
        "rationale": "Detects whether the model performs worse in already under-served areas.",
    },
    "time": {
        "builder": time_groups,
        "label": "Time of day",
        "rationale": "Coverage must not collapse after dark, when risk is highest.",
    },
    "land_use": {
        "builder": land_use_groups,
        "label": "Land-use designation",
        "rationale": "Residential areas must not be systematically under-assessed.",
    },
    "policing": {
        "builder": policing_groups,
        "label": "Distance-to-police quartile",
        "rationale": "Errors are most costly where help is furthest away.",
    },
}


# ==========================================================================
# Metrics
# ==========================================================================
def _expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                                n_bins: int = 10) -> float:
    """Weighted gap between stated confidence and realised accuracy."""
    if len(confidence) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        weight = mask.mean()
        total += weight * abs(confidence[mask].mean() - correct[mask].mean())
    return round(float(total), 5)


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   confidence: np.ndarray, positive: int) -> dict:
    """One group's confusion-derived statistics for a chosen positive class."""
    actual_pos = y_true == positive
    pred_pos = y_pred == positive

    tp = int(np.sum(actual_pos & pred_pos))
    fp = int(np.sum(~actual_pos & pred_pos))
    fn = int(np.sum(actual_pos & ~pred_pos))
    tn = int(np.sum(~actual_pos & ~pred_pos))

    correct = (y_true == y_pred).astype(float)

    return {
        "n": len(y_true),
        "selection_rate": round(float(pred_pos.mean()), 5),
        "base_rate": round(float(actual_pos.mean()), 5),
        "accuracy": round(float(correct.mean()), 5),
        "tpr": round(tp / (tp + fn), 5) if (tp + fn) else None,
        "fpr": round(fp / (fp + tn), 5) if (fp + tn) else None,
        "precision": round(tp / (tp + fp), 5) if (tp + fp) else None,
        "false_omission_rate": round(fn / (fn + tn), 5) if (fn + tn) else None,
        "mean_confidence": round(float(confidence.mean()), 5) if len(confidence) else None,
        "calibration_error": _expected_calibration_error(confidence, correct),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _gap(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(max(clean) - min(clean), 5) if len(clean) >= 2 else None


def _ratio(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None and v > 0]
    if len(clean) < 2:
        return None
    return round(min(clean) / max(clean), 5)


# ==========================================================================
# Audit
# ==========================================================================
def audit_group(raw: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                confidence: np.ndarray, groups: pd.Series, *,
                positive_class: str = "High") -> dict:
    """Fairness statistics for one grouping dimension."""
    positive = C.CLASS_TO_INT[positive_class]
    per_group: dict[str, dict] = {}

    for name in sorted(groups.dropna().unique()):
        mask = (groups == name).to_numpy()
        if mask.sum() < MIN_GROUP_SIZE:
            continue
        per_group[str(name)] = _group_metrics(
            y_true[mask], y_pred[mask], confidence[mask], positive
        )

    if len(per_group) < 2:
        return {"available": False, "reason": "Not enough populated groups to compare."}

    selection = [g["selection_rate"] for g in per_group.values()]
    tprs = [g["tpr"] for g in per_group.values()]
    fprs = [g["fpr"] for g in per_group.values()]
    accs = [g["accuracy"] for g in per_group.values()]
    eces = [g["calibration_error"] for g in per_group.values()]

    dp_difference = _gap(selection)
    disparate_impact = _ratio(selection)
    tpr_gap = _gap(tprs)
    fpr_gap = _gap(fprs)
    accuracy_gap = _gap(accs)
    calibration_gap = _gap(eces)
    equalised_odds = max(v for v in (tpr_gap or 0.0, fpr_gap or 0.0))

    findings: list[dict] = []

    if disparate_impact is not None and disparate_impact < DISPARATE_IMPACT_FLOOR:
        worst = min(per_group, key=lambda k: per_group[k]["selection_rate"])
        best = max(per_group, key=lambda k: per_group[k]["selection_rate"])
        # Selection-rate differences track real differences in danger, so this
        # is context, not a verdict — unless the base rates say otherwise.
        base_ratio = _ratio([g["base_rate"] for g in per_group.values()])
        justified = base_ratio is not None and abs(base_ratio - disparate_impact) < 0.15
        findings.append({
            "severity": "info" if justified else "medium",
            "metric": "disparate_impact",
            "message": (
                f"'{best}' is flagged High {per_group[best]['selection_rate'] / max(per_group[worst]['selection_rate'], 1e-9):.1f}× "
                f"more often than '{worst}' (ratio {disparate_impact:.2f}, below the 0.80 four-fifths threshold)."
                + (" This tracks the underlying base rates, so it reflects real risk differences rather than bias."
                   if justified else
                   " The gap is larger than the difference in actual risk, which warrants investigation.")
            ),
        })

    if tpr_gap is not None and tpr_gap > EQUALISED_ODDS_TOLERANCE:
        worst = min((k for k in per_group if per_group[k]["tpr"] is not None),
                    key=lambda k: per_group[k]["tpr"])
        findings.append({
            "severity": "high",
            "metric": "equal_opportunity",
            "message": (
                f"Recall on genuinely {positive_class}-risk areas varies by {tpr_gap:.1%} "
                f"across groups; '{worst}' is worst at {per_group[worst]['tpr']:.1%}. "
                "Missed danger is concentrated in one group."
            ),
        })

    if fpr_gap is not None and fpr_gap > EQUALISED_ODDS_TOLERANCE:
        worst = max((k for k in per_group if per_group[k]["fpr"] is not None),
                    key=lambda k: per_group[k]["fpr"])
        findings.append({
            "severity": "medium",
            "metric": "false_positive_parity",
            "message": (
                f"False-alarm rate varies by {fpr_gap:.1%}; '{worst}' is over-flagged most. "
                "Sustained over-flagging misdirects patrol and lighting budget."
            ),
        })

    if calibration_gap is not None and calibration_gap > CALIBRATION_TOLERANCE:
        findings.append({
            "severity": "medium",
            "metric": "calibration_parity",
            "message": (
                f"Calibration error differs by {calibration_gap:.3f} across groups — "
                "a stated confidence does not mean the same thing everywhere."
            ),
        })

    if accuracy_gap is not None and accuracy_gap > 0.05:
        worst = min(per_group, key=lambda k: per_group[k]["accuracy"])
        findings.append({
            "severity": "high",
            "metric": "accuracy_parity",
            "message": (
                f"Accuracy spans {accuracy_gap:.1%} across groups; '{worst}' is lowest "
                f"at {per_group[worst]['accuracy']:.1%}."
            ),
        })

    severities = {f["severity"] for f in findings}
    if "high" in severities:
        verdict = "fail"
    elif {"medium"} & severities:
        verdict = "review"
    else:
        verdict = "pass"

    return {
        "available": True,
        "positive_class": positive_class,
        "verdict": verdict,
        "groups": per_group,
        "metrics": {
            "demographic_parity_difference": dp_difference,
            "disparate_impact_ratio": disparate_impact,
            "equalised_odds_difference": round(equalised_odds, 5),
            "true_positive_rate_gap": tpr_gap,
            "false_positive_rate_gap": fpr_gap,
            "accuracy_gap": accuracy_gap,
            "calibration_gap": calibration_gap,
        },
        "findings": findings,
    }


def audit(model, raw: pd.DataFrame, y_true: np.ndarray, *,
          positive_class: str = "High",
          dimensions: list[str] | None = None) -> dict:
    """Full fairness report across every grouping dimension."""
    proba = np.asarray(model.predict_proba(raw), dtype=float)
    y_pred = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    y_true = np.asarray(y_true).astype(int)

    raw = raw.reset_index(drop=True)
    selected = dimensions or list(GROUP_DEFINITIONS)

    results: dict[str, Any] = {}
    for key in selected:
        spec = GROUP_DEFINITIONS.get(key)
        if spec is None:
            continue
        try:
            builder: Callable[[pd.DataFrame], pd.Series] = spec["builder"]
            groups = builder(raw).reset_index(drop=True)
            report = audit_group(
                raw, y_true, y_pred, confidence, groups, positive_class=positive_class
            )
            report["label"] = spec["label"]
            report["rationale"] = spec["rationale"]
            results[key] = report
        except Exception as exc:  # pragma: no cover
            log.warning("fairness dimension %s failed: %s", key, exc)
            results[key] = {"available": False, "reason": str(exc)}

    verdicts = [r.get("verdict") for r in results.values() if r.get("available")]
    if "fail" in verdicts:
        overall = "fail"
    elif "review" in verdicts:
        overall = "review"
    elif verdicts:
        overall = "pass"
    else:
        overall = "unavailable"

    all_findings = [
        {**f, "dimension": key}
        for key, r in results.items()
        for f in r.get("findings", [])
    ]
    order = {"high": 0, "medium": 1, "info": 2}
    all_findings.sort(key=lambda f: order.get(f["severity"], 3))

    return {
        "available": bool(verdicts),
        "n_evaluated": len(y_true),
        "positive_class": positive_class,
        "overall_verdict": overall,
        "dimensions": results,
        "findings": all_findings,
        "summary": _summary(overall, all_findings),
        "method_note": (
            "Groups are operational strata (density, infrastructure provision, time, "
            "land use, policing distance), not demographic attributes — the dataset "
            "carries none, and fabricating them would be worse than omitting them. "
            "These are the accepted equity-of-service proxies in urban planning."
        ),
    }


def _summary(verdict: str, findings: list[dict]) -> str:
    if verdict == "pass":
        return (
            "No disparity above threshold. Error rates, calibration and accuracy are "
            "consistent across density, infrastructure, time, land use and policing strata."
        )
    if verdict == "unavailable":
        return "Not enough data to audit fairness."
    high = [f for f in findings if f["severity"] == "high"]
    medium = [f for f in findings if f["severity"] == "medium"]
    parts = []
    if high:
        parts.append(f"{len(high)} serious disparity finding(s) in model error rates.")
    if medium:
        parts.append(f"{len(medium)} finding(s) needing review.")
    parts.append(
        "Error-rate gaps matter more than selection-rate gaps: flagging genuinely "
        "dangerous areas more often is correct behaviour, but missing danger more "
        "often in one group is not."
    )
    return " ".join(parts)


def audit_from_dataset(model, *, sample: int | None = 20_000,
                       positive_class: str = "High") -> dict:
    """Run the audit on the canonical hold-out split."""
    from .data import load_dataset, split_xy, stratified_split

    df = load_dataset()
    X, y = split_xy(df)
    _, X_test, _, y_test = stratified_split(X, y)

    if sample and len(X_test) > sample:
        rng = np.random.default_rng(C.RANDOM_STATE)
        idx = rng.choice(len(X_test), size=sample, replace=False)
        X_test = X_test.iloc[idx]
        y_test = y_test[idx]

    return audit(model, X_test, y_test, positive_class=positive_class)
