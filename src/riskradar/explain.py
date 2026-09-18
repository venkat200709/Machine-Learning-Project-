"""Explainability: why did the model say what it said?

A safety system that outputs "HIGH RISK" and nothing else is not deployable —
an operator has to be able to challenge the verdict. This module produces
per-prediction attributions using SHAP (exact Shapley values via
``TreeExplainer`` on the gradient-boosted member of the ensemble) with a
deterministic fallback so the API never hard-fails when SHAP is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import engineer, pretty
from .models import SoftVoteEnsemble

try:
    import shap

    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False


def _tree_member(model):
    """Find the tree-based estimator that carries most of the ensemble's weight.

    SHAP's ``TreeExplainer`` gives exact, fast attributions but only for tree
    models. In the production ensemble the LightGBM member holds ~83% of the
    vote, so explaining it is a faithful account of the ensemble's behaviour —
    and we say so explicitly in the response payload.
    """
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model

    if isinstance(est, SoftVoteEnsemble):
        order = np.argsort(est._w)[::-1]
        for i in order:
            sub = est.estimators[i]
            inner = sub.named_steps["model"] if hasattr(sub, "named_steps") else sub
            if hasattr(inner, "feature_importances_"):
                return inner, float(est._w[i]), est.member_names[i]
        return None, 0.0, ""

    if hasattr(est, "feature_importances_"):
        return est, 1.0, "model"
    return None, 0.0, ""


class RiskExplainer:
    """Lazily-built SHAP explainer with a graceful degradation path."""

    def __init__(self, model, background: pd.DataFrame | None = None):
        self.model = model
        self.background = background
        self._explainer = None
        self._member, self._weight, self._member_name = _tree_member(model)
        self._baseline: np.ndarray | None = None
        self.mode = "unavailable"

    # ------------------------------------------------------------------
    def _ensure(self) -> None:
        if self._explainer is not None or self._member is None:
            return
        if not HAS_SHAP:
            self.mode = "importance-weighted deviation (SHAP not installed)"
            return
        try:
            self._explainer = shap.TreeExplainer(self._member)
            ev = np.atleast_1d(np.asarray(self._explainer.expected_value, dtype=float))
            self._baseline = ev
            self.mode = (
                "exact SHAP (TreeExplainer)"
                if self._weight >= 0.999
                else f"exact SHAP (TreeExplainer on '{self._member_name}', "
                     f"{self._weight:.0%} of ensemble weight)"
            )
        except Exception:  # pragma: no cover
            self._explainer = None
            self.mode = "importance-weighted deviation (SHAP unavailable for this model)"

    # ------------------------------------------------------------------
    def explain(self, raw: pd.DataFrame, class_index: int, top_k: int = 10,
                features: pd.DataFrame | None = None) -> dict:
        """Attribute one prediction to its most influential inputs.

        Returns a payload the frontend renders as a waterfall: each entry has
        the engineered feature, a human label, its value and a signed impact
        (positive = pushed *toward* the predicted class). Pass ``features`` to
        reuse an already-engineered matrix instead of rebuilding it.
        """
        self._ensure()
        features = engineer(raw) if features is None else features
        names = list(features.columns)
        values = features.iloc[0].to_numpy(dtype=float)

        contributions, baseline = self._shap_contributions(features, class_index)
        if contributions is None:
            contributions, baseline = self._fallback_contributions(features, class_index)

        order = np.argsort(np.abs(contributions))[::-1][:top_k]
        drivers = [
            {
                "feature": names[i],
                "label": pretty(names[i]),
                "value": round(float(values[i]), 4),
                "impact": round(float(contributions[i]), 5),
                "direction": "increases" if contributions[i] > 0 else "decreases",
            }
            for i in order
        ]

        pushing_up = [d for d in drivers if d["impact"] > 0][:5]
        pulling_down = [d for d in drivers if d["impact"] < 0][:5]

        return {
            "method": self.mode,
            "predicted_class": C.INT_TO_CLASS[class_index],
            "baseline": round(float(baseline), 5),
            "total_contribution": round(float(contributions.sum()), 5),
            "drivers": drivers,
            "risk_factors": pushing_up,
            "protective_factors": pulling_down,
            "narrative": self._narrative(pushing_up, pulling_down, class_index),
        }

    # ------------------------------------------------------------------
    def _shap_contributions(self, features: pd.DataFrame, class_index: int):
        if self._explainer is None:
            return None, 0.0
        try:
            sv = self._explainer.shap_values(features)
            arr = np.asarray(sv)
            # Modern SHAP returns (n_samples, n_features, n_classes);
            # older versions return a list of per-class arrays.
            if arr.ndim == 3:
                contrib = arr[0, :, class_index]
            elif isinstance(sv, list):
                contrib = np.asarray(sv[class_index])[0]
            else:
                contrib = arr[0]
            base = float(self._baseline[class_index]) if self._baseline is not None and \
                len(self._baseline) > class_index else 0.0
            return np.asarray(contrib, dtype=float), base
        except Exception:  # pragma: no cover
            return None, 0.0

    def _fallback_contributions(self, features: pd.DataFrame, class_index: int):
        """Importance-weighted z-score deviation from the background median.

        Not Shapley values, and labelled as such — but it still answers "which
        inputs are unusual for this record, among the features the model cares
        about", which is the operationally useful question.
        """
        imp = getattr(self._member, "feature_importances_", None)
        imp = np.ones(features.shape[1]) if imp is None else np.asarray(imp, dtype=float)
        imp = imp / (imp.sum() or 1.0)

        if self.background is not None and len(self.background):
            bg = engineer(self.background)
            med = bg.median().to_numpy(dtype=float)
            scale = bg.std().replace(0, 1.0).to_numpy(dtype=float)
        else:
            med = np.zeros(features.shape[1])
            scale = np.ones(features.shape[1])

        z = (features.iloc[0].to_numpy(dtype=float) - med) / np.where(scale == 0, 1.0, scale)
        sign = 1.0 if class_index >= 1 else -1.0
        return np.clip(z, -6, 6) * imp * sign, 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _narrative(up: list[dict], down: list[dict], class_index: int) -> str:
        label = C.INT_TO_CLASS[class_index]
        if not up and not down:
            return f"Model predicts {label} risk."
        parts = [f"Assessed as {label} risk."]
        if up:
            parts.append(
                "Driven upward by " + ", ".join(d["label"].lower() for d in up[:3]) + "."
            )
        if down:
            parts.append(
                "Offset by " + ", ".join(d["label"].lower() for d in down[:3]) + "."
            )
        return " ".join(parts)


LEVERS: list[tuple[str, str, list[float], str]] = [
    ("Working_Streetlights", "Repair broken streetlights",
     [0.25, 0.5, 1.0], "repair_lights"),
    ("CCTV_Count", "Install additional CCTV cameras",
     [10, 25, 50, 100], "add"),
    ("Police_Distance_km", "Add a police outpost nearby",
     [0.5, 0.3, 0.1], "scale"),
    ("Footfall", "Improve street activity (lighting, vendors, transit)",
     [1.5, 2.5, 4.0], "scale"),
]


def _apply_lever(row: pd.Series, kind: str, column: str, step: float) -> tuple[pd.Series, str]:
    trial = row.copy()
    if kind == "repair_lights":
        broken = float(trial["Broken_Streetlights"])
        fixed = round(broken * step)
        trial["Working_Streetlights"] = trial["Working_Streetlights"] + fixed
        trial["Broken_Streetlights"] = max(0.0, broken - fixed)
        return trial, f"repair {int(fixed)} of {int(broken)} broken lights"
    if kind == "add":
        trial[column] = trial[column] + step
        return trial, f"add {int(step)} cameras (to {int(trial[column])})"
    if kind == "scale" and column == "Police_Distance_km":
        trial[column] = trial[column] * step
        return trial, f"reduce police distance to {float(trial[column]):.1f} km"
    trial[column] = trial[column] * step
    return trial, f"raise footfall to {int(trial[column])}/hour"


def counterfactual_scan(model, raw: pd.DataFrame, current_class: int) -> list[dict]:
    """What would actually make this place safer?

    Sweeps the levers a city authority can genuinely pull — lighting, CCTV,
    police proximity, street activity — and reports the smallest intervention
    per lever that moves the prediction down a risk band. This is what turns a
    classifier into a planning tool.

    All trials are scored in a **single batched forward pass**; looping one
    prediction at a time made this the slowest part of the request by an order
    of magnitude.
    """
    if current_class == 0:
        return []

    base = raw.iloc[0]
    trials, meta = [], []
    for column, action, steps, kind in LEVERS:
        for step in steps:
            trial, detail = _apply_lever(base, kind, column, step)
            trials.append(trial)
            meta.append((action, detail))

    if not trials:
        return []

    predictions = np.asarray(model.predict(pd.DataFrame(trials).reset_index(drop=True)))

    results: list[dict] = []
    seen: set[str] = set()
    for (action, detail), new_class in zip(meta, predictions, strict=True):
        new_class = int(new_class)
        if new_class < current_class and action not in seen:
            seen.add(action)
            results.append({
                "action": action,
                "detail": detail,
                "new_risk": C.INT_TO_CLASS[new_class],
                "bands_improved": current_class - new_class,
            })
    return sorted(results, key=lambda r: -r["bands_improved"])
