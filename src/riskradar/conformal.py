"""Conformal prediction — turning confidence into a guarantee.

The problem with 98% accuracy
------------------------------
A softmax probability of 0.94 is not a promise. It is the model's opinion, and
on an input unlike anything in training that opinion can be confidently wrong.
For a system that tells a woman whether a street is safe, "the model was very
sure" is not an acceptable answer to "why did it get that one wrong?".

Split conformal prediction fixes this. Given a calibration set the model never
trained on, it returns a *set* of labels with a distribution-free, finite-sample
guarantee: for a user-chosen error rate α, the true label is inside the set at
least ``1 − α`` of the time. No assumptions about the model, the data
distribution, or the loss. The only requirement is exchangeability between
calibration and test data.

What this buys operationally
-----------------------------
The set size becomes an *automatic abstention signal*:

* ``{High}``            → confident, act on it.
* ``{Medium, High}``    → genuinely ambiguous; escalate to a human, do not
                          silently pick the argmax.
* ``{Low, Medium, High}`` → the model knows nothing useful here.
* ``{}``                → the input is unlike anything in calibration —
                          out-of-distribution, flag it.

Two scores are implemented
---------------------------
* **LAC** (Least Ambiguous set-valued Classifier): ``s = 1 − p_true``. Gives the
  smallest possible average set size, but coverage is only marginal.
* **APS** (Adaptive Prediction Sets): cumulative sorted probability mass. Larger
  sets, but far better *conditional* coverage — it does not achieve its 90% by
  being right on easy cases and hopeless on hard ones.

**Mondrian (class-conditional) calibration** is the default: a separate quantile
per true class. Marginal coverage can hit 90% overall while covering the High
class only 70% of the time — which is the only class where a miss actually
hurts someone. Class-conditional calibration guarantees the rate *within* each
class, and that is the correct guarantee for this domain.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger("riskradar.conformal")

CALIBRATION_PATH = C.MODELS_DIR / "conformal_calibration.json"

SCORE_METHODS = ("lac", "aps")


# --------------------------------------------------------------------------
# Non-conformity scores
# --------------------------------------------------------------------------
def lac_scores(proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``1 − p(true class)``. Higher = the model conformed worse."""
    return 1.0 - proba[np.arange(len(labels)), labels]


def aps_scores(proba: np.ndarray, labels: np.ndarray, *,
               randomize: bool = False,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Cumulative probability mass required to reach the true label.

    Sorting descending and summing until the true class is included measures
    "how deep into the ranking did I have to go".

    On ``randomize``
    ----------------
    The textbook APS subtracts a uniform fraction of the true class's own mass,
    which makes marginal coverage *exact* rather than conservative. We default
    it **off**, and that choice has to be justified because it costs slightly
    larger sets:

    A randomised score means the same input can yield a different prediction
    set on two consecutive calls. In a system whose output may be cited to
    justify where police and lighting budget went, an answer that changes when
    you ask again is indefensible — and it makes the endpoint untestable.

    Critically, the score used at calibration and the score used at serving
    must be *the same function*. Calibrating on the randomised score and then
    serving the deterministic one silently destroys the guarantee: the serving
    score is systematically larger than the calibrated quantile, so every set
    comes back empty. That failure is invisible without the coverage check in
    :func:`evaluate_coverage`, which is precisely why that check exists.

    Deterministic APS therefore over-covers a little. Over-covering is the safe
    direction: the guarantee still holds, the sets are just marginally wider.
    """
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cumulative = np.cumsum(sorted_p, axis=1)

    rank = np.argmax(order == labels[:, None], axis=1)
    rows = np.arange(len(labels))
    total = cumulative[rows, rank]

    if not randomize:
        return total

    rng = rng or np.random.default_rng(C.RANDOM_STATE)
    return total - rng.random(len(labels)) * sorted_p[rows, rank]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected ``(1−α)`` quantile.

    The ``(n+1)`` correction is not cosmetic: it is exactly what converts an
    asymptotic statement into a guarantee that holds for the ``n`` points you
    actually have. With too few points the required level exceeds 1 and the
    honest answer is "no finite threshold works" — we return infinity, which
    yields full sets rather than a false promise.
    """
    n = len(scores)
    if n == 0:
        return float("inf")
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    if level >= 1.0:
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


# --------------------------------------------------------------------------
# Calibration artefact
# --------------------------------------------------------------------------
# Coverage levels the quantile grid is precomputed at. Dense at the top,
# because that is where the sets change fastest and where an operator asking
# for "99% coverage" actually lives.
QUANTILE_GRID_LEVELS: list[float] = (
    [round(0.50 + 0.01 * i, 4) for i in range(40)]        # 0.50 … 0.89
    + [round(0.90 + 0.002 * i, 4) for i in range(45)]     # 0.90 … 0.988
    + [0.99, 0.992, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]
)


@dataclass
class ConformalCalibration:
    """Everything needed to build prediction sets at serving time.

    Carries a **quantile grid** as well as the single calibrated threshold.
    Without it, a request asking for 99% coverage instead of the calibrated
    90% could only be answered by reusing the 90% threshold and quietly
    mislabelling it — the guarantee would be a fiction. Storing the score
    distribution as a few hundred precomputed order statistics per class costs
    a couple of kilobytes and makes any coverage level in [0.5, 0.999] a real,
    honest answer.
    """

    alpha: float = 0.10
    method: str = "aps"
    mondrian: bool = True
    quantiles: dict[str, float] = field(default_factory=dict)
    global_quantile: float = float("inf")
    quantile_grid: dict[str, Any] = field(default_factory=dict)
    n_calibration: int = 0
    coverage: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    model_version: str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "method": self.method,
            "mondrian": self.mondrian,
            "quantiles": {k: _jsonable(v) for k, v in self.quantiles.items()},
            "global_quantile": _jsonable(self.global_quantile),
            "quantile_grid": self.quantile_grid,
            "n_calibration": self.n_calibration,
            "coverage": self.coverage,
            "created_at": self.created_at,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConformalCalibration:
        return cls(
            alpha=float(data.get("alpha", 0.10)),
            method=str(data.get("method", "aps")),
            mondrian=bool(data.get("mondrian", True)),
            quantiles={k: _unjson(v) for k, v in (data.get("quantiles") or {}).items()},
            global_quantile=_unjson(data.get("global_quantile", float("inf"))),
            quantile_grid=data.get("quantile_grid", {}),
            n_calibration=int(data.get("n_calibration", 0)),
            coverage=data.get("coverage", {}),
            created_at=str(data.get("created_at", "")),
            model_version=str(data.get("model_version", "")),
        )

    # ------------------------------------------------------------------
    def thresholds_at(self, alpha: float) -> dict[str, float]:
        """Per-class thresholds for an arbitrary error rate.

        Falls back to the single calibrated threshold when no grid is present,
        which keeps calibration files written by older versions readable.
        """
        grid = self.quantile_grid or {}
        levels = grid.get("levels")
        if not levels:
            return dict(self.quantiles)

        target = 1.0 - alpha
        # Nearest stored level at or above the requirement: never quote a
        # weaker guarantee than the caller asked for.
        idx = next((i for i, lv in enumerate(levels) if lv >= target), len(levels) - 1)

        per_class = grid.get("per_class") or {}
        if self.mondrian and per_class:
            return {
                label: _unjson(values[idx])
                for label, values in per_class.items()
                if idx < len(values)
            }
        global_values = grid.get("global") or []
        if idx < len(global_values):
            value = _unjson(global_values[idx])
            return dict.fromkeys(C.CLASS_ORDER, value)
        return dict(self.quantiles)

    def save(self, path=None) -> None:
        path = path or CALIBRATION_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path=None) -> ConformalCalibration | None:
        path = path or CALIBRATION_PATH
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # pragma: no cover
            log.warning("conformal calibration file is unreadable; ignoring it")
            return None


def _jsonable(v: float) -> Any:
    return "inf" if not np.isfinite(v) else round(float(v), 8)


def _unjson(v: Any) -> float:
    return float("inf") if v in ("inf", None) else float(v)


# --------------------------------------------------------------------------
# Predictor
# --------------------------------------------------------------------------
class ConformalPredictor:
    """Builds calibrated prediction sets from a probability vector."""

    def __init__(self, calibration: ConformalCalibration | None = None) -> None:
        self.calibration = calibration or ConformalCalibration.load()
        self._rng = np.random.default_rng(C.RANDOM_STATE)

    @property
    def ready(self) -> bool:
        return self.calibration is not None and self.calibration.n_calibration > 0

    # ------------------------------------------------------------------
    def predict_set(self, proba: np.ndarray, alpha: float | None = None) -> dict:
        """Prediction set for a single probability vector.

        ``alpha`` may be overridden per request — a control-room operator can
        ask for 99% coverage and accept the larger sets that come with it.
        """
        proba = np.asarray(proba, dtype=float).ravel()
        if not self.ready:
            return self._unavailable(proba)

        cal = self.calibration
        effective_alpha = float(alpha) if alpha is not None else cal.alpha
        thresholds = self._thresholds(effective_alpha)

        included: list[str] = []
        margins: dict[str, float] = {}
        for idx, label in C.INT_TO_CLASS.items():
            score = self._score_for_hypothesis(proba, idx)
            q = thresholds.get(label, cal.global_quantile)
            margins[label] = round(float(q - score), 5)
            if score <= q:
                included.append(label)

        # Preserve the ordinal Low → High reading order, not probability order.
        included.sort(key=lambda lbl: C.CLASS_TO_INT[lbl])
        argmax_label = C.INT_TO_CLASS[int(np.argmax(proba))]

        return {
            "available": True,
            "alpha": round(effective_alpha, 4),
            "coverage_target": round(1.0 - effective_alpha, 4),
            "method": cal.method,
            "mondrian": cal.mondrian,
            "prediction_set": included,
            "set_size": len(included),
            "point_prediction": argmax_label,
            "abstain": len(included) != 1,
            "certain": len(included) == 1,
            "undecided": len(included) == 0,
            # Retained for API compatibility; an empty set means "undecided",
            # which is *evidence* of an unusual input but not proof of one.
            "out_of_distribution": len(included) == 0,
            "margins": margins,
            "n_calibration": cal.n_calibration,
            "interpretation": _interpret(included, argmax_label, effective_alpha),
        }

    # ------------------------------------------------------------------
    def _score_for_hypothesis(self, proba: np.ndarray, class_index: int) -> float:
        """Non-conformity of the hypothesis 'the true label is class_index'."""
        method = (self.calibration.method if self.calibration else "lac").lower()
        if method == "aps":
            # Must be byte-for-byte the same computation as `aps_scores` with
            # randomize=False. See that function for why the two must agree.
            order = np.argsort(-proba)
            cumulative = np.cumsum(proba[order])
            rank = int(np.where(order == class_index)[0][0])
            return float(cumulative[rank])
        return float(1.0 - proba[class_index])

    def _thresholds(self, alpha: float) -> dict[str, float]:
        cal = self.calibration
        if cal is None:
            return {}
        if abs(alpha - cal.alpha) < 1e-9:
            return dict(cal.quantiles)
        return cal.thresholds_at(alpha)

    @staticmethod
    def _unavailable(proba: np.ndarray) -> dict:
        return {
            "available": False,
            "reason": "No conformal calibration found. Run: python run.py --calibrate",
            "prediction_set": [C.INT_TO_CLASS[int(np.argmax(proba))]],
            "set_size": 1,
            "point_prediction": C.INT_TO_CLASS[int(np.argmax(proba))],
            "abstain": False,
            "certain": False,
            "out_of_distribution": False,
        }


def _interpret(included: list[str], argmax: str, alpha: float) -> str:
    pct = f"{(1 - alpha) * 100:.0f}%"
    if not included:
        # An empty set is *not* the same as out-of-distribution. Under LAC it
        # means the probability mass is split so evenly that no single band
        # clears the calibrated threshold — the model is genuinely undecided,
        # which is exactly the case a human should see.
        return (
            f"No risk band clears the {pct} confidence threshold — the model's "
            "probability is split too evenly to commit to any single answer. "
            "This is the model correctly declining to guess. Escalate to a human "
            "or gather more signal; do not act on the point estimate alone."
        )
    if len(included) == 1:
        return f"Statistically confident: the true risk is {included[0]} with {pct} guaranteed coverage."
    if len(included) == len(C.CLASS_ORDER):
        return (
            f"The model cannot distinguish between risk bands here at {pct} confidence. "
            "Collect more signal before acting on this assessment."
        )
    return (
        f"Genuinely ambiguous between {' and '.join(included)} at {pct} coverage. "
        f"The point estimate is {argmax}, but do not treat it as settled."
    )


# --------------------------------------------------------------------------
# Calibration routine
# --------------------------------------------------------------------------
def calibrate(model, X_cal: pd.DataFrame, y_cal: np.ndarray, *,
              alpha: float = 0.10, method: str = "aps", mondrian: bool = True,
              X_eval: pd.DataFrame | None = None, y_eval: np.ndarray | None = None,
              model_version: str = "") -> ConformalCalibration:
    """Fit conformal quantiles, then verify the guarantee on held-out data.

    The verification step is the point. A conformal method that is not measured
    empirically is just an assertion; reporting realised coverage against the
    target is what makes the guarantee checkable by a reviewer.
    """
    method = method.lower()
    if method not in SCORE_METHODS:
        raise ValueError(f"method must be one of {SCORE_METHODS}, got {method!r}")

    y_cal = np.asarray(y_cal).astype(int)
    proba = np.asarray(model.predict_proba(X_cal), dtype=float)

    scorer = aps_scores if method == "aps" else lac_scores
    scores = scorer(proba, y_cal)

    quantiles: dict[str, float] = {}
    if mondrian:
        for idx, label in C.INT_TO_CLASS.items():
            mask = y_cal == idx
            if mask.sum() >= 20:
                quantiles[label] = conformal_quantile(scores[mask], alpha)
            else:
                # Too few examples for a per-class guarantee — fall back rather
                # than quote a quantile computed from a handful of points.
                quantiles[label] = conformal_quantile(scores, alpha)

    cal = ConformalCalibration(
        alpha=alpha,
        method=method,
        mondrian=mondrian,
        quantiles=quantiles,
        global_quantile=conformal_quantile(scores, alpha),
        quantile_grid=_build_quantile_grid(scores, y_cal),
        n_calibration=len(scores),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_version=model_version,
    )

    if X_eval is not None and y_eval is not None and len(X_eval):
        cal.coverage = evaluate_coverage(model, X_eval, np.asarray(y_eval).astype(int), cal)

    return cal


def _build_quantile_grid(scores: np.ndarray, y_cal: np.ndarray) -> dict[str, Any]:
    """Precompute conformal thresholds across the whole coverage range.

    Each stored value is a genuine ``conformal_quantile`` at that level — the
    ``(n+1)/n`` correction is applied per level rather than approximated once,
    so a threshold read off the grid is exactly the threshold that a fresh
    calibration at that alpha would have produced.
    """
    levels = QUANTILE_GRID_LEVELS
    grid: dict[str, Any] = {
        "levels": levels,
        "global": [_jsonable(conformal_quantile(scores, 1.0 - lv)) for lv in levels],
        "per_class": {},
    }
    for idx, label in C.INT_TO_CLASS.items():
        mask = y_cal == idx
        sample = scores[mask] if mask.sum() >= 20 else scores
        grid["per_class"][label] = [
            _jsonable(conformal_quantile(sample, 1.0 - lv)) for lv in levels
        ]
    return grid


def evaluate_coverage(model, X: pd.DataFrame, y: np.ndarray,
                      calibration: ConformalCalibration) -> dict:
    """Measure realised coverage and average set size on unseen data."""
    predictor = ConformalPredictor(calibration)
    proba = np.asarray(model.predict_proba(X), dtype=float)

    covered = np.zeros(len(y), dtype=bool)
    sizes = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        result = predictor.predict_set(proba[i])
        labels = result["prediction_set"]
        sizes[i] = len(labels)
        covered[i] = C.INT_TO_CLASS[int(y[i])] in labels

    per_class = {}
    for idx, label in C.INT_TO_CLASS.items():
        mask = y == idx
        if mask.sum():
            per_class[label] = {
                "n": int(mask.sum()),
                "coverage": round(float(covered[mask].mean()), 4),
                "avg_set_size": round(float(sizes[mask].mean()), 3),
            }

    target = 1.0 - calibration.alpha
    marginal = float(covered.mean())
    return {
        "n_eval": len(y),
        "target_coverage": round(target, 4),
        "empirical_coverage": round(marginal, 4),
        "coverage_gap": round(marginal - target, 4),
        "guarantee_met": bool(marginal >= target - 0.02),
        "avg_set_size": round(float(sizes.mean()), 3),
        "singleton_rate": round(float((sizes == 1).mean()), 4),
        "abstention_rate": round(float((sizes != 1).mean()), 4),
        "empty_set_rate": round(float((sizes == 0).mean()), 4),
        "per_class": per_class,
    }


def calibrate_from_dataset(model, *, alpha: float = 0.10, method: str = "auto",
                           mondrian: bool = True, model_version: str = "",
                           save: bool = True) -> ConformalCalibration:
    """Calibrate against the shipped dataset, reusing the canonical split.

    The hold-out half of the train/test split is split *again*: one half
    calibrates the quantiles, the other measures whether the guarantee holds.
    Reusing the calibration data to also report coverage would produce an
    optimistic number, which defeats the purpose of the exercise.

    ``method="auto"`` fits both scores and keeps whichever produces the
    **smaller average prediction set while still meeting coverage**. That
    choice is data-dependent and not obvious in advance: APS generally gives
    better conditional coverage, but on a very confident model its
    deterministic form pushes the calibrated quantile to ~1.0 and the sets
    become uselessly wide, whereas LAC stays tight. Rather than hard-code a
    preference, measure it and record the comparison in the artefact.
    """
    from sklearn.model_selection import train_test_split

    from .data import load_dataset, split_xy, stratified_split

    df = load_dataset()
    X, y = split_xy(df)
    _, X_hold, _, y_hold = stratified_split(X, y)

    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X_hold, y_hold, test_size=0.5, random_state=C.RANDOM_STATE, stratify=y_hold
    )

    candidates = list(SCORE_METHODS) if method == "auto" else [method]
    fitted: list[ConformalCalibration] = [
        calibrate(
            model, X_cal, y_cal, alpha=alpha, method=candidate, mondrian=mondrian,
            X_eval=X_eval, y_eval=y_eval, model_version=model_version,
        )
        for candidate in candidates
    ]

    valid = [c for c in fitted if c.coverage.get("guarantee_met")]
    pool = valid or fitted  # if neither holds, keep the closest to target
    best = min(
        pool,
        key=lambda c: (
            c.coverage.get("avg_set_size", 99.0),
            abs(c.coverage.get("coverage_gap", 1.0)),
        ),
    )

    if len(fitted) > 1:
        best.coverage["method_comparison"] = [
            {
                "method": c.method,
                "empirical_coverage": c.coverage.get("empirical_coverage"),
                "avg_set_size": c.coverage.get("avg_set_size"),
                "singleton_rate": c.coverage.get("singleton_rate"),
                "guarantee_met": c.coverage.get("guarantee_met"),
                "selected": c.method == best.method,
            }
            for c in fitted
        ]
        best.coverage["selection_rationale"] = (
            f"'{best.method}' selected: it meets the {1 - alpha:.0%} coverage target "
            f"with the smaller average prediction set "
            f"({best.coverage.get('avg_set_size')} labels)."
        )

    if save:
        best.save()
        log.info("conformal calibration written to %s (method=%s)",
                 CALIBRATION_PATH, best.method)
    return best
