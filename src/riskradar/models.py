"""Model zoo: pipeline builders and a pre-fitted soft-voting ensemble.

Keeping the estimator definitions in one registry means the trainer, the tests
and the documentation all describe the *same* models — there is no second
place where hyper-parameters can drift out of sync.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .features import engineer

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


# ==========================================================================
# Pipeline assembly
# ==========================================================================
def fe_step() -> FunctionTransformer:
    """Feature engineering as a pipeline step — identical at train and serve time."""
    return FunctionTransformer(engineer, validate=False)


def make_pipeline(estimator, scale: bool = False) -> Pipeline:
    """Wrap an estimator so it consumes *raw* RiskRadar records."""
    steps = [("features", fe_step())]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


# ==========================================================================
# Pre-fitted soft-voting ensemble
# ==========================================================================
class SoftVoteEnsemble(BaseEstimator, ClassifierMixin):
    """Weighted probability averaging over already-fitted pipelines.

    Unlike ``sklearn.ensemble.VotingClassifier`` this does **not** refit its
    members. Base learners are trained once, versioned independently and then
    composed — the pattern real serving stacks use, and it makes ensembling
    essentially free at build time.
    """

    def __init__(self, estimators: list, weights: list[float] | None = None,
                 names: list[str] | None = None):
        self.estimators = estimators
        self.weights = weights
        self.names = names

    # -- sklearn contract ------------------------------------------------
    def fit(self, X, y=None):
        self.classes_ = np.array(sorted(range(len(C.CLASS_ORDER))))
        return self

    @property
    def _w(self) -> np.ndarray:
        w = np.ones(len(self.estimators)) if self.weights is None else np.asarray(
            self.weights, dtype=float
        )
        return w / w.sum()

    def predict_proba(self, X) -> np.ndarray:
        """Average member probabilities, engineering features exactly once.

        Every member pipeline starts with the same ``features`` step, so naively
        calling ``member.predict_proba(X)`` would rebuild the same 79-column
        matrix once per member — the dominant cost for single-row requests. We
        transform once and hand the result to the remaining pipeline steps.
        """
        shared = self._shared_features(X)
        if shared is not None:
            probas = [
                (est[1:] if hasattr(est, "steps") else est).predict_proba(shared)
                for est in self.estimators
            ]
        else:
            probas = [est.predict_proba(X) for est in self.estimators]
        return np.average(np.stack(probas, axis=0), axis=0, weights=self._w)

    def _shared_features(self, X):
        """Engineered matrix, if every member agrees on the same first step."""
        try:
            firsts = [est.steps[0][0] for est in self.estimators]
            if len(set(firsts)) == 1 and firsts[0] == "features":
                return self.estimators[0].named_steps["features"].transform(X)
        except Exception:
            pass
        return None

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @property
    def member_names(self) -> list[str]:
        return self.names or [f"member_{i}" for i in range(len(self.estimators))]


# ==========================================================================
# Registry
# ==========================================================================
def _lgbm(**kw):
    base = {
        "n_jobs": -1, "random_state": C.RANDOM_STATE, "verbose": -1,
        "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.85,
        "min_child_samples": 25, "reg_lambda": 1.0,
    }
    base.update(kw)
    return LGBMClassifier(**base)


def registry(fast: bool = False) -> dict[str, dict]:
    """All benchmark candidates.

    Each entry: ``{"label": display name, "build": callable -> Pipeline}``.
    Hyper-parameters below are the winners of the tuning sweep documented in
    ``docs/PROJECT_REPORT.md``.
    """
    scale = 0.35 if fast else 1.0

    def n(x: int) -> int:
        return max(20, int(x * scale))

    reg: dict[str, dict] = {
        "baseline": {
            "label": "Baseline (Majority Class)",
            "build": lambda: make_pipeline(DummyClassifier(strategy="most_frequent")),
        },
        "logreg": {
            "label": "Logistic Regression",
            "build": lambda: make_pipeline(
                LogisticRegression(max_iter=1200, C=1.0, n_jobs=-1), scale=True
            ),
        },
        "dtree": {
            "label": "Decision Tree",
            "build": lambda: make_pipeline(
                DecisionTreeClassifier(
                    max_depth=16, min_samples_leaf=15, random_state=C.RANDOM_STATE
                )
            ),
        },
        "rf": {
            "label": "Random Forest",
            "build": lambda: make_pipeline(
                RandomForestClassifier(
                    n_estimators=n(300), min_samples_leaf=2, max_features="sqrt",
                    n_jobs=-1, random_state=C.RANDOM_STATE,
                )
            ),
        },
        "extratrees": {
            "label": "Extra Trees",
            "build": lambda: make_pipeline(
                ExtraTreesClassifier(
                    n_estimators=n(300), min_samples_leaf=2, max_features="sqrt",
                    n_jobs=-1, random_state=C.RANDOM_STATE,
                )
            ),
        },
        "histgb": {
            "label": "Hist Gradient Boosting",
            "build": lambda: make_pipeline(
                HistGradientBoostingClassifier(
                    max_iter=n(500), learning_rate=0.08, max_leaf_nodes=31,
                    min_samples_leaf=25, l2_regularization=0.5,
                    early_stopping=True, validation_fraction=0.12,
                    n_iter_no_change=50, random_state=C.RANDOM_STATE,
                )
            ),
        },
    }

    if HAS_LGBM:
        reg["lgbm"] = {
            "label": "LightGBM (tuned)",
            "build": lambda: make_pipeline(
                _lgbm(n_estimators=n(1800), learning_rate=0.05, num_leaves=31)
            ),
        }
        reg["lgbm_wide"] = {
            "label": "LightGBM (wide trees)",
            "build": lambda: make_pipeline(
                _lgbm(n_estimators=n(900), learning_rate=0.05, num_leaves=63,
                      random_state=C.RANDOM_STATE + 7)
            ),
        }
    if HAS_XGB:
        reg["xgb"] = {
            "label": "XGBoost",
            "build": lambda: make_pipeline(
                XGBClassifier(
                    n_estimators=n(900), learning_rate=0.06, max_depth=7,
                    min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
                    reg_lambda=1.5, tree_method="hist", n_jobs=-1,
                    random_state=C.RANDOM_STATE, eval_metric="mlogloss",
                )
            ),
        }
    return reg


# --------------------------------------------------------------------------
# Ensemble composition
# --------------------------------------------------------------------------
# Candidate pool. Membership is *earned*: a model only joins if its hold-out
# accuracy is within ENSEMBLE_TOLERANCE of the best single model. Averaging in
# a weak learner actively hurts a soft vote, so the pool is filtered, not fixed.
ENSEMBLE_POOL: list[str] = ["lgbm", "lgbm_wide", "histgb", "xgb", "logreg", "rf", "extratrees"]

ENSEMBLE_TOLERANCE = 0.010   # 1.0 accuracy point behind the leader
ENSEMBLE_TEMPERATURE = 0.0015  # softmax temperature over accuracy
ENSEMBLE_MIN_MEMBERS = 2


def ensemble_weights(scores: dict[str, float]) -> dict[str, float]:
    """Pick members and weight them with a temperature-scaled softmax on accuracy.

    A plain average is the wrong prior here: the candidates are separated by
    fractions of a point, so linear weights end up near-uniform and the vote
    gets dragged toward its weakest member. A low-temperature softmax keeps the
    leader dominant while still letting diverse runners-up contribute the
    variance reduction that makes voting worthwhile.
    """
    eligible = {s: a for s, a in scores.items() if s in ENSEMBLE_POOL}
    if not eligible:
        return {}
    best = max(eligible.values())
    chosen = {s: a for s, a in eligible.items() if a >= best - ENSEMBLE_TOLERANCE}
    if len(chosen) < ENSEMBLE_MIN_MEMBERS:
        chosen = dict(sorted(eligible.items(), key=lambda kv: -kv[1])[:ENSEMBLE_MIN_MEMBERS])

    logits = {s: (a - best) / ENSEMBLE_TEMPERATURE for s, a in chosen.items()}
    exps = {s: float(np.exp(v)) for s, v in logits.items()}
    total = sum(exps.values())
    return {s: round(v / total, 6) for s, v in exps.items()}
