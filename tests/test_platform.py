"""Tests for the platform layer: conformal, drift, fairness, registry, security.

These are deliberately *behavioural*, not smoke tests. A test that only asserts
"the endpoint returned 200" would have passed while conformal prediction was
returning empty sets for every input — which is exactly the bug that shipped
and was caught by measuring coverage instead. So each test here asserts the
property the component actually promises:

* conformal → the coverage guarantee holds on held-out data
* drift     → PSI is ~0 on identical data and large on shifted data
* fairness  → group metrics are computed and disparities are surfaced
* registry  → content hashing detects a tampered artefact
* security  → keys are never stored in plaintext, limits actually limit
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import requires_model

from riskradar import config as C


# ==========================================================================
# Settings
# ==========================================================================
class TestSettings:
    def test_env_overrides_are_typed(self, monkeypatch):
        from riskradar import settings as S

        monkeypatch.setenv("RISKRADAR_RATE_LIMIT_PER_MINUTE", "999")
        monkeypatch.setenv("RISKRADAR_AUTH_ENABLED", "yes")
        fresh = S.Settings()
        assert fresh.rate_limit_per_minute == 999
        assert fresh.auth_enabled is True

    def test_bad_boolean_is_rejected_loudly(self, monkeypatch):
        """A typo'd flag must fail at boot, not silently default to False."""
        from riskradar import settings as S

        monkeypatch.setenv("RISKRADAR_AUTH_ENABLED", "maybe")
        with pytest.raises(ValueError, match="not a boolean"):
            S.Settings()

    def test_secrets_never_appear_in_public_dict(self, monkeypatch):
        from riskradar import settings as S

        monkeypatch.setenv("RISKRADAR_ADMIN_KEY", "super-secret-value")
        monkeypatch.setenv(
            "RISKRADAR_DATABASE_URL", "postgresql://user:hunter2@db:5432/rr"
        )
        public = S.Settings().public_dict()
        blob = str(public)
        assert "super-secret-value" not in blob
        assert "hunter2" not in blob
        assert public["admin_key"] == "set"

    def test_production_audit_flags_open_configuration(self, monkeypatch):
        from riskradar import settings as S

        monkeypatch.setenv("RISKRADAR_ENV", "production")
        monkeypatch.setenv("RISKRADAR_AUTH_ENABLED", "false")
        findings = S.Settings().audit()
        assert any(f["level"] == "critical" and f["setting"] == "AUTH_ENABLED"
                   for f in findings)


# ==========================================================================
# Cache
# ==========================================================================
class TestCache:
    def test_key_is_order_and_float_noise_invariant(self):
        from riskradar.cache import payload_key

        a = payload_key({"Hour": 22, "Crime_Count": 40.0}, model_version="v1")
        b = payload_key({"Crime_Count": 40.0000000001, "Hour": 22}, model_version="v1")
        assert a == b

    def test_model_version_partitions_the_cache(self):
        """A promotion must not let the old model's answers leak through."""
        from riskradar.cache import payload_key

        assert payload_key({"Hour": 1}, model_version="v1") != \
               payload_key({"Hour": 1}, model_version="v2")

    def test_lru_evicts_and_reports(self):
        from riskradar.cache import TTLCache

        cache = TTLCache(maxsize=3, ttl=60)
        for i in range(5):
            cache.set(f"k{i}", i)
        assert cache.stats()["size"] == 3
        assert cache.stats()["evictions"] == 2
        assert cache.get("k0") is None      # evicted
        assert cache.get("k4") == 4

    def test_entries_expire(self):
        from riskradar.cache import TTLCache

        cache = TTLCache(maxsize=10, ttl=-1)  # already expired on write
        cache.set("k", "v")
        assert cache.get("k") is None
        assert cache.stats()["expirations"] == 1


# ==========================================================================
# Conformal prediction
# ==========================================================================
class TestConformal:
    def test_quantile_applies_finite_sample_correction(self):
        from riskradar.conformal import conformal_quantile

        scores = np.linspace(0, 1, 100)
        # With n=100 and alpha=0.1 the level is ceil(101*0.9)/100 = 0.91
        q = conformal_quantile(scores, 0.10)
        assert 0.89 <= q <= 0.93

    def test_too_few_points_refuses_to_promise(self):
        """Rather than quote a threshold it cannot justify, it returns inf."""
        from riskradar.conformal import conformal_quantile

        assert conformal_quantile(np.array([0.1, 0.2]), 0.01) == float("inf")

    def test_lac_and_serving_score_agree(self):
        """The calibration score and the serving score must be one function.

        They diverged once — calibration used randomised APS, serving used the
        deterministic form — and every prediction set came back empty.
        """
        from riskradar.conformal import ConformalPredictor, aps_scores, lac_scores

        proba = np.array([[0.7, 0.2, 0.1]])
        for method, fn in (("lac", lac_scores), ("aps", aps_scores)):
            predictor = ConformalPredictor.__new__(ConformalPredictor)
            predictor.calibration = type("Cal", (), {"method": method})()
            for k in range(3):
                serving = predictor._score_for_hypothesis(proba[0], k)
                calibration = float(fn(proba, np.array([k]))[0])
                assert serving == pytest.approx(calibration, abs=1e-9), method

    def test_sets_grow_monotonically_with_coverage(self):
        from riskradar.conformal import ConformalCalibration, ConformalPredictor

        cal = ConformalCalibration(
            alpha=0.10, method="lac", mondrian=False,
            quantiles=dict.fromkeys(C.CLASS_ORDER, 0.5),
            quantile_grid={
                "levels": [0.5, 0.7, 0.9, 0.99],
                "global": [0.05, 0.2, 0.5, 0.95],
                "per_class": {},
            },
            n_calibration=500,
        )
        proba = np.array([0.55, 0.35, 0.10])
        sizes = [
            ConformalPredictor(cal).predict_set(proba, alpha=1 - lv)["set_size"]
            for lv in (0.5, 0.7, 0.9, 0.99)
        ]
        assert sizes == sorted(sizes), f"set size must not shrink as coverage rises: {sizes}"

    @requires_model
    def test_coverage_guarantee_holds_on_held_out_data(self):
        """The headline promise, measured rather than asserted."""
        from riskradar.conformal import ConformalPredictor

        predictor = ConformalPredictor()
        if not predictor.ready:
            pytest.skip("no conformal calibration — run `python run.py --calibrate`")

        coverage = predictor.calibration.coverage
        assert coverage, "calibration was written without a coverage evaluation"
        assert coverage["empirical_coverage"] >= coverage["target_coverage"] - 0.02
        assert coverage["guarantee_met"] is True
        # Per-class (Mondrian) coverage is the one that matters for safety: a
        # marginal guarantee can be met while systematically failing on High.
        for label, stats in coverage["per_class"].items():
            assert stats["coverage"] >= coverage["target_coverage"] - 0.05, label


# ==========================================================================
# Drift
# ==========================================================================
class TestDrift:
    def test_psi_is_zero_for_identical_distributions(self):
        from riskradar.drift import population_stability_index

        rng = np.random.default_rng(0)
        x = rng.normal(size=4000)
        psi, _ = population_stability_index(x, x)
        assert psi < 0.01

    def test_psi_grows_with_the_size_of_the_shift(self):
        from riskradar.drift import population_stability_index

        rng = np.random.default_rng(0)
        ref = rng.normal(size=4000)
        psis = [
            population_stability_index(ref, rng.normal(loc=shift, size=4000))[0]
            for shift in (0.25, 1.0, 3.0)
        ]
        assert psis == sorted(psis)
        assert psis[0] < 0.25 < psis[-1]

    def test_ks_and_js_agree_with_psi_on_direction(self):
        from riskradar.drift import jensen_shannon_distance, ks_statistic

        rng = np.random.default_rng(1)
        ref = rng.normal(size=2000)
        assert ks_statistic(ref, ref) < 0.05
        assert ks_statistic(ref, rng.normal(loc=2.5, size=2000)) > 0.5
        assert jensen_shannon_distance([0.5, 0.5], [0.5, 0.5]) < 1e-6
        assert jensen_shannon_distance([1.0, 0.0], [0.0, 1.0]) > 0.9

    def test_classification_thresholds(self):
        from riskradar.drift import STATUS_ALERT, STATUS_STABLE, STATUS_WARNING, classify_psi

        assert classify_psi(0.02) == STATUS_STABLE
        assert classify_psi(0.15) == STATUS_WARNING
        assert classify_psi(0.40) == STATUS_ALERT

    def test_monitor_declines_to_report_without_enough_data(self):
        from riskradar.drift import DriftMonitor, ReferenceDistribution

        monitor = DriftMonitor(
            reference=ReferenceDistribution(
                features={"Crime_Count": {"bins": [0, 1, 2], "proportions": [.5, .5],
                                          "mean": 1, "std": 1, "p05": 0, "p50": 1, "p95": 2,
                                          "min": 0, "max": 2}},
                class_distribution={}, n_reference=10, created_at="",
            )
        )
        report = monitor.report(min_samples=50)
        assert report["available"] is False
        assert "50" in report["reason"]


# ==========================================================================
# Fairness
# ==========================================================================
@requires_model
class TestFairness:
    @pytest.fixture(scope="class")
    def sample(self):
        from riskradar.data import load_dataset, split_xy, stratified_split

        X, y = split_xy(load_dataset())
        _, X_test, _, y_test = stratified_split(X, y)
        return X_test.head(3000).reset_index(drop=True), y_test[:3000]

    def test_group_builders_partition_every_row(self, sample):
        from riskradar.fairness import GROUP_DEFINITIONS

        X, _ = sample
        for key, spec in GROUP_DEFINITIONS.items():
            groups = spec["builder"](X)
            assert len(groups) == len(X), key
            assert groups.isna().sum() == 0, key
            assert groups.nunique() >= 2, key

    def test_audit_reports_every_dimension(self, sample):
        from riskradar.fairness import GROUP_DEFINITIONS, audit
        from riskradar.service import RiskService

        X, y = sample
        report = audit(RiskService.instance().model, X, y)
        assert report["available"]
        assert set(report["dimensions"]) == set(GROUP_DEFINITIONS)
        assert report["overall_verdict"] in {"pass", "review", "fail"}

        for key, dim in report["dimensions"].items():
            if not dim.get("available"):
                continue
            for name, stats in dim["groups"].items():
                assert 0.0 <= stats["accuracy"] <= 1.0, (key, name)
                assert 0.0 <= stats["selection_rate"] <= 1.0, (key, name)

    def test_a_deliberately_biased_predictor_is_caught(self, sample):
        """The audit must be capable of failing, or it proves nothing."""
        from riskradar.fairness import audit_group

        X, y = sample
        groups = pd.Series(
            np.where(X["Population_Density"] > X["Population_Density"].median(), "A", "B")
        )
        # A predictor that is accurate in group A and blind in group B.
        y_pred = np.array(y, copy=True)
        mask = (groups == "B").to_numpy()
        y_pred[mask] = C.CLASS_TO_INT["Low"]
        confidence = np.full(len(y), 0.9)

        report = audit_group(X, np.asarray(y), y_pred, confidence, groups)
        assert report["verdict"] == "fail"
        assert any(f["metric"] == "equal_opportunity" for f in report["findings"])


# ==========================================================================
# Registry
# ==========================================================================
class TestRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        from riskradar.registry import ModelRegistry

        return ModelRegistry(root=tmp_path / "registry")

    @pytest.fixture
    def artefact(self, tmp_path):
        path = tmp_path / "model.joblib"
        path.write_bytes(b"pretend this is a pickled pipeline")
        return path

    def test_register_hashes_and_versions(self, registry, artefact):
        mv = registry.register(artefact, model_name="Test", metrics={"accuracy": 0.9})
        assert mv.version == "v1"
        assert len(mv.sha256) == 64
        assert registry.get("v1").stage == "staging"
        assert registry.register(artefact, model_name="Test").version == "v2"

    def test_tampering_is_detected(self, registry, artefact):
        mv = registry.register(artefact, model_name="Test")
        from pathlib import Path

        Path(mv.artefact_path).write_bytes(b"a different model entirely")
        ok, message = mv.verify()
        assert ok is False
        assert "Integrity failure" in message

    def test_only_one_champion_at_a_time(self, registry, artefact):
        registry.register(artefact, model_name="A", version="v1")
        registry.register(artefact, model_name="B", version="v2", copy=False)
        registry.set_stage("v1", "champion")
        registry.set_stage("v2", "champion")
        champions = [v for v in registry.list_versions() if v["stage"] == "champion"]
        assert len(champions) == 1
        assert champions[0]["version"] == "v2"
        assert registry.get("v1").stage == "archived"

    def test_promotion_is_gated_without_shadow_evidence(self, registry, artefact):
        registry.register(artefact, model_name="A", version="v1")
        result = registry.promote("v1")
        assert result["promoted"] is False
        assert any("shadow" in b.lower() for b in result["gate"]["blockers"])

    def test_force_overrides_the_gate_and_says_so(self, registry, artefact):
        registry.register(artefact, model_name="A", version="v1")
        result = registry.promote("v1", force=True)
        assert result["promoted"] is True
        assert result["forced"] is True

    def test_rollback_restores_the_previous_champion(self, registry, artefact):
        registry.register(artefact, model_name="A", version="v1")
        registry.register(artefact, model_name="B", version="v2", copy=False)
        registry.set_stage("v1", "champion")
        registry.set_stage("v2", "champion")
        result = registry.rollback()
        assert result["rolled_back"] is True
        assert registry.champion().version == "v1"

    def test_shadow_stats_flag_severe_disagreements(self):
        from riskradar.registry import ShadowEvaluator

        shadow = ShadowEvaluator()
        for champ, chall in [("Low", "High")] * 3 + [("High", "High")] * 20:
            shadow._agree.append(champ == chall)
            shadow._confidence.append(0.9)
            shadow._champion_labels.append(champ)
            shadow._challenger_labels.append(chall)
        stats = shadow.stats()
        assert stats["severe_disagreements"] == 3
        assert "Low↔High" in stats["verdict"] or "investigate" in stats["verdict"].lower()


# ==========================================================================
# Security
# ==========================================================================
class TestSecurity:
    def test_keys_are_hashed_not_stored(self):
        from riskradar.security import generate_key, hash_key

        raw, digest, prefix = generate_key()
        assert raw.startswith("rr_")
        assert digest == hash_key(raw)
        assert raw not in digest
        assert len(digest) == 64
        assert prefix == raw[:12]

    def test_role_ladder(self):
        from riskradar.security import role_satisfies

        assert role_satisfies("admin", "viewer")
        assert role_satisfies("analyst", "viewer")
        assert not role_satisfies("viewer", "analyst")
        assert not role_satisfies("analyst", "admin")

    def test_rate_limiter_permits_burst_then_blocks(self):
        from riskradar.security import TokenBucketLimiter

        limiter = TokenBucketLimiter(rate_per_minute=60, burst=5)
        allowed = [limiter.check("client")[0] for _ in range(8)]
        assert allowed[:5] == [True] * 5
        assert allowed[5:] == [False] * 3
        assert limiter.stats()["rejected_total"] == 3

    def test_limiter_isolates_identities(self):
        from riskradar.security import TokenBucketLimiter

        limiter = TokenBucketLimiter(rate_per_minute=60, burst=2)
        assert limiter.check("a")[0] and limiter.check("a")[0]
        assert limiter.check("a")[0] is False
        assert limiter.check("b")[0] is True, "one noisy client must not starve another"

    def test_limiter_refills_over_time(self, monkeypatch):
        import riskradar.security as sec

        limiter = sec.TokenBucketLimiter(rate_per_minute=6000, burst=1)
        clock = {"t": 1000.0}
        monkeypatch.setattr(sec.time, "monotonic", lambda: clock["t"])
        assert limiter.check("c")[0] is True
        assert limiter.check("c")[0] is False
        clock["t"] += 1.0  # 100 tokens/second
        assert limiter.check("c")[0] is True


# ==========================================================================
# Optimiser
# ==========================================================================
@requires_model
class TestOptimiser:
    def test_allocation_respects_the_budget(self):
        from riskradar.optimizer import InterventionOptimiser, sample_areas
        from riskradar.service import RiskService

        model = RiskService.instance().model
        areas = sample_areas(n=12, seed=7)
        result = InterventionOptimiser(model).optimise(areas, budget=1_500_000)
        assert result["spent"] <= result["budget"] + 1e-6
        assert result["n_areas"] == 12

    def test_reported_reduction_is_measured_not_accumulated(self):
        """The report re-scores the final plan, so it cannot drift from reality."""
        from riskradar.optimizer import CATALOGUE, InterventionOptimiser, sample_areas
        from riskradar.service import RiskService

        model = RiskService.instance().model
        areas = sample_areas(n=10, seed=3)
        result = InterventionOptimiser(model).optimise(areas, budget=2_000_000)

        rebuilt = areas.copy()
        for entry in result["plan"]:
            row = rebuilt.iloc[entry["area"]]
            for action in entry["actions"]:
                row = CATALOGUE[action["intervention"]].apply(row, action["units"])
            rebuilt.iloc[entry["area"]] = row

        proba = np.asarray(model.predict_proba(rebuilt[C.RAW_FEATURE_COLUMNS]))
        bands = [C.INT_TO_CLASS[int(i)] for i in proba.argmax(axis=1)]
        after = {c: bands.count(c) for c in C.CLASS_ORDER}
        assert after == result["distribution"]["after"]

    def test_a_bigger_budget_never_does_worse(self):
        from riskradar.optimizer import InterventionOptimiser, sample_areas
        from riskradar.service import RiskService

        model = RiskService.instance().model
        areas = sample_areas(n=10, seed=11)
        optimiser = InterventionOptimiser(model)
        small = optimiser.optimise(areas, budget=500_000)
        large = optimiser.optimise(areas, budget=5_000_000)
        assert large["objective"]["reduction"] >= small["objective"]["reduction"] - 1e-9

    def test_rejects_an_unknown_lever(self):
        from riskradar.optimizer import InterventionOptimiser, sample_areas
        from riskradar.service import RiskService

        with pytest.raises(ValueError, match="No valid interventions"):
            InterventionOptimiser(RiskService.instance().model).optimise(
                sample_areas(n=5), budget=100_000, levers=["teleporters"]
            )


# ==========================================================================
# Routing
# ==========================================================================
@requires_model
class TestRouting:
    def test_grid_scores_every_cell(self):
        from riskradar.routing import RouteEngine
        from riskradar.service import RiskService

        grid = RouteEngine().build_grid(RiskService.instance().model, resolution=16, hour=22)
        assert grid.shape == (16, 16)
        assert set(np.unique(grid.labels)) <= set(C.CLASS_ORDER)
        assert np.all((grid.risk >= 0) & (grid.risk <= 1))

    def test_safest_route_is_never_more_exposed_than_the_shortest(self):
        from riskradar.routing import RouteEngine
        from riskradar.service import RiskService

        engine = RouteEngine()
        model = RiskService.instance().model
        b = engine.build_grid(model, resolution=24, hour=23).bounds()
        result = engine.route(
            model,
            origin=(b["min_lat"] + (b["max_lat"] - b["min_lat"]) * 0.15,
                    b["min_lon"] + (b["max_lon"] - b["min_lon"]) * 0.15),
            destination=(b["min_lat"] + (b["max_lat"] - b["min_lat"]) * 0.85,
                         b["min_lon"] + (b["max_lon"] - b["min_lon"]) * 0.85),
            risk_aversion=4.0, hour=23, resolution=24,
        )
        assert result["available"]
        safe, fast = result["safest_route"], result["shortest_route"]
        assert safe["risk_exposure"] <= fast["risk_exposure"] + 1e-6
        # The safe route buys that with distance; it can never be shorter.
        assert safe["distance_km"] >= fast["distance_km"] - 1e-6

    def test_zero_aversion_reproduces_the_shortest_path(self):
        from riskradar.routing import RouteEngine
        from riskradar.service import RiskService

        engine = RouteEngine()
        model = RiskService.instance().model
        b = engine.build_grid(model, resolution=20, hour=12).bounds()
        result = engine.route(
            model,
            origin=(b["min_lat"] + (b["max_lat"] - b["min_lat"]) * 0.2,
                    b["min_lon"] + (b["max_lon"] - b["min_lon"]) * 0.2),
            destination=(b["min_lat"] + (b["max_lat"] - b["min_lat"]) * 0.8,
                         b["min_lon"] + (b["max_lon"] - b["min_lon"]) * 0.8),
            risk_aversion=0.0, hour=12, resolution=20,
        )
        assert result["comparison"]["same_route"] is True
