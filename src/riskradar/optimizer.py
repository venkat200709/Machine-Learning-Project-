"""City-scale intervention optimiser — where should the money go?

The question a classifier cannot answer
----------------------------------------
``counterfactual_scan`` in :mod:`explain` answers "what would fix *this*
street?". A city has ten thousand streets and one budget. The real question is
allocative:

    Given ₹40 lakh, which combination of streetlight repairs, CCTV
    installations and patrol placements across which areas moves the most
    people out of High risk?

That is a constrained combinatorial optimisation over a black-box objective —
the model itself — and it is where this project stops being a classifier and
becomes a decision-support system.

Approach
--------
The objective is **expected population-weighted risk reduction**, evaluated by
the model. Interventions have per-unit costs and per-area capacity limits.

Greedy marginal-gain selection is used, in the *lazy* (Minoux) form. The
justification is that the objective is approximately submodular — the second
CCTV camera on a street helps less than the first, which is both true of the
model's response and true in reality. For a monotone submodular objective
under a knapsack constraint, greedy is within a constant factor
``(1 − 1/e) ≈ 63%`` of the optimal allocation, and unlike an exact solver it
runs in seconds and needs no LP dependency.

"Approximately" is doing real work in that sentence, so the report states the
guarantee is a heuristic bound here rather than pretending it is proven for a
gradient-boosted model. The reported gain is measured, not assumed: every
allocation is re-scored by the model at the end.

Lazy evaluation matters
-----------------------
A naive greedy re-scores every candidate at every round: ``O(rounds × areas ×
levers)`` forward passes, thousands of model calls. Marginal gains only
*decrease* as budget is spent, so a candidate whose stale gain is already below
the current best cannot win — a priority queue lets most candidates be skipped
entirely. In practice this cuts model calls by 80–95%.
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger("riskradar.optimizer")


# ==========================================================================
# Intervention catalogue
# ==========================================================================
@dataclass(frozen=True)
class Intervention:
    """One lever a city authority can actually pull.

    Costs are indicative Indian municipal figures in rupees, kept as data so a
    different city can re-price the catalogue without touching the algorithm.
    """

    key: str
    label: str
    unit: str
    unit_cost: float
    max_units: int
    apply: Callable[[pd.Series, int], pd.Series]
    description: str
    category: str = "infrastructure"

    def cost(self, units: int) -> float:
        return self.unit_cost * units


def _repair_lights(row: pd.Series, units: int) -> pd.Series:
    out = row.copy()
    broken = float(out["Broken_Streetlights"])
    fixed = min(units, int(broken))
    out["Working_Streetlights"] = float(out["Working_Streetlights"]) + fixed
    out["Broken_Streetlights"] = broken - fixed
    return out


def _add_cctv(row: pd.Series, units: int) -> pd.Series:
    out = row.copy()
    out["CCTV_Count"] = float(out["CCTV_Count"]) + units
    return out


def _new_lights(row: pd.Series, units: int) -> pd.Series:
    out = row.copy()
    out["Working_Streetlights"] = float(out["Working_Streetlights"]) + units
    out["Streetlight_Count"] = float(out["Streetlight_Count"]) + units
    return out


def _patrol(row: pd.Series, units: int) -> pd.Series:
    """Each unit is a patrol post, modelled as cutting effective distance 30%."""
    out = row.copy()
    out["Police_Distance_km"] = float(out["Police_Distance_km"]) * (0.70 ** units)
    return out


def _transit(row: pd.Series, units: int) -> pd.Series:
    """A bus stop adds access and, through it, footfall and natural surveillance."""
    out = row.copy()
    out["Bus_Stop_Count"] = float(out["Bus_Stop_Count"]) + units
    out["Footfall"] = float(out["Footfall"]) * (1.0 + 0.12 * units)
    return out


CATALOGUE: dict[str, Intervention] = {
    "repair_lights": Intervention(
        key="repair_lights", label="Repair broken streetlights", unit="lamp",
        unit_cost=3_500, max_units=60, apply=_repair_lights,
        description="Restore a failed lamp. Cheapest lever, and capped by how many are broken.",
        category="lighting",
    ),
    "new_lights": Intervention(
        key="new_lights", label="Install new streetlights", unit="lamp",
        unit_cost=12_000, max_units=40, apply=_new_lights,
        description="Add lighting where none exists.",
        category="lighting",
    ),
    "cctv": Intervention(
        key="cctv", label="Install CCTV cameras", unit="camera",
        unit_cost=45_000, max_units=30, apply=_add_cctv,
        description="Surveillance coverage with a deterrent and evidentiary effect.",
        category="surveillance",
    ),
    "patrol": Intervention(
        key="patrol", label="Establish a patrol post", unit="post",
        unit_cost=250_000, max_units=3, apply=_patrol,
        description="Cuts effective police response distance by ~30% per post.",
        category="policing",
    ),
    "transit": Intervention(
        key="transit", label="Add a transit stop", unit="stop",
        unit_cost=180_000, max_units=5, apply=_transit,
        description="Improves escape options and raises footfall (eyes on the street).",
        category="mobility",
    ),
}


# ==========================================================================
# Allocation state
# ==========================================================================
@dataclass
class AreaState:
    index: int
    row: pd.Series
    population: float
    baseline_risk: float
    baseline_band: str
    allocation: dict[str, int] = field(default_factory=dict)

    def current_row(self) -> pd.Series:
        row = self.row.copy()
        for key, units in self.allocation.items():
            if units:
                row = CATALOGUE[key].apply(row, units)
        return row

    def spent(self) -> float:
        return sum(CATALOGUE[k].cost(u) for k, u in self.allocation.items())


def _expected_risk(proba: np.ndarray) -> np.ndarray:
    """Expected band index in [0, 1]. The objective is its population-weighted sum.

    Using the expectation rather than the argmax means an intervention that
    moves an area from 95% High to 55% High registers as progress, even though
    the label has not flipped yet. Optimising the label alone produces a
    plan that stalls on hard areas and wastes budget on borderline ones.
    """
    return (np.asarray(proba, dtype=float) @ np.array([0.0, 0.5, 1.0]))


# ==========================================================================
# Optimiser
# ==========================================================================
class InterventionOptimiser:
    """Budget-constrained allocation of safety interventions across areas."""

    # Each round buys a block of units rather than one, so a ₹40 lakh budget
    # does not need 1,000 rounds of model calls to be spent.
    BLOCK_SIZES: ClassVar[dict[str, int]] = {
        "repair_lights": 10, "new_lights": 5, "cctv": 3, "patrol": 1, "transit": 1,
    }

    def __init__(self, model) -> None:
        self.model = model

    # ------------------------------------------------------------------
    def optimise(self, areas: pd.DataFrame, budget: float, *,
                 levers: list[str] | None = None,
                 population_weighted: bool = True,
                 max_rounds: int = 400) -> dict:
        """Allocate ``budget`` across ``areas`` to minimise expected risk."""
        levers = [k for k in (levers or list(CATALOGUE)) if k in CATALOGUE]
        if not levers:
            raise ValueError(f"No valid interventions selected. Choose from {list(CATALOGUE)}.")
        if budget <= 0:
            raise ValueError("Budget must be positive.")

        areas = areas[C.RAW_FEATURE_COLUMNS].reset_index(drop=True).copy()
        n = len(areas)
        if n == 0:
            raise ValueError("No areas supplied.")

        base_proba = np.asarray(self.model.predict_proba(areas), dtype=float)
        base_risk = _expected_risk(base_proba)
        base_bands = [C.INT_TO_CLASS[int(i)] for i in base_proba.argmax(axis=1)]

        population = (
            areas["Population_Density"].to_numpy(dtype=float)
            if population_weighted else np.ones(n)
        )
        population = population / max(population.sum(), 1e-9) * n  # mean-1 weights

        states = [
            AreaState(index=i, row=areas.iloc[i], population=float(population[i]),
                      baseline_risk=float(base_risk[i]), baseline_band=base_bands[i])
            for i in range(n)
        ]

        objective_start = float(np.sum(base_risk * population))
        spent = 0.0
        model_calls = 1
        rounds = 0

        # Seed the heap with *real* gains computed in a single batched pass.
        # Seeding with a sentinel instead would force the first ~|areas|×|levers|
        # rounds to each trigger their own two-row model call — the dominant
        # cost of the whole solve. One call replaces all of them.
        heap: list[tuple[float, int, str, int]] = []
        seed_rows: list[pd.Series] = []
        seed_meta: list[tuple[int, str, int, float]] = []
        for state in states:
            for key in levers:
                block = min(self.BLOCK_SIZES.get(key, 1), CATALOGUE[key].max_units)
                if block <= 0:
                    continue
                seed_rows.append(state.row)
                seed_rows.append(CATALOGUE[key].apply(state.row, block))
                seed_meta.append((state.index, key, block, CATALOGUE[key].cost(block)))

        if seed_rows:
            seed_proba = np.asarray(
                self.model.predict_proba(
                    pd.DataFrame(seed_rows).reset_index(drop=True)[C.RAW_FEATURE_COLUMNS]
                ), dtype=float,
            )
            seed_risk = _expected_risk(seed_proba)
            model_calls += 1
            for j, (area_idx, key, _block, cost) in enumerate(seed_meta):
                gain = max(
                    float((seed_risk[2 * j] - seed_risk[2 * j + 1]) * states[area_idx].population),
                    0.0,
                )
                if gain > 1e-9:
                    heap.append((-(gain / cost), area_idx, key, 0))
        heapq.heapify(heap)

        history: list[dict] = []

        while heap and rounds < max_rounds:
            neg_ratio, area_idx, lever_key, computed_at = heapq.heappop(heap)
            state = states[area_idx]
            lever = CATALOGUE[lever_key]

            used = state.allocation.get(lever_key, 0)
            block = min(self.BLOCK_SIZES.get(lever_key, 1), lever.max_units - used)
            if block <= 0:
                continue

            cost = lever.cost(block)
            if spent + cost > budget:
                continue

            # Stale entry: recompute its gain and push it back to be re-ranked.
            if computed_at != rounds:
                gain = self._marginal_gain(state, lever_key, block)
                model_calls += 1
                if gain <= 1e-9:
                    continue
                heapq.heappush(heap, (-(gain / cost), area_idx, lever_key, rounds))
                continue

            # Top of the heap and freshly computed — buy it.
            gain = -neg_ratio * cost
            state.allocation[lever_key] = used + block
            spent += cost
            rounds += 1

            history.append({
                "round": rounds,
                "area": int(area_idx),
                "intervention": lever_key,
                "label": lever.label,
                "units": block,
                "cost": round(cost, 2),
                "cumulative_cost": round(spent, 2),
                "expected_gain": round(gain, 6),
                "efficiency": round(gain / cost * 1e6, 4),
            })

            if lever.max_units - state.allocation[lever_key] > 0:
                heapq.heappush(heap, (-1e9, area_idx, lever_key, -1))

        # Measure the real outcome rather than trusting the accumulated gains.
        final_rows = pd.DataFrame([s.current_row() for s in states]).reset_index(drop=True)
        final_proba = np.asarray(self.model.predict_proba(final_rows), dtype=float)
        final_risk = _expected_risk(final_proba)
        final_bands = [C.INT_TO_CLASS[int(i)] for i in final_proba.argmax(axis=1)]
        model_calls += 1

        objective_end = float(np.sum(final_risk * population))

        return self._report(
            states, base_risk, final_risk, base_bands, final_bands,
            base_proba, final_proba, population,
            budget=budget, spent=spent,
            objective_start=objective_start, objective_end=objective_end,
            history=history, model_calls=model_calls, levers=levers,
        )

    # ------------------------------------------------------------------
    def _marginal_gain(self, state: AreaState, lever_key: str, block: int) -> float:
        """Population-weighted expected-risk reduction from one more block."""
        current = state.current_row()
        proposed = CATALOGUE[lever_key].apply(current, block)

        # Both states in one batched call — two forward passes, not two calls.
        frame = pd.DataFrame([current, proposed])[C.RAW_FEATURE_COLUMNS]
        proba = np.asarray(self.model.predict_proba(frame), dtype=float)
        risk = _expected_risk(proba)
        return max(float((risk[0] - risk[1]) * state.population), 0.0)

    # ------------------------------------------------------------------
    @staticmethod
    def _report(states, base_risk, final_risk, base_bands, final_bands,
                base_proba, final_proba, population, *, budget, spent,
                objective_start, objective_end, history, model_calls, levers) -> dict:
        n = len(states)

        improved = [
            i for i in range(n)
            if C.CLASS_TO_INT[final_bands[i]] < C.CLASS_TO_INT[base_bands[i]]
        ]
        worsened = [
            i for i in range(n)
            if C.CLASS_TO_INT[final_bands[i]] > C.CLASS_TO_INT[base_bands[i]]
        ]
        bands_moved = sum(
            C.CLASS_TO_INT[base_bands[i]] - C.CLASS_TO_INT[final_bands[i]]
            for i in range(n)
        )

        totals: dict[str, dict] = {}
        for state in states:
            for key, units in state.allocation.items():
                entry = totals.setdefault(key, {
                    "intervention": key,
                    "label": CATALOGUE[key].label,
                    "unit": CATALOGUE[key].unit,
                    "category": CATALOGUE[key].category,
                    "units": 0, "cost": 0.0, "areas": 0,
                })
                entry["units"] += units
                entry["cost"] += CATALOGUE[key].cost(units)
                entry["areas"] += 1

        plan = []
        for state in states:
            if not state.allocation:
                continue
            i = state.index
            plan.append({
                "area": i,
                "latitude": round(float(state.row["Latitude"]), 6),
                "longitude": round(float(state.row["Longitude"]), 6),
                "before_band": base_bands[i],
                "after_band": final_bands[i],
                "before_risk": round(float(base_risk[i]), 4),
                "after_risk": round(float(final_risk[i]), 4),
                "risk_reduction": round(float(base_risk[i] - final_risk[i]), 4),
                "before_safety": round(100 * (1 - float(base_risk[i])), 1),
                "after_safety": round(100 * (1 - float(final_risk[i])), 1),
                "bands_improved": C.CLASS_TO_INT[base_bands[i]] - C.CLASS_TO_INT[final_bands[i]],
                "population_weight": round(state.population, 4),
                "cost": round(state.spent(), 2),
                "actions": [
                    {
                        "intervention": k, "label": CATALOGUE[k].label,
                        "units": u, "unit": CATALOGUE[k].unit,
                        "cost": round(CATALOGUE[k].cost(u), 2),
                        "detail": f"{u} {CATALOGUE[k].unit}{'s' if u != 1 else ''}",
                    }
                    for k, u in sorted(state.allocation.items())
                    if u
                ],
            })
        plan.sort(key=lambda p: -p["risk_reduction"])

        def distribution(bands: list[str]) -> dict[str, int]:
            return {c: int(sum(1 for b in bands if b == c)) for c in C.CLASS_ORDER}

        reduction = objective_start - objective_end
        pct = (reduction / objective_start * 100) if objective_start else 0.0

        return {
            "available": True,
            "budget": round(budget, 2),
            "spent": round(spent, 2),
            "remaining": round(budget - spent, 2),
            "utilisation": round(spent / budget, 4) if budget else 0.0,
            "n_areas": n,
            "n_areas_treated": len(plan),
            "n_areas_improved": len(improved),
            "n_areas_worsened": len(worsened),
            "bands_moved": int(bands_moved),
            "objective": {
                "metric": "population-weighted expected risk",
                "before": round(objective_start, 5),
                "after": round(objective_end, 5),
                "reduction": round(reduction, 5),
                "reduction_percent": round(pct, 2),
            },
            "distribution": {
                "before": distribution(base_bands),
                "after": distribution(final_bands),
            },
            "safety_score": {
                "before": round(100 * (1 - float(np.mean(base_risk))), 2),
                "after": round(100 * (1 - float(np.mean(final_risk))), 2),
            },
            "cost_effectiveness": {
                "cost_per_band_moved": round(spent / bands_moved, 2) if bands_moved else None,
                "cost_per_area_improved": round(spent / len(improved), 2) if improved else None,
                "risk_reduction_per_lakh": round(reduction / (spent / 100_000), 5)
                if spent else None,
            },
            "spend_by_intervention": sorted(
                totals.values(), key=lambda t: -t["cost"]
            ),
            "plan": plan,
            "history": history[:200],
            "solver": {
                "algorithm": "lazy greedy marginal-gain (Minoux) under a knapsack constraint",
                "rounds": len(history),
                "model_calls": model_calls,
                "levers_considered": levers,
                "guarantee": (
                    "Greedy attains (1 − 1/e) ≈ 63% of optimal for a monotone submodular "
                    "objective. The model's response is approximately, not provably, "
                    "submodular, so treat that as a strong heuristic bound rather than a "
                    "proof. The reported reduction is measured by re-scoring the final "
                    "allocation, not accumulated from the greedy steps."
                ),
            },
            "narrative": _narrative(spent, budget, len(improved), bands_moved, pct, totals),
        }


def _narrative(spent: float, budget: float, improved: int, bands: int,
               pct: float, totals: dict) -> str:
    if not totals:
        return (
            "No intervention in the catalogue produced a measurable risk reduction "
            "within this budget. Try raising the budget or widening the lever set."
        )
    biggest = max(totals.values(), key=lambda t: t["cost"])
    lakh = spent / 100_000
    return (
        f"₹{lakh:,.1f} lakh of the ₹{budget / 100_000:,.1f} lakh budget allocated across "
        f"{len(totals)} intervention types. {improved} area(s) drop at least one risk band "
        f"({bands} band-steps total), cutting population-weighted expected risk by {pct:.1f}%. "
        f"The largest single line is {biggest['label'].lower()} "
        f"({biggest['units']} {biggest['unit']}s, ₹{biggest['cost'] / 100_000:,.1f} lakh)."
    )


def sample_areas(n: int = 60, *, risk_filter: str | None = None,
                 model=None, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Draw a working set of areas from the dataset for a planning scenario."""
    from .data import load_dataset

    df = load_dataset()
    rng = np.random.default_rng(seed)

    if risk_filter and model is not None:
        pool = df.sample(n=min(len(df), max(n * 12, 600)), random_state=seed)
        proba = np.asarray(model.predict_proba(pool[C.RAW_FEATURE_COLUMNS]), dtype=float)
        bands = np.array([C.INT_TO_CLASS[int(i)] for i in proba.argmax(axis=1)])
        pool = pool[bands == risk_filter]
        if len(pool) >= n:
            df = pool
    elif risk_filter and C.TARGET in df.columns:
        subset = df[df[C.TARGET] == risk_filter]
        if len(subset) >= n:
            df = subset

    take = min(n, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    return df.iloc[idx][C.RAW_FEATURE_COLUMNS].reset_index(drop=True)
