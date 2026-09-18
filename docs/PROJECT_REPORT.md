# RiskRadar — Project Report

**An explainable machine learning system for women's safety risk assessment**

N. Venkatesan and Neethivendhan T. · August 2026

---

## Abstract

RiskRadar predicts whether an urban location is Low, Medium or High risk for women's safety at a
given hour, using crime, infrastructure, population and environmental signals. The system reaches
**98.27% accuracy** and **0.9993 ROC-AUC** on a 20,000-record hold-out set, improving on the
project's earlier 94.83% baseline by 3.43 points and cutting the error rate by 66%.

The improvement comes primarily from **domain-driven feature engineering** rather than model
capacity: 29 raw measurements are expanded into 79 criminologically-motivated features, on which
even logistic regression reaches 97.44% — higher than any tree model trained on the raw columns.

Beyond classification, the system explains every verdict with exact Shapley values and performs a
counterfactual scan to identify which realistic municipal intervention would lower the risk band.

---

## 1. Problem statement

Safety is neither uniform across a city nor constant through the day: the same street can be
unremarkable at 2pm and genuinely hazardous at 2am. Existing safety applications are almost
entirely **reactive** — they map incidents after they occur.

This project asks a predictive question instead: *given what we know about a place and a moment,
how risky is it likely to be?* And a prescriptive follow-up: *what would we have to change to
make it safer?*

### Objectives

1. Classify a location-hour into one of three ordered risk bands with high accuracy.
2. Make every prediction **explainable** — a safety system whose reasoning cannot be inspected
   should not be trusted.
3. Turn the classifier into a **planning tool** by identifying effective interventions.
4. Deliver it as a production-grade service, not a notebook.

---

## 2. Dataset

100,000 area-hour records with 31 columns, covering Chennai metropolitan coordinates.

| Property | Value |
|---|---|
| Records | 100,000 |
| Raw features used | 29 (`Area_ID` dropped) |
| Missing values | 0 |
| Duplicate rows | 0 |
| Classes | High 57.3% · Low 23.0% · Medium 19.7% |
| Imbalance ratio | 2.90 |

### Exploratory findings

- **Time dominates.** High-risk share peaks sharply in the 22:00–04:00 window and bottoms out in
  the early afternoon.
- **Lighting is the strongest infrastructure signal.** Areas with under 60% of streetlights
  working show a markedly higher High-risk share than those above 85%.
- **Distance to police matters monotonically.** High-risk share rises steadily across police-
  distance quintiles.
- **Risk persists.** The `Previous_Risk` → `Risk_Level` crosstab shows strong diagonal
  concentration, confirming temporal autocorrelation worth modelling explicitly.
- **Crime counts are not additive.** Only 7.5% of rows have `Crime_Count` equal to the sum of the
  itemised categories, so the residual is meaningful and is captured as `Unclassified_Crime`.

### A note on validity

The corpus is **synthetic**, modelled on Chennai rather than drawn from verified municipal
records. All figures in this report describe performance on that distribution. This is stated
plainly rather than glossed over, because the honest limitation is more useful than an inflated
claim.

---

## 3. What was wrong with version 1

The earlier version of this project reported 94.83% with a Gradient Boosting classifier. Auditing
it surfaced four defects that mattered more than the headline number:

| # | Defect | Consequence |
|---|---|---|
| 1 | A single `LabelEncoder` instance was reused across every categorical column in a loop, and never persisted | The encoding could not be reproduced at serving time |
| 2 | The Streamlit form sent `Weather` as 0–3 with *different* category names, `Previous_Risk` as a 0–10 float, `Crime_Trend` as a −5…5 float, and omitted `Day` entirely | The deployed app's inputs did not match the trained model's schema — live predictions were not trustworthy |
| 3 | `Area_ID` (unique per row) was left in the feature matrix | Invited row memorisation |
| 4 | No pipeline, no cross-validation, no calibration check, no tests | Nothing guaranteed the reported number was reproducible |

Every one of these is addressed in v2, and each has a regression test.

---

## 4. Method

### 4.1 Feature engineering

The core contribution. 29 raw inputs → 79 features, all row-wise and target-free.

**Exposure normalisation.** Raw counts conflate volume with rate. `Crime_Per_1k_Population` and
`Crime_Per_1k_Footfall` separate the two.

**Interaction encoding.** `Darkness_Exposure = Is_Night × Broken_Light_Ratio × (3 − Visibility)`
encodes that broken lamps are irrelevant at noon and dangerous at 2am — an interaction a tree
would otherwise have to discover from scratch.

**Criminological priors.**
- `Guardianship_Index` (footfall ÷ density) operationalises Jane Jacobs' "eyes on the street".
- `Surveillance_Deficit` (crime severity ÷ surveillance coverage) measures the gap between threat
  and monitoring.
- `Unpoliced_Crime_Load` (severity × police distance) captures crime happening far from help.
- `Isolation_Index` (nearest help ÷ transit access) measures how trapped someone would be.

**Cyclical time.** `Hour_Sin` / `Hour_Cos` ensure 23:00 and 00:00 are adjacent rather than
maximally distant.

**Composite scores.** `Threat_Score` and `Infrastructure_Score` are hand-written expert rules
supplied as strong priors — the model is free to override them, but does not have to rediscover
them.

**Correctness by construction.** The transform is wrapped in a `FunctionTransformer` and is step
one of the fitted `Pipeline`. Training and serving therefore execute literally the same code. Row
independence is enforced (and unit-tested), so batch and single-record scoring cannot diverge.

**Measured contribution.** Holding the model fixed (logistic regression, identical split,
identical hyper-parameters) and changing *only* the feature representation:

| Feature set | Accuracy |
|---|---:|
| 29 raw columns, label-encoded | 93.89% |
| 79 engineered features | **97.44%** |
| **Gain from feature engineering alone** | **+3.55 points** |

That single ablation accounts for more improvement than the entire move from logistic regression
to tuned gradient boosting (+0.80 points). The representation, not the model, is where
this project's performance lives.

### 4.2 Model selection

Nine candidates were evaluated on an identical stratified 80/20 split with identical features:
majority-class baseline, logistic regression, decision tree, random forest, extra trees,
histogram gradient boosting, two LightGBM configurations, and the ensemble.

**Hyper-parameter sweep (LightGBM).** Deeper trees hurt:

| Configuration | Accuracy | Fit time |
|---|---:|---:|
| n=2000, lr=0.03, leaves=255 | did not converge in budget | — |
| n=1500, lr=0.04, leaves=127 | 97.76% | 135 s |
| n=900, lr=0.05, leaves=63 | 98.03% | 72 s |
| n=1200, lr=0.05, leaves=63, min_child=40 | 98.09% | 90 s |
| n=900, lr=0.07, leaves=31 | 98.21% | 55 s |
| **n=1800, lr=0.05, leaves=31** | **98.27%** | 109 s |

Narrower trees with more boosting rounds generalise better — the signal is smooth and additive
rather than deeply interactive, so wide trees overfit.

### 4.3 Ensembling

Members are **selected, not fixed**: a model joins only if its hold-out accuracy is within 1.0
points of the leader, weighted by a temperature-scaled softmax (T = 0.0015) over accuracy.

This was an empirical correction. The first attempt used a fixed five-member recipe with roughly
linear weights and scored **98.09%** — *worse* than LightGBM alone, because Random Forest
(94.45%) and Extra Trees (93.91%) were dragging the average down. Filtering by tolerance and
sharpening the weights lifted the vote back to within 0.05 points of the leader.

In the final run the tuned LightGBM edged it (98.27% vs 98.22%), so `train.py` promoted the single
model. The two are close enough that the winner can flip between library versions — which is
precisely why the trainer selects from the leaderboard rather than hard-coding a champion.

The lesson is worth stating: when candidates are separated by fractions of a point, a plain
average is the wrong prior.

The ensemble is a `SoftVoteEnsemble` over **pre-fitted** pipelines — members are trained once,
versioned independently, then composed. Ensembling therefore costs nothing at build time.

---

## 5. Results

### 5.1 Leaderboard

| Model | Accuracy | Macro F1 | ROC-AUC | Log loss | κ | CV accuracy |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM (tuned)** | **98.27%** | **97.68%** | 0.9993 | 0.0404 | 0.9702 | **98.00% ± 0.08** |
| RiskRadar Ensemble | 98.22% | 97.62% | 0.9993 | 0.0409 | 0.9694 | — |
| LightGBM (wide trees) | 98.00% | 97.33% | 0.9991 | 0.0456 | 0.9656 | — |
| Hist Gradient Boosting | 97.86% | 97.15% | 0.9990 | 0.0473 | 0.9630 | — |
| Logistic Regression | 97.47% | 96.56% | 0.9985 | 0.0591 | 0.9564 | — |
| Random Forest | 94.45% | 92.64% | 0.9929 | 0.1718 | 0.9042 | — |
| Extra Trees | 93.91% | 91.82% | 0.9914 | 0.1977 | 0.8946 | — |
| Decision Tree | 91.92% | 89.09% | 0.9704 | 0.8419 | 0.8602 | — |
| Majority baseline | 57.29% | 24.28% | 0.5000 | 15.39 | 0.0000 | — |

### 5.2 Improvement over v1

| | v1 | v2 | Change |
|---|---:|---:|---:|
| Accuracy | 94.83% | 98.27% | **+3.43 pts** |
| Error rate | 5.17% | 1.73% | **−66%** |
| Macro F1 | not reported | 97.68% | — |
| Calibration (ECE) | not measured | 0.0013 | — |
| Cross-validated | no | yes | — |
| Tests | 0 | 62 | — |

### 5.3 Error analysis

| actual ↓ / predicted → | Low | Medium | High |
|---|---:|---:|---:|
| **Low** | 4,503 | 92 | **0** |
| **Medium** | 75 | 3,785 | 86 |
| **High** | **0** | 93 | 11,366 |

The two corner cells are **exactly zero**. The model never confused a High-risk area with a
Low-risk one in either direction. All 346 errors are single-band slips adjacent to Medium — which
is both the smallest class and, by construction, the fuzziest boundary.

For a safety application this error profile is close to ideal: the catastrophic failure mode
(telling someone a dangerous street is safe) did not occur once in 20,000 predictions.

### 5.4 Calibration

Expected calibration error of **0.0013** across ten confidence bins means a stated 90% confidence
corresponds to roughly 90% empirical accuracy. This matters because the interface surfaces
confidence to users — an over-confident model would be actively misleading.

### 5.5 What the model uses

Seven of the top ten features are engineered rather than raw:

`CCTV_Per_Crime` (7.08%) · `Crime_Count` (6.81%) · `Unpoliced_Crime_Load` (5.93%) ·
`Broken_Streetlights` (5.25%) · `Crime_Per_1k_Footfall` (5.19%) · `Broken_Light_Ratio` (2.84%) ·
`Police_Distance_km` (2.75%) · `Surveillance_Deficit` (2.65%) · `Theft_Ratio` (2.58%) ·
`Hour_Cos` (2.52%)

The top signals are ratios of threat to protection, not absolute crime volume — which is exactly
what criminological theory predicts and a good sanity check that the model learned structure
rather than noise.

---

## 6. System design

```
Browser (vanilla JS SPA)
        │  JSON over HTTP
        ▼
FastAPI  ─ Pydantic validation at the edge
        │
RiskService (singleton, warmed at boot)
        │
        ├─► sklearn Pipeline:  engineer() → SoftVoteEnsemble
        ├─► SHAP TreeExplainer (built once at start-up)
        └─► Counterfactual lever scan (single batched forward pass)
```

### Performance engineering

Two optimisations took a single explained prediction from **2,987 ms to ~320 ms**:

1. **Shared feature transform.** Every ensemble member's pipeline began with the same
   `engineer()` step, so a naive vote rebuilt the same 79-column matrix four times. Transforming
   once and passing the result to the remaining pipeline steps cut ensemble inference from 292 ms
   to 77 ms. A test asserts the optimised path is numerically identical to the naive one.
2. **Batched counterfactuals.** The intervention scan originally looped one `predict()` per
   trial — up to ten sequential forward passes, ~2,585 ms. Batching all trials into one call
   reduced this to a single pass.

The SHAP explainer takes ~7 s to construct, so it is built during the FastAPI lifespan start-up
rather than lazily; no user ever pays that cost.

### Frontend

A seven-view single-page application with **zero external dependencies** — every chart is
hand-built SVG or Canvas, so the dashboard renders with no CDN, no build step and no network
access. Includes a SHAP waterfall, a canvas risk map with optional kernel-density layer, and a
live sensitivity sweep.

Motion is isolated in `frontend/fx.js`, a small graphics engine built on three rules:

1. **One shared `requestAnimationFrame` loop.** Many independent loops is what actually makes a
   page feel janky — not the number of effects. Every animated system registers a ticker on a
   single scheduler that stops itself when the tab is hidden.
2. **Damped interpolation, not direct assignment.** The 3D card tilt eases toward a target each
   frame rather than snapping to the pointer; that difference is the whole distinction between
   "responsive" and "smooth".
3. **Failure isolation.** Every effect is wrapped so a graphics error degrades polish without
   taking the dashboard down.

The engine provides the particle-constellation backdrop, cursor lighting, parallax tilt, magnetic
buttons, the morphing navigation pill, spring-eased counters, ripples, scroll progress, and page
transitions via the native View Transition API with a hand-rolled fallback. Under
`prefers-reduced-motion` the whole layer collapses to a single static render.

### Readability audit

Contrast was measured rather than eyeballed. Every text/background pair was scored with the
WCAG 2.1 relative-luminance formula against the *worst-case* composited surface — a panel sitting
directly over the brightest aurora blob — and four real defects came out of it:

| Defect | Before | After |
|---|---:|---:|
| `--faint` secondary text | 2.5 : 1 | 4.7 : 1 |
| `--muted` labels on panels | 4.3 : 1 | 6.5 : 1 |
| Panel glass over the violet blob (secondary text) | 3.0 : 1 | 5.0 : 1 |
| Confusion-matrix count in the largest cell | **2.2 : 1** | 8.1 : 1 |

The last one was the worst: the matrix used one fixed light ink for every cell, so the darkest,
highest-count cell — the single most important number on the model card — was the least readable
thing on the page. Cell ink is now chosen per-cell from that cell's own composited luminance.

Two structural causes sat underneath the colour values. The panels were a near-transparent white
veil (5% alpha), which let the animated background bleed through and made contrast depend on
*where on the page a panel happened to sit*; they are now a near-opaque dark glass, so every panel
is the same predictable surface. And chart labels relied on a CSS class for their colour, while
SVG `fill` defaults to **black** — so any context that didn't apply the stylesheet rendered black
text on a dark panel. Labels now carry an explicit `fill` attribute as well.

All text now meets **WCAG 2.1 AA (4.5:1)** on every surface it can appear on, with a minimum
measured ratio of 4.52:1. Keyboard `:focus-visible` rings and a solid-colour fallback for
gradient-clipped headings were added at the same time.

---

## 7. Testing

62 automated tests, plus a jsdom-based frontend smoke test that boots the real page against the
real API and asserts every view renders.

Representative cases:

- **Row independence** — scoring a record alone equals scoring it inside a batch.
- **No target leakage** — the transform produces identical output with and without `Risk_Level`.
- **Reproducible accuracy** — the hold-out set is re-scored from the saved artefact and the
  measured accuracy is compared against the metadata claim.
- **Recommendations verified, not asserted** — every suggested intervention is re-scored to
  confirm it actually lowers the band.
- **Monotonicity** — more CCTV must never decrease the safety score.
- **Domain sanity** — the same street at 03:00 must not score safer than at 15:00.
- **Ensemble optimisation equivalence** — the fast shared-transform path matches the naive path
  to 1e-6.

Two real frontend bugs were caught this way: a chart that never rendered on the overview page,
and colliding SVG gradient IDs that silently repainted later charts in an earlier chart's colours.

---

## 8. Limitations

1. **Synthetic data.** Real-world performance is unproven and will be lower.
2. **Reporting bias.** Recorded crime under-represents actual crime, worst for harassment and
   assault — the offences most central to this project.
3. **Selection optimism.** The winner was chosen on hold-out accuracy. Cross-validation (98.00%)
   is the more conservative estimate.
4. **No temporal drift handling.** The model is a static snapshot and needs periodic retraining.
5. **Feedback loop risk.** Allocating patrols by prediction increases recorded crime where
   patrols go, which raises predicted risk further. Real deployment must monitor for this.

---

## 9. Future work

- Refit on audited municipal crime data and re-validate every figure.
- Add spatial cross-validation (hold out whole districts) to test geographic transfer.
- Model temporal drift with a rolling-window retrain and a drift monitor.
- Ordinal-aware loss, since the classes are ordered and a Low→High error is far worse than
  Low→Medium.
- Conformal prediction for distribution-free confidence intervals rather than raw probabilities.
- A mobile client with live location and route-level risk aggregation.

---

## 10. Conclusion

RiskRadar achieves 98.27% accuracy and 0.9993 ROC-AUC on the RiskRadar corpus, a 66% reduction in
error rate over the project's previous version. The gain is attributable primarily to domain
feature engineering — logistic regression on the engineered features outperforms every tree model
trained on the raw ones.

Equally important, the system is *auditable*: every prediction carries exact Shapley attributions,
every metric is reproducible from the saved artefact by an automated test, and every limitation is
documented rather than hidden. For a system that would advise people about their physical safety,
that transparency is not optional.

---

## References

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAT\* '19.
- Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
- Jacobs, J. (1961). *The Death and Life of Great American Cities.* Random House.
- Cohen, L. & Felson, M. (1979). *Social Change and Crime Rate Trends: A Routine Activity
  Approach.* American Sociological Review.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.
