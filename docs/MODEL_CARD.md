# Model Card — RiskRadar v2.0.0

*Following the model-card framework of Mitchell et al. (2019).*

---

## 1. Model details

| | |
|---|---|
| **Name** | LightGBM (tuned) |
| **Version** | 2.0.0 |
| **Type** | Multi-class classifier (3 ordered classes: Low → Medium → High) |
| **Architecture** | Gradient-boosted trees — 1,800 estimators, 31 leaves, lr 0.05, subsample 0.85, colsample 0.85, L2 1.0 |
| **Runner-up** | RiskRadar Ensemble (weighted soft-vote), 98.22% — see §4.3 of the project report |
| **Pipeline** | `FunctionTransformer(engineer)` → estimator. The artefact consumes **raw** records. |
| **Framework** | scikit-learn 1.9 · LightGBM 4.x · Python 3.13 |
| **Artefact** | `models/riskradar_model.joblib` (~19 MB) |
| **Licence** | MIT |
| **Authors** | N. Venkatesan and Neethivendhan T. |

### Why this model, and why not the ensemble?

The trainer benchmarks nine candidates and promotes whichever scores best — the choice is made by
`train.py`, not by hand. Here the tuned LightGBM won at 98.27%, with the soft-vote ensemble a
close second at 98.22%.

The ensemble is still worth describing, because building it surfaced a real result. Its members
are selected automatically: only models within 1.0 accuracy point of the leader join the vote,
weighted by a temperature-scaled softmax over hold-out accuracy. Random Forest (94.45%) and Extra
Trees (93.91%) are **excluded by that rule** — an earlier near-uniform weighting included them and
dragged the vote down *below* its own best member. When candidates are separated by fractions of a
point, a plain average is the wrong prior.

The two finish within 0.05 points and the winner can flip between library versions, so the
deployed artefact is simply whichever won on the machine that trained it.

---

## 2. Intended use

**Primary use.** Estimating environmental safety risk for a location at a point in time, to
support: route planning for individuals, patrol allocation for authorities, and prioritising
where lighting and CCTV budgets should go.

**Users.** Urban-safety researchers, municipal planners, and developers building safety features
on top of the API.

**Out of scope.**

- ❌ Identifying, profiling or scoring **individual people**. The model takes no personal data.
- ❌ Justifying **withdrawal** of services, transport or policing from an area.
- ❌ Insurance pricing, property valuation, or any use that penalises residents of a predicted-
  high-risk area.
- ❌ Sole basis for a safety-critical decision. It is decision *support*, not a decision maker.

---

## 3. Data

| | |
|---|---|
| **Source** | RiskRadar dataset — 100,000 area-hour records, Chennai metropolitan coordinates |
| **Split** | 80,000 train / 20,000 test, stratified, `random_state=42` |
| **Missing values** | 0 |
| **Duplicate rows** | 0 |
| **Class balance** | High 57,296 (57.3%) · Low 22,974 (23.0%) · Medium 19,730 (19.7%) — imbalance ratio 2.90 |

### Raw inputs (29 used, 1 dropped)

**Crime** — total incidents, violent, theft, assault, harassment, emergency calls
**Time** — hour, day of week, month, weekend flag
**Infrastructure** — total / working / broken streetlights, CCTV count
**Population** — density, footfall
**Access** — bus stops, distance to metro / police / hospital, school count
**Land use** — commercial flag, residential flag
**Environment** — weather, visibility, previously recorded risk, crime trend

`Area_ID` is **dropped**: it is unique per row, so retaining it would let a tree memorise
individual records rather than learn generalisable structure.

### ⚠️ The dataset is synthetic

This corpus is modelled on Chennai but is **not** verified municipal crime data. Every number in
this card describes performance on that synthetic distribution. Before any real deployment the
model must be refitted and re-validated on audited data, and the metrics below should be assumed
to be optimistic.

---

## 4. Feature engineering

79 features are derived row-wise from the 29 raw inputs. The transform is the first step of the
fitted pipeline, so training and serving are guaranteed identical.

**Design constraints, enforced by tests:**

1. **No target usage** — nothing reads `Risk_Level`.
2. **No cross-row statistics** — every feature is a pure function of one row, so there is no
   train/test leakage and a single record can be scored in isolation.
3. **Exposure normalisation** — 40 incidents means something different in a dense commercial hub
   than in a quiet lane, so counts are divided by population and footfall.

**Representative engineered features:**

| Feature | Idea |
|---|---|
| `Darkness_Exposure` | Broken lights × night flag × poor visibility — encodes that lamp failures only matter after dark |
| `Surveillance_Deficit` | Weighted crime severity ÷ (CCTV + working lights) — the gap between threat and coverage |
| `Unpoliced_Crime_Load` | Crime severity × distance to police — crime far from help |
| `Guardianship_Index` | Footfall ÷ population density — Jane Jacobs' "eyes on the street" |
| `Isolation_Index` | Distance to nearest help ÷ transit access |
| `Gendered_Crime_Ratio` | Severity-weighted share of assault + harassment + violent crime |
| `Hour_Sin` / `Hour_Cos` | Cyclical encoding so 23:00 sits next to 00:00 |
| `Threat_Score` / `Infrastructure_Score` | Composite expert priors handed to the model |

**Ablation — feature engineering, not model size, drove the gain.** With the model, split and
hyper-parameters held fixed (logistic regression) and only the representation changed:

| Feature set | Accuracy |
|---|---:|
| 29 raw columns, label-encoded | 93.89% |
| 79 engineered features | **97.44%** |
| **Gain from representation alone** | **+3.55 pts** |

For comparison, upgrading from logistic regression to tuned gradient boosting adds only +0.80
points on top of that.

---

## 5. Performance

Hold-out test set, 20,000 records never seen during training or model selection.

| Metric | Value |
|---|---|
| Accuracy | **98.27%** |
| Balanced accuracy | 97.70% |
| Macro F1 | 97.68% |
| Weighted F1 | 98.27% |
| ROC-AUC (one-vs-rest, macro) | 0.9993 |
| Log loss | 0.0404 |
| Cohen's κ | 0.9702 |
| Matthews correlation | 0.9702 |
| Expected calibration error | **0.0013** |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Low | 98.36% | 98.00% | 98.18% | 4,595 |
| Medium | 95.34% | 95.92% | 95.63% | 3,946 |
| High | 99.25% | 99.19% | 99.22% | 11,459 |

### Confusion matrix

| actual ↓ / predicted → | Low | Medium | High |
|---|---:|---:|---:|
| **Low** | 4,503 | 92 | **0** |
| **Medium** | 75 | 3,785 | 86 |
| **High** | **0** | 93 | 11,366 |

**The critical off-diagonal cells are zero.** The model never once called a genuinely High-risk
area "Low", and never called a Low-risk area "High". All 346 errors are single-band slips
involving Medium — the least costly kind of mistake this system can make.

### Cross-validation

3-fold stratified CV on the training set, measured directly on the deployed model:
**98.00% ± 0.08** (folds: 98.11%, 97.97%, 97.93%).

The 0.08-point spread and the closeness of CV to hold-out accuracy indicate the model is not
overfitting the split.

### Feature importance (top 10)

| Rank | Feature | Weight |
|---:|---|---:|
| 1 | `CCTV_Per_Crime` | 7.08% |
| 2 | `Crime_Count` | 6.81% |
| 3 | `Unpoliced_Crime_Load` | 5.93% |
| 4 | `Broken_Streetlights` | 5.25% |
| 5 | `Crime_Per_1k_Footfall` | 5.19% |
| 6 | `Broken_Light_Ratio` | 2.84% |
| 7 | `Police_Distance_km` | 2.75% |
| 8 | `Surveillance_Deficit` | 2.65% |
| 9 | `Theft_Ratio` | 2.58% |
| 10 | `Hour_Cos` | 2.52% |

Seven of the top ten are **engineered**, not raw — direct evidence that the domain modelling is
carrying the performance.

### Latency

| Operation | Time |
|---|---|
| Prediction only | ~30–80 ms |
| Prediction + SHAP + counterfactual scan | ~150–320 ms |
| Batch, 1,000 rows | ~400–715 ms |

The SHAP explainer is built once at server start-up, so no user pays the ~7 s construction cost.

---

## 6. Explainability

Attribution uses **exact Shapley values** via `shap.TreeExplainer` on the deployed gradient-
boosted model. When an ensemble is deployed instead, the explainer targets its highest-weighted
tree member and the response says so explicitly, rather than implying every member was explained.

If SHAP is unavailable, the service falls back to importance-weighted deviation from the
background median — clearly labelled as such in the response, never silently substituted.

---

## 7. Limitations and risks

**Data**

- Synthetic corpus; real-world performance is unproven and will be lower.
- Reported crime under-represents actual crime, systematically and worst for harassment and
  assault — the offences most central to this project.
- Geographically limited to Chennai coordinates; other cities need refitting.

**Model**

- Predicts *reported-risk band*, not personal danger. A Low prediction is not a safety guarantee.
- Trained on a static snapshot with no temporal drift handling. Requires periodic retraining.
- The 3-class boundary is coarse; Medium is the weakest class (95.6% F1) and absorbs almost all
  the residual error.

**Deployment**

- Model selection used hold-out accuracy, which introduces mild optimism into the reported figure.
  Cross-validation (98.00% ± 0.08) is the more conservative estimate.
- Counterfactual recommendations describe *model* behaviour, not a causal guarantee that the
  intervention will reduce real crime.

**Societal**

- **Feedback loop risk.** If patrols are allocated by prediction, more crime is recorded where
  patrols go, which raises predicted risk further. Any real deployment must monitor for this.
- **Stigma risk.** Publishing area-level risk can depress property values and reputations.
  Outputs should inform investment, never disinvestment.
- **Automation bias.** A confident-looking 98% figure invites over-trust. The safety score is a
  prior, not a substitute for situational judgement.

---

## 8. Ethical considerations

No personal, biometric or identifying data is used at any stage. Features are environmental
(lighting, cameras, distances, aggregate counts).

The explainability layer is a deliberate safeguard, not a feature: an operator can always see why
a verdict was reached and challenge it. A safety model that cannot be interrogated should not be
deployed.

---

## 9. Reproducibility

```bash
pip install -r requirements.txt
python run.py --train
pytest                          # 62 tests, including a check that the claimed
                                # accuracy is reproducible from the artefact
```

Fixed `random_state=42` throughout; the stratified split is deterministic. Artefacts are written
to `models/` and `reports/` with a full metadata record including training timestamp, library
versions and the complete feature list.

---

*Model card last updated: 10 August 2026 · artefact trained 10 August 2026 on Python 3.13*
