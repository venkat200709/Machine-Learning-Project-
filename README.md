# 🛡️ RiskRadar

**Explainable machine learning for women's safety in urban areas.**

RiskRadar classifies any location-hour as **Low**, **Medium** or **High** risk from crime,
infrastructure, population and environmental signals — then explains *why*, and tells you which
realistic intervention would change the answer.

<p align="center">
  <img alt="accuracy"  src="https://img.shields.io/badge/accuracy-98.27%25-2DD4A7?style=flat-square" />
  <img alt="macro f1"  src="https://img.shields.io/badge/macro%20F1-97.68%25-8B7CFF?style=flat-square" />
  <img alt="roc auc"   src="https://img.shields.io/badge/ROC--AUC-0.9993-22D3EE?style=flat-square" />
  <img alt="coverage"  src="https://img.shields.io/badge/conformal%20coverage-89.4%25%20%2F%2090%25%20target-8B7CFF?style=flat-square" />
  <img alt="fairness"  src="https://img.shields.io/badge/fairness%20audit-pass-2DD4A7?style=flat-square" />
  <img alt="tests"     src="https://img.shields.io/badge/tests-101%20passing-2DD4A7?style=flat-square" />
  <img alt="python"    src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square" />
  <img alt="licence"   src="https://img.shields.io/badge/licence-MIT-lightgrey?style=flat-square" />
</p>

---

## Quick start

**Windows:** double-click `start.bat`.

**Any platform:**

```bash
pip install -r requirements.txt
python run.py                 # a trained model ships with the project
python run.py --train         # or retrain the full suite first (~10 min)
```

Then open **http://127.0.0.1:8000** — the dashboard opens automatically.
Interactive API docs live at **http://127.0.0.1:8000/docs**.

> If `python` isn't recognised on Windows, use the launcher instead: `py run.py`
> (and `py -m pip install -r requirements.txt`). Keep the window open — it *is* the server.
> Something not loading? Run `python check.py` in a second window for a full diagnosis.

---

## What it does

| | |
|---|---|
| **Assess an area** | 29 inputs → risk band, confidence, a 0–100 safety score, and a SHAP breakdown of the decision |
| **Explain the verdict** | Exact Shapley values per feature, split into risk factors and protective factors |
| **Quantify what it doesn't know** | Conformal prediction sets with a distribution-free coverage *guarantee* — and automatic abstention when the model genuinely cannot decide |
| **Prescribe an action** | Counterfactual scan: "repair 18 of 25 broken lights → risk falls to Medium" |
| **Plan a safe journey** | Risk-weighted route search over a scored city grid: safest path vs shortest, with the trade-off quantified |
| **Allocate a budget** | Knapsack-constrained optimiser: where ₹40 lakh of lighting, CCTV and patrols buys the most safety |
| **Sweep a variable** | Trace how risk responds as a single input moves across its range |
| **Score in bulk** | Upload a CSV, get every row scored plus an annotated file back |
| **Explore the data** | Nine analytics views + a hand-drawn geospatial risk map |
| **Audit the model** | Full model card, plus a five-dimension fairness audit and a production-readiness report |
| **Operate it** | Live command centre: health score, drift (PSI/KS/JS), alerts, cache, model registry, streamed over a WebSocket |

---

## Results

Trained on 100,000 area-hour records (80/20 stratified split, identical features for every model).

| Model | Accuracy | Macro F1 | ROC-AUC | Log loss | Kappa | CV accuracy |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM (tuned)** ⭐ | **98.27%** | **97.68%** | **0.9993** | 0.0404 | 0.9702 | **98.00% ± 0.08** |
| RiskRadar Ensemble | 98.22% | 97.62% | 0.9993 | 0.0409 | 0.9694 | — |
| LightGBM (wide trees) | 98.00% | 97.33% | 0.9991 | 0.0456 | 0.9656 | — |
| Hist Gradient Boosting | 97.86% | 97.15% | 0.9990 | 0.0473 | 0.9630 | — |
| Logistic Regression | 97.47% | 96.56% | 0.9985 | 0.0591 | 0.9564 | — |
| Random Forest | 94.45% | 92.64% | 0.9929 | 0.1718 | 0.9042 | — |
| Extra Trees | 93.91% | 91.82% | 0.9914 | 0.1977 | 0.8946 | — |
| Decision Tree | 91.92% | 89.09% | 0.9704 | 0.8419 | 0.8602 | — |
| Majority-class baseline | 57.29% | 24.28% | 0.5000 | 15.39 | 0.0000 | — |

The tuned LightGBM and the soft-vote ensemble finish within 0.05 points of each other, and which
one wins can flip between library versions. `train.py` picks the winner from the leaderboard
automatically, so the deployed artefact is always the one that actually scored best on *your*
machine — check the Model Card page in the dashboard to see which it chose.

**Against the previous version of this project (94.83%): +3.43 points, and the error rate falls
from 5.17% to 1.73% — a 66% reduction in mistakes.**

Cross-validation agrees with the hold-out result (98.00% ± 0.08 over 3 folds), and probability
calibration is strong: expected calibration error is **0.0013**, so a stated 90% confidence really
does mean about 90%.

In 20,000 hold-out predictions there were **zero** Low↔High confusions — the model never once
called a genuinely dangerous area safe. All 346 errors are single-band slips involving Medium.

**Most of that gain is feature engineering, not model capacity.** Holding the model fixed
(logistic regression) and changing only the representation: **93.89% → 97.44%, a +3.55 point
gain** — more than the entire move from logistic regression to tuned gradient boosting
(+0.80 points).

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
   Browser  ───────►│  FastAPI  ·  Pydantic-validated edge     │
   (vanilla JS)     │  /predict  /batch  /what-if  /analytics  │
                    └───────────────────┬──────────────────────┘
                                        │
                              ┌─────────▼──────────┐
                              │   RiskService      │  singleton, warm at boot
                              └─────────┬──────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
      ┌───────▼────────┐      ┌─────────▼─────────┐     ┌─────────▼─────────┐
      │ sklearn        │      │ SHAP              │     │ Counterfactual    │
      │ Pipeline       │      │ TreeExplainer     │     │ lever scan        │
      │ ┌────────────┐ │      └───────────────────┘     └───────────────────┘
      │ │ engineer() │ │  ← 29 raw → 79 features, row-wise, no target leakage
      │ └────────────┘ │
      │ ┌────────────┐ │
      │ │ best model │ │  ← promoted from a 9-model leaderboard (LightGBM / ensemble)
      │ └────────────┘ │
      └────────────────┘
```

The feature transform is **step one of the fitted pipeline**, so training and serving run
literally the same code. This removes the most common production ML bug by construction.

The dashboard is deliberately **one self-contained HTML file** with the stylesheet and controller
inlined. A split frontend needs the browser to fetch two further URLs, and if either fails — wrong
base path, stale cache, a static mount shadowing a route — the page renders as raw unstyled HTML
and looks catastrophically broken while the server is perfectly healthy. With a single file there
is nothing left to fail: if the page renders at all, you have the whole interface. It also means
no `StaticFiles` mount, so nothing can shadow an `/api/` route.

---

## The platform layer

The model above is the easy half. Everything below is what separates a notebook
that scores 98% from something a city could actually run.

### Conformal prediction — a guarantee, not an opinion

A softmax score of 0.94 is the model's *opinion*, and on an input unlike
anything it trained on that opinion can be confidently wrong. Split conformal
prediction replaces it with a distribution-free promise: choose an error rate
α, and the true band lands inside the returned set at least `1 − α` of the time
— regardless of whether the model is any good.

The set size becomes an automatic abstention signal:

| Set | Meaning |
|---|---|
| `{High}` | Confident. Act on it. |
| `{Medium, High}` | Genuinely ambiguous — escalate, don't silently take the argmax. |
| `{}` | No band clears the bar. The model is correctly declining to guess. |

Calibration is **class-conditional (Mondrian)**, because marginal coverage can
hit 90% overall while covering the High class only 70% of the time — and High is
the only class where a miss hurts someone. Measured per-class coverage on
held-out data: **89.0% / 90.7% / 89.1%** against a 90% target, with an 89.6%
singleton rate.

The scoring rule (LAC vs APS) is **selected on measured evidence**, not
assumed: both are fitted, and the one meeting the coverage target with the
smaller average set wins. The comparison is recorded in the artefact.

### Drift monitoring

Accuracy is measured once, on the day of training, against data that no longer
exists. Drift monitoring is the only thing standing between "deployed" and
"quietly wrong for eight months".

* **Covariate drift** — PSI per feature, corroborated by KS and Jensen–Shannon.
* **Prediction drift** — has the output mix moved? Needs no labels.
* **Confidence decay** — often twitches before PSI does.
* **Concept drift** — accuracy on records with reported ground truth.

`POST /api/v2/monitor/drift/simulate` injects a synthetic degradation so you can
*watch the detector fire*. A monitor nobody has seen alarm is indistinguishable
from one that doesn't work.

### Fairness audit

This model decides where public money goes, so two failure modes matter:
over-flagging poor dense neighbourhoods (reported crime tracks policing
intensity, which tracks income — a feedback loop that looks like accuracy), and
under-serving the areas that need help most.

The dataset carries no demographic attributes and inventing them would be worse
than omitting them, so the audit uses the accepted equity-of-service proxies:
**density**, **infrastructure provision**, **time of day**, **land use** and
**distance to police**. It reports independence (disparate impact), separation
(equalised odds) and sufficiency (calibration parity), and it deliberately
separates *selection-rate* gaps — which may be legitimate — from *error-rate*
gaps, which are much harder to defend.

Current verdict: **pass** on all five dimensions, TPR gaps under 1%.
`python run.py --audit` exits non-zero on a failure, so CI can block a release.

### Model registry

Retraining used to overwrite `riskradar_model.joblib`, which made rollback,
reproduction and safe evaluation all impossible. Artefacts are now
content-addressed by SHA-256 with stages — `staging → challenger → champion →
archived`. A challenger sees every request the champion sees and its answer is
recorded and compared but never returned, which is the only honest way to
evaluate on production traffic. Promotion is **gated** on shadow volume,
agreement rate and stability; `--force` overrides it and says so in the audit log.

### Security, persistence, observability

* API keys stored as SHA-256 digests, compared in constant time, with roles
  (`viewer` / `analyst` / `admin`), expiry and revocation. Off by default so a
  fresh clone just works; one env var turns it on.
* Per-identity token-bucket rate limiting.
* SQLAlchemy 2.0 persistence — predictions, feedback, drift snapshots, alerts,
  audit log. SQLite by default, PostgreSQL by URL. **Degrades, never crashes**:
  if the database is unreachable the safety service keeps serving and reports
  the outage through `/api/v2/system/readiness`.
* Prometheus metrics at `/api/v2/metrics`, structured JSON logs, request IDs
  propagated end to end.

---

## Project layout

```
ML PROJECT/
├── run.py                      one-command launcher
├── start.bat / check.bat       Windows double-click launcher + diagnostic
├── check.py                    self-diagnostic (files, packages, endpoints)
├── requirements.txt
├── Dockerfile
├── src/riskradar/
│   │  ── model ──────────────────────────────────────────────────
│   ├── config.py               paths, schema, class order — one source of truth
│   ├── data.py                 loading, validation, stratified splitting
│   ├── features.py             79 engineered features (the heart of the project)
│   ├── models.py               model zoo + pre-fitted soft-vote ensemble
│   ├── train.py                benchmark → tune → ensemble → report
│   ├── explain.py              SHAP attribution + counterfactual levers
│   ├── bootstrap.py            post-training artefacts (calibration, reference)
│   │  ── trust ──────────────────────────────────────────────────
│   ├── conformal.py            prediction sets with a coverage guarantee
│   ├── drift.py                PSI / KS / JS monitoring + alerting
│   ├── fairness.py             five-dimension bias audit
│   ├── registry.py             versioning, champion/challenger, shadow, rollback
│   │  ── product ────────────────────────────────────────────────
│   ├── routing.py              risk-weighted route search over a scored grid
│   ├── optimizer.py            budget-constrained intervention allocation
│   │  ── platform ───────────────────────────────────────────────
│   ├── settings.py             twelve-factor runtime config (zero-dependency)
│   ├── db.py                   SQLAlchemy persistence + audit trail
│   ├── security.py             API keys, RBAC, rate limiting
│   ├── cache.py                TTL + LRU inference cache
│   ├── observability.py        Prometheus metrics, JSON logs, request IDs
│   ├── service.py              inference singleton
│   ├── schemas.py              Pydantic request/response contracts
│   ├── api.py                  FastAPI application (v1 surface, unchanged)
│   └── api_v2.py               the platform API
├── frontend/
│   └── index.html              the ENTIRE dashboard - one self-contained file
│                               (CSS + JS inlined; nothing external to fail)
├── deploy/                     prometheus scrape config, alert rules, grafana
├── .github/workflows/ci.yml    lint · test matrix · frontend · fairness · docker
├── tests/                      100 pytest cases + a 60-check jsdom smoke test
├── scripts/                    notebook builder, UI snapshot tool
├── notebooks/                  EDA and modelling walkthrough
├── docs/                       model card, project report, review decks
├── data/                       riskradar_dataset.csv + runtime database
├── models/                     trained artefacts + registry (gitignored)
├── reports/                    leaderboard, analytics, geo sample (generated)
└── legacy/                     the previous Streamlit version, kept for reference
```

---

## Usage

### Train

```bash
python -m riskradar.train                # full suite (from src/, or via run.py --train)
python -m riskradar.train --fast         # reduced budget smoke run
```

Training is also **resumable** — useful on a laptop or in CI, where one long job may be
interrupted:

```bash
python -m riskradar.train --stage model --slug lgbm
python -m riskradar.train --stage cv    --slug lgbm --fold 0 --folds 3
python -m riskradar.train --stage ensemble
python -m riskradar.train --stage finalize
```

### Predict from the API

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"Crime_Count":71,"Harassment_Count":12,"Hour":2,"Working_Streetlights":21,
       "Broken_Streetlights":29,"CCTV_Count":4,"Police_Distance_km":9.3,
       "Visibility":"Poor","Previous_Risk":"High","Crime_Trend":"Increasing"}'
```

Every field has a sensible default — send only what you know.

### Predict from Python

```python
import joblib, pandas as pd
model = joblib.load("models/riskradar_model.joblib")
model.predict(pd.DataFrame([{...raw columns...}]))   # no preprocessing needed
```

### Rebuild the platform artefacts

Conformal calibration, the drift reference and the registry entry are derived
from the model and are regenerated automatically at the end of training. To
rebuild them against an existing model — the path you take after cloning a repo
that ships a model but not these derived files:

```bash
python run.py --calibrate                # conformal + drift reference + registry
python run.py --calibrate --alpha 0.05   # 95% coverage instead of 90%
python run.py --audit                    # fairness report; exits 1 on failure
```

### Test

```bash
pytest                                   # 100 backend tests
node tests/smoke_frontend.js             # 60-check frontend smoke test (npm i jsdom)
```

### Docker

```bash
docker compose up                        # app only, SQLite, zero setup
docker compose --profile full up         # + PostgreSQL, Prometheus, Grafana
```

Grafana lands on <http://localhost:3000> with Prometheus already provisioned.

### Configure

Every deployment setting is an environment variable prefixed `RISKRADAR_`; see
[`.env.example`](.env.example) for the annotated list. Nothing is required —
the defaults give a working dashboard with no setup.

```bash
RISKRADAR_AUTH_ENABLED=true
RISKRADAR_ADMIN_KEY=<a long random string>
RISKRADAR_DATABASE_URL=postgresql+psycopg://user:pw@db:5432/riskradar
RISKRADAR_CORS_ORIGINS=https://dashboard.example.org
```

`GET /api/v2/system/readiness` audits the running configuration and tells you
what would still block a public deployment.

---

## API reference

### v1 — scoring (unchanged; every existing client keeps working)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/predict` | Score one area, with explanation and recommendations |
| `POST` | `/api/predict/batch` | Score an uploaded CSV, return an annotated file |
| `POST` | `/api/what-if` | Sweep one input, trace the risk response curve |
| `GET` | `/api/model` | Model card, leaderboard, feature importance |
| `GET` | `/api/analytics` | Pre-aggregated dataset statistics |
| `GET` | `/api/geo` | Stratified geo sample for the map |
| `GET` | `/api/health` | Liveness, readiness, uptime, request count |
| `GET` | `/docs` | Interactive OpenAPI documentation |

### v2 — the platform

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v2/predict` | Score with a conformal prediction set |
| `POST` | `/api/v2/predict/uncertainty` | How the set widens as the guarantee tightens |
| `POST` | `/api/v2/feedback` | Report ground truth — closes the learning loop |
| `POST` | `/api/v2/route` | Safest vs shortest route, trade-off quantified |
| `GET` | `/api/v2/route/grid` | The scored risk field behind the map |
| `GET` | `/api/v2/route/timelapse` | The city's risk across 24 hours |
| `POST` | `/api/v2/optimise` | Allocate a safety budget across areas |
| `POST` | `/api/v2/optimise/curve` | Risk reduction vs budget — where returns diminish |
| `GET` | `/api/v2/monitor/drift` | PSI / KS / JS across the live window |
| `POST` | `/api/v2/monitor/drift/simulate` | Inject drift and watch the detector fire |
| `GET` | `/api/v2/monitor/live` | Recent predictions from the audit store |
| `GET` | `/api/v2/monitor/alerts` | Operational alerts |
| `WS` | `/api/v2/monitor/stream` | Live telemetry |
| `GET` | `/api/v2/fairness` | Bias audit across five operational strata |
| `GET` | `/api/v2/conformal` | Calibration artefact and verified coverage |
| `GET` | `/api/v2/registry` | Versions, stages, shadow evaluation |
| `GET` | `/api/v2/registry/{v}/gate` | Would this version be allowed to promote? |
| `POST` | `/api/v2/registry/{v}/promote` | Promote to champion (gated) |
| `POST` | `/api/v2/registry/rollback` | Return to the previous champion |
| `GET` | `/api/v2/admin/keys` | Manage API credentials |
| `GET` | `/api/v2/admin/audit` | Append-only trail of state changes |
| `GET` | `/api/v2/system/status` | Everything the control room needs, one call |
| `GET` | `/api/v2/system/readiness` | Production-readiness audit |
| `GET` | `/api/v2/metrics` | Prometheus exposition |

---

## Responsible use

RiskRadar estimates **environmental** risk. It never profiles people, and no personal data is
collected, stored or transmitted.

Reported crime is a biased proxy for actual crime, and under-reporting is worst for exactly the
offences this project cares about — so a Low prediction is *not* a guarantee of safety.

Predicted risk is an argument for **investing** in an area — better lighting, more cameras, closer
patrols — never for withdrawing services from it or stigmatising the people who live there.

Full disclosure of metrics, limitations and intended use is in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## Author

**N. Venkatesan and Neethivendhan T.** · Machine Learning project, 2026 · MIT licence
