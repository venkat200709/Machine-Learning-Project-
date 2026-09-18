# ▶ Start here

## Easiest way

**Double-click `start.bat`** in this folder. That's it.

It finds Python, installs anything missing, and starts the server.
**Keep that window open** — it *is* the server. Closing it stops the site.

## From the command line

On this machine `python` is **not** on your PATH — only the `py` launcher is.
So use `py`, not `python`:

```cmd
cd /d "C:\Users\venka\Documents\Venkat\ML PROJECT"
py run.py
```

Your browser opens at **http://127.0.0.1:8000** automatically. Press **Ctrl+F5** on first
load so the browser doesn't show you a cached copy. Press **Ctrl+C** in the window to stop.

The trained model is already included, so this works immediately.
To retrain from scratch (~10 minutes): `py run.py --train`

## If something looks wrong

While the server is running, open a **second** window and run:

```cmd
py check.py
```

(or double-click `check.bat`). It tests every file, package and endpoint and tells you
exactly what's broken. If it says **"The server is NOT running"**, the page you're looking
at is a cached copy — start the server and press Ctrl+F5.

> **If startup says "The saved model could not be loaded"** — nothing is broken.
> scikit-learn model files aren't always portable across library versions, so `run.py`
> detects it and retrains against *your* installed versions. Takes ~10 minutes, happens
> once, same 98.27% result.

---

## What changed from your version

| | Before | Now |
|---|---|---|
| Accuracy | 94.83% | **98.27%** |
| Error rate | 5.17% | **1.73%** (−66%) |
| Frontend | Streamlit pages | Custom animated dashboard, 7 views |
| Backend | none | FastAPI with OpenAPI docs |
| Explainability | none | SHAP per prediction |
| Tests | none | 61 backend + 42 frontend |
| Features | 30 raw | 29 raw → 79 engineered |

Your original Streamlit app is preserved in `legacy/streamlit_v1/` — nothing was thrown away.

---

## The five-minute demo (for your review)

1. **Overview** — open with this. Point at 98.27% and the leaderboard: nine models,
   same split, same features, fairly compared.

2. **Assess Area** → click **"Isolated night street"** → **Run assessment.**
   You get High risk, a safety score of 0, and a SHAP waterfall showing *exactly* which
   inputs drove it. This is the moment that separates the project from a normal classifier.

3. Scroll to **"What would change the outcome"** — the model names a specific municipal
   intervention that would lower the risk band. It's a planning tool, not just a predictor.

4. Change **Sensitivity sweep** to `CCTV_Count` — watch the risk curve respond live.

5. **Model Card** → the confusion matrix. Say this out loud: *"the Low↔High corners are zero —
   in 20,000 predictions the model never once called a dangerous area safe."* That error
   profile is the single strongest thing you can claim about a safety system.

6. **Risk Map** → hit **Density** for the heat layer.

Have `http://127.0.0.1:8000/docs` open in a second tab. If anyone asks whether it's a real
backend, show them the live OpenAPI page.

---

## Questions you'll probably be asked

**"Why is 98% believable — isn't that overfitting?"**
Three-fold cross-validation gives 98.00% ± 0.08, which matches the hold-out 98.27% almost
exactly. If it were overfitting, CV would be far lower. There's also a test
(`test_reported_accuracy_is_reproducible`) that re-scores the hold-out set from the saved
model file and fails if the claimed number can't be reproduced.

**"What actually made it better?"**
Feature engineering, not a bigger model. Same logistic regression, same split — 93.89% on the
raw columns, 97.44% on the engineered ones. That +3.55 is more than the entire gain from
switching to gradient boosting (+0.80). It's in the report as a controlled ablation.

**"How does it explain itself?"**
Exact Shapley values via SHAP's TreeExplainer on the deployed gradient-boosted model. Every
response states the attribution method it used, so the explanation never overclaims.

**"What's wrong with it?"**
The dataset is synthetic. Reported crime under-represents real crime, worst for harassment and
assault. And predicted risk must never be used to justify pulling services out of an area —
it's an argument for investment. All of this is written down in `docs/MODEL_CARD.md`, and
saying it yourself is stronger than being asked.

---

## Where things are

| File | What it is |
|---|---|
| `README.md` | Full project documentation |
| `docs/PROJECT_REPORT.md` | The written report — method, results, limitations |
| `docs/MODEL_CARD.md` | Formal model card (metrics, intended use, risks) |
| `notebooks/RiskRadar_EDA_and_Modeling.ipynb` | EDA + modelling walkthrough, runs end to end |
| `src/riskradar/features.py` | The 79 engineered features — the heart of the project |
| `src/riskradar/train.py` | Benchmark → ensemble → report pipeline |
| `frontend/index.html` | The entire dashboard - one self-contained file |
| `tests/` | 61 backend tests + 42 frontend checks |
| `check.py` | one-command diagnostic |

---

## Troubleshooting

**"The saved model could not be loaded"** — expected on a different Python/scikit-learn version.
`run.py` retrains automatically; just let it finish. Nothing is wrong with the project.

**`'python' is not recognized`** — use `py` instead of `python` (that is the whole fix).

**`'pip' is not recognized`** — use `py -m pip install -r requirements.txt`.

**`ModuleNotFoundError: lightgbm`** — run `py -m pip install -r requirements.txt` again.

**Port 8000 already in use** — an old server is still running. Close that window, or use
`py run.py --port 8080`.

**Blank dashboard / 'Model offline'** — the server isn't running. Run `py check.py`; if
section 4 fails, start the server with `py run.py` and hard-refresh with Ctrl+F5.

**Browser didn't open** — go to http://127.0.0.1:8000 manually.

**Changes don't appear** — press Ctrl+F5, or fully close and reopen the browser.

**Unstyled / raw-looking page** — you opened `index.html` from the folder instead of going through
the server. Run `py run.py` and browse to http://127.0.0.1:8000. (The page now detects this and
tells you.)
