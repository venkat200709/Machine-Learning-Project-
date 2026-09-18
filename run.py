#!/usr/bin/env python
"""RiskRadar launcher.

    python run.py            # start the API + dashboard on http://127.0.0.1:8000
    python run.py --train    # train first, then start
    python run.py --port 9000

Puts ``src/`` on the path so the project runs from a clean checkout with no
``pip install -e .`` step.
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import sys
import webbrowser
from pathlib import Path
from threading import Timer

# ── Console encoding ──────────────────────────────────────────────────
# Windows consoles still default to a legacy code page (cp437/cp850) that
# cannot represent characters like an em-dash. A single such character in a
# print() raises UnicodeEncodeError and kills the launcher *before* the server
# starts — which looks to the user like "the app just doesn't work". Force
# UTF-8 and fall back to replacement characters rather than ever crashing.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))


def port_in_use(host: str, port: int) -> bool:
    """True if something is already listening — the usual cause of a silent
    'my changes didn't apply' is an older server still holding the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0


def preflight(auto_train: bool = True) -> None:
    """Prove the model actually loads *before* we open a browser at it.

    The shipped artefact is a pickled scikit-learn pipeline, which is only
    guaranteed to deserialise under the library versions it was written with.
    If the user's scikit-learn / numpy / LightGBM differ enough, loading fails —
    and the honest fix is a local retrain, which we offer to do for them rather
    than leaving a stack trace on screen.
    """
    from riskradar import config as C
    from riskradar.service import RiskService

    if not C.MODEL_PATH.exists():
        print("=" * 68)
        print("  No trained model found.")
        print("=" * 68)
        if not auto_train:
            print("  Train one with:  python run.py --train")
            sys.exit(1)
        print("  Training one now - this takes about 10 minutes.\n")
        from riskradar.train import run as train_run

        train_run()
        RiskService._instance = None

    svc = RiskService.instance()
    if svc.ready:
        return

    print("=" * 68)
    print("  The saved model could not be loaded.")
    print("=" * 68)
    print(f"  {svc.load_error}\n")

    if not auto_train:
        sys.exit(1)

    print("  Retraining locally against your installed library versions.")
    print("  This takes about 10 minutes and only has to happen once.\n")
    from riskradar.train import run as train_run

    train_run()

    RiskService._instance = None
    if not RiskService.instance().ready:
        print("  Retraining did not fix it. Please check the errors above.")
        sys.exit(1)
    print("\n  Fixed - model retrained and loaded.\n")


def run_fairness_audit() -> None:
    """Print the bias audit to the console — the CI-friendly entry point."""
    import joblib

    from riskradar import config as C
    from riskradar.fairness import audit_from_dataset

    if not C.MODEL_PATH.exists():
        print("No trained model. Run:  python run.py --train")
        sys.exit(1)

    print("=" * 72)
    print("  RiskRadar - Algorithmic Fairness Audit")
    print("=" * 72)

    report = audit_from_dataset(joblib.load(C.MODEL_PATH))
    print(f"\nEvaluated {report['n_evaluated']:,} hold-out records")
    print(f"Overall verdict: {report['overall_verdict'].upper()}\n")

    for dim in report["dimensions"].values():
        if not dim.get("available"):
            continue
        print(f"  {dim['label']:<48} {dim['verdict'].upper()}")
        m = dim["metrics"]
        print(f"      disparate impact {m['disparate_impact_ratio']}  "
              f"TPR gap {m['true_positive_rate_gap']}  "
              f"accuracy gap {m['accuracy_gap']}")

    if report["findings"]:
        print("\nFindings")
        print("-" * 72)
        for f in report["findings"]:
            print(f"  [{f['severity']:<6}] {f['dimension']}: {f['message']}")

    print("\n" + report["summary"])
    print("=" * 72)
    # Non-zero exit on a failing audit so CI can gate a release on it.
    sys.exit(1 if report["overall_verdict"] == "fail" else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RiskRadar platform.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--train", action="store_true", help="train before serving")
    parser.add_argument("--fast", action="store_true", help="reduced training budget")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code change")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-auto-train", action="store_true",
                        help="fail instead of retraining if the artefact won't load")
    parser.add_argument("--calibrate", action="store_true",
                        help="rebuild conformal calibration, drift reference and "
                             "registry entry against the existing model, then exit")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="conformal error rate (0.10 = 90%% coverage)")
    parser.add_argument("--audit", action="store_true",
                        help="run the fairness audit and print the report, then exit")
    args = parser.parse_args()

    if args.train:
        from riskradar.train import run as train_run

        train_run(fast=args.fast)

    if args.calibrate:
        from riskradar.bootstrap import bootstrap_all

        bootstrap_all(alpha=args.alpha)
        return

    if args.audit:
        run_fairness_audit()
        return

    preflight(auto_train=not args.no_auto_train)

    import uvicorn

    url = f"http://{args.host}:{args.port}"

    if port_in_use(args.host, args.port):
        print("=" * 68)
        print(f"  Port {args.port} is already in use.")
        print("=" * 68)
        print("  Another RiskRadar server is probably still running in a")
        print("  different window. Your browser would keep talking to THAT one,")
        print("  so any new changes would appear to have no effect.")
        print()
        print("  Fix it by either:")
        print("    - pressing Ctrl+C in the other window, then re-running this, or")
        print(f"    - starting on a free port:  python run.py --port {args.port + 1}")
        print("=" * 68)
        sys.exit(1)

    print("=" * 68)
    print("  RiskRadar - AI Women's Safety Intelligence Platform")
    print("=" * 68)
    print(f"  Dashboard : {url}")
    print(f"  API docs  : {url}/docs")
    print("  Stop with Ctrl+C")
    print("=" * 68)
    print("  Tip: press Ctrl+F5 in the browser to bypass its cache.")
    print()

    if not args.no_browser:
        Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "riskradar.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
