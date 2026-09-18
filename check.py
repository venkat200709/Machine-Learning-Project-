#!/usr/bin/env python
"""RiskRadar self-diagnostic.

Run this while the server is up (in a second CMD window):

    python check.py
    python check.py --port 8001

It probes every artefact and every endpoint the dashboard depends on and
prints exactly what is wrong, rather than leaving you to infer it from a blank
page. Everything is ASCII so no console code page can break it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"
problems: list[str] = []


def line(status: str, label: str, detail: str = "") -> None:
    print(f" {status} {label:<34} {detail}")
    if status == BAD:
        problems.append(label)


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * 70)


def probe(url: str, timeout: float = 25.0):
    """Return (status_code, content_type, body_text or None, error or None)."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read().decode("utf-8", "replace"), None
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read().decode("utf-8", "replace"), None
    except Exception as e:
        return None, "", None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose a running RiskRadar server.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    print("=" * 70)
    print("  RiskRadar diagnostic")
    print("=" * 70)
    print(f"  project : {ROOT}")
    print(f"  server  : {base}")
    print(f"  python  : {sys.version.split()[0]}")

    # ---------------------------------------------------------------- files
    section("1. Files on disk")
    files = {
        "data/riskradar_dataset.csv": True,
        "models/riskradar_model.joblib": True,
        "models/model_metadata.json": True,
        "reports/model_leaderboard.json": True,
        "reports/analytics.json": True,
        "reports/geo_sample.json": True,
        "frontend/index.html": True,
        "frontend/app.js": True,
        "frontend/fx.js": True,
        "frontend/styles.css": True,
    }
    for rel, required in files.items():
        p = ROOT / rel
        if p.exists():
            line(OK, rel, f"{p.stat().st_size:,} bytes")
        else:
            line(BAD if required else WARN, rel, "MISSING")

    # ------------------------------------------------------------- packages
    section("2. Packages")
    for mod in ["numpy", "pandas", "sklearn", "lightgbm", "shap",
                "fastapi", "uvicorn", "pydantic", "joblib", "starlette"]:
        try:
            m = __import__(mod)
            line(OK, mod, getattr(m, "__version__", "?"))
        except Exception as e:
            line(WARN if mod == "shap" else BAD, mod, f"NOT IMPORTABLE ({e})")

    # -------------------------------------------------------------- artefact
    section("3. Model artefact loads")
    try:
        from riskradar.service import RiskService
        svc = RiskService()
        if svc.ready:
            acc = svc.metadata.get("metrics", {}).get("accuracy")
            line(OK, "model loads", svc.metadata.get("model_name", "?"))
            line(OK, "metadata parsed", f"accuracy {acc*100:.2f}%" if acc else "no accuracy")
            line(OK if svc.leaderboard else BAD, "leaderboard parsed",
                 f"{len(svc.leaderboard)} rows")
            line(OK if svc.analytics else BAD, "analytics parsed",
                 f"{len(svc.analytics)} keys")
            line(OK if svc.geo else BAD, "geo sample parsed", f"{len(svc.geo)} points")
        else:
            line(BAD, "model loads", svc.load_error or "unknown reason")
    except Exception as e:
        line(BAD, "model loads", f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------- endpoints
    section("4. HTTP endpoints (server must be running)")
    status, ctype, body, err = probe(f"{base}/api/health", timeout=8)
    if err:
        line(BAD, "server reachable", err)
        print("\n" + "=" * 70)
        print("  The server is NOT running (or is on a different port).")
        print("  Start it in another window:   python run.py")
        print(f"  If you used a custom port:    python check.py --port <port>")
        print("=" * 70)
        return 1
    line(OK, "server reachable", f"HTTP {status}")

    endpoints = [
        ("/api/health", True), ("/api/model", True), ("/api/analytics", True),
        ("/api/geo?limit=5", True), ("/api/template.csv", False),
        ("/", False), ("/app.js", False), ("/fx.js", False), ("/styles.css", False),
    ]
    for path, must_be_json in endpoints:
        status, ctype, body, err = probe(base + path)
        if err:
            line(BAD, path, err); continue
        if status != 200:
            snippet = (body or "")[:160].replace("\n", " ")
            line(BAD, path, f"HTTP {status} :: {snippet}")
            continue
        if must_be_json:
            if "json" not in ctype.lower():
                line(BAD, path, f"HTTP 200 but Content-Type is '{ctype}' "
                                f"(expected JSON) :: {(body or '')[:80]!r}")
                continue
            try:
                data = json.loads(body)
            except Exception as e:
                line(BAD, path, f"HTTP 200 but body is not JSON: {e}")
                continue
            size = len(body)
            hint = ""
            if path == "/api/model":
                hint = data.get("metadata", {}).get("model_name", "")
            elif path == "/api/geo?limit=5":
                hint = f"{data.get('count', 0)} points"
            line(OK, path, f"HTTP 200 · {size:,} bytes {hint}")
        else:
            line(OK, path, f"HTTP {status} · {len(body or ''):,} bytes · {ctype.split(';')[0]}")

    # -- the exact call the dashboard makes first --------------------------
    section("5. The call the dashboard makes on load")
    status, ctype, body, err = probe(f"{base}/api/model")
    if err or status != 200 or "json" not in ctype.lower():
        line(BAD, "GET /api/model", err or f"HTTP {status}, type {ctype}")
        print("\n  This is the call that decides 'Model online' vs 'Model offline'.")
        print("  Raw first 400 characters of the response:\n")
        print("   ", (body or "<no body>")[:400].replace("\n", "\n    "))
    else:
        meta = json.loads(body).get("metadata", {})
        line(OK, "GET /api/model", f"{meta.get('model_name')} "
                                   f"{meta.get('metrics', {}).get('accuracy', 0)*100:.2f}%")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 70)
    if problems:
        print(f"  {len(problems)} PROBLEM(S): " + ", ".join(problems[:6]))
        print("  Copy this whole output and send it back for a precise fix.")
    else:
        print("  ALL CHECKS PASSED - the dashboard should be fully live.")
        print("  If the browser still looks empty, press Ctrl+F5 to bypass its cache.")
    print("=" * 70)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
