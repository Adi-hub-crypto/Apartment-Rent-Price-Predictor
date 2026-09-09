"""Local web interface for the rent price model.

    python3 web_app.py

Serves http://127.0.0.1:8000 and opens it.  Standard library only - no Flask,
no CDN, no API keys, no network access.

    GET  /                 the interface
    GET  /api/state        model status, metrics, districts
    POST /api/estimate     {area, district, floor, distance} -> price + breakdown
    POST /api/batch        {csv} -> a predicted rent per row
    GET  /api/submission   the last batch as a CSV download
    POST /api/train        {method} -> refit and reload
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from rent_data import (DISTRICT_LABELS, DISTRICTS, TARGET, load_csv, load_train)
from rent_model import METHODS, RentModel, fit_full

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "model.pkl")
METRICS_FILE = os.path.join(HERE, "metrics.json")
INDEX_FILE = os.path.join(HERE, "static", "index.html")

REQUIRED = ["area", "district", "floor", "distance_to_center"]


class Service:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.submission = None
        if os.path.exists(MODEL_FILE):
            try:
                self.model = RentModel.load(MODEL_FILE)
            except Exception:                              # noqa: BLE001
                self.model = None

    def metrics(self):
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE) as fh:
                return json.load(fh)
        return None

    def state(self):
        return {
            "ready": self.model is not None,
            "method": getattr(self.model, "method", None),
            "methods": METHODS,
            "districts": [{"key": d, "label": DISTRICT_LABELS[d]} for d in DISTRICTS],
            "reference": getattr(self.model, "reference_", None),
            "metrics": self.metrics(),
        }

    def estimate(self, area, district, floor, distance):
        if self.model is None:
            raise RuntimeError("No model loaded. Train one from the Model tab.")
        if district not in DISTRICTS:
            raise ValueError(f"{district!r} is not one of the ten districts.")
        if not (0 < float(area) <= 1000):
            raise ValueError("Area must be between 0 and 1000 m².")
        if not (1 <= int(floor) <= 200):
            raise ValueError("Floor must be between 1 and 200.")
        if not (0 <= float(distance) <= 100):
            raise ValueError("Distance must be between 0 and 100 km.")

        detail = self.model.explain(area, district, floor, distance)
        return {
            "price": detail["price"],
            "base": detail["base"],
            "reference": detail["reference"],
            "steps": detail["steps"],
            "district_label": DISTRICT_LABELS[district],
        }

    def batch(self, csv_text):
        if self.model is None:
            raise RuntimeError("No model loaded. Train one from the Model tab.")
        df = load_csv(io.StringIO(csv_text))

        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError("That CSV is missing: " + ", ".join(missing))

        unknown = sorted(set(df["district"].astype(str)) - set(DISTRICTS))
        preds = self.model.predict(df)
        ids = df["id"].tolist() if "id" in df.columns else list(range(len(preds)))
        self.submission = pd.DataFrame({"id": ids, TARGET: preds})

        check = {"columns": list(self.submission.columns), "rows": len(preds),
                 "unknown_districts": unknown}
        sample_path = os.path.join(HERE, "data", "sample_submission.csv")
        if os.path.exists(sample_path):
            sample = load_csv(sample_path)
            check["sample_rows"] = len(sample)
            check["matches_sample"] = bool(
                list(sample.columns) == list(self.submission.columns)
                and len(sample) == len(self.submission)
                and sample["id"].tolist() == self.submission["id"].tolist())

        preview = [{"id": int(i), "price": float(p),
                    "area": float(a), "district": str(d), "floor": int(f),
                    "distance": float(dc)}
                   for i, p, a, d, f, dc in zip(
                       ids[:12], preds[:12], df["area"][:12], df["district"][:12],
                       df["floor"][:12], df["distance_to_center"][:12])]

        summary = {
            "min": float(preds.min()), "max": float(preds.max()),
            "median": float(pd.Series(preds).median()),
            "mean": float(preds.mean()),
        }
        return {"rows": len(preds), "check": check, "preview": preview,
                "summary": summary}

    def train(self, method):
        with self.lock:
            train = load_train()
            model = fit_full(train, method=method)
            model.save(MODEL_FILE)
            self.model = model
        return {"method": method, "label": METHODS[method],
                "n_train": len(train), "n_features": int(model.n_features_)}


class Handler(BaseHTTPRequestHandler):
    service: Service = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, body, content_type="application/json", status=200, extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(INDEX_FILE, encoding="utf-8") as fh:
                self._send(fh.read(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(self.service.state())
        elif self.path.startswith("/data/") and self.path.endswith(".csv"):
            path = os.path.join(HERE, "data", os.path.basename(self.path))
            if not os.path.exists(path):
                self._send({"error": f"{os.path.basename(self.path)} not found"},
                           status=404)
                return
            with open(path, encoding="utf-8") as fh:
                self._send(fh.read(), "text/csv; charset=utf-8")
        elif self.path == "/api/submission":
            if self.service.submission is None:
                self._send({"error": "Nothing classified yet."}, status=404)
                return
            # CRLF, matching sample_submission.csv
            csv_text = self.service.submission.to_csv(index=False, lineterminator="\r\n")
            self._send(csv_text, "text/csv; charset=utf-8", extra={
                "Content-Disposition": 'attachment; filename="submission.csv"'})
        else:
            self._send({"error": "Not found"}, status=404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or "{}")
            if self.path == "/api/estimate":
                result = self.service.estimate(
                    data.get("area"), data.get("district"),
                    data.get("floor"), data.get("distance"))
            elif self.path == "/api/batch":
                result = self.service.batch(data.get("csv", ""))
            elif self.path == "/api/train":
                result = self.service.train(data.get("method", "ols"))
            else:
                self._send({"error": "Not found"}, status=404)
                return
            self._send(result)
        except Exception as exc:                           # noqa: BLE001
            self._send({"error": str(exc)}, status=400)


def free_port(preferred):
    for port in range(preferred, preferred + 20):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    Handler.service = Service()
    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\n  Rent price estimator -> {url}")
    print("  Press Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
