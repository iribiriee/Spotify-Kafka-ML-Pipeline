"""
dashboard/backend.py
--------------------
Backend for the drift dashboard. Two jobs in one process:

  1. A background thread runs a Kafka consumer subscribed to BOTH 'predictions'
     and 'drift-alerts', folding each message into an in-memory summary.
  2. Flask serves the page ('/') and the live summary ('/api/state' as JSON),
     which the browser polls every couple of seconds.

A lock guards the shared state so an HTTP read never sees a half-written update.

Run on the host:
    python dashboard/backend.py
Then open http://localhost:5000
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from flask import Flask, jsonify, send_from_directory
from confluent_kafka import Consumer, KafkaError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from shared.schemas import Prediction, DriftAlert   # noqa: E402

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_HOST", "localhost:9092")
TOPIC_PRED = os.getenv("TOPIC_PREDICTIONS", "predictions")
TOPIC_ALERTS = os.getenv("TOPIC_DRIFT_ALERTS", "drift-alerts")
PORT = int(os.getenv("DASHBOARD_PORT", "5000"))

ACC_WINDOW = 200          # rolling accuracy window
FEATURE = "acousticness"  # feature shown in the distribution-shift panel
FEATURE_WINDOW = 300      # live window size for that feature
ACC_SERIES_MAX = 2000     # plot points kept (every 25 msgs => covers ~50k tracks)
ALERTS_MAX = 60           # rich alert records for the LOG panel (recent, scrollable)
ALERT_ROWS_MAX = 5000     # alert row positions for CHART markers (just integers)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR))


class DashboardState:
    """Shared, lock-guarded summary the consumer writes and Flask reads."""
    def __init__(self):
        self.lock = threading.Lock()
        self.processed = 0
        self.current_row = 0
        self.current_phase = 0
        self.correct_window = deque(maxlen=ACC_WINDOW)
        self.acc_series = deque(maxlen=ACC_SERIES_MAX)      # (row, accuracy%)
        self.feature_window = deque(maxlen=FEATURE_WINDOW)
        self.alerts = deque(maxlen=ALERTS_MAX)
        self.alert_rows_all = deque(maxlen=ALERT_ROWS_MAX)  # for chart markers
        self.alert_count = 0
        self.last_pred_time = 0.0   # wall-clock of most recent prediction
        self.model_version = 1      # bumps when a retrain event arrives
        self.retrain_rows = []      # stream rows where retraining happened
        # phase-1 reference for the chosen feature (histogram baseline)
        self.reference = self._load_reference()

    def _load_reference(self) -> list:
        csv = REPO_ROOT / "data" / "genre_ordered_tracks.csv"
        df = pd.read_csv(csv)
        return df[df["genre_phase"] == 1][FEATURE].astype(float).tolist()

    def on_prediction(self, p: Prediction) -> None:
        with self.lock:
            self.processed += 1
            self.current_row = p.row_index
            self.current_phase = p.genre_phase
            self.last_pred_time = time.time()
            self.correct_window.append(1 if p.correct else 0)
            if FEATURE in p.features:
                self.feature_window.append(float(p.features[FEATURE]))
            # Record a rolling-accuracy point every 25 messages (keeps it light).
            if self.processed % 25 == 0 and self.correct_window:
                acc = 100.0 * sum(self.correct_window) / len(self.correct_window)
                self.acc_series.append([p.row_index, round(acc, 1)])

    def on_alert(self, a: DriftAlert) -> None:
        with self.lock:
            if a.detector == "retrain":
                # statistic carries the new version number.
                if a.statistic:
                    self.model_version = int(a.statistic)
                self.retrain_rows.append(a.row_index)
                self.alerts.append({
                    "detector": "retrain",
                    "row_index": a.row_index,
                    "phase": a.genre_phase,
                    "feature": None,
                    "statistic": None,
                    "description": a.description,
                })
                return
            self.alert_count += 1
            self.alert_rows_all.append(a.row_index)
            self.alerts.append({
                "detector": a.detector,
                "row_index": a.row_index,
                "phase": a.genre_phase,
                "feature": a.feature,
                "statistic": a.statistic,
                "description": a.description,
            })

    def snapshot(self) -> dict:
        with self.lock:
            acc = (100.0 * sum(self.correct_window) / len(self.correct_window)
                   if self.correct_window else 0.0)
            # Live KS for the feature panel (only once the window is full enough).
            live = list(self.feature_window)
            if len(live) >= 50:
                ks_stat, _ = ks_2samp(self.reference, live)
                ks_stat = round(float(ks_stat), 3)
            else:
                ks_stat = None
            # Alert row positions for the accuracy chart markers.
            alert_rows = list(self.alert_rows_all)
            # Status from staleness: nothing yet = idle; recent message = ongoing;
            # messages seen but quiet for a while = finished.
            if self.processed == 0:
                status = "idle"
            elif time.time() - self.last_pred_time < 5.0:
                status = "ongoing"
            else:
                status = "finished"
            return {
                "processed": self.processed,
                "current_row": self.current_row,
                "current_phase": self.current_phase,
                "status": status,
                "model_version": self.model_version,
                "retrain_rows": list(self.retrain_rows),
                "accuracy": round(acc, 1),
                "alert_count": self.alert_count,
                "acc_series": list(self.acc_series),
                "alert_rows": alert_rows,
                "reference": self.reference,
                "feature_window": live,
                "feature_name": FEATURE,
                "ks_stat": ks_stat,
                "alerts": list(self.alerts)[-30:][::-1],  # newest first
            }


state = DashboardState()


def consume_loop() -> None:
    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": "dashboard-group",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([TOPIC_PRED, TOPIC_ALERTS])
    print(f"[dashboard] consuming '{TOPIC_PRED}' + '{TOPIC_ALERTS}' at {BOOTSTRAP}")
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"[dashboard] consumer error: {msg.error()}",
                          file=sys.stderr)
                continue
            if msg.topic() == TOPIC_PRED:
                state.on_prediction(Prediction.from_json(msg.value()))
            else:
                state.on_alert(DriftAlert.from_json(msg.value()))
    finally:
        consumer.close()


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/state")
def api_state():
    return jsonify(state.snapshot())


def main() -> None:
    # Kafka consumer runs in a daemon thread; Flask owns the main thread.
    threading.Thread(target=consume_loop, daemon=True).start()
    print(f"[dashboard] serving on http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()