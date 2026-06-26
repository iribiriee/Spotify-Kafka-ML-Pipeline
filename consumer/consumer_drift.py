"""
consumer/consumer_drift.py
--------------------------
The drift monitor — the academic core. Consumes Prediction messages (which now
carry features + prediction + label) and runs TWO detectors in parallel:

  KS test (data drift): per feature, compares a sliding window of recent values
    against a phase-1 reference distribution. A small p-value means the input
    distribution has shifted. Needs no labels.

  ADWIN (concept drift): feeds the model's error stream (1=wrong, 0=right) to
    river's ADWIN, which flags when the error RATE changes — i.e. the model is
    actually degrading. Needs labels (which we echo in the message).

When either fires, a DriftAlert is written to drift-alerts. Alerts include the
row_index so they can be validated against the known phase boundaries.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from river import drift
from confluent_kafka import Consumer, Producer, KafkaError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from shared.schemas import Prediction, DriftAlert   # noqa: E402

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_HOST", "localhost:9092")
TOPIC_IN = os.getenv("TOPIC_PREDICTIONS", "predictions")
TOPIC_OUT = os.getenv("TOPIC_DRIFT_ALERTS", "drift-alerts")

WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "300"))
KS_PVALUE = float(os.getenv("DRIFT_KS_PVALUE", "0.05"))
# Run the KS test once per this many messages (not every message — that's noisy
# and wasteful, since a 300-window barely changes message to message).
KS_CHECK_EVERY = int(os.getenv("DRIFT_KS_CHECK_EVERY", "50"))
# Don't re-alert on the same feature until this many rows have passed (cooldown).
KS_COOLDOWN = int(os.getenv("DRIFT_KS_COOLDOWN", "300"))

# Continuous features worth testing — the ones that actually shift across genres.
KS_FEATURES = ["acousticness", "energy", "danceability",
               "valence", "loudness", "speechiness"]


class DriftMonitor:
    def __init__(self):
        self.consumer = Consumer({
            "bootstrap.servers": BOOTSTRAP,
            "group.id": "drift-group",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        })
        self.producer = Producer({"bootstrap.servers": BOOTSTRAP})

        # --- KS: reference distributions from phase-1 training data ---
        self.reference = self._load_phase1_reference()
        # Sliding windows of recent feature values, one deque per feature.
        self.windows = {f: deque(maxlen=WINDOW_SIZE) for f in KS_FEATURES}
        self.last_ks_alert_row = {f: -KS_COOLDOWN for f in KS_FEATURES}

        # --- ADWIN: one detector over the model's error stream ---
        self.adwin = drift.ADWIN()

        self.seen = 0

    def _load_phase1_reference(self) -> dict:
        csv = REPO_ROOT / "data" / "genre_ordered_tracks.csv"
        df = pd.read_csv(csv)
        p1 = df[df["genre_phase"] == 1]
        ref = {f: p1[f].astype(float).values for f in KS_FEATURES}
        print(f"[drift] loaded phase-1 reference from {len(p1)} rows "
              f"for features: {KS_FEATURES}")
        return ref

    def _emit(self, alert: DriftAlert) -> None:
        self.producer.produce(topic=TOPIC_OUT, key=alert.detector,
                              value=alert.to_json())
        self.producer.poll(0)
        print(f"[drift] ALERT {alert.detector.upper()} @ row {alert.row_index}: "
              f"{alert.description}")

    def check_ks(self, row_index: int, genre_phase: int) -> None:
        """Run KS per feature; emit an alert for any that has drifted."""
        for feature in KS_FEATURES:
            window = self.windows[feature]
            if len(window) < WINDOW_SIZE:
                continue  # wait for a full window before testing
            if row_index - self.last_ks_alert_row[feature] < KS_COOLDOWN:
                continue  # still cooling down from a recent alert on this feature

            stat, p_value = ks_2samp(self.reference[feature], list(window))
            if p_value < KS_PVALUE:
                self.last_ks_alert_row[feature] = row_index
                self._emit(DriftAlert(
                    detector="ks",
                    row_index=row_index,
                    genre_phase=genre_phase,
                    feature=feature,
                    statistic=round(float(stat), 4),
                    p_value=round(float(p_value), 6),
                    description=(f"data drift in '{feature}' "
                                 f"(KS={stat:.3f}, p={p_value:.2e})"),
                ))

    def check_adwin(self, pred: Prediction) -> None:
        """Feed the error signal to ADWIN; emit if it flags concept drift."""
        error = 0 if pred.correct else 1
        self.adwin.update(error)
        if self.adwin.drift_detected:
            self._emit(DriftAlert(
                detector="adwin",
                row_index=pred.row_index,
                genre_phase=pred.genre_phase,
                feature=None,
                statistic=round(float(self.adwin.estimation), 4),
                p_value=None,
                description=(f"concept drift: error rate shifted to "
                             f"~{self.adwin.estimation:.3f}"),
            ))

    def run(self) -> None:
        self.consumer.subscribe([TOPIC_IN])
        print(f"[drift] consuming '{TOPIC_IN}' -> '{TOPIC_OUT}' at {BOOTSTRAP}")
        print(f"[drift] KS window={WINDOW_SIZE}, p<{KS_PVALUE}, "
              f"check every {KS_CHECK_EVERY}; ADWIN on error stream")
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    print(f"[drift] consumer error: {msg.error()}", file=sys.stderr)
                    continue

                pred = Prediction.from_json(msg.value())

                # ADWIN sees every message (it's cheap and adaptive).
                self.check_adwin(pred)

                # KS: append to windows every message, but only TEST periodically.
                for feature in KS_FEATURES:
                    if feature in pred.features:
                        self.windows[feature].append(float(pred.features[feature]))
                self.seen += 1
                if self.seen % KS_CHECK_EVERY == 0:
                    self.check_ks(pred.row_index, pred.genre_phase)
                    self.producer.flush(5)
        except KeyboardInterrupt:
            print("\n[drift] stopping, flushing...")
        finally:
            self.producer.flush(10)
            self.consumer.close()


def main() -> None:
    DriftMonitor().run()


if __name__ == "__main__":
    main()