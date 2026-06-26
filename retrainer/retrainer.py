"""
retrainer/retrainer.py
----------------------
Closes the loop. Consumes drift-alerts; when ADWIN reports concept drift (the
model is actually failing), it arms a retrain. It then waits until the rolling
buffer has accumulated RETRAIN_DELAY_ROWS of fresh post-drift data (checked by
reading the buffer's latest row_index), trains a new model on the RECENT buffer
only, and publishes it as a new version. The inference consumer hot-reloads it.

A cooldown after each retrain prevents repeated retraining during one drift
episode (e.g. the second ADWIN alert won't fire a redundant retrain mid-recovery).

Design choices (decided with the user):
  - Trigger: ADWIN only (real performance degradation, not mere input shift).
  - Data: recent buffer only (full adaptation to new genres; accepts that the
    model "forgets" old genres — a deliberate trade-off).
  - Timing: short delay after the alert so the buffer holds new-genre data,
    tracked via the buffer file (single source of truth for stream progress).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import Consumer, Producer, KafkaError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from shared.schemas import DriftAlert, FEATURE_NAMES   # noqa: E402
from retrainer.train_model import train_and_publish     # noqa: E402

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_HOST", "localhost:9092")
TOPIC_ALERTS = os.getenv("TOPIC_DRIFT_ALERTS", "drift-alerts")

BUFFER_PATH = REPO_ROOT / "data" / "buffer" / "recent_tracks.csv"
RETRAIN_DELAY_ROWS = int(os.getenv("RETRAIN_DELAY_ROWS", "500"))
RETRAIN_COOLDOWN_ROWS = int(os.getenv("RETRAIN_COOLDOWN_ROWS", "1500"))
RETRAIN_MIN_ROWS = int(os.getenv("RETRAIN_MIN_ROWS", "300"))   # need enough to train
POLL_BUFFER_EVERY = float(os.getenv("RETRAIN_POLL_SECONDS", "2.0"))


def buffer_latest_row() -> int:
    """Max row_index currently in the buffer (stream-progress clock). -1 if none."""
    if not BUFFER_PATH.exists():
        return -1
    try:
        df = pd.read_csv(BUFFER_PATH)
        return int(df["row_index"].max()) if len(df) else -1
    except Exception:
        return -1


def load_recent_buffer(drift_row: int | None = None,
                       min_rows: int = 0) -> pd.DataFrame:
    """Load training data from the buffer.

    If drift_row is given, prefer rows AT OR AFTER it (pure new-genre data, no
    old-genre contamination). If that yields fewer than min_rows, fall back to
    the full buffer so we still have enough to train on.
    """
    df = pd.read_csv(BUFFER_PATH)
    if drift_row is not None:
        post = df[df["row_index"] >= drift_row]
        if len(post) >= max(min_rows, 1):
            df = post
    return df[FEATURE_NAMES + ["label"]].copy()


class Retrainer:
    def __init__(self):
        self.consumer = Consumer({
            "bootstrap.servers": BOOTSTRAP,
            "group.id": "retrainer-group",
            "auto.offset.reset": "latest",   # only react to NEW alerts
            "enable.auto.commit": True,
        })
        self.producer = Producer({"bootstrap.servers": BOOTSTRAP})
        self.pending_target_row = None    # retrain once buffer reaches this row
        self.pending_drift_row = None     # the ADWIN row — train on data after this
        self.cooldown_until_row = -1      # ignore triggers until buffer passes this

    def arm(self, alert: DriftAlert) -> None:
        latest = buffer_latest_row()
        if latest < self.cooldown_until_row:
            print(f"[retrainer] ADWIN @ row {alert.row_index} ignored (cooldown "
                  f"until row {self.cooldown_until_row})")
            return
        if self.pending_target_row is not None:
            return  # already armed
        self.pending_target_row = alert.row_index + RETRAIN_DELAY_ROWS
        self.pending_drift_row = alert.row_index
        print(f"[retrainer] ARMED by ADWIN @ row {alert.row_index}; will retrain "
              f"once buffer reaches row {self.pending_target_row}")

    def maybe_retrain(self) -> None:
        if self.pending_target_row is None:
            return
        latest = buffer_latest_row()
        if latest < self.pending_target_row:
            return  # still waiting for the buffer to fill with post-drift data

        df = load_recent_buffer(drift_row=self.pending_drift_row,
                                min_rows=RETRAIN_MIN_ROWS)
        if len(df) < RETRAIN_MIN_ROWS:
            print(f"[retrainer] buffer only {len(df)} rows; need {RETRAIN_MIN_ROWS}; "
                  f"waiting")
            return

        print(f"[retrainer] RETRAINING on {len(df)} tracks since drift row "
              f"{self.pending_drift_row} (buffer at row {latest})...")
        meta = train_and_publish(df, source_label=f"retrain_at_row_{latest}")
        print(f"[retrainer] published model v{meta['version']} "
              f"f1={meta['metrics']['f1']}; inference will hot-reload")

        # Emit a retrain event so the dashboard can mark it (reuses drift-alerts).
        event = DriftAlert(
            detector="retrain",
            row_index=latest,
            genre_phase=0,
            feature=None,
            statistic=float(meta["version"]),     # carry the new version number
            p_value=None,
            description=(f"retrained -> model v{meta['version']} "
                        f"(f1={meta['metrics']['f1']}) on {len(df)} recent tracks"),
        )
        self.producer.produce(topic=TOPIC_ALERTS, key="retrain",
                              value=event.to_json())
        self.producer.flush(5)

        self.cooldown_until_row = latest + RETRAIN_COOLDOWN_ROWS
        self.pending_target_row = None
        self.pending_drift_row = None

    def run(self) -> None:
        self.consumer.subscribe([TOPIC_ALERTS])
        print(f"[retrainer] watching '{TOPIC_ALERTS}' at {BOOTSTRAP}")
        print(f"[retrainer] trigger=ADWIN, delay={RETRAIN_DELAY_ROWS} rows, "
              f"cooldown={RETRAIN_COOLDOWN_ROWS} rows")
        last_poll = 0.0
        try:
            while True:
                msg = self.consumer.poll(1.0)
                # Check the buffer on a timer even when no new alert arrives.
                if time.time() - last_poll > POLL_BUFFER_EVERY:
                    self.maybe_retrain()
                    last_poll = time.time()
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        print(f"[retrainer] consumer error: {msg.error()}",
                              file=sys.stderr)
                    continue
                alert = DriftAlert.from_json(msg.value())
                if alert.detector == "adwin":
                    self.arm(alert)
        finally:
            self.consumer.close()


def main() -> None:
    Retrainer().run()


if __name__ == "__main__":
    main()