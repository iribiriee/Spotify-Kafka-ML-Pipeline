"""
consumer/consumer_inference.py
------------------------------
The real-time prediction service. Reads RawTrack messages from raw-tracks,
runs the current model, and writes Prediction messages to predictions.

Designed to run either standalone (python consumer/consumer_inference.py) or
as a thread launched by consumer/main.py.

Key behaviours:
  - Feature vector built in FEATURE_NAMES order (must match training).
  - Scales with the SAVED scaler from the model bundle (never refits).
  - HOT-RELOAD: every RELOAD_CHECK_EVERY messages, checks current.txt; if a
    newer model version was published, reloads the bundle mid-stream.
  - Joins consumer group 'inference-group' and reads from earliest, so it
    processes the whole stream and resumes cleanly after a restart.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from confluent_kafka import Consumer, Producer, KafkaError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from shared.schemas import RawTrack, Prediction, FEATURE_NAMES   # noqa: E402
from shared import model_registry as reg                          # noqa: E402

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_HOST", "localhost:9092")
TOPIC_IN = os.getenv("TOPIC_RAW_TRACKS", "raw-tracks")
TOPIC_OUT = os.getenv("TOPIC_PREDICTIONS", "predictions")
RELOAD_CHECK_EVERY = int(os.getenv("INFERENCE_RELOAD_CHECK_EVERY", "100"))


class InferenceService:
    def __init__(self):
        self.bundle = reg.load_current()
        print(f"[inference] loaded model v{self.bundle.version} "
              f"(f1={self.bundle.metadata['metrics']['f1']})")
        self.consumer = Consumer({
            "bootstrap.servers": BOOTSTRAP,
            "group.id": "inference-group",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        })
        self.producer = Producer({"bootstrap.servers": BOOTSTRAP})
        self.seen = 0

    def maybe_reload(self) -> None:
        """Swap to a newer model version if one has been published."""
        latest = reg.get_current_version()
        if latest is not None and latest > self.bundle.version:
            self.bundle = reg.load_version(latest)
            print(f"[inference] HOT-RELOADED to model v{self.bundle.version}")

    def predict(self, track: RawTrack) -> Prediction:
        # Build the vector in the SAME order the model was trained on.
        vec = np.array([[track.features[name] for name in FEATURE_NAMES]],
                       dtype=float)
        vec_scaled = self.bundle.scaler.transform(vec)
        pred = int(self.bundle.model.predict(vec_scaled)[0])
        # predict_proba[:, 1] = probability of class "popular".
        proba = float(self.bundle.model.predict_proba(vec_scaled)[0][1])
        return Prediction(
            track_id=track.track_id,
            row_index=track.row_index,
            genre_phase=track.genre_phase,
            label=track.label,          # echo ground truth for the drift monitor
            prediction=pred,
            probability=round(proba, 4),
            features=track.features,    # carry inputs forward for the KS drift test
        )

    def run(self) -> None:
        self.consumer.subscribe([TOPIC_IN])
        print(f"[inference] consuming '{TOPIC_IN}' -> '{TOPIC_OUT}' at {BOOTSTRAP}")
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    print(f"[inference] consumer error: {msg.error()}",
                          file=sys.stderr)
                    continue

                track = RawTrack.from_json(msg.value())
                prediction = self.predict(track)
                self.producer.produce(
                    topic=TOPIC_OUT,
                    key=prediction.track_id,
                    value=prediction.to_json(),
                )
                self.producer.poll(0)

                self.seen += 1
                if self.seen % RELOAD_CHECK_EVERY == 0:
                    self.maybe_reload()
                    self.producer.flush(5)
                    print(f"[inference] processed {self.seen} tracks "
                          f"(at row {prediction.row_index}, phase {prediction.genre_phase})")
        except KeyboardInterrupt:
            print("\n[inference] stopping, flushing...")
        finally:
            self.producer.flush(10)
            self.consumer.close()


def main() -> None:
    InferenceService().run()


if __name__ == "__main__":
    main()