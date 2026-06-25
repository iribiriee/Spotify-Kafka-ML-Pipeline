"""
producer/producer.py
--------------------
Streams genre_ordered_tracks.csv into the raw-tracks topic, one track at a
time, preserving order. This is the entry point of the live pipeline.

Run on the host (during development):
    python producer/producer.py
It reads config from .env (KAFKA_BOOTSTRAP_HOST, topic name, delay, loop mode).

Key streaming details handled here:
  - confluent-kafka's produce() is ASYNC: it only queues a message. We call
    poll() each iteration to serve delivery callbacks, and flush() at the end
    to guarantee everything is actually sent before exit.
  - A delivery callback surfaces send failures (otherwise they're silent).
  - Messages are keyed by track_id so ordering survives if partitions grow.
  - Ctrl-C flushes pending messages before exiting (no lost tail).
"""

from __future__ import annotations

import os
import sys
import time
import signal
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

# Make the repo root importable so `shared` resolves when run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from shared.schemas import RawTrack, FEATURE_NAMES  # noqa: E402


# --- Config from environment (with host-friendly defaults) -------------------
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_HOST", "localhost:9092")
TOPIC = os.getenv("TOPIC_RAW_TRACKS", "raw-tracks")
DELAY = float(os.getenv("PRODUCER_DELAY_SECONDS", "0.1"))
LOOP_FOREVER = os.getenv("PRODUCER_LOOP_FOREVER", "false").lower() == "true"
CSV_PATH = REPO_ROOT / "data" / "genre_ordered_tracks.csv"


def delivery_report(err, msg) -> None:
    """Called once per message when its send succeeds or permanently fails."""
    if err is not None:
        # Surface failures loudly — don't let them pass silently.
        print(f"[producer] DELIVERY FAILED for key={msg.key()}: {err}",
              file=sys.stderr)


def build_raw_track(row: pd.Series) -> RawTrack:
    """Turn one CSV row into a RawTrack message."""
    features = {}
    for name in FEATURE_NAMES:
        val = row[name]
        # explicit is a bool in the CSV; cast to int so JSON is clean and the
        # model sees a numeric feature.
        if name == "explicit":
            features[name] = int(bool(val))
        else:
            features[name] = float(val)
    return RawTrack(
        track_id=str(row["track_id"]),
        row_index=int(row["row_index"]),
        genre_phase=int(row["genre_phase"]),
        label=int(row["label"]),
        features=features,
    )


def stream_once(producer: Producer, df: pd.DataFrame) -> int:
    """Send every row of df in order. Returns count sent."""
    sent = 0
    for _, row in df.iterrows():
        track = build_raw_track(row)
        producer.produce(
            topic=TOPIC,
            key=track.track_id,
            value=track.to_json(),
            callback=delivery_report,
        )
        # Serve delivery callbacks for already-sent messages (non-blocking).
        producer.poll(0)
        sent += 1

        # Light progress signal at each phase boundary and every 500 rows.
        if track.row_index % 500 == 0:
            print(f"[producer] sent row {track.row_index} "
                  f"(phase {track.genre_phase})")
        time.sleep(DELAY)
    return sent


def main() -> None:
    if not CSV_PATH.exists():
        print(f"[producer] ERROR: {CSV_PATH} not found. "
              f"Run `python data/genre_ordering.py` first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"[producer] loaded {len(df)} tracks from {CSV_PATH.name}")
    print(f"[producer] -> topic '{TOPIC}' at {BOOTSTRAP} "
          f"| delay={DELAY}s | loop_forever={LOOP_FOREVER}")

    producer = Producer({
        "bootstrap.servers": BOOTSTRAP,
        # Keep ordering strict: don't reorder on retry.
        "enable.idempotence": True,
    })

    # Flush on Ctrl-C so the last queued messages aren't lost.
    def handle_sigint(signum, frame):
        print("\n[producer] interrupted — flushing pending messages...")
        producer.flush(10)
        print("[producer] done.")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    total = 0
    pass_num = 0
    while True:
        pass_num += 1
        if LOOP_FOREVER:
            print(f"[producer] --- pass {pass_num} ---")
        total += stream_once(producer, df)
        if not LOOP_FOREVER:
            break

    # Block until every queued message has been delivered.
    remaining = producer.flush(30)
    if remaining:
        print(f"[producer] WARNING: {remaining} messages still queued at exit",
              file=sys.stderr)
    print(f"[producer] finished — {total} tracks sent.")


if __name__ == "__main__":
    main()