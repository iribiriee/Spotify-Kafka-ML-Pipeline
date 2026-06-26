"""
consumer/main.py
----------------
Container entry point. Runs the inference consumer and the drift monitor as two
concurrent daemon threads in a single container (per the assignment's design).

Each consumer is its own long-running loop; threads let them share one process.
If either thread dies, we log it; the container stays up as long as the main
thread lives, so the other consumer keeps working.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from consumer.consumer_inference import InferenceService   # noqa: E402
from consumer.consumer_drift import DriftMonitor           # noqa: E402


def run_inference():
    try:
        InferenceService().run()
    except Exception as exc:  # noqa: BLE001
        print(f"[main] inference thread crashed: {exc}", file=sys.stderr)


def run_drift():
    try:
        DriftMonitor().run()
    except Exception as exc:  # noqa: BLE001
        print(f"[main] drift thread crashed: {exc}", file=sys.stderr)


def main() -> None:
    threads = [
        threading.Thread(target=run_inference, name="inference", daemon=True),
        threading.Thread(target=run_drift, name="drift", daemon=True),
    ]
    for t in threads:
        t.start()
        print(f"[main] started {t.name} thread")

    # Keep the main thread alive so the daemon threads keep running.
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[main] shutting down")


if __name__ == "__main__":
    main()