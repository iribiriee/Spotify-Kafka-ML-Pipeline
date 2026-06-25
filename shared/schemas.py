"""
shared/schemas.py
-----------------
The single source of truth for Kafka message shapes. Every service imports
from here, so the producer, inference consumer, drift monitor, and dashboard
all agree on field names and types — no copy-paste drift.

Messages are dataclasses serialized to JSON on the wire. JSON keeps messages
human-readable in the Kafka UI, which is invaluable for debugging.

Three message types, one per topic:
    RawTrack    -> raw-tracks     (producer emits)
    Prediction  -> predictions    (inference emits)
    DriftAlert  -> drift-alerts   (drift monitor emits)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional
import json
import time


# The model's input features — these names match the dataset columns exactly.
# Kept as a module-level list so the producer, trainer, and inference consumer
# all iterate the SAME ordered set of features (order matters for the model).
FEATURE_NAMES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "duration_ms",
    "explicit",
]


@dataclass
class RawTrack:
    """A single track streamed into raw-tracks."""
    track_id: str
    row_index: int          # position in the genre-ordered stream
    genre_phase: int        # 1, 2, or 3 — ground-truth drift marker
    label: int              # 1 if popular (popularity >= 60) else 0
    features: dict          # {feature_name: value} for all FEATURE_NAMES
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "RawTrack":
        return cls(**json.loads(raw))


@dataclass
class Prediction:
    """The model's output for one track, written to predictions.

    Echoes row_index, label, and genre_phase from the input so the drift
    monitor can pair prediction-vs-truth and locate the row, without joining
    across topics.
    """
    track_id: str
    row_index: int
    genre_phase: int
    label: int              # echoed ground truth
    prediction: int         # model's predicted class (0/1)
    probability: float      # model's confidence for class 1
    timestamp: float = field(default_factory=time.time)

    @property
    def correct(self) -> bool:
        """True if the prediction matched the label. ADWIN consumes (1-correct)."""
        return self.prediction == self.label

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Prediction":
        return cls(**json.loads(raw))


@dataclass
class DriftAlert:
    """A drift event written to drift-alerts when a detector fires."""
    detector: str                   # "ks" or "adwin"
    row_index: int                  # stream position where drift was detected
    genre_phase: int                # phase active when detected
    description: str                # human-readable summary for the dashboard
    feature: Optional[str] = None   # which feature drifted (KS); None for ADWIN
    statistic: Optional[float] = None   # KS statistic, or ADWIN error estimate
    p_value: Optional[float] = None     # KS p-value; None for ADWIN
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "DriftAlert":
        return cls(**json.loads(raw))