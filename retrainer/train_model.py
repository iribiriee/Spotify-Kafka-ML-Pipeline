"""
retrainer/train_model.py
------------------------
The training logic, as one reusable function: train_and_publish(). It is called
two ways:
  - manually / from the notebook, to build the baseline (phase-1-only) model;
  - by the retrainer service later, to build a new version from buffered data.

Pipeline (and the reasoning, inline):
  1. Load training data (a DataFrame with FEATURE_NAMES + 'label').
  2. Train/test split BEFORE any resampling, so the test set stays real.
  3. Fit a StandardScaler on the training features only; save it with the model.
  4. SMOTE the *training* split only, to fix class imbalance.
  5. Train a RandomForest.
  6. Evaluate on the untouched test split (precision/recall/F1, not just accuracy).
  7. Publish the bundle (model + scaler + metadata) through the registry,
     which versions it atomically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from shared.schemas import FEATURE_NAMES          # noqa: E402
from shared import model_registry as reg          # noqa: E402

RANDOM_STATE = 42


def train_and_publish(df: pd.DataFrame,
                      source_label: str = "baseline_phase1",
                      n_estimators: int = 100,
                      use_smote: bool = True) -> dict:
    """Train on df (must contain FEATURE_NAMES + 'label'), evaluate, publish.

    Returns the metadata dict (including metrics and the published version).
    """
    X = df[FEATURE_NAMES].astype(float).values
    y = df["label"].astype(int).values

    # --- Split BEFORE resampling: the test set must reflect reality, never
    #     contain synthetic SMOTE points. Stratify to keep class ratio in both.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # --- Scale: fit on train only, then reuse everywhere (saved in the bundle).
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- SMOTE the training split only, to balance popular vs not-popular.
    #     Guard: SMOTE needs at least a few minority samples; skip if too few.
    n_minority = int(min(np.bincount(y_train)))
    if use_smote and n_minority >= 6:
        X_train_s, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(
            X_train_s, y_train
        )
        smote_applied = True
    else:
        smote_applied = False

    # --- Train the forest. class_weight balanced as a backstop if SMOTE skipped.
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        class_weight=None if smote_applied else "balanced",
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    # --- Evaluate on the untouched, real test split.
    y_pred = model.predict(X_test_s)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # --- Feature importances (handy for the writeup: what drives popularity).
    importances = dict(sorted(
        zip(FEATURE_NAMES, (round(float(i), 4) for i in model.feature_importances_)),
        key=lambda kv: kv[1], reverse=True,
    ))

    metadata = {
        "source": source_label,
        "features": FEATURE_NAMES,
        "n_train_rows": int(len(df)),
        "smote_applied": smote_applied,
        "class_balance_raw": {int(k): int(v) for k, v in
                              zip(*np.unique(y, return_counts=True))},
        "metrics": metrics,
        "feature_importances": importances,
    }

    version = reg.publish_new_version(model, scaler, metadata)
    metadata["version"] = version

    print(f"[train] published model v{version} from '{source_label}'")
    print(f"[train] metrics: {metrics}")
    return metadata


def load_phase1_baseline_df() -> pd.DataFrame:
    """Convenience loader: the phase-1-only slice used for the baseline model."""
    csv = REPO_ROOT / "data" / "genre_ordered_tracks.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Run `python data/genre_ordering.py` first."
        )
    df = pd.read_csv(csv)
    return df[df["genre_phase"] == 1].copy()


if __name__ == "__main__":
    # Running this file directly builds the baseline (phase-1-only) model.
    df = load_phase1_baseline_df()
    print(f"[train] baseline training on {len(df)} phase-1 rows")
    train_and_publish(df, source_label="baseline_phase1")