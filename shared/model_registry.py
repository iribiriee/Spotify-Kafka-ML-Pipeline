"""
shared/model_registry.py
-------------------------
Single source of truth for model versioning and hot-reload.

A "version" is a BUNDLE of three files that must always travel together:
    model_v{N}.joblib     - the trained classifier
    scaler_v{N}.joblib     - the StandardScaler fit on the SAME training data
    metadata_v{N}.json     - feature order, metrics, training info

The active version is named in a one-line pointer file, current.txt. Hot-reload
works by polling that integer: if it's higher than the loaded version, reload.

Publishing a new version is ATOMIC with respect to readers: all three files
are written first, then current.txt is updated last. A reader keyed on
current.txt therefore never observes a half-written version.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json
import os
import tempfile

import joblib

# models/ lives at the repo root, regardless of which service imports this.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CURRENT_FILE = MODELS_DIR / "current.txt"


@dataclass
class ModelBundle:
    """An loaded model version: classifier + scaler + metadata + its version."""
    version: int
    model: Any
    scaler: Any
    metadata: dict


def _paths(version: int) -> tuple[Path, Path, Path]:
    return (
        MODELS_DIR / f"model_v{version}.joblib",
        MODELS_DIR / f"scaler_v{version}.joblib",
        MODELS_DIR / f"metadata_v{version}.json",
    )


def get_current_version() -> Optional[int]:
    """Return the active version number, or None if nothing is published yet."""
    if not CURRENT_FILE.exists():
        return None
    text = CURRENT_FILE.read_text().strip()
    return int(text) if text else None


def load_current() -> ModelBundle:
    """Load the active model bundle. Raises if nothing has been published."""
    version = get_current_version()
    if version is None:
        raise FileNotFoundError(
            f"No model published yet (missing {CURRENT_FILE}). "
            f"Run the training step to create the baseline."
        )
    return load_version(version)


def load_version(version: int) -> ModelBundle:
    model_p, scaler_p, meta_p = _paths(version)
    missing = [p.name for p in (model_p, scaler_p, meta_p) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Version {version} incomplete; missing: {missing}")
    return ModelBundle(
        version=version,
        model=joblib.load(model_p),
        scaler=joblib.load(scaler_p),
        metadata=json.loads(meta_p.read_text()),
    )


def next_version() -> int:
    """The version number a new publish should use (current + 1, or 1)."""
    cur = get_current_version()
    return 1 if cur is None else cur + 1


def publish_new_version(model: Any, scaler: Any, metadata: dict,
                        version: Optional[int] = None) -> int:
    """Write a new version bundle, then atomically make it current.

    Returns the published version number. Readers using get_current_version()
    will not see this version until current.txt is updated on the last line.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if version is None:
        version = next_version()

    model_p, scaler_p, meta_p = _paths(version)

    # 1) Write the three artifacts FIRST (before touching current.txt).
    joblib.dump(model, model_p)
    joblib.dump(scaler, scaler_p)
    meta = dict(metadata)
    meta["version"] = version
    meta_p.write_text(json.dumps(meta, indent=2))

    # 2) Flip current.txt atomically: write to a temp file in the same dir,
    #    then os.replace (atomic on the same filesystem) over current.txt.
    fd, tmp = tempfile.mkstemp(dir=MODELS_DIR, prefix=".current_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(version))
        os.replace(tmp, CURRENT_FILE)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return version