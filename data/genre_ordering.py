"""
genre_ordering.py
-----------------
Run ONCE before starting the stack. Produces data/genre_ordered_tracks.csv:
the input the producer streams and the training notebook learns from.

What it does, and why:
  1. Ensures the raw dataset exists (auto-download via download_dataset.py).
  2. Keeps only the 12 genres that make up our 3 drift phases. This shrinks
     114k -> ~12k rows of *clean, meaningful* drift instead of 114 muddled genres.
  3. Tags each row with genre_phase (1, 2, or 3).
  4. Shuffles WITHIN each phase, then concatenates phases in order. This is the
     key trick: order ACROSS phases is fixed (so drift happens at known row
     boundaries), but order WITHIN a phase is random (so there's no alphabetical
     or popularity artifact for the detector to accidentally latch onto).
  5. Adds the binary label: 1 if popularity >= 60 else 0.
  6. Writes the result with a clean RangeIndex as row_index.
"""

from pathlib import Path
import pandas as pd

from download_dataset import ensure_dataset

# --- The three drift phases. Each is a list of genres present in the dataset. ---
# Phase boundaries are where we EXPECT the detectors to fire. Genres are chosen
# to be acoustically COHERENT within a phase but DISTINCT across phases, so drift
# is clean and dramatic. ~8 genres per phase (~1000 tracks each) => balanced
# ~8000-track phases. Genre names not present are skipped automatically.
PHASE1_GENRES = ["acoustic", "classical", "folk", "piano"]   # calm, acoustic
PHASE2_GENRES = ["edm", "techno", "house", "trance"]   # high energy, electronic
PHASE3_GENRES = ["hip-hop", "r-n-b", "reggaeton", "dancehall"]  # vocal, rhythmic

POPULARITY_THRESHOLD = 60     # popularity >= 60 -> "popular" (label 1)
SHUFFLE_SEED = 42             # reproducible within-phase shuffles

OUT_PATH = Path(__file__).parent / "genre_ordered_tracks.csv"


def build_ordered_dataset() -> pd.DataFrame:
    raw_path = ensure_dataset()
    # The HF CSV has an unnamed leading index column; index_col=0 drops it cleanly.
    df = pd.read_csv(raw_path, index_col=0)

    phase_map = {}
    for g in PHASE1_GENRES:
        phase_map[g] = 1
    for g in PHASE2_GENRES:
        phase_map[g] = 2
    for g in PHASE3_GENRES:
        phase_map[g] = 3

    # Keep only the genres we care about, and attach their phase number.
    available = set(df["track_genre"].unique())
    requested = set(phase_map)
    missing = sorted(requested - available)
    if missing:
        print(f"[genre_ordering] NOTE: these requested genres are not in the "
              f"dataset and were skipped: {missing}")
    used = sorted(requested & available)
    print(f"[genre_ordering] using {len(used)} genres: {used}")

    df = df[df["track_genre"].isin(phase_map)].copy()
    df["genre_phase"] = df["track_genre"].map(phase_map)

    # Shuffle within each phase, then stack phases in order 1 -> 2 -> 3.
    ordered_parts = []
    for phase in (1, 2, 3):
        part = df[df["genre_phase"] == phase]
        part = part.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)
        ordered_parts.append(part)
    ordered = pd.concat(ordered_parts, ignore_index=True)

    # Binary target.
    ordered["label"] = (ordered["popularity"] >= POPULARITY_THRESHOLD).astype(int)

    # row_index = position in the genre-ordered stream (matches the Kafka schema).
    ordered = ordered.reset_index(drop=True)
    ordered["row_index"] = ordered.index

    return ordered


def main() -> None:
    ordered = build_ordered_dataset()
    ordered.to_csv(OUT_PATH, index=False)

    # --- Report so you can eyeball that it worked (and find phase boundaries) ---
    print(f"\nWrote {OUT_PATH}  ({len(ordered)} rows)")
    print("\nRows per phase:")
    print(ordered["genre_phase"].value_counts().sort_index().to_string())

    print("\nPhase boundaries (row_index where phase changes):")
    changed = ordered["genre_phase"].diff().fillna(0) != 0
    boundaries = [b for b in ordered.index[changed] if b != 0]  # exclude row 0
    for b in boundaries:
        print(f"  row {b}: phase -> {ordered.loc[b, 'genre_phase']}")

    print("\nLabel balance overall:")
    print(ordered["label"].value_counts(normalize=True).round(3).to_string())
    print("\nLabel balance per phase (fraction popular):")
    print(ordered.groupby("genre_phase")["label"].mean().round(3).to_string())


if __name__ == "__main__":
    main()