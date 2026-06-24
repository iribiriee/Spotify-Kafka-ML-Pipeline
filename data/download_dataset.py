"""
download_dataset.py
-------------------
Fetches the Spotify Tracks dataset (maharshipandya/spotify-tracks-dataset) from
Hugging Face if it is not already present locally. No authentication required.

Design notes:
- Idempotent: if a valid CSV already exists, it does nothing (fast, offline-safe).
- The HF file is stored with Xet/LFS, so a failed/partial fetch can silently
  return a tiny "pointer" file instead of the real 20 MB CSV. We guard against
  that by checking the downloaded file size, and re-raise a clear error otherwise.
"""

from pathlib import Path
import sys
import urllib.request

# Direct, no-auth download URL (the file is named dataset.csv in the HF repo root).
HF_URL = (
    "https://huggingface.co/datasets/maharshipandya/"
    "spotify-tracks-dataset/resolve/main/dataset.csv"
)

# Where we keep the raw dataset locally.
RAW_CSV = Path(__file__).parent / "dataset.csv"

# The real file is ~20 MB. Anything under 1 MB is almost certainly an LFS
# pointer or an error page, not the dataset.
MIN_VALID_BYTES = 1_000_000


def ensure_dataset(path: Path = RAW_CSV, url: str = HF_URL) -> Path:
    """Return the path to a valid local copy of the dataset, downloading if needed."""
    if path.exists() and path.stat().st_size >= MIN_VALID_BYTES:
        print(f"[download] Using cached dataset: {path} "
              f"({path.stat().st_size / 1_000_000:.1f} MB)")
        return path

    print(f"[download] Fetching dataset from Hugging Face ...\n           {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)

    size = path.stat().st_size
    if size < MIN_VALID_BYTES:
        # Clean up the bad file so a re-run will retry instead of trusting it.
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file was only {size} bytes — expected >= "
            f"{MIN_VALID_BYTES}. The download likely returned an LFS pointer "
            f"or an error page. Check your network and try again."
        )

    print(f"[download] Saved {path} ({size / 1_000_000:.1f} MB)")
    return path


if __name__ == "__main__":
    try:
        ensure_dataset()
    except Exception as exc:
        print(f"[download] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)