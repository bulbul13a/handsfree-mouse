#!/usr/bin/env python3
"""Download dlib's pre-trained 68-point face landmark model into ./models/.

The model (~95 MB extracted) is intentionally NOT committed to the repository.
Note: it was trained on the iBUG 300-W dataset, whose licence excludes
commercial use - check the terms before using it commercially.
"""
import argparse
import bz2
import sys
import urllib.request
from pathlib import Path

URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
NAME = "shape_predictor_68_face_landmarks.dat"


def main() -> int:
    default_dest = Path(__file__).resolve().parents[1] / "models"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=default_dest, help="output directory (default: ./models)")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    target = args.dest / NAME
    if target.exists() and not args.force:
        print(f"Already present: {target}")
        return 0

    archive = args.dest / (NAME + ".bz2")
    print(f"Downloading {URL}")

    def progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(URL, archive, progress)
    print("\nExtracting...")
    with bz2.open(archive, "rb") as src, open(target, "wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    archive.unlink()
    print(f"Saved model to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
