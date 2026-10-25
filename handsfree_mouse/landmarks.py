"""Face detection + 68-point landmark prediction (dlib)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

MODEL_NAME = "shape_predictor_68_face_landmarks.dat"


def resolve_model_path(path: str) -> Path:
    """Find the dlib model file or raise a helpful ``FileNotFoundError``."""
    candidates = []
    env = os.environ.get("HANDSFREE_MODEL")
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [
        Path(path).expanduser(),
        Path(__file__).resolve().parent.parent / "models" / MODEL_NAME,
        Path.home() / ".handsfree_mouse" / MODEL_NAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {MODEL_NAME}.\n"
        "Download it with:  python scripts/download_model.py\n"
        "or point to it with --model PATH / the HANDSFREE_MODEL environment variable."
    )


class LandmarkDetector:
    """Wraps dlib's HOG face detector and the ensemble-of-regression-trees shape predictor."""

    def __init__(self, model_path, enhance_lighting: bool = True):
        import cv2
        import dlib  # imported lazily so the rest of the package works without it

        self._cv2 = cv2
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(model_path))
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if enhance_lighting else None

    def detect(self, frame_bgr) -> Optional[np.ndarray]:
        """Return a (68, 2) float array for the largest face, or ``None``."""
        gray = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2GRAY)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        faces = self._detector(gray, 0)
        if not faces:
            return None
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self._predictor(gray, face)
        return np.array([(p.x, p.y) for p in shape.parts()], dtype=float)
