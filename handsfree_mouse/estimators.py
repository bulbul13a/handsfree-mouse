"""Facial movement estimators computed from dlib's 68 landmarks.

Indices below are the 0-based dlib indices. The report lists them 1-based
(eyes 37-48, mouth 49-68); subtract one to compare.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# 0-based dlib landmark groups
JAW = slice(0, 17)
NOSE_TIP = 30
IMAGE_LEFT_EYE = slice(36, 42)   # eye that appears on the left side of the image
IMAGE_RIGHT_EYE = slice(42, 48)  # eye that appears on the right side of the image
INNER_MOUTH = slice(60, 68)      # 8 inner-lip points


@dataclass(frozen=True)
class Features:
    """Per-frame measurements consumed by the gesture controller."""

    ear_left: float   # the *user's* left eye
    ear_right: float  # the *user's* right eye
    mar: float
    nose: Tuple[float, float]
    face_width: float


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def eye_aspect_ratio(eye) -> float:
    """EAR (Soukupova & Cech, 2016) for six eye points p1..p6.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    ~0.25-0.35 when open, close to 0 when closed.
    """
    eye = np.asarray(eye, dtype=float)
    horizontal = _dist(eye[0], eye[3])
    if horizontal == 0:
        return 0.0
    vertical = _dist(eye[1], eye[5]) + _dist(eye[2], eye[4])
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(mouth) -> float:
    """MAR for the eight inner-lip points p1..p8 (as in the project report).

    MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
    ~0 when the lips are closed and grows as the mouth opens.
    """
    mouth = np.asarray(mouth, dtype=float)
    horizontal = _dist(mouth[0], mouth[4])
    if horizontal == 0:
        return 0.0
    vertical = _dist(mouth[1], mouth[7]) + _dist(mouth[2], mouth[6]) + _dist(mouth[3], mouth[5])
    return vertical / (2.0 * horizontal)


def extract_features(landmarks, mirror: bool = True) -> Features:
    """Turn a (68, 2) landmark array into :class:`Features`.

    ``mirror`` must match whether the frame was flipped horizontally *before*
    landmark detection. In a mirrored (selfie) frame, the eye on the left of the
    image is the user's left eye.
    """
    pts = np.asarray(landmarks, dtype=float)
    if pts.shape != (68, 2):
        raise ValueError(f"expected landmarks of shape (68, 2), got {pts.shape}")

    ear_img_left = eye_aspect_ratio(pts[IMAGE_LEFT_EYE])
    ear_img_right = eye_aspect_ratio(pts[IMAGE_RIGHT_EYE])
    if mirror:
        ear_left, ear_right = ear_img_left, ear_img_right
    else:
        ear_left, ear_right = ear_img_right, ear_img_left

    return Features(
        ear_left=ear_left,
        ear_right=ear_right,
        mar=mouth_aspect_ratio(pts[INNER_MOUTH]),
        nose=(float(pts[NOSE_TIP][0]), float(pts[NOSE_TIP][1])),
        face_width=_dist(pts[0], pts[16]),
    )
