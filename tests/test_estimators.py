import numpy as np
import pytest

from handsfree_mouse.estimators import (
    eye_aspect_ratio,
    extract_features,
    mouth_aspect_ratio,
)


def eye(cx, cy, w=30.0, h=9.0):
    """Six eye points p1..p6 with EAR == h / w."""
    return [
        (cx - w / 2, cy), (cx - w / 6, cy - h / 2), (cx + w / 6, cy - h / 2),
        (cx + w / 2, cy), (cx + w / 6, cy + h / 2), (cx - w / 6, cy + h / 2),
    ]


def mouth(cx, cy, w=40.0, h=0.0):
    """Eight inner-lip points p1..p8 with MAR == 3h / (2w)."""
    return [
        (cx - w / 2, cy), (cx - w / 6, cy - h / 2), (cx, cy - h / 2), (cx + w / 6, cy - h / 2),
        (cx + w / 2, cy), (cx + w / 6, cy + h / 2), (cx, cy + h / 2), (cx - w / 6, cy + h / 2),
    ]


def face(img_left_h=9.0, img_right_h=9.0, mouth_h=0.0):
    pts = np.zeros((68, 2))
    pts[0] = (100, 200)
    pts[16] = (260, 200)          # face width = 160
    pts[30] = (180, 220)          # nose tip
    pts[36:42] = eye(140, 180, h=img_left_h)
    pts[42:48] = eye(220, 180, h=img_right_h)
    pts[60:68] = mouth(180, 260, h=mouth_h)
    return pts


def test_ear_open_and_closed():
    assert eye_aspect_ratio(eye(0, 0, w=30, h=9)) == pytest.approx(0.30)
    assert eye_aspect_ratio(eye(0, 0, h=0)) == pytest.approx(0.0)


def test_ear_degenerate_eye_is_zero():
    assert eye_aspect_ratio(np.zeros((6, 2))) == 0.0


def test_mar_closed_and_open():
    assert mouth_aspect_ratio(mouth(0, 0, h=0)) == pytest.approx(0.0)
    assert mouth_aspect_ratio(mouth(0, 0, w=40, h=20)) == pytest.approx(0.75)


def test_extract_features_basic():
    f = extract_features(face())
    assert f.ear_left == pytest.approx(0.30)
    assert f.ear_right == pytest.approx(0.30)
    assert f.mar == pytest.approx(0.0)
    assert f.nose == (180.0, 220.0)
    assert f.face_width == pytest.approx(160.0)


def test_mirrored_frame_maps_image_left_eye_to_user_left():
    lm = face(img_left_h=0.0, img_right_h=9.0)  # image-left eye closed
    mirrored = extract_features(lm, mirror=True)
    raw = extract_features(lm, mirror=False)
    assert mirrored.ear_left == pytest.approx(0.0) and mirrored.ear_right == pytest.approx(0.30)
    assert raw.ear_right == pytest.approx(0.0) and raw.ear_left == pytest.approx(0.30)


def test_extract_features_rejects_bad_shape():
    with pytest.raises(ValueError):
        extract_features(np.zeros((5, 2)))
