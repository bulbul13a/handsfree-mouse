"""All tunable parameters in one place."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- camera / detection -------------------------------------------------
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    #: Flip the frame horizontally (selfie view). Head movement then matches
    #: what the user sees on screen: turn right -> cursor moves right.
    mirror: bool = True
    #: Apply CLAHE contrast enhancement before detection (helps in dim rooms).
    enhance_lighting: bool = True
    model_path: str = "models/shape_predictor_68_face_landmarks.dat"

    # --- calibration ----------------------------------------------------------
    #: Number of face frames used to learn the user's neutral face.
    calibration_frames: int = 45

    # --- eyes (ratios are relative to the calibrated open-eye EAR) -----------
    #: EAR below this fraction of the open-eye baseline => eye is closed.
    closed_ratio: float = 0.60
    #: Both eyes between closed_ratio and this fraction => squint.
    squint_ratio: float = 0.82
    #: Seconds one eye must stay closed (other open) to count as a wink/click.
    wink_hold: float = 0.20
    #: Minimum seconds between two clicks.
    click_cooldown: float = 0.60
    #: Seconds a squint must be held to toggle scroll mode.
    squint_hold: float = 0.80

    # --- mouth ------------------------------------------------------------------
    #: MAR increase over the neutral (closed-mouth) baseline that counts as "open".
    mar_open_delta: float = 0.30
    #: Seconds the mouth must stay open to toggle mouse control.
    mouth_hold: float = 0.40
    #: Minimum seconds between two control toggles.
    toggle_cooldown: float = 1.00

    # --- head -> cursor -----------------------------------------------------------
    #: Nose offset (as a fraction of face width) ignored around the neutral position.
    dead_zone: float = 0.04
    #: Cursor speed (pixels/second) = cursor_gain * speed * offset ** cursor_power
    cursor_gain: float = 5000.0
    cursor_power: float = 1.4
    speed: float = 1.0
    #: Exponential smoothing of the nose position (1.0 = no smoothing).
    smoothing: float = 0.5
    #: Scroll notches per second per unit of (dead-zone-corrected) offset.
    scroll_gain: float = 80.0
