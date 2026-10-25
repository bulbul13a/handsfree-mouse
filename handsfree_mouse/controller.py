"""Gesture state machine: facial features in, mouse actions out.

This module is pure logic (no camera, no OS mouse calls) so it can be unit
tested with synthetic :class:`~handsfree_mouse.estimators.Features`.

Gestures (see README):
    * mouth open (held)        -> toggle mouse control ON/OFF
    * head movement            -> move cursor (or scroll, in scroll mode)
    * left / right eye wink    -> left / right click
    * squint (both eyes)       -> toggle scroll mode
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from statistics import median
from typing import List, Optional, Tuple

from .config import Config
from .estimators import Features

log = logging.getLogger(__name__)

# Sanity limits for the calibration pose.
_MIN_OPEN_EAR = 0.12
_MAX_NEUTRAL_MAR = 0.30


class ActionType(Enum):
    MOVE = auto()
    CLICK_LEFT = auto()
    CLICK_RIGHT = auto()
    SCROLL = auto()
    CONTROL_ON = auto()
    CONTROL_OFF = auto()
    SCROLL_ON = auto()
    SCROLL_OFF = auto()


@dataclass(frozen=True)
class Action:
    type: ActionType
    dx: int = 0  # MOVE: pixels right
    dy: int = 0  # MOVE: pixels down; SCROLL: notches (positive = scroll up)


@dataclass(frozen=True)
class Baseline:
    """The user's neutral face, learned during calibration."""

    ear_left: float
    ear_right: float
    mar: float
    anchor: Tuple[float, float]


class Calibrator:
    """Collects neutral-face samples and reduces them to a :class:`Baseline`."""

    def __init__(self, frames: int):
        self.frames = max(1, int(frames))
        self._samples: List[Features] = []

    def add(self, features: Features) -> None:
        self._samples.append(features)

    def reset(self) -> None:
        self._samples.clear()

    @property
    def progress(self) -> float:
        return min(1.0, len(self._samples) / self.frames)

    @property
    def done(self) -> bool:
        return len(self._samples) >= self.frames

    def result(self) -> Baseline:
        s = self._samples
        return Baseline(
            ear_left=median(f.ear_left for f in s),
            ear_right=median(f.ear_right for f in s),
            mar=median(f.mar for f in s),
            anchor=(median(f.nose[0] for f in s), median(f.nose[1] for f in s)),
        )


def _dead_zone(value: float, zone: float) -> float:
    if abs(value) <= zone:
        return 0.0
    return value - math.copysign(zone, value)


class MouseController:
    """Consumes one :class:`Features` per frame and emits :class:`Action` lists."""

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()
        self.calibrator = Calibrator(self.cfg.calibration_frames)
        self.baseline: Optional[Baseline] = None
        self.anchor: Optional[Tuple[float, float]] = None
        self.control_active = False
        self.scroll_mode = False
        self.message = ""
        self._last_t: Optional[float] = None
        self._toggle_lock_until = 0.0
        self._click_lock_until = 0.0
        self._reset_motion()
        self._reset_gestures()
        self._reanchor_pending = False

    # ------------------------------------------------------------------ state
    @property
    def calibrating(self) -> bool:
        return self.baseline is None

    @property
    def status_text(self) -> str:
        if self.calibrating:
            return f"CALIBRATING {int(self.calibrator.progress * 100)}%"
        if not self.control_active:
            return "MOUSE OFF"
        return "MOUSE ON - SCROLL" if self.scroll_mode else "MOUSE ON - CURSOR"

    def _reset_motion(self) -> None:
        self._nose_s: Optional[Tuple[float, float]] = None
        self._acc_x = 0.0
        self._acc_y = 0.0
        self._acc_scroll = 0.0

    def _reset_gestures(self) -> None:
        self._mouth_since: Optional[float] = None
        self._mouth_armed = False
        self._squint_since: Optional[float] = None
        self._squint_armed = False
        self._wink_side: Optional[str] = None
        self._wink_since = 0.0
        self._wink_armed = False

    # ------------------------------------------------------------ calibration
    def start_calibration(self) -> None:
        """(Re)start learning the neutral face. Mouse control is switched off."""
        self.baseline = None
        self.anchor = None
        self.control_active = False
        self.scroll_mode = False
        self.calibrator.reset()
        self._reset_motion()
        self._reset_gestures()
        self._reanchor_pending = False
        self.message = "Look at the camera: eyes open, mouth closed, head still"
        log.info("Calibrating - %s", self.message)

    def _finish_calibration(self) -> None:
        base = self.calibrator.result()
        if min(base.ear_left, base.ear_right) < _MIN_OPEN_EAR or base.mar > _MAX_NEUTRAL_MAR:
            self.message = "Calibration failed: keep eyes open and mouth closed"
            log.warning(self.message)
            self.calibrator.reset()
            return
        self.baseline = base
        self.anchor = base.anchor
        self.message = "Calibrated. Open your mouth to switch mouse control on."
        log.info(
            "Calibrated: EAR L=%.2f R=%.2f, MAR=%.2f",
            base.ear_left, base.ear_right, base.mar,
        )

    # ---------------------------------------------------------------- toggles
    def _set_control(self, active: bool) -> List[Action]:
        if active == self.control_active:
            return []
        self.control_active = active
        self.scroll_mode = False
        self._reset_motion()
        self._reset_gestures()
        self._reanchor_pending = active
        log.info("Mouse control %s", "ON" if active else "OFF")
        return [Action(ActionType.CONTROL_ON if active else ActionType.CONTROL_OFF)]

    def toggle_control(self) -> List[Action]:
        """Manual toggle (e.g. a caregiver pressing space)."""
        if self.calibrating:
            return []
        return self._set_control(not self.control_active)

    # ----------------------------------------------------------------- update
    def update(self, features: Optional[Features], t: float) -> List[Action]:
        """Process one frame. ``t`` is a monotonic timestamp in seconds."""
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), 0.25)
        self._last_t = t

        if features is None:  # no face this frame
            self._reset_motion()
            self._mouth_since = None
            self._squint_since = None
            self._wink_side = None
            return []

        if self.baseline is None:
            self.calibrator.add(features)
            if self.calibrator.done:
                self._finish_calibration()
            return []

        mouth_open, actions = self._handle_mouth(features, t)
        if mouth_open or not self.control_active:
            return actions

        rl = features.ear_left / self.baseline.ear_left
        rr = features.ear_right / self.baseline.ear_right
        actions += self._handle_squint(rl, rr, t)
        actions += self._handle_winks(rl, rr, t)

        # Freeze the cursor while an eye is closed so a click does not drift.
        if min(rl, rr) >= self.cfg.closed_ratio:
            actions += self._handle_head(features, dt)
        return actions

    # --------------------------------------------------------------- gestures
    def _handle_mouth(self, f: Features, t: float) -> Tuple[bool, List[Action]]:
        cfg = self.cfg
        is_open = (f.mar - self.baseline.mar) > cfg.mar_open_delta
        actions: List[Action] = []
        if is_open:
            if self._mouth_since is None:
                self._mouth_since = t
            held = t - self._mouth_since
            if self._mouth_armed and held >= cfg.mouth_hold and t >= self._toggle_lock_until:
                actions += self._set_control(not self.control_active)
                self._mouth_armed = False
                self._toggle_lock_until = t + cfg.toggle_cooldown
        else:
            self._mouth_since = None
            self._mouth_armed = True
            if self._reanchor_pending and self.control_active:
                # "Grab the mouse": the current head position becomes the centre.
                self.anchor = f.nose
                self._nose_s = None
                self._reanchor_pending = False
        return is_open, actions

    def _handle_squint(self, rl: float, rr: float, t: float) -> List[Action]:
        cfg = self.cfg
        lo, hi = cfg.closed_ratio, cfg.squint_ratio
        if rl >= hi and rr >= hi:
            self._squint_armed = True
        squinting = lo <= rl < hi and lo <= rr < hi
        if not squinting:
            self._squint_since = None
            return []
        if self._squint_since is None:
            self._squint_since = t
        if self._squint_armed and t - self._squint_since >= cfg.squint_hold:
            self._squint_armed = False
            self.scroll_mode = not self.scroll_mode
            self._reset_motion()
            log.info("Scroll mode %s", "ON" if self.scroll_mode else "OFF")
            return [Action(ActionType.SCROLL_ON if self.scroll_mode else ActionType.SCROLL_OFF)]
        return []

    def _handle_winks(self, rl: float, rr: float, t: float) -> List[Action]:
        cfg = self.cfg
        left_closed = rl < cfg.closed_ratio
        right_closed = rr < cfg.closed_ratio
        if not left_closed and not right_closed:
            self._wink_armed = True  # both eyes open again: ready for the next wink

        side: Optional[str] = None
        if left_closed and not right_closed:
            side = "left"
        elif right_closed and not left_closed:
            side = "right"  # both closed = ordinary blink -> ignored

        if side != self._wink_side:
            self._wink_side = side
            self._wink_since = t
        if side is None or not self._wink_armed:
            return []
        if t - self._wink_since >= cfg.wink_hold and t >= self._click_lock_until:
            self._wink_armed = False
            self._click_lock_until = t + cfg.click_cooldown
            return [Action(ActionType.CLICK_LEFT if side == "left" else ActionType.CLICK_RIGHT)]
        return []

    def _handle_head(self, f: Features, dt: float) -> List[Action]:
        cfg = self.cfg
        a = cfg.smoothing
        if self._nose_s is None:
            self._nose_s = f.nose
        else:
            self._nose_s = (
                a * f.nose[0] + (1 - a) * self._nose_s[0],
                a * f.nose[1] + (1 - a) * self._nose_s[1],
            )
        if dt <= 0 or f.face_width <= 0 or self.anchor is None:
            return []

        # Offsets from the neutral position, normalised by face size so the
        # behaviour does not depend on the distance to the camera.
        ox = (self._nose_s[0] - self.anchor[0]) / f.face_width
        oy = (self._nose_s[1] - self.anchor[1]) / f.face_width
        if not cfg.mirror:
            ox = -ox
        mx = _dead_zone(ox, cfg.dead_zone)
        my = _dead_zone(oy, cfg.dead_zone)

        if mx == 0.0 and my == 0.0:
            self._acc_x = self._acc_y = self._acc_scroll = 0.0
            return []

        if self.scroll_mode:
            self._acc_scroll += -my * cfg.scroll_gain * dt  # head up -> scroll up
            notches = int(self._acc_scroll)
            if notches == 0:
                return []
            self._acc_scroll -= notches
            return [Action(ActionType.SCROLL, dy=notches)]

        gain = cfg.cursor_gain * cfg.speed
        vx = math.copysign(gain * abs(mx) ** cfg.cursor_power, mx) if mx else 0.0
        vy = math.copysign(gain * abs(my) ** cfg.cursor_power, my) if my else 0.0
        self._acc_x += vx * dt
        self._acc_y += vy * dt
        ix, iy = int(self._acc_x), int(self._acc_y)  # truncate toward zero
        self._acc_x -= ix
        self._acc_y -= iy
        if ix or iy:
            return [Action(ActionType.MOVE, dx=ix, dy=iy)]
        return []
