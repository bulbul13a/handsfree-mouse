from dataclasses import replace

import pytest

from handsfree_mouse.config import Config
from handsfree_mouse.controller import ActionType as A
from handsfree_mouse.controller import MouseController
from handsfree_mouse.estimators import Features

DT = 0.04  # 25 fps
BASE = Features(ear_left=0.30, ear_right=0.30, mar=0.05, nose=(320.0, 240.0), face_width=160.0)


def feat(**kw):
    return replace(BASE, **kw)


def types(actions):
    return [a.type for a in actions]


def feed(ctrl, f, t, duration):
    """Feed the same features for `duration` seconds. Returns (actions, next_t)."""
    out = []
    end = t + duration
    while t < end - 1e-9:
        out += ctrl.update(f, t)
        t += DT
    return out, t


@pytest.fixture
def cfg():
    return Config(calibration_frames=5)


@pytest.fixture
def ctrl(cfg):
    c = MouseController(cfg)
    c.start_calibration()
    for i in range(5):
        c.update(BASE, i * DT)
    assert not c.calibrating
    c.update(BASE, 5 * DT)  # a closed-mouth frame arms the mouth gesture
    return c


def activate(ctrl, t=1.0):
    acts, t = feed(ctrl, feat(mar=0.5), t, 0.6)      # open mouth
    assert A.CONTROL_ON in types(acts)
    _, t = feed(ctrl, BASE, t, 0.2)                  # close mouth -> re-anchor
    return t


# ------------------------------------------------------------- calibration
def test_calibration_learns_baseline(ctrl):
    assert ctrl.baseline.ear_left == pytest.approx(0.30)
    assert ctrl.anchor == (320.0, 240.0)
    assert not ctrl.control_active


def test_calibration_fails_with_closed_eyes(cfg):
    c = MouseController(cfg)
    c.start_calibration()
    for i in range(5):
        c.update(feat(ear_left=0.02, ear_right=0.02), i * DT)
    assert c.calibrating and "failed" in c.message


def test_no_actions_while_calibrating(cfg):
    c = MouseController(cfg)
    c.start_calibration()
    assert c.update(feat(mar=0.9), 0.0) == []
    assert c.update(None, DT) == []


# -------------------------------------------------------- activation (mouth)
def test_mouth_open_toggles_control_on_and_off(ctrl):
    t = activate(ctrl)
    assert ctrl.control_active
    _, t = feed(ctrl, BASE, t, 1.2)                  # wait out the cooldown
    acts, t = feed(ctrl, feat(mar=0.5), t, 0.6)
    assert A.CONTROL_OFF in types(acts)
    assert not ctrl.control_active


def test_short_mouth_open_does_not_toggle(ctrl):
    acts, _ = feed(ctrl, feat(mar=0.5), 1.0, 0.2)
    assert acts == [] and not ctrl.control_active


def test_holding_mouth_open_toggles_only_once(ctrl):
    acts, _ = feed(ctrl, feat(mar=0.5), 1.0, 3.0)
    assert types(acts).count(A.CONTROL_ON) == 1


# ------------------------------------------------------------------ clicks
def test_left_wink_clicks_once(ctrl):
    t = activate(ctrl)
    acts, _ = feed(ctrl, feat(ear_left=0.10), t, 2.0)
    assert types(acts).count(A.CLICK_LEFT) == 1
    assert A.CLICK_RIGHT not in types(acts)


def test_right_wink_right_clicks(ctrl):
    t = activate(ctrl)
    acts, _ = feed(ctrl, feat(ear_right=0.10), t, 1.0)
    assert types(acts).count(A.CLICK_RIGHT) == 1


def test_second_wink_after_reopening_clicks_again(ctrl):
    t = activate(ctrl)
    a1, t = feed(ctrl, feat(ear_left=0.10), t, 0.5)
    _, t = feed(ctrl, BASE, t, 0.8)
    a2, _ = feed(ctrl, feat(ear_left=0.10), t, 0.5)
    assert types(a1 + a2).count(A.CLICK_LEFT) == 2


def test_normal_blink_is_ignored(ctrl):
    t = activate(ctrl)
    acts, _ = feed(ctrl, feat(ear_left=0.05, ear_right=0.05), t, 1.0)
    assert not {A.CLICK_LEFT, A.CLICK_RIGHT} & set(types(acts))


def test_very_short_wink_is_ignored(ctrl):
    t = activate(ctrl)
    acts, _ = feed(ctrl, feat(ear_left=0.10), t, 0.08)
    assert acts == []


def test_no_clicks_while_control_off(ctrl):
    acts, _ = feed(ctrl, feat(ear_left=0.10), 1.0, 1.0)
    assert acts == []


# ------------------------------------------------------------ cursor moves
def test_head_right_moves_cursor_right(ctrl):
    t = activate(ctrl)
    acts, _ = feed(ctrl, feat(nose=(340.0, 240.0)), t, 0.5)
    moves = [a for a in acts if a.type is A.MOVE]
    assert moves and sum(a.dx for a in moves) > 0 and sum(a.dy for a in moves) == 0


def test_head_down_moves_cursor_down_and_up_moves_up(ctrl):
    t = activate(ctrl)
    down, t = feed(ctrl, feat(nose=(320.0, 260.0)), t, 0.5)
    _, t = feed(ctrl, BASE, t, 0.3)
    up, _ = feed(ctrl, feat(nose=(320.0, 220.0)), t, 0.5)
    assert sum(a.dy for a in down if a.type is A.MOVE) > 0
    assert sum(a.dy for a in up if a.type is A.MOVE) < 0


def test_dead_zone_prevents_jitter(ctrl):
    t = activate(ctrl)
    acts, _ = feed(ctrl, feat(nose=(323.0, 242.0)), t, 1.0)  # 3 px of 160 = 1.9 %
    assert acts == []


def test_farther_offset_moves_faster(ctrl):
    t = activate(ctrl)
    near, t = feed(ctrl, feat(nose=(332.0, 240.0)), t, 1.0)
    _, t = feed(ctrl, BASE, t, 0.3)
    far, _ = feed(ctrl, feat(nose=(350.0, 240.0)), t, 1.0)
    assert sum(a.dx for a in far) > sum(a.dx for a in near) > 0


def test_no_movement_while_control_off(ctrl):
    acts, _ = feed(ctrl, feat(nose=(400.0, 240.0)), 1.0, 1.0)
    assert acts == []


def test_cursor_frozen_while_winking(ctrl):
    t = activate(ctrl)
    acts, _ = feed(ctrl, feat(ear_left=0.10, nose=(360.0, 240.0)), t, 0.5)
    assert A.MOVE not in types(acts)


def test_reanchor_uses_position_at_activation(ctrl):
    # user has drifted right before switching the mouse on
    acts, t = feed(ctrl, feat(mar=0.5, nose=(360.0, 240.0)), 1.0, 0.6)
    _, t = feed(ctrl, feat(nose=(360.0, 240.0)), t, 0.2)
    assert ctrl.anchor == (360.0, 240.0)
    acts, _ = feed(ctrl, feat(nose=(360.0, 240.0)), t, 0.5)
    assert A.MOVE not in types(acts)


def test_without_mirror_x_is_inverted(cfg):
    c = MouseController(replace(cfg, mirror=False))
    c.start_calibration()
    for i in range(6):
        c.update(BASE, i * DT)
    t = activate(c)
    acts, _ = feed(c, feat(nose=(340.0, 240.0)), t, 0.5)
    assert sum(a.dx for a in acts if a.type is A.MOVE) < 0


# ---------------------------------------------------------- squint / scroll
SQUINT = dict(ear_left=0.21, ear_right=0.21)  # ratio 0.70


def test_squint_toggles_scroll_mode(ctrl):
    t = activate(ctrl)
    acts, t = feed(ctrl, feat(**SQUINT), t, 1.2)
    assert types(acts).count(A.SCROLL_ON) == 1 and ctrl.scroll_mode
    _, t = feed(ctrl, BASE, t, 0.3)                   # eyes open again re-arms
    acts, _ = feed(ctrl, feat(**SQUINT), t, 1.2)
    assert A.SCROLL_OFF in types(acts) and not ctrl.scroll_mode


def test_head_scrolls_instead_of_moving_in_scroll_mode(ctrl):
    t = activate(ctrl)
    _, t = feed(ctrl, feat(**SQUINT), t, 1.2)
    _, t = feed(ctrl, BASE, t, 0.3)
    up, t = feed(ctrl, feat(nose=(320.0, 215.0)), t, 1.0)
    assert A.MOVE not in types(up)
    assert sum(a.dy for a in up if a.type is A.SCROLL) > 0     # head up -> scroll up
    _, t = feed(ctrl, BASE, t, 0.3)
    down, _ = feed(ctrl, feat(nose=(320.0, 265.0)), t, 1.0)
    assert sum(a.dy for a in down if a.type is A.SCROLL) < 0   # head down -> scroll down


def test_full_blink_is_not_a_squint(ctrl):
    t = activate(ctrl)
    acts, _ = feed(ctrl, feat(ear_left=0.05, ear_right=0.05), t, 2.0)
    assert A.SCROLL_ON not in types(acts)


# ----------------------------------------------------------- robustness
def test_face_lost_returns_no_actions(ctrl):
    t = activate(ctrl)
    assert ctrl.update(None, t) == []
    assert ctrl.control_active


def test_manual_toggle(ctrl):
    assert types(ctrl.toggle_control()) == [A.CONTROL_ON]
    assert types(ctrl.toggle_control()) == [A.CONTROL_OFF]


def test_recalibration_switches_control_off(ctrl):
    activate(ctrl)
    ctrl.start_calibration()
    assert ctrl.calibrating and not ctrl.control_active
