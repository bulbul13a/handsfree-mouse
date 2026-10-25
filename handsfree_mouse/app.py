"""Command-line application: webcam -> landmarks -> gestures -> mouse."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from typing import Optional, Sequence

import cv2

from .config import Config
from .controller import ActionType, MouseController
from .estimators import extract_features
from .landmarks import LandmarkDetector, resolve_model_path
from .mouse import MouseDriver

log = logging.getLogger("handsfree_mouse")

WINDOW = "Handsfree Mouse"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, AMBER, WHITE = (80, 200, 80), (60, 60, 230), (0, 180, 255), (255, 255, 255)


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(
        prog="handsfree-mouse",
        description="Control the mouse cursor with facial movements through a webcam.",
    )
    p.add_argument("--camera", type=int, default=d.camera_index, help="camera index (default: %(default)s)")
    p.add_argument("--width", type=int, default=d.frame_width)
    p.add_argument("--height", type=int, default=d.frame_height)
    p.add_argument("--model", default=d.model_path, help="path to shape_predictor_68_face_landmarks.dat")
    p.add_argument("--no-mirror", action="store_true", help="do not flip the camera image")
    p.add_argument("--no-enhance", action="store_true", help="disable low-light contrast enhancement")
    p.add_argument("--speed", type=float, default=d.speed, help="cursor speed multiplier (default: %(default)s)")
    p.add_argument("--dead-zone", type=float, default=d.dead_zone, help="head dead zone as fraction of face width")
    p.add_argument("--closed-ratio", type=float, default=d.closed_ratio, help="eye 'closed' threshold vs. baseline EAR")
    p.add_argument("--squint-ratio", type=float, default=d.squint_ratio, help="squint upper threshold vs. baseline EAR")
    p.add_argument("--mar-delta", type=float, default=d.mar_open_delta, help="MAR increase that counts as mouth open")
    p.add_argument("--dry-run", action="store_true", help="never move/click the real mouse; just show detections")
    p.add_argument("--no-preview", action="store_true", help="run without the preview window (stop with Ctrl+C)")
    p.add_argument("--show-landmarks", action="store_true", help="draw the 68 landmarks on the preview")
    p.add_argument("-v", "--verbose", action="store_true", help="log every click")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return replace(
        Config(),
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        model_path=args.model,
        mirror=not args.no_mirror,
        enhance_lighting=not args.no_enhance,
        speed=args.speed,
        dead_zone=args.dead_zone,
        closed_ratio=args.closed_ratio,
        squint_ratio=args.squint_ratio,
        mar_open_delta=args.mar_delta,
    )


def draw_overlay(frame, controller, features, landmarks, show_landmarks, dry_run) -> None:
    h, w = frame.shape[:2]
    if show_landmarks and landmarks is not None:
        for x, y in landmarks.astype(int):
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    if controller.calibrating:
        color = AMBER
    elif controller.control_active:
        color = GREEN
    else:
        color = RED
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    label = controller.status_text + ("   [DRY RUN]" if dry_run else "")
    cv2.putText(frame, label, (10, 24), FONT, 0.7, color, 2, cv2.LINE_AA)

    if features is None:
        info = "No face detected"
    else:
        info = f"EAR L {features.ear_left:.2f}  R {features.ear_right:.2f}   MAR {features.mar:.2f}"
    cv2.putText(frame, info, (10, h - 32), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "q quit | c recalibrate | space toggle mouse", (10, h - 12), FONT, 0.45, WHITE, 1, cv2.LINE_AA)

    if controller.calibrating:
        cv2.putText(frame, controller.message, (10, 56), FONT, 0.5, AMBER, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (10, 66), (w - 10, 76), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, 66), (10 + int((w - 20) * controller.calibrator.progress), 76), AMBER, -1)


def run(cfg: Config, dry_run: bool, preview: bool, show_landmarks: bool) -> int:
    model = resolve_model_path(cfg.model_path)
    detector = LandmarkDetector(model, cfg.enhance_lighting)
    controller = MouseController(cfg)
    driver = MouseDriver(dry_run=dry_run)

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        log.error("Cannot open camera %d", cfg.camera_index)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low

    controller.start_calibration()
    log.info("Running. Open your mouth (hold ~0.4 s) to switch mouse control on/off.")
    if dry_run:
        log.info("DRY RUN: the real mouse will not be touched.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("Camera returned no frame")
                return 1
            if cfg.mirror:
                frame = cv2.flip(frame, 1)

            landmarks = detector.detect(frame)
            features = extract_features(landmarks, cfg.mirror) if landmarks is not None else None
            actions = controller.update(features, time.monotonic())
            driver.execute(actions)
            for action in actions:
                if action.type in (ActionType.CLICK_LEFT, ActionType.CLICK_RIGHT):
                    log.debug("%s", action.type.name)

            if preview:
                draw_overlay(frame, controller, features, landmarks, show_landmarks, dry_run)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    controller.start_calibration()
                elif key == ord(" "):
                    driver.execute(controller.toggle_control())
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    try:
        return run(config_from_args(args), args.dry_run, not args.no_preview, args.show_landmarks)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"Missing dependency: {exc}\nInstall with: pip install -r requirements.txt", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
