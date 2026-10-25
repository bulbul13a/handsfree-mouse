#!/usr/bin/env python3
"""Technical Module I demos from the report (Appendix A), in one script.

    python examples/module1_demo.py show      face.jpg   # load & present an image
    python examples/module1_demo.py faces     face.jpg   # face detection (green box)
    python examples/module1_demo.py nose      face.jpg   # nose-tip landmark
    python examples/module1_demo.py landmarks face.jpg   # all 68 landmarks
    python examples/module1_demo.py live                 # real-time webcam landmarks (Esc to quit)
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handsfree_mouse.landmarks import resolve_model_path  # noqa: E402

WINDOW = "Front_Face"


def load(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    return img


def wait_and_close():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["show", "faces", "nose", "landmarks", "live"])
    ap.add_argument("image", nargs="?", help="image path (not needed for 'live')")
    ap.add_argument("--model", default="models/shape_predictor_68_face_landmarks.dat")
    args = ap.parse_args()

    if args.mode == "show":
        cv2.imshow(WINDOW, load(args.image))
        return wait_and_close()

    import dlib

    detector = dlib.get_frontal_face_detector()
    predictor = None
    if args.mode != "faces":
        predictor = dlib.shape_predictor(str(resolve_model_path(args.model)))

    if args.mode == "live":
        cap = cv2.VideoCapture(0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for face in detector(gray):
                shape = predictor(gray, face)
                for n in range(68):
                    cv2.circle(frame, (shape.part(n).x, shape.part(n).y), 3, (0, 255, 0), -1)
            cv2.imshow("Face", frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    img = load(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for face in detector(gray):
        if args.mode == "faces":
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 4)
            continue
        shape = predictor(gray, face)
        if args.mode == "nose":
            cv2.circle(img, (shape.part(30).x, shape.part(30).y), 5, (0, 255, 0), -1)
        else:
            for n in range(68):
                cv2.circle(img, (shape.part(n).x, shape.part(n).y), 4, (153, 255, 51), -1)
    cv2.imshow(WINDOW, img)
    wait_and_close()


if __name__ == "__main__":
    main()
