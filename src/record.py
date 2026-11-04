"""Record labeled landmark samples into data/<mode>.csv.

Webcam:  python src/record.py --label jotaro            (SPACE = start/stop capture, Q = quit)
Video:   python src/record.py --label jotaro --video clips/jotaro.mov
ASL:     python src/record.py --mode asl --label A

Tips: record `neutral` too. Vary distance, angle and lighting a little. Aim for 200+ samples
per label. For videos, trim them so every frame shows the pose.
"""
import argparse
import csv
import sys

import cv2

import config
from landmarks import FEATURE_DIM, Extractor


def open_writer(mode):
    path = config.data_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["label"] + [f"f{i}" for i in range(FEATURE_DIM[mode])])
    return f, w, path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=config.MODES, default="pose")
    ap.add_argument("--label", required=True, help="e.g. jotaro, neutral, or an ASL letter")
    ap.add_argument("--video", help="process a video file instead of the webcam")
    ap.add_argument("--flip", action="store_true",
                    help="mirror video frames (use if your clip isn't mirrored but the webcam is)")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--every", type=int, default=2, help="keep every Nth frame (default 2)")
    args = ap.parse_args()

    label = args.label.strip()
    if args.mode == "pose" and label not in config.POSES and label != config.NEUTRAL_LABEL:
        print(f"Warning: '{label}' isn't in config.POSES; it'll be recorded anyway.")

    if args.video:
        cap = cv2.VideoCapture(args.video)
        mirror, capturing = args.flip, True
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        mirror, capturing = config.MIRROR_CAMERA, False
    if not cap.isOpened():
        sys.exit("Could not open the camera/video. Check --camera or --video.")

    extractor = Extractor(args.mode)
    f, writer, path = open_writer(args.mode)
    saved = frame_i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            frame_i += 1
            feats, results = extractor.process(frame)

            if capturing and feats is not None and frame_i % args.every == 0:
                writer.writerow([label] + [f"{v:.5f}" for v in feats])
                saved += 1

            extractor.draw(frame, results)
            status = "REC" if capturing else "paused (SPACE to record)"
            color = (0, 0, 255) if capturing else (200, 200, 200)
            cv2.putText(frame, f"{label}: {saved} saved  [{status}]", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            if feats is None:
                cv2.putText(frame, "no body/hands detected", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.imshow("record - Q to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and not args.video:
                capturing = not capturing
    finally:
        f.close()
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()
    print(f"Saved {saved} '{label}' samples to {path}")


if __name__ == "__main__":
    main()
