"""Live demo.

python src/app.py                      # pose mode, webcam
python src/app.py --mode asl           # ASL fingerspelling mode
python src/app.py --video clip.mov     # run on a video (no webcam needed)

Keys: Q quit · D toggle landmark debug view · C clear ASL text
"""
import argparse
import sys
import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

import config
import effects
from landmarks import Extractor


def put(frame, text, org, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=config.MODES, default="pose")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--video", help="use a video file instead of the webcam")
    ap.add_argument("--flip", action="store_true", help="mirror video frames")
    args = ap.parse_args()

    mpath = config.model_path(args.mode)
    if not mpath.exists():
        sys.exit(f"No model at {mpath}. Run: python src/train.py --mode {args.mode}")
    bundle = joblib.load(mpath)
    clf = bundle["model"]

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    mirror = args.flip if args.video else config.MIRROR_CAMERA
    if not cap.isOpened():
        sys.exit("Could not open the camera/video.")

    extractor = Extractor(args.mode)
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) if args.mode == "pose" else None
    sound = effects.Sound()

    history = deque(maxlen=config.STABLE_FRAMES)
    active, active_since, last_seen = None, 0.0, 0.0
    asl_text, asl_last = "", None
    debug = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            now = time.time()
            h, w = frame.shape[:2]

            feats, results = extractor.process(frame)
            guess, conf = None, 0.0
            if feats is not None:
                proba = clf.predict_proba(feats[None, :])[0]
                i = int(np.argmax(proba))
                guess, conf = clf.classes_[i], float(proba[i])
                if conf < config.CONFIDENCE_THRESHOLD:
                    guess = None
            history.append(guess)
            stable = guess if (len(history) == history.maxlen and len(set(history)) == 1) else None

            if args.mode == "pose":
                if stable and stable != config.NEUTRAL_LABEL:
                    if stable != active:
                        active, active_since = stable, now
                        sound.play(active)
                    last_seen = now
                if active and now - last_seen > config.HOLD_SECONDS:
                    active = None

                if active:
                    alpha = min(1.0, (now - active_since) / config.FADE_SECONDS)
                    mask = None
                    seg_res = seg.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if seg_res.segmentation_mask is not None:
                        mask = cv2.GaussianBlur(seg_res.segmentation_mask, (15, 15), 0)
                    effects.draw_stand(frame, active, extractor.anchor_x(results, w), mask, alpha)
                    effects.draw_menacing(frame, now, alpha)
                    effects.draw_stand_name(frame, active, alpha)
            else:  # ASL
                if stable and stable != asl_last and stable != config.NEUTRAL_LABEL:
                    if stable == "del":
                        asl_text = asl_text[:-1]
                    elif stable == "space":
                        asl_text = (asl_text + " ")[-30:]
                    else:
                        asl_text = (asl_text + stable)[-30:]
                asl_last = stable
                if stable:
                    put(frame, stable, (w - 160, 130), scale=4, thick=6, color=(90, 230, 255))
                put(frame, asl_text, (20, h - 30), scale=1.4, thick=3)

            if debug:
                extractor.draw(frame, results)
            put(frame, f"{guess or '-'} {conf:.2f}", (20, 40), scale=0.7, color=(200, 200, 200))
            cv2.imshow("Do You UnderSTAND? - Q to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("c"):
                asl_text = ""
    finally:
        cap.release()
        extractor.close()
        if seg:
            seg.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
