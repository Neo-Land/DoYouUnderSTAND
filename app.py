"""Step 3: strike a pose, summon your Stand.

Usage:
    python app.py                         # webcam
    python app.py --video data/videos/jotaro_s1.mp4   # test on a recorded clip (no webcam needed)
    python app.py --camera 1 --no-sound --debug

Keys:  q / Esc quit    d toggle skeleton    m mute
Without a trained model it still runs and shows the skeleton, which is a handy setup check.
"""
import argparse
import sys
import time

import cv2
import joblib
import numpy as np

from dyu.config import load_config, resolve
from dyu.effects import SoundPlayer, StandRenderer
from dyu.features import FEATURE_DIM, featurize
from dyu.landmarks import Detector, draw_landmarks
from dyu.smoothing import PoseSmoother

WINDOW = "Do You UnderSTAND"


def open_capture(args, cam_cfg):
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        idx = args.camera if args.camera is not None else cam_cfg.get("index", 0)
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(idx, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))
    if not cap.isOpened():
        sys.exit("Couldn't open the camera/video. Try --camera 1, or check the file path.")
    return cap


def load_model(path):
    p = resolve(path)
    if not p.exists():
        print(f"No trained model at {p}. Showing landmarks only (run scripts/train.py first).")
        return None
    bundle = joblib.load(p)
    if bundle.get("feature_dim") != FEATURE_DIM:
        sys.exit("The saved model was trained with different features. Re-run scripts/train.py.")
    print(f"Loaded {bundle['model_name']} model: {', '.join(bundle['labels'])}")
    return bundle


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="run on a video file instead of the webcam")
    ap.add_argument("--camera", type=int, help="camera index (default from config.yaml)")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--debug", action="store_true", help="start with the skeleton overlay on")
    args = ap.parse_args()

    cfg = load_config()
    cam_cfg, rec, fx = cfg["camera"], cfg["recognition"], cfg["effects"]
    stands = cfg["stands"]

    bundle = load_model(cfg["training"]["model_path"])
    smoother = PoseSmoother(rec["window"], rec["min_votes"], rec["threshold"], rec["linger_seconds"])
    renderer = StandRenderer(stands, fx)
    sound = SoundPlayer({k: v.get("sound") for k, v in stands.items()}, fx.get("default_sound"),
                        fx.get("volume", 0.8), enabled=fx.get("sound", True) and not args.no_sound)

    cap = open_capture(args, cam_cfg)
    mirror = cam_cfg.get("mirror", True) and not args.video
    debug = args.debug or bundle is None
    # Same cap as extraction: frames taller than this are shrunk, so the models see what they were
    # trained on and phone clips (1080x1920) don't run at a third of the speed.
    max_height = int(cfg["extraction"].get("max_height", 720))
    detector = Detector(segmentation=True)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    t0 = prev = time.perf_counter()
    fps, prev_active = 0.0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[0] > max_height:
                s = max_height / frame.shape[0]
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if mirror:
                frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            dt, prev = now - prev, now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)

            res = detector.process(frame, (now - t0) * 1000.0)

            label, prob = None, 0.0
            if bundle is not None and res.pose is not None:
                proba = bundle["model"].predict_proba([featurize(res.pose, res.left_hand, res.right_hand)])[0]
                i = int(np.argmax(proba))
                label, prob = bundle["labels"][i], float(proba[i])
            active = smoother.update(label, prob, now)
            if active and active != prev_active:
                sound.play(active)
            prev_active = active

            renderer.update(active, dt)
            out = renderer.draw(frame, res, now - t0)
            if debug:
                draw_landmarks(out, res)

            # HUD
            if active:
                s = stands.get(active, {})
                put_text(out, f"{s.get('user', active)}  -  {s.get('stand', '')}", (20, 45), 1.0, (230, 120, 255))
                put_text(out, s.get("pose", ""), (20, 80), 0.6)
            elif res.pose is None:
                put_text(out, "Step into frame (whole body)", (20, 45), 0.8, (120, 200, 255))
            if debug and label:
                put_text(out, f"frame guess: {label} {prob:.2f}", (20, out.shape[0] - 50), 0.6)
            put_text(out, f"{fps:4.1f} fps   [q] quit  [d] skeleton  [m] mute"
                     + ("  (muted)" if sound.muted else ""), (20, out.shape[0] - 20), 0.5)

            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("m"):
                sound.toggle_mute()
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
