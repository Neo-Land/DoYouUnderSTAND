"""The live demo, in two modes.

    python app.py                          # JoJo poses: hold a pose to summon its Stand (default)
    python app.py --mode signs             # ASL word signs: sign a word, see the top 5 guesses
    python app.py --video <clip>           # play a recorded clip instead of the webcam
    python app.py --camera 1 --debug       # another camera; skeleton + per-frame guess shown

Keys:  q / Esc quit    d toggle skeleton    m mute (poses)    c clear the transcript (signs)
Poses: hold a pose ~1.5 s to summon its Stand, both arms up in a "Y" ~1 s to dismiss it.
Signs: start with your hands down, sign one word, lower your hands; the guess appears right after.
Without a trained model the poses mode still runs and shows the skeleton, a handy setup check.
"""
import argparse
import sys
import time

import cv2
import joblib
import numpy as np

from dyu.config import load_config, resolve
from dyu.effects import SoundPlayer, StandRenderer
from dyu.features import FEATURE_DIM, body_in_frame, featurize
from dyu.landmarks import Detector, draw_landmarks
from dyu.sign_live import SignRecognizer, SignSegmenter, frame_row
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
        print(f"No trained model at {p}. Showing landmarks only (run scripts/poses/train.py first).")
        return None
    bundle = joblib.load(p)
    if bundle.get("feature_dim") != FEATURE_DIM:
        sys.exit("The saved model was trained with different features. Re-run scripts/poses/train.py.")
    print(f"Loaded {bundle['model_name']} model: {', '.join(bundle['labels'])}")
    return bundle


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


def run_poses(args, cfg):
    """JoJo mode: recognize poses, summon and dismiss Stands."""
    cam_cfg, rec, fx = cfg["camera"], cfg["recognition"], cfg["effects"]
    stands = cfg["stands"]

    bundle = load_model(cfg["training"]["model_path"])
    smoother = PoseSmoother.from_config(rec)
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
            # Only classify when the body is framed like the training clips (shoulders, hips and
            # knees in view). A close-up upper body is unlike any training frame, and the model
            # would still name a pose for it with high confidence.
            framed = res.pose is not None and body_in_frame(res.pose, frame.shape[1] / frame.shape[0])
            if bundle is not None and framed:
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
            elif not framed:
                put_text(out, "Step back: show yourself at least head to knees", (20, 45), 0.8, (120, 200, 255))
            # Hold bar: how long the current pose still has to be held to summon (or dismiss).
            holding, frac = smoother.progress(now)
            if holding is not None and frac > 0:
                name = "Dismissing" if holding == smoother.dismiss else stands.get(holding, {}).get("stand", holding)
                y = 115 if active else 80
                x0, x1 = 20, 20 + int(260 * frac)
                cv2.rectangle(out, (20, y), (280, y + 12), (40, 40, 40), -1)
                cv2.rectangle(out, (x0, y), (x1, y + 12), (230, 120, 255), -1)
                put_text(out, name, (290, y + 12), 0.5)
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


def run_signs(args, cfg):
    """ASL mode: cut the stream into signs, classify each, show the top 5 and a transcript."""
    cam_cfg = cfg["camera"]
    try:
        recognizer = SignRecognizer(resolve(cfg["signs"]["models_dir"]))
    except FileNotFoundError as e:
        sys.exit(str(e))
    print(f"Loaded {recognizer.name} sign model: {len(recognizer.classes)} signs")
    segmenter = SignSegmenter()

    cap = open_capture(args, cam_cfg)
    mirror = cam_cfg.get("mirror", True) and not args.video
    debug = args.debug
    max_height = int(cfg["extraction"].get("max_height", 720))
    detector = Detector(segmentation=False)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    t0 = prev = time.perf_counter()
    fps, guesses, transcript = 0.0, [], []
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
            sign = segmenter.update(frame_row(res), now - t0)
            if sign is not None:
                guesses = recognizer.top(sign, k=5)
                transcript = (transcript + [guesses[0][0]])[-8:]

            out = frame
            if debug:
                draw_landmarks(out, res)
            h, w = out.shape[:2]

            # status, top left
            if res.pose is None:
                put_text(out, "Step into frame (head and shoulders)", (20, 45), 0.8, (120, 200, 255))
            elif segmenter.active:
                cv2.circle(out, (32, 38), 10, (40, 40, 230), -1)
                put_text(out, "Signing...", (52, 47), 0.9, (40, 40, 230))
            else:
                put_text(out, "Sign a word, then lower your hands", (20, 45), 0.8)

            # top 5 for the last sign, right side
            if guesses:
                x0 = w - 330
                put_text(out, "Last sign", (x0, 45), 0.7, (230, 120, 255))
                for i, (word, p) in enumerate(guesses):
                    y = 80 + 34 * i
                    cv2.rectangle(out, (x0, y - 16), (x0 + int(200 * p), y + 4), (230, 120, 255) if i == 0 else (120, 90, 150), -1)
                    put_text(out, f"{word}  {p:.0%}", (x0 + 6, y), 0.6 if i else 0.75)

            # transcript, bottom
            if transcript:
                put_text(out, "  ".join(transcript), (20, h - 55), 0.9, (255, 255, 255), 2)
            put_text(out, f"{fps:4.1f} fps   {recognizer.name}   [q] quit  [d] skeleton  [c] clear",
                     (20, h - 20), 0.5)

            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("c"):
                guesses, transcript = [], []
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["poses", "signs"], default="poses")
    ap.add_argument("--video", help="run on a video file instead of the webcam")
    ap.add_argument("--camera", type=int, help="camera index (default from config.yaml)")
    ap.add_argument("--no-sound", action="store_true", help="poses mode: no sound")
    ap.add_argument("--debug", action="store_true", help="start with the skeleton overlay on")
    args = ap.parse_args()
    cfg = load_config()
    if args.mode == "signs":
        run_signs(args, cfg)
    else:
        run_poses(args, cfg)


if __name__ == "__main__":
    main()
