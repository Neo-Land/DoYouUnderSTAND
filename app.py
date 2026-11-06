"""The live demo: a start menu, then JoJo poses, ASL signs, or Learn ASL practice.

    python app.py                          # start menu (webcam)
    python app.py --mode poses             # straight into a mode: menu, poses, signs, practice
    python app.py --video <clip>           # play a recorded clip instead of the webcam
    python app.py --camera 1 --debug       # another camera; skeleton overlay on

Keys, anywhere:  q / Esc quit   m menu   Tab next mode   d skeleton overlay
In the menu:     j JoJo poses   a ASL signs   p Learn ASL
JoJo:  hold a pose ~1.5 s to summon its Stand; both arms up in a "Y" ~1 s to dismiss it.  s mutes sound.
ASL:   start with your hands down, sign one word, lower your hands; the top 5 appear.  c clears.
Learn: sign the word shown; green = top guess, yellow = in the top 5.  n skips to the next word.

The screens themselves live in dyu/modes.py; this file is the camera loop and the mode switching.
"""
import argparse
import sys
import time

import cv2

from dyu.config import load_config
from dyu.landmarks import Detector
from dyu.modes import AppContext, MenuMode, PosesMode, PracticeMode, SignsMode
from dyu.ui import put_text

WINDOW = "Do You UnderSTAND"
MODE_KEYS = {"poses": "j", "signs": "a", "practice": "p"}
CYCLE = ["poses", "signs", "practice"]       # the order Tab goes through


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


class App:
    def __init__(self, args, cfg):
        self.ctx = AppContext(cfg, debug=args.debug, sound=not args.no_sound)
        self.modes = {"menu": MenuMode(self.ctx), "poses": PosesMode(self.ctx),
                      "signs": SignsMode(self.ctx), "practice": PracticeMode(self.ctx)}
        self.current = None
        self.switch(args.mode)

    @property
    def mode(self):
        return self.modes[self.current]

    def switch(self, name):
        """Enter a mode; an ASL mode without a sign model falls back to the menu with a notice."""
        mode = self.modes[name]
        mode.enter()
        if getattr(mode, "recognizer", True) is None:
            name = "menu"
            self.modes[name].enter()
        self.current = name

    def handle_key(self, key):
        """Global keys first, then the mode's own. Returns False to quit."""
        if key in (ord("q"), 27):
            return False
        if key == ord("m"):
            self.switch("menu")
        elif key == 9:                                        # Tab
            nxt = CYCLE[(CYCLE.index(self.current) + 1) % len(CYCLE)] if self.current in CYCLE else CYCLE[0]
            self.switch(nxt)
        elif key == ord("d"):
            self.ctx.debug = not self.ctx.debug
        elif self.current == "menu" and key in {ord(k) for k in MODE_KEYS.values()}:
            self.switch(next(m for m, k in MODE_KEYS.items() if ord(k) == key))
        else:
            self.mode.on_key(key)
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["menu", "poses", "signs", "practice"], default="menu")
    ap.add_argument("--video", help="run on a video file instead of the webcam")
    ap.add_argument("--camera", type=int, help="camera index (default from config.yaml)")
    ap.add_argument("--no-sound", action="store_true", help="JoJo mode: no sound")
    ap.add_argument("--debug", action="store_true", help="start with the skeleton overlay on")
    args = ap.parse_args()
    cfg = load_config()

    app = App(args, cfg)
    cap = open_capture(args, cfg["camera"])
    mirror = cfg["camera"].get("mirror", True) and not args.video
    # Same cap as extraction: frames taller than this are shrunk, so the models see what they were
    # trained on and phone clips (1080x1920) don't run at a third of the speed.
    max_height = int(cfg["extraction"].get("max_height", 720))
    detector = Detector(segmentation=True)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    t0 = prev = time.perf_counter()
    fps = 0.0
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

            detector.mask_enabled = app.mode.needs_mask          # the mask is only needed to draw Stands
            res = detector.process(frame, (now - t0) * 1000.0)
            out = app.mode.process(frame, res, now - t0, dt)
            put_text(out, f"{fps:4.1f} fps   {app.mode.name}   {app.mode.hints}", (20, out.shape[0] - 20), 0.5)

            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            if key != 255 and not app.handle_key(key):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
