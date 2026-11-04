"""Render one still per Stand from its pose clip, to check Stand placement without a webcam.

Usage:
    python scripts/render_stand_previews.py                  # all Stands, session s2 clips
    python scripts/render_stand_previews.py --session s3 --at 0.6
    python scripts/render_stand_previews.py diego kira       # only these Stands

For each Stand it plays <pose>_<session>.* up to `--at` (fraction of the clip) with that Stand
forced on, so the image has fully faded in and the anchor has settled, then saves the frame with
the same compositing as app.py (Stand behind the person mask, menacing text).
Output: reports/stand_previews/<pose>.jpg (git-ignored: the Stand art isn't ours to publish).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from dyu.clips import is_video, parse_clip_name  # noqa: E402
from dyu.config import load_config, resolve  # noqa: E402
from dyu.effects import StandRenderer  # noqa: E402
from dyu.landmarks import Detector  # noqa: E402


def find_clip(videos_dir, pose, session):
    for p in sorted(videos_dir.iterdir()):
        if p.is_file() and is_video(p) and parse_clip_name(p) == (pose, session):
            return p
    return None


def render(path, pose, renderer, at, max_height):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = int(total * at)
    warmup = max(int(fps * 1.5), 15)  # long enough for the fade-in and anchor smoothing
    start = max(0, target - warmup)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_out = None
    with Detector(segmentation=True) as det:
        for i in range(start, target + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[0] > max_height:
                s = max_height / frame.shape[0]
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            t = i / fps
            res = det.process(frame, t * 1000.0)
            renderer.update(pose, 1.0 / fps)
            frame_out = renderer.draw(frame, res, t)
    cap.release()
    return frame_out


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stands", nargs="*", help="Stand keys to render (default: all in config.yaml)")
    ap.add_argument("--session", default="s2")
    ap.add_argument("--at", type=float, default=0.5, help="where in the clip to grab the frame (0-1)")
    ap.add_argument("--out", default="reports/stand_previews")
    args = ap.parse_args()

    videos_dir = resolve(cfg["extraction"]["videos_dir"])
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer_cfg = dict(cfg["effects"])
    for pose in args.stands or list(cfg["stands"]):
        clip = find_clip(videos_dir, pose, args.session)
        if clip is None:
            print(f"- {pose}: no {pose}_{args.session} clip, skipped")
            continue
        renderer = StandRenderer({pose: cfg["stands"][pose]}, renderer_cfg)  # fresh fade + anchor
        img = render(clip, pose, renderer, args.at, cfg["extraction"]["max_height"])
        if img is None:
            print(f"- {pose}: couldn't read {clip.name}")
            continue
        s = cfg["stands"][pose]
        cv2.putText(img, f"{pose}  scale {s.get('scale')}  x {s.get('offset_x')}  y {s.get('offset_y')}",
                    (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        dest = out_dir / f"{pose}.jpg"
        cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"- {pose}: {clip.name} -> {dest}")


if __name__ == "__main__":
    main()
