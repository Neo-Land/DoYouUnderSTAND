"""Step 1: turn every training video into rows of landmark coordinates.

Usage:
    python scripts/extract_landmarks.py
    python scripts/extract_landmarks.py --videos data/videos --out data/landmarks.csv

Clip names must look like  <pose>_s<session>.<ext>   e.g. jotaro_s1.mp4, none_s2.mov
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from dyu.clips import is_video, parse_clip_name  # noqa: E402
from dyu.config import load_config, resolve  # noqa: E402
from dyu.features import RAW_COLUMNS, raw_to_row  # noqa: E402
from dyu.landmarks import Detector  # noqa: E402

META_COLUMNS = ["video", "label", "session", "frame"]


def extract_video(path, label, session, stride, trim, max_height):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  ! couldn't open {path.name}")
        return [], 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(total * trim) if total > 0 else 0
    end = total - int(total * trim) if total > 0 else 10**9

    rows, no_pose, idx = [], 0, 0
    with Detector(segmentation=False) as det:
        while True:
            ok, frame = cap.read()
            if not ok or idx >= end:
                break
            if idx >= start and (idx - start) % stride == 0:
                if frame.shape[0] > max_height:
                    s = max_height / frame.shape[0]
                    frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                res = det.process(frame, idx * 1000.0 / fps)
                if res.pose is None:
                    no_pose += 1
                else:
                    rows.append([path.name, label, session, idx] + raw_to_row(res.pose, res.left_hand, res.right_hand).tolist())
            idx += 1
    cap.release()
    return rows, no_pose


def main():
    cfg = load_config()
    ex = cfg["extraction"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=ex["videos_dir"])
    ap.add_argument("--out", default=ex["output_csv"])
    args = ap.parse_args()

    videos_dir, out_path = resolve(args.videos), resolve(args.out)
    known = set(cfg["stands"]) | {"none", "dismiss"}
    videos = sorted(p for p in videos_dir.iterdir() if p.is_file() and is_video(p)) if videos_dir.exists() else []
    if not videos:
        sys.exit(f"No videos found in {videos_dir}. Put clips like jotaro_s1.mp4 there.")

    all_rows, t0 = [], time.time()
    for path in videos:
        parsed = parse_clip_name(path)
        if parsed is None:
            print(f"- skipping {path.name}: name should look like jotaro_s1.mp4")
            continue
        label, session = parsed
        if label not in known:
            print(f"- warning: '{label}' isn't in config.yaml stands (still extracting it)")
        rows, no_pose = extract_video(path, label, session, ex["stride"], ex["trim_fraction"], ex["max_height"])
        flag = "  <- check framing!" if rows and no_pose > len(rows) * 0.2 else ""
        lh_i = len(META_COLUMNS) + RAW_COLUMNS.index("lh_present")
        rh_i = len(META_COLUMNS) + RAW_COLUMNS.index("rh_present")
        lh = sum(r[lh_i] for r in rows) / max(1, len(rows))
        rh = sum(r[rh_i] for r in rows) / max(1, len(rows))
        print(f"- {path.name:28s} {label:12s} {session}  kept {len(rows):4d} frames, no body in {no_pose:3d}, "
              f"hands L {lh:4.0%} R {rh:4.0%}{flag}")
        all_rows += rows

    if not all_rows:
        sys.exit("No landmarks extracted. Is your whole body visible in the clips?")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows, columns=META_COLUMNS + RAW_COLUMNS)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} frames to {out_path}  ({time.time() - t0:.0f}s)")
    print(df.groupby(["label", "session"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
