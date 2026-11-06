"""ASL step 1: turn every ASL Citizen video in the sign subset into per-frame landmarks.

Usage:
    python scripts/signs/extract_landmarks.py               # all videos in datasets/asl_citizen/subset.csv
    python scripts/signs/extract_landmarks.py --limit 20    # quick test on 20 videos
    python scripts/signs/extract_landmarks.py --workers 8

Needs datasets/asl_citizen/subset.csv and the videos it lists (scripts/signs/extract_asl_citizen.py makes both).
Each video goes through the same MediaPipe Detector as the JoJo app (pose + both hands, with the
wrist-crop fallback), and its frames are saved as one float32 array of shape (frames, len(RAW_COLUMNS))
in data/asl_signs/<video>.npy, NaN rows where no body was found. data/asl_signs/index.csv lists every
video with its sign, split, signer and detection rates. Safe to stop and re-run: finished videos are
skipped.

License (Microsoft Research, ASL Citizen): these landmarks are derived from the dataset, so they must
never be committed or published. They stay in data/ (git-ignored).
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dyu.config import load_config, resolve
from dyu.features import RAW_COLUMNS, raw_to_row

_CFG = load_config()["signs"]
SUBSET = resolve(_CFG["dataset_dir"]) / "subset.csv"
VIDEOS = resolve(_CFG["dataset_dir"]) / "ASL_Citizen" / "videos"
OUT = resolve(_CFG["landmarks_dir"])
MAX_HEIGHT = 720
GAP_MS = 10_000  # timestamp gap between videos, so VIDEO-mode tracking starts fresh on each clip

_det = None
_ts = 0.0


def _init_worker():
    global _det
    from dyu.landmarks import Detector

    _det = Detector(segmentation=False)


def out_path(video_file):
    return OUT / (Path(video_file).stem + ".npy")


def extract_one(video_file):
    """Landmarks for one video -> saved .npy; returns a summary dict for the index."""
    import cv2

    global _ts
    cap = cv2.VideoCapture(str(VIDEOS / video_file))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps < 5 or fps > 120:
        fps = 30.0
    rows, w, h = [], 0, 0
    nan_row = np.full(len(RAW_COLUMNS), np.nan, np.float32)
    start = _ts + GAP_MS
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] > MAX_HEIGHT:
            s = MAX_HEIGHT / frame.shape[0]
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        _ts = start + i * 1000.0 / fps
        res = _det.process(frame, _ts)
        rows.append(raw_to_row(res.pose, res.left_hand, res.right_hand) if res.pose is not None else nan_row)
        i += 1
    cap.release()
    arr = np.asarray(rows, np.float32).reshape(len(rows), len(RAW_COLUMNS))
    np.save(out_path(video_file), arr)
    has = ~np.isnan(arr[:, 0]) if len(arr) else np.zeros(0, bool)
    lh = arr[has, RAW_COLUMNS.index("lh_present")] if has.any() else np.zeros(0)
    rh = arr[has, RAW_COLUMNS.index("rh_present")] if has.any() else np.zeros(0)
    return {"file": video_file, "frames": len(arr), "fps": round(fps, 2), "width": w, "height": h,
            "pose_rate": float(has.mean()) if len(arr) else 0.0,
            "lh_rate": float(lh.mean()) if len(lh) else 0.0, "rh_rate": float(rh.mean()) if len(rh) else 0.0,
            "any_hand_rate": float(np.maximum(lh, rh).mean()) if len(lh) else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only the first N videos (for testing)")
    args = ap.parse_args()

    if not SUBSET.exists():
        sys.exit(f"Missing {SUBSET}. Run scripts/signs/extract_asl_citizen.py first.")
    subset = pd.read_csv(SUBSET)
    if args.limit:
        subset = subset.sample(args.limit, random_state=0)
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "index.csv"
    done = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame({"file": pd.Series(dtype=str)})
    done = done[[out_path(f).exists() for f in done["file"]]] if len(done) else done
    finished = set(done["file"])
    todo = [f for f in subset.file if f not in finished]
    print(f"{len(subset)} videos, {len(subset) - len(todo)} already done, {len(todo)} to go "
          f"({args.workers} workers)", flush=True)

    rows, t0 = [], time.time()
    if todo:
        with mp.Pool(args.workers, initializer=_init_worker) as pool:
            for k, r in enumerate(pool.imap_unordered(extract_one, todo, chunksize=4), 1):
                rows.append(r)
                if k % 100 == 0 or k == len(todo):
                    rate = k / (time.time() - t0)
                    print(f"  {k}/{len(todo)}  {rate:.1f} videos/s, ~{(len(todo) - k) / rate / 60:.0f} min left",
                          flush=True)
                if k % 500 == 0:  # checkpoint the index so a stop loses little
                    pd.concat([done, pd.DataFrame(rows)]).to_csv(index_path, index=False)

    index = pd.concat([done, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("file", keep="last")
    index = index.drop(columns=[c for c in ("gloss", "split", "participant") if c in index.columns])
    index = subset.merge(index, on="file", how="inner")
    index.to_csv(index_path, index=False)

    print(f"\nSaved {len(index)} videos to {OUT} ({time.time() - t0:.0f}s this run)")
    print(f"frames per video: median {index.frames.median():.0f}, total {index.frames.sum():,}")
    print(f"body found in {index.pose_rate.mean():.1%} of frames; a hand in {index.any_hand_rate.mean():.1%}; "
          f"left {index.lh_rate.mean():.1%}, right {index.rh_rate.mean():.1%}")
    bad = index[(index.pose_rate < 0.5) | (index.any_hand_rate < 0.3)]
    print(f"videos with poor detection (body < 50% or hands < 30% of frames): {len(bad)}")
    print(index.groupby("split").agg(videos=("file", "count"), signers=("participant", "nunique"),
                                     hand_rate=("any_hand_rate", "mean")).to_string())


if __name__ == "__main__":
    main()
