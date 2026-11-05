"""Replay the recorded clips through the live app pipeline and measure what a user would see.

Usage:
    python scripts/eval_live.py                     # uses cached landmarks if present
    python scripts/eval_live.py --refresh           # re-run MediaPipe on every frame first
    python scripts/eval_live.py --clips kira none   # only clips with these labels

Step 1 (slow, cached): every frame of every clip goes through the app's Detector (with the
wrist-crop fallback) and the raw landmarks are saved to data/live_frames.csv.
Step 2 (fast): each clip is scored by a model trained on the OTHER sessions (leave-one-session-out,
like scripts/train.py), frames failing the app's body_in_frame check are skipped, and the per-frame
predictions go through PoseSmoother with the config.yaml `recognition` settings, exactly as in app.py.

Reported per clip:
  frame acc  per-frame classifier accuracy on the held part of the clip (trim_fraction cut off)
  on-time    share of the held part where the correct Stand is showing (none clips: no Stand)
  wrong      share of the whole clip where a wrong Stand is showing
  trigger    seconds from the start of the clip until the correct Stand first appears
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import train  # noqa: E402  (scripts/train.py: same features and candidate models)
from dyu.clips import is_video, parse_clip_name  # noqa: E402
from dyu.config import load_config, resolve  # noqa: E402
from dyu.features import RAW_COLUMNS, body_in_frame, raw_to_row, row_to_raw  # noqa: E402
from dyu.smoothing import PoseSmoother  # noqa: E402

CACHE = "data/live_frames.csv"
META = ["video", "label", "session", "frame", "t", "total", "has_pose", "ms"]


# ---------------------------------------------------------------- step 1: landmarks for every frame

def collect_clip(path, label, session, max_height):
    import cv2
    from dyu.landmarks import Detector

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, empty = [], [0.0] * len(RAW_COLUMNS)
    with Detector(segmentation=False) as det:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[0] > max_height:
                s = max_height / frame.shape[0]
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            t = idx / fps
            t0 = time.perf_counter()
            res = det.process(frame, t * 1000.0)
            ms = 1000 * (time.perf_counter() - t0)
            has = res.pose is not None
            raw = raw_to_row(res.pose, res.left_hand, res.right_hand).tolist() if has else empty
            rows.append([path.name, label, session, idx, t, total, int(has), ms] + raw)
            idx += 1
    cap.release()
    return rows


def load_frames(cfg, refresh=False, labels=None):
    cache = resolve(CACHE)
    if cache.exists() and not refresh:
        df = pd.read_csv(cache)
    else:
        ex = cfg["extraction"]
        videos_dir = resolve(ex["videos_dir"])
        rows = []
        for path in sorted(p for p in videos_dir.iterdir() if p.is_file() and is_video(p)):
            parsed = parse_clip_name(path)
            if parsed is None:
                continue
            t0 = time.time()
            clip_rows = collect_clip(path, *parsed, ex["max_height"])
            print(f"  landmarks: {path.name:20s} {len(clip_rows):4d} frames ({time.time() - t0:.0f}s)", flush=True)
            rows += clip_rows
        df = pd.DataFrame(rows, columns=META + RAW_COLUMNS)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"Cached {len(df)} frames to {cache}")
    if labels:
        df = df[df.label.isin(labels)]
    return df


# ---------------------------------------------------------------- step 2: held-out predictions

def holdout_probs(train_df, frames, model_name="logistic_regression", fit=None):
    """Per-frame class probabilities for `frames`, each session predicted by a model trained on the
    other sessions of `train_df`. Returns (probs array with NaN rows where no body, class list).
    `fit(X, y)` can replace the default training (to try other models)."""
    X, X_flip = train.build_features(train_df)
    y, sess = train_df["label"].to_numpy(), train_df["session"].to_numpy()
    F, _ = train.build_features(frames)
    probs, classes = None, None
    for held in sorted(frames.session.unique()):
        tr = sess != held
        Xtr, ytr = np.vstack([X[tr], X_flip[tr]]), np.concatenate([y[tr], y[tr]])
        if fit is None:
            m = train.candidates()[model_name]
            m.fit(Xtr, ytr)
        else:
            m = fit(Xtr, ytr)
        if classes is None:
            classes = list(m.classes_)
            probs = np.full((len(frames), len(classes)), np.nan, np.float32)
        te = (frames.session == held).to_numpy()
        probs[te] = m.predict_proba(F[te])
    probs[frames.has_pose.to_numpy() == 0] = np.nan
    probs[~framed_mask(frames)] = np.nan  # the app skips frames where the body isn't framed
    return probs, classes


def framed_mask(frames):
    """Per frame: does the body pass the app's body_in_frame check (shoulders, hips, knees in view)?"""
    import cv2

    videos_dir = resolve(load_config()["extraction"]["videos_dir"])
    aspect = {}
    for v in frames.video.unique():
        cap = cv2.VideoCapture(str(videos_dir / v))
        aspect[v] = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / max(1.0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    raw = frames[RAW_COLUMNS].to_numpy(dtype=np.float32)
    return np.array([has and body_in_frame(row_to_raw(r)[0], aspect[v])
                     for r, v, has in zip(raw, frames.video.to_numpy(), frames.has_pose.to_numpy())])


# ---------------------------------------------------------------- step 3: smoothing + metrics

def simulate(frames, probs, classes, rec, trim, smoother_cls=PoseSmoother):
    """Run the smoother over each clip and return one metrics row per clip."""
    out = []
    for video, idx in frames.groupby("video", sort=True).indices.items():
        clip = frames.iloc[idx]
        label, total = clip.label.iloc[0], int(clip.total.iloc[0]) or len(clip)
        start, end = int(total * trim), total - int(total * trim)
        sm = smoother_cls(rec["window"], rec["min_votes"], rec["threshold"], rec["linger_seconds"])
        want = None if label == "none" else label
        n = held = held_frame_ok = held_on = wrong = 0
        trigger = None
        for k, (fr, t) in enumerate(zip(clip.frame.to_numpy(), clip.t.to_numpy())):
            p = probs[idx[k]]
            guess, conf = (None, 0.0) if np.isnan(p[0]) else (classes[int(np.argmax(p))], float(np.max(p)))
            active = sm.update(guess, conf, t)
            if active is not None and active != want:
                wrong += 1
            if want is not None and active == want and trigger is None:
                trigger = t
            if start <= fr < end:
                held += 1
                held_frame_ok += guess == label
                held_on += active == want
            n += 1
        out.append({"clip": video, "label": label, "session": clip.session.iloc[0], "frames": n,
                    "frame_acc": held_frame_ok / max(1, held), "on_time": held_on / max(1, held),
                    "wrong": wrong / max(1, n), "trigger_s": trigger,
                    "ms_per_frame": float(np.median(clip.ms.to_numpy()[5:]))})
    return pd.DataFrame(out)


def summarize(res):
    poses, nones = res[res.label != "none"], res[res.label == "none"]
    return {
        "pose_on": poses.on_time.mean(),
        "pose_wrong": poses.wrong.mean(),
        "triggered": int(poses.trigger_s.notna().sum()),
        "n_pose": len(poses),
        "trigger_median": poses.trigger_s.median(),
        "none_false": nones.wrong.mean() if len(nones) else float("nan"),
    }


def print_report(res):
    print(f"\n{'clip':20s} {'frames':>6s} {'frame acc':>9s} {'on-time':>8s} {'wrong':>6s} {'trigger':>8s} {'ms/frame':>8s}")
    for r in res.itertuples():
        trig = "-" if r.label == "none" else ("never" if pd.isna(r.trigger_s) else f"{r.trigger_s:.1f}s")
        print(f"{r.clip:20s} {r.frames:6d} {r.frame_acc:9.0%} {r.on_time:8.0%} {r.wrong:6.0%} {trig:>8s} {r.ms_per_frame:8.1f}")
    s = summarize(res)
    print("\nSummary")
    print(f"  pose clips: correct Stand showing {s['pose_on']:.0%} of the held pose, wrong Stand "
          f"{s['pose_wrong']:.0%} of the time, triggered in {s['triggered']}/{s['n_pose']} clips, "
          f"median {s['trigger_median']:.1f}s after the clip starts")
    print(f"  none clips: a Stand was showing {s['none_false']:.0%} of the time (should be 0%)")
    ms = res.ms_per_frame.median()
    print(f"  speed: median {ms:.0f} ms/frame (~{1000 / ms:.0f} fps for detection, mask off)")


def main():
    cfg = load_config()
    ex, rec = cfg["extraction"], cfg["recognition"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ex["output_csv"], help="landmarks CSV to train the held-out models on")
    ap.add_argument("--model", default="logistic_regression", choices=list(train.candidates()))
    ap.add_argument("--clips", nargs="*", help="only score clips with these labels")
    ap.add_argument("--refresh", action="store_true", help="re-run MediaPipe instead of using the cache")
    ap.add_argument("--out", default="reports/live_eval.csv")
    args = ap.parse_args()

    frames = load_frames(cfg, args.refresh, args.clips).reset_index(drop=True)
    probs, classes = holdout_probs(pd.read_csv(resolve(args.data)), frames, args.model)
    res = simulate(frames, probs, classes, rec, ex["trim_fraction"])
    print(f"recognition: threshold {rec['threshold']}, window {rec['window']}, min_votes {rec['min_votes']}, "
          f"linger {rec['linger_seconds']}s")
    print_report(res)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
