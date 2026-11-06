"""ASL step 3: replay the test signers' videos through the app's live sign pipeline and score it.

Usage:
    python scripts/signs/eval_live.py            # test split, the saved GRU (or logreg) model

In the app, nobody hands the model a neatly cut clip: SignSegmenter has to find where the sign starts
and ends in the stream, and SignRecognizer classifies whatever it cut out. This replays the cached
landmarks of every test-split video (11 signers the model never saw) frame by frame through exactly
that code, at each video's own frame rate, and reports:
  detected   share of videos where the segmenter found a sign at all
  top-1/5    accuracy of the guess for the longest sign found (missed videos count as wrong)
Numbers only are written (reports/signs/live.json): no dataset frames, per the ASL Citizen license.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dyu.config import ROOT, load_config, resolve
from dyu.sign_live import SignRecognizer, SignSegmenter, display_name


def replay(frames, fps):
    """Feed one video's raw rows through a fresh segmenter; return the signs it cut out."""
    seg, signs = SignSegmenter(), []
    for i, row in enumerate(frames):
        sign = seg.update(None if np.isnan(row[0]) else row, i / fps)
        if sign is not None:
            signs.append(sign)
    # flush: the video ends while the sign may still be "open"
    t = len(frames) / fps
    while seg.active:
        t += 1 / fps
        sign = seg.update(None, t)
        if sign is not None:
            signs.append(sign)
    return signs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--no-rest-trim", action="store_true", help="classify the whole cut-out sign")
    ap.add_argument("--no-mirror-average", action="store_true", help="score the sign only as signed")
    ap.add_argument("--out", default="live.json", help="report file name in reports/signs/")
    args = ap.parse_args()

    cfg = load_config()["signs"]
    data = resolve(cfg["landmarks_dir"])
    recognizer = SignRecognizer(resolve(cfg["models_dir"]), rest_trim=not args.no_rest_trim,
                                mirror_average=not args.no_mirror_average)
    index = pd.read_csv(data / "index.csv")
    index = index[index.split == args.split]
    print(f"{recognizer.name} model; {len(index)} {args.split} videos, {index.participant.nunique()} signers")

    rows = []
    for r in index.itertuples():
        frames = np.load(data / (Path(r.file).stem + ".npy"))
        signs = replay(frames, r.fps if r.fps > 0 else 30.0)
        truth = display_name(r.gloss)
        if not signs:
            rows.append({"gloss": r.gloss, "detected": False, "segments": 0, "top1": False, "top5": False})
            continue
        longest = max(signs, key=len)
        words = [w for w, _ in recognizer.top(longest, k=5)]
        rows.append({"gloss": r.gloss, "detected": True, "segments": len(signs),
                     "top1": words[0] == truth, "top5": truth in words})
    res = pd.DataFrame(rows)

    report = {"model": recognizer.name, "split": args.split, "videos": len(res),
              "rest_trim": recognizer.rest_trim, "mirror_average": recognizer.mirror_average,
              "detected": float(res.detected.mean()), "more_than_one_segment": float((res.segments > 1).mean()),
              "top1": float(res.top1.mean()), "top5": float(res.top5.mean()),
              "top1_when_detected": float(res[res.detected].top1.mean())}
    print(f"  sign detected in {report['detected']:.1%} of videos "
          f"(split into more than one sign: {report['more_than_one_segment']:.1%})")
    print(f"  live top-1 {report['top1']:.1%}  top-5 {report['top5']:.1%}  "
          f"(top-1 when detected: {report['top1_when_detected']:.1%})")
    out = resolve(cfg["reports_dir"]) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
