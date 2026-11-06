"""JoJo evaluation: replay the recorded clips through the live app logic and measure what a user would see.

Usage:
    python scripts/poses/eval_live.py                     # uses cached landmarks if present
    python scripts/poses/eval_live.py --refresh           # re-run MediaPipe on every frame first
    python scripts/poses/eval_live.py --clips kira none   # only clips with these labels

Step 1 (slow, cached): every frame of every clip goes through the app's Detector (with the
wrist-crop fallback) and the raw landmarks are saved to data/live_frames.csv.
Step 2 (fast): each clip is scored by a model trained on the OTHER sessions (leave-one-session-out,
like scripts/poses/train.py), frames failing the app's body_in_frame check are skipped, and the per-frame
predictions go through PoseSmoother with the config.yaml `recognition` settings, exactly as in app.py.

Reported per clip:
  frame acc  per-frame classifier accuracy on the held part of the clip (trim_fraction cut off)
  on-time    share of the held part where the correct Stand is showing (none/dismiss: no Stand)
  wrong      share of the whole clip where a wrong Stand is showing
  summons    how many times a wrong Stand was summoned (it then stays until dismissed)
  trigger    pose clips: seconds until the correct Stand first appears
             dismiss clips: a Stand is showing at the start; seconds until it is dismissed
"""
import argparse

import pandas as pd

from dyu import live_eval as L
from dyu import pose_model
from dyu.config import load_config, resolve


def main():
    cfg = load_config()
    ex, rec = cfg["extraction"], cfg["recognition"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ex["output_csv"], help="landmarks CSV to train the held-out models on")
    ap.add_argument("--model", default=None, choices=list(pose_model.candidates()),
                    help="default: the best model in reports/poses/metrics.json")
    ap.add_argument("--clips", nargs="*", help="only score clips with these labels")
    ap.add_argument("--refresh", action="store_true", help="re-run MediaPipe instead of using the cache")
    ap.add_argument("--out", default="reports/poses/live_eval.csv")
    args = ap.parse_args()

    frames = L.load_frames(cfg, args.refresh, args.clips).reset_index(drop=True)
    model = args.model or pose_model.best_model_name(cfg)
    probs, classes = L.holdout_probs(pd.read_csv(resolve(args.data)), frames, model)
    res = L.simulate(frames, probs, classes, rec, ex["trim_fraction"])
    print(f"model {model}; recognition: " + ", ".join(f"{k} {v}" for k, v in rec.items()))
    L.print_report(res)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
