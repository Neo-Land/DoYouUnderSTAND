"""Grid-search the live recognition settings on held-out predictions.

Usage:
    python scripts/tune_smoothing.py                 # needs data/live_frames.csv (run eval_live.py once)
    python scripts/tune_smoothing.py --C 0.1 1.0     # also compare classifier regularization strengths

For every combination of logistic-regression C and the smoother's threshold / window / min_votes /
linger, it replays the cached frames of all clips (each session scored by a model trained on the
other sessions) and reports:
  none_false  share of the time a Stand shows during normal movement (none clips)      lower = better
  pose_on     share of the held pose where the correct Stand is showing (pose clips)   higher = better
  triggered   pose clips where the Stand appeared at all
  trig_s      median seconds until the Stand appears
Settings are picked on the same 21 clips they're scored on, so expect live numbers to be a bit worse.
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import eval_live as E  # noqa: E402
from dyu.config import load_config, resolve  # noqa: E402


def main():
    cfg = load_config()
    ex, rec0 = cfg["extraction"], cfg["recognition"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--C", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--window", type=int, nargs="+", default=[10, 15, 20, 30])
    ap.add_argument("--vote-share", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--linger", type=float, nargs="+", default=[0.3, 0.75, 1.5])
    ap.add_argument("--max-trigger", type=float, default=1.5, help="drop settings slower than this (s)")
    ap.add_argument("--out", default="reports/smoothing_grid.csv")
    args = ap.parse_args()

    frames = E.load_frames(cfg).reset_index(drop=True)
    train_df = pd.read_csv(resolve(ex["output_csv"]))
    rows = []
    for C in args.C:
        def fit(X, y, C=C):
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C)).fit(X, y)

        probs, classes = E.holdout_probs(train_df, frames, fit=fit)
        for th, win, share, linger in itertools.product(args.threshold, args.window, args.vote_share, args.linger):
            rec = {"threshold": th, "window": win, "min_votes": max(1, round(win * share)), "linger_seconds": linger}
            s = E.summarize(E.simulate(frames, probs, classes, rec, ex["trim_fraction"]))
            rows.append({"C": C, **rec, **s})
        print(f"  C={C}: done", flush=True)

    res = pd.DataFrame(rows)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    cur = res[(res.C == 1.0) & (res.threshold == rec0["threshold"]) & (res.window == rec0["window"])
              & (res.min_votes == rec0["min_votes"]) & (res.linger_seconds == rec0["linger_seconds"])]
    cols = ["C", "threshold", "window", "min_votes", "linger_seconds", "none_false", "pose_on", "triggered", "trigger_median"]
    if len(cur):
        print("\nCurrent settings:")
        print(cur[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    ok = res[res.trigger_median <= args.max_trigger]
    best_trig = ok.triggered.max()
    front = ok[ok.triggered == best_trig].sort_values(["none_false", "pose_on"], ascending=[True, False])
    print(f"\nBest by none_false among settings that trigger {best_trig}/{int(res.n_pose.iloc[0])} pose clips "
          f"within {args.max_trigger}s:")
    print(front[cols].head(15).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nAll {len(res)} combinations saved to {out}")


if __name__ == "__main__":
    main()
