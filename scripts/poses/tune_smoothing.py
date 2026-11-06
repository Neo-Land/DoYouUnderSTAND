"""JoJo tuning: grid-search the live recognition settings on held-out predictions.

Usage:
    python scripts/poses/tune_smoothing.py                  # needs data/live_frames.csv (run scripts/poses/eval_live.py once)
    python scripts/poses/tune_smoothing.py --hold 1.0 1.5   # narrow or widen any axis

For every combination of the smoother's threshold / window / vote share / hold_seconds, it replays the
cached frames of all clips (each session scored by a model trained on the other sessions, using the
model scripts/poses/train.py picked) and reports:
  none_summons  false summons during normal movement (none clips)                     lower = better
  none_false    share of normal-movement time a Stand is showing                       lower = better
  pose_on       share of the held pose where the correct Stand is showing (pose clips)  higher = better
  triggered     pose clips where the Stand appeared at all
  trig_s        median seconds until the Stand appears
  dismissed     dismiss clips where the Stand was sent away
Settings are picked on the same clips they're scored on, so expect live numbers to be a bit worse.
"""
import argparse
import itertools

import pandas as pd

from dyu import live_eval as E
from dyu.config import load_config, resolve
from dyu.pose_model import best_model_name


def main():
    cfg = load_config()
    ex, rec0 = cfg["extraction"], cfg["recognition"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--window", type=int, nargs="+", default=[10, 15, 20, 30])
    ap.add_argument("--vote-share", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--hold", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--max-trigger", type=float, default=3.0, help="drop settings slower than this (s)")
    ap.add_argument("--out", default="reports/poses/smoothing_grid.csv")
    args = ap.parse_args()

    frames = E.load_frames(cfg).reset_index(drop=True)
    model = best_model_name(cfg)
    probs, classes = E.holdout_probs(pd.read_csv(resolve(ex["output_csv"])), frames, model)
    print(f"model: {model}")
    rows = []
    for th, win, share, hold in itertools.product(args.threshold, args.window, args.vote_share, args.hold):
        rec = dict(rec0, threshold=th, window=win, min_votes=max(1, round(win * share)), hold_seconds=hold)
        s = E.summarize(E.simulate(frames, probs, classes, rec, ex["trim_fraction"]))
        rows.append({k: rec[k] for k in ("threshold", "window", "min_votes", "hold_seconds")} | s)

    res = pd.DataFrame(rows)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    cols = ["threshold", "window", "min_votes", "hold_seconds", "none_summons", "none_false", "pose_on",
            "triggered", "trigger_median", "dismissed", "dismiss_median"]
    fmt = {"float_format": lambda v: f"{v:.2f}", "index": False}
    cur = res[(res.threshold == rec0["threshold"]) & (res.window == rec0["window"])
              & (res.min_votes == rec0["min_votes"]) & (res.hold_seconds == rec0["hold_seconds"])]
    if len(cur):
        print("\nCurrent settings:")
        print(cur[cols].to_string(**fmt))
    ok = res[(res.trigger_median <= args.max_trigger) & (res.dismissed == res.n_dismiss)]
    best_trig = ok.triggered.max()
    front = ok[ok.triggered == best_trig].sort_values(["none_summons", "none_false", "pose_on"],
                                                     ascending=[True, True, False])
    print(f"\nFewest false summons among settings that summon {best_trig}/{int(res.n_pose.iloc[0])} pose clips "
          f"within {args.max_trigger}s and dismiss every dismiss clip:")
    print(front[cols].head(15).to_string(**fmt))
    print(f"\nAll {len(res)} combinations saved to {out}")


if __name__ == "__main__":
    main()
