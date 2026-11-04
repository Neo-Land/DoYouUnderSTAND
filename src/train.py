"""Train a classifier on data/<mode>.csv -> models/<mode>.joblib.

python src/train.py --mode pose
python src/train.py --mode asl

Note: the test split is random frames, so neighboring frames from one recording can land in
both sets and the score runs optimistic. The real test is the live app.
"""
import argparse
import sys
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import config
from landmarks import FEATURE_DIM, MIRROR


def load(mode):
    path = config.data_path(mode)
    if not path.exists():
        sys.exit(f"No data at {path}. Record some with src/record.py first.")
    raw = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=1)
    if raw.ndim == 1:
        raw = raw[None, :]
    y = raw[:, 0]
    X = raw[:, 1:].astype(np.float32)
    if X.shape[1] != FEATURE_DIM[mode]:
        sys.exit(f"Feature size {X.shape[1]} != expected {FEATURE_DIM[mode]}. Re-record the data.")
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=config.MODES, default="pose")
    ap.add_argument("--no-mirror", action="store_true",
                    help="don't add left/right mirrored copies of samples")
    args = ap.parse_args()

    X, y = load(args.mode)
    counts = Counter(y)
    print("Samples per label:", dict(sorted(counts.items())))
    if len(counts) < 2:
        sys.exit("Need at least 2 labels to train.")
    low = [k for k, v in counts.items() if v < 50]
    if low:
        print(f"Heads up: few samples for {low}; record more for better accuracy.")
    if args.mode == "pose" and config.NEUTRAL_LABEL not in counts:
        print("Heads up: no 'neutral' samples, so the app may trigger Stands when you're not posing.")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    mirror = MIRROR[args.mode]
    if not args.no_mirror:
        X_tr = np.vstack([X_tr, np.array([mirror(v) for v in X_tr])])
        y_tr = np.concatenate([y_tr, y_tr])

    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42,
                                 class_weight="balanced")
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_te)
    print(classification_report(y_te, pred, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print("labels:", list(clf.classes_))
    print(confusion_matrix(y_te, pred, labels=clf.classes_))

    # Refit on everything for the final model
    X_all, y_all = X, y
    if not args.no_mirror:
        X_all = np.vstack([X, np.array([mirror(v) for v in X])])
        y_all = np.concatenate([y, y])
    clf.fit(X_all, y_all)

    out = config.model_path(args.mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "mode": args.mode, "dim": FEATURE_DIM[args.mode]}, out)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
