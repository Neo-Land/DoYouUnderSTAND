"""Step 2: train the pose classifier and measure it honestly.

Usage:
    python scripts/train.py

Evaluation is leave-one-session-out: train on sessions s1+s2, test on s3, and so on.
Frames from one clip are near-duplicates, so a random split would let the model
"memorize" the test set and report fake-high accuracy. Holding out a whole session
measures how well it handles a new day, new clothes, new lighting.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from dyu.config import load_config, resolve  # noqa: E402
from dyu.features import FEATURE_DIM, RAW_COLUMNS, featurize, flip_raw, row_to_raw  # noqa: E402

FEATURE_VERSION = 1


def candidates():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
    }


def build_features(df):
    raw = df[RAW_COLUMNS].to_numpy(dtype=np.float32)
    X, X_flip = np.empty((len(raw), FEATURE_DIM), np.float32), np.empty((len(raw), FEATURE_DIM), np.float32)
    for i, row in enumerate(raw):
        frame = row_to_raw(row)
        X[i] = featurize(*frame)
        X_flip[i] = featurize(*flip_raw(*frame))
    return X, X_flip


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cfg["extraction"]["output_csv"])
    ap.add_argument("--out", default=cfg["training"]["model_path"])
    args = ap.parse_args()

    df = pd.read_csv(resolve(args.data))
    y, sessions = df["label"].to_numpy(), df["session"].to_numpy()
    labels = sorted(np.unique(y))
    print(f"{len(df)} frames, {len(labels)} classes: {', '.join(labels)}")
    print(f"sessions: {', '.join(sorted(np.unique(sessions)))}\n")
    if "none" not in labels:
        print("Warning: no 'none' clips. Without them, any movement may summon a Stand.\n")

    X, X_flip = build_features(df)
    reports_dir = resolve(cfg["training"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    unique_sessions = sorted(np.unique(sessions))
    results = {}
    if len(unique_sessions) < 2:
        print("Only one session found: can't hold one out. Record s2/s3 clips for an honest score.")
    else:
        for name in candidates():
            y_true, y_pred, fold_acc = [], [], {}
            for held in unique_sessions:
                tr, te = sessions != held, sessions == held
                model = candidates()[name]
                model.fit(np.vstack([X[tr], X_flip[tr]]), np.concatenate([y[tr], y[tr]]))
                pred = model.predict(X[te])
                fold_acc[held] = float(accuracy_score(y[te], pred))
                y_true += list(y[te])
                y_pred += list(pred)
            results[name] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "per_session_accuracy": fold_acc,
                "_true": y_true, "_pred": y_pred,
            }
            print(f"{name:20s} accuracy {results[name]['accuracy']:.3f}   macro-F1 {results[name]['macro_f1']:.3f}   "
                  + "  ".join(f"{s}:{a:.2f}" for s, a in fold_acc.items()))

    best = max(results, key=lambda n: results[n]["macro_f1"]) if results else "logistic_regression"
    print(f"\nBest model: {best}")
    if results:
        r = results[best]
        print("\nPer-class results on held-out sessions:")
        print(classification_report(r["_true"], r["_pred"], digits=3, zero_division=0))
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(r["_true"], r["_pred"], labels=labels, ax=ax,
                                                xticks_rotation=45, colorbar=False, cmap="Purples")
        ax.set_title(f"Held-out sessions ({best})")
        fig.tight_layout()
        fig.savefig(reports_dir / "confusion_matrix.png", dpi=150)
        print(f"Confusion matrix saved to {reports_dir / 'confusion_matrix.png'}")
        metrics = {n: {k: v for k, v in m.items() if not k.startswith("_")} for n, m in results.items()}
        metrics["best"] = best
        (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Final model: train on everything (originals + mirrored).
    final = candidates()[best]
    final.fit(np.vstack([X, X_flip]), np.concatenate([y, y]))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "labels": list(final.classes_), "model_name": best,
                 "feature_version": FEATURE_VERSION, "feature_dim": FEATURE_DIM}, out)
    print(f"Model saved to {out}")


if __name__ == "__main__":
    main()
