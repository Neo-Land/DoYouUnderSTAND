"""The pose classifier: candidate models, feature matrices, and which model training picked.

Shared by scripts/poses/train.py (training + leave-one-session-out evaluation) and dyu.live_eval
(replaying clips through the app's logic), so both always use the same features and models.
"""
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import resolve
from .features import FEATURE_DIM, RAW_COLUMNS, featurize, flip_raw, row_to_raw

FEATURE_VERSION = 1


def candidates():
    """Fresh, untrained instances of every model the training script compares."""
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
    }


def build_features(df):
    """Raw landmark rows (a DataFrame with RAW_COLUMNS) -> (features, features of the mirrored frames)."""
    raw = df[RAW_COLUMNS].to_numpy(dtype=np.float32)
    X, X_flip = np.empty((len(raw), FEATURE_DIM), np.float32), np.empty((len(raw), FEATURE_DIM), np.float32)
    for i, row in enumerate(raw):
        frame = row_to_raw(row)
        X[i] = featurize(*frame)
        X_flip[i] = featurize(*flip_raw(*frame))
    return X, X_flip


def best_model_name(cfg):
    """The model scripts/poses/train.py picked last time (from metrics.json), so replays and tuning
    match what the app runs. Falls back to logistic regression if training hasn't run."""
    path = resolve(cfg["training"]["reports_dir"]) / "metrics.json"
    try:
        return json.loads(path.read_text())["best"]
    except (OSError, KeyError, ValueError):
        return "logistic_regression"


def fast_predict_proba(model, X):
    """predict_proba for one frame (or a few) without scikit-learn's per-tree overhead.

    For a random forest, RandomForestClassifier.predict_proba runs each of its trees through joblib
    and re-validates it, which costs ~16 ms per frame for 300 trees even on one core. Calling each
    tree's compiled routine directly and averaging gives the same numbers in a fraction of that.
    Any other model falls back to its own predict_proba.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    if not isinstance(model, RandomForestClassifier):
        return model.predict_proba(X)
    total = None
    for tree in model.estimators_:
        counts = tree.tree_.predict(X)                     # leaf class weights per sample
        if counts.ndim == 3:                               # older scikit-learn: (n, n_outputs, n_classes)
            counts = counts[:, 0, :]
        sums = counts.sum(axis=1, keepdims=True)
        proba = counts / np.where(sums == 0, 1.0, sums)
        total = proba if total is None else total + proba
    return total / len(model.estimators_)
