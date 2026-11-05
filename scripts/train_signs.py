"""ASL step 2: baseline word-sign classifier on ASL Citizen landmarks, signer-independent.

Usage:
    python scripts/train_signs.py

Needs data/asl_signs/ from scripts/extract_signs.py. Uses ASL Citizen's official splits, which put
every signer in exactly one of train / val / test, so test accuracy is on people the model has never
seen. Each video becomes a fixed-length sequence (dyu.signs.sequence), training adds left/right
mirrored copies, and regularization is picked on the validation signers. The final model is refit on
train + val and scored once on test.

Writes:
  reports/signs_baseline.json         top-1 / top-5 on val and test (numbers only: OK to publish)
  reports/signs_baseline_per_sign.csv test accuracy per sign
  models/sign_baseline.joblib         the model (trained on ASL Citizen: never publish it)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from dyu.config import ROOT  # noqa: E402
from dyu.signs import SEQ_LEN, mirror_sequence, sequence  # noqa: E402

DATA = ROOT / "data" / "asl_signs"
REPORTS = ROOT / "reports"
MODEL = ROOT / "models" / "sign_baseline.joblib"


def load():
    index = pd.read_csv(DATA / "index.csv")
    seqs = np.stack([sequence(np.load(DATA / (Path(f).stem + ".npy"))) for f in index.file])
    return index, seqs


def topk(model, X, y, k):
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    top = classes[np.argsort(-proba, axis=1)[:, :k]]
    return float(np.mean([t in row for t, row in zip(y, top)]))


def fit(X, y, C):
    Xm = np.concatenate([X, np.stack([mirror_sequence(s) for s in X])])
    ym = np.concatenate([y, y])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    model.fit(Xm.reshape(len(Xm), -1), ym)
    return model


def main():
    t0 = time.time()
    index, seqs = load()
    y = index.gloss.to_numpy()
    split = index.split.to_numpy()
    flat = seqs.reshape(len(seqs), -1)
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"{len(index)} videos, {index.gloss.nunique()} signs, {SEQ_LEN} frames each "
          f"({seqs.shape[2]} features per frame)")
    print(f"train {tr.sum()} videos / {index[tr].participant.nunique()} signers, "
          f"val {va.sum()} / {index[va].participant.nunique()}, test {te.sum()} / {index[te].participant.nunique()}")

    best = None
    for C in (0.003, 0.01, 0.03, 0.1):
        m = fit(seqs[tr], y[tr], C)
        acc1, acc5 = topk(m, flat[va], y[va], 1), topk(m, flat[va], y[va], 5)
        print(f"  C={C:<6} val top-1 {acc1:.1%}  top-5 {acc5:.1%}", flush=True)
        if best is None or acc1 > best[1]:
            best = (C, acc1, acc5)
    C, val1, val5 = best

    final = fit(seqs[tr | va], y[tr | va], C)
    test1, test5 = topk(final, flat[te], y[te], 1), topk(final, flat[te], y[te], 5)
    pred = final.predict(flat[te])
    per_sign = (pd.DataFrame({"gloss": y[te], "correct": pred == y[te]})
                .groupby("gloss").correct.agg(["mean", "count"]).rename(columns={"mean": "accuracy", "count": "videos"})
                .sort_values("accuracy"))
    chance = 1 / index.gloss.nunique()
    print(f"\nLogistic regression on {SEQ_LEN}-frame sequences, C={C} (picked on val)")
    print(f"  val  top-1 {val1:.1%}  top-5 {val5:.1%}")
    print(f"  TEST top-1 {test1:.1%}  top-5 {test5:.1%}   (unseen signers; chance {chance:.1%})")
    print("\nHardest signs on test:")
    print(per_sign.head(10).to_string(float_format=lambda v: f"{v:.0%}"))
    print("\nEasiest:")
    print(per_sign.tail(5).to_string(float_format=lambda v: f"{v:.0%}"))

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "signs_baseline.json").write_text(json.dumps({
        "model": "logistic_regression", "C": C, "seq_len": SEQ_LEN, "signs": int(index.gloss.nunique()),
        "videos": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "val": {"top1": val1, "top5": val5}, "test": {"top1": test1, "top5": test5},
    }, indent=2))
    per_sign.to_csv(REPORTS / "signs_baseline_per_sign.csv")
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "classes": list(final.classes_), "seq_len": SEQ_LEN}, MODEL)
    print(f"\nSaved reports/signs_baseline.json and {MODEL.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
