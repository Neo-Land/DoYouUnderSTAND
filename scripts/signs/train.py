"""ASL step 2: train a word-sign classifier on ASL Citizen landmarks, scored on unseen signers.

Usage:
    python scripts/signs/train.py --model logreg            # baseline: logistic regression
    python scripts/signs/train.py --model gru               # GRU sequence model (needs torch), 3 seeds
    python scripts/signs/train.py --model gru --seeds 0     # one seed, faster

Needs data/asl_signs/ from scripts/signs/extract_landmarks.py. Uses ASL Citizen's official splits,
which put every signer in exactly one of train / val / test, so test accuracy is on people the model
has never seen. Each video becomes a fixed-length sequence (dyu.signs.sequence).

Protocol (same for both models): settings are chosen on the validation signers only (logreg: the
regularization C; gru: the number of epochs, by early stopping); the final model is refit on
train + val with those settings and scored once on test. The GRU is trained with several seeds and
reported as mean ± std, since one neural-network run can be lucky or unlucky.

Writes:
  reports/signs/<model>.json            top-1 / top-5 on val and test (numbers only: OK to publish)
  reports/signs/<model>_per_sign.csv    test accuracy per sign
  models/sign_<model>.joblib / .pt      the model (trained on ASL Citizen: never publish it)
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dyu.config import ROOT, load_config, resolve
from dyu.signs import SEQ_LEN, mirror_sequence, sequence

_CFG = load_config()["signs"]
DATA = resolve(_CFG["landmarks_dir"])
REPORTS = resolve(_CFG["reports_dir"])
MODELS = resolve(_CFG["models_dir"])


def load():
    """All videos as (N, SEQ_LEN, FRAME_DIM) sequences, plus the index (gloss, split, signer)."""
    index = pd.read_csv(DATA / "index.csv")
    seqs = np.stack([sequence(np.load(DATA / (Path(f).stem + ".npy"))) for f in index.file])
    return index, seqs


def top_k(proba, classes, y, k):
    top = np.asarray(classes)[np.argsort(-proba, axis=1)[:, :k]]
    return float(np.mean([t in row for t, row in zip(y, top, strict=True)]))


# ---------------------------------------------------------------- logistic regression baseline

def fit_logreg(X, y, C):
    Xm = np.concatenate([X, np.stack([mirror_sequence(s) for s in X])])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    model.fit(Xm.reshape(len(Xm), -1), np.concatenate([y, y]))
    return model


def run_logreg(seqs, y, tr, va, te):
    flat = seqs.reshape(len(seqs), -1)
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        m = fit_logreg(seqs[tr], y[tr], C)
        p = m.predict_proba(flat[va])
        acc1, acc5 = top_k(p, m.classes_, y[va], 1), top_k(p, m.classes_, y[va], 5)
        print(f"  C={C:<5} val top-1 {acc1:.1%}  top-5 {acc5:.1%}", flush=True)
        if best is None or acc1 > best[1]:
            best = (C, acc1, acc5)
    C, val1, val5 = best
    final = fit_logreg(seqs[tr | va], y[tr | va], C)
    p = final.predict_proba(flat[te])
    result = {"settings": {"C": C}, "val": {"top1": val1, "top5": val5},
              "test": {"top1": top_k(p, final.classes_, y[te], 1), "top5": top_k(p, final.classes_, y[te], 5)}}
    pred = final.classes_[p.argmax(1)]
    MODELS.mkdir(exist_ok=True)
    path = MODELS / "sign_logreg.joblib"
    joblib.dump({"model": final, "classes": list(final.classes_), "seq_len": SEQ_LEN}, path)
    return result, pred, path


# ---------------------------------------------------------------- GRU sequence model

def run_gru(seqs, y, tr, va, te, seeds):
    from dyu.sign_model import GRUConfig, SignClassifier

    runs, saved, pred0 = [], None, None
    for seed in seeds:
        cfg = GRUConfig(seed=seed)
        print(f"  seed {seed}: training on train, early stopping on val", flush=True)
        m = SignClassifier(cfg).fit(seqs[tr], y[tr], seqs[va], y[va])
        epochs = m.best_epoch
        val1, val5 = m.top_k(seqs[va], y[va], 1), m.top_k(seqs[va], y[va], 5)
        print(f"  seed {seed}: best epoch {epochs}, val top-1 {val1:.1%} top-5 {val5:.1%}; refitting on train + val",
              flush=True)
        final = SignClassifier(cfg).fit(seqs[tr | va], y[tr | va], epochs=epochs, log=None)
        test1, test5 = final.top_k(seqs[te], y[te], 1), final.top_k(seqs[te], y[te], 5)
        print(f"  seed {seed}: TEST top-1 {test1:.1%} top-5 {test5:.1%}", flush=True)
        runs.append({"seed": seed, "epochs": epochs, "val_top1": val1, "val_top5": val5,
                     "test_top1": test1, "test_top5": test5})
        if saved is None:  # keep the first seed's model and predictions
            MODELS.mkdir(exist_ok=True)
            saved = MODELS / "sign_gru.pt"
            final.save(saved)
            pred0 = final.predict(seqs[te])

    r = pd.DataFrame(runs)
    result = {"settings": {**GRUConfig().__dict__, "seed": list(seeds)}, "runs": runs,
              "val": {"top1": float(r.val_top1.mean()), "top5": float(r.val_top5.mean())},
              "test": {"top1": float(r.test_top1.mean()), "top5": float(r.test_top5.mean()),
                       "top1_std": float(r.test_top1.std(ddof=0)), "top5_std": float(r.test_top5.std(ddof=0))}}
    return result, pred0, saved


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["logreg", "gru"], default="gru")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="GRU only")
    args = ap.parse_args()

    t0 = time.time()
    index, seqs = load()
    y, split = index.gloss.to_numpy(), index.split.to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"{len(index)} videos, {index.gloss.nunique()} signs, {SEQ_LEN} frames x {seqs.shape[2]} features")
    print(f"train {tr.sum()} videos / {index[tr].participant.nunique()} signers, "
          f"val {va.sum()} / {index[va].participant.nunique()}, test {te.sum()} / {index[te].participant.nunique()}\n")

    if args.model == "logreg":
        result, pred, model_path = run_logreg(seqs, y, tr, va, te)
    else:
        result, pred, model_path = run_gru(seqs, y, tr, va, te, args.seeds)

    test, chance = result["test"], 1 / index.gloss.nunique()
    spread = f" (± {test['top1_std']:.1%} over {len(args.seeds)} seeds)" if "top1_std" in test else ""
    print(f"\n{args.model}: val top-1 {result['val']['top1']:.1%}  top-5 {result['val']['top5']:.1%}")
    print(f"{args.model}: TEST top-1 {test['top1']:.1%}{spread}  top-5 {test['top5']:.1%}   "
          f"(unseen signers; chance {chance:.1%})")

    per_sign = (pd.DataFrame({"gloss": y[te], "correct": pred == y[te]})
                .groupby("gloss").correct.agg(["mean", "count"])
                .rename(columns={"mean": "accuracy", "count": "videos"}).sort_values("accuracy"))
    print("\nHardest signs on test:")
    print(per_sign.head(8).to_string(float_format=lambda v: f"{v:.0%}"))

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "seq_len": SEQ_LEN, "signs": int(index.gloss.nunique()),
              "videos": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}, **result}
    (REPORTS / f"{args.model}.json").write_text(json.dumps(report, indent=2))
    per_sign.to_csv(REPORTS / f"{args.model}_per_sign.csv")
    print(f"\nSaved {(REPORTS / f'{args.model}.json').relative_to(ROOT)} and {model_path.relative_to(ROOT)} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
