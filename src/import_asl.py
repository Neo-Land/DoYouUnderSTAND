"""Turn a folder of labeled ASL images into hand-landmark samples in data/asl.csv.

python src/import_asl.py                       # uses the downloaded Kaggle dataset
python src/import_asl.py --src path/to/folder  # any folder laid out as <label>/<images>
python src/import_asl.py --per-class 500       # faster, fewer samples

Images where MediaPipe finds no hand are skipped (the "nothing" class is skipped entirely).
Appends to data/asl.csv, so your own webcam recordings (record.py --mode asl) can go in too.
Use --fresh to start the CSV over.
"""
import argparse
import csv
import sys
import time

import cv2

import config
from download_asl import find_train_dir
from landmarks import FEATURE_DIM, Extractor

SKIP = {"nothing"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="folder with one subfolder per label")
    ap.add_argument("--per-class", type=int, default=1500, help="max images per label (default 1500)")
    ap.add_argument("--fresh", action="store_true", help="overwrite data/asl.csv instead of appending")
    args = ap.parse_args()

    from pathlib import Path
    src = Path(args.src) if args.src else find_train_dir()
    if not src or not src.is_dir():
        sys.exit("No dataset found. Run: python src/download_asl.py")

    out = config.data_path("asl")
    out.parent.mkdir(parents=True, exist_ok=True)
    new = args.fresh or not out.exists() or out.stat().st_size == 0
    f = open(out, "w" if args.fresh else "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["label"] + [f"f{i}" for i in range(FEATURE_DIM["asl"])])

    ex = Extractor("asl", static=True)
    start = time.time()
    total_saved = 0
    try:
        for d in sorted(p for p in src.iterdir() if p.is_dir()):
            raw = d.name
            if raw.lower() in SKIP:
                continue
            label = raw.upper() if len(raw) == 1 else raw.lower()   # A..Z, "space", "del"
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)[: args.per_class]
            saved = 0
            for p in imgs:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                feats, _ = ex.process(img)
                if feats is None:
                    continue
                w.writerow([label] + [f"{v:.5f}" for v in feats])
                saved += 1
            total_saved += saved
            rate = saved / max(len(imgs), 1)
            print(f"{label:>6}: {saved}/{len(imgs)} usable ({rate:.0%})  [{time.time() - start:.0f}s]", flush=True)
    finally:
        f.close()
        ex.close()
    print(f"Saved {total_saved} samples to {out}")


if __name__ == "__main__":
    main()
