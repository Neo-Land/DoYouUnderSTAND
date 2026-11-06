"""ASL step 0: unpack only the videos of the chosen signs from the full ASL Citizen zip (~46 GB).

Usage:
    python scripts/signs/extract_asl_citizen.py              # the core 100 signs
    python scripts/signs/extract_asl_citizen.py --vocab 300  # core + extra_vocabulary

The signs are `signs.vocabulary` in config.yaml; paths are under `signs:` too.
Expects datasets/asl_citizen/ASL_Citizen.zip. Writes:
  datasets/asl_citizen/ASL_Citizen/splits/*.csv       (full splits, small)
  datasets/asl_citizen/ASL_Citizen/videos/*.mp4       (chosen signs only, ~2 GB for 100 signs)
  datasets/asl_citizen/subset.csv                     (split, participant, file, gloss for those videos)

Safe to re-run: videos already on disk with the right size are skipped. Edit the vocabulary and
re-run to add signs. License: non-commercial research only, nothing here may be committed.
"""
import argparse
import csv
import io
import sys
import zipfile

from dyu.config import load_config, resolve
from dyu.signs import VOCAB_SIZES, vocabulary

PREFIX = "ASL_Citizen/"
SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", type=int, choices=VOCAB_SIZES, default=100)
    args = ap.parse_args()
    cfg = load_config()["signs"]
    zpath = resolve(cfg["dataset_zip"])
    if not zpath.exists():
        sys.exit(f"Not found: {zpath}\nDownload ASL Citizen and put the zip there.")
    out = resolve(cfg["dataset_dir"])
    wanted = set(vocabulary(cfg, args.vocab))

    with zipfile.ZipFile(zpath) as z:
        rows = []
        for split in SPLITS:
            name = f"{PREFIX}splits/{split}.csv"
            z.extract(name, out)
            with z.open(name) as f:
                for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                    if r["Gloss"] in wanted:
                        rows.append({"split": split, "participant": r["Participant ID"],
                                     "file": r["Video file"], "gloss": r["Gloss"]})

        missing = wanted - {r["gloss"] for r in rows}
        if missing:
            sys.exit(f"Glosses not in ASL Citizen: {sorted(missing)}")

        infos = {i.filename: i for i in z.infolist()}
        done = skipped = 0
        absent = []
        for k, r in enumerate(rows, 1):
            info = infos.get(f"{PREFIX}videos/{r['file']}")
            if info is None:
                absent.append(r["file"])
                continue
            dest = out / info.filename
            if dest.exists() and dest.stat().st_size == info.file_size:
                skipped += 1
            else:
                z.extract(info, out)
                done += 1
            if k % 250 == 0:
                print(f"  {k}/{len(rows)}")

    rows = [r for r in rows if r["file"] not in set(absent)]
    with open(out / "subset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "participant", "file", "gloss"])
        w.writeheader()
        w.writerows(rows)

    print(f"Extracted {done}, already present {skipped}, missing from zip {len(absent)}.")
    for split in SPLITS:
        n = sum(r["split"] == split for r in rows)
        p = len({r["participant"] for r in rows if r["split"] == split})
        print(f"  {split:5s}: {n} videos, {p} signers")
    print(f"Wrote {out / 'subset.csv'}")


if __name__ == "__main__":
    main()
