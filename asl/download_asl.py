"""Download the Kaggle "ASL Alphabet" dataset (grassknoted/asl-alphabet, ~1 GB) into datasets/.

Needs a Kaggle API token: kaggle.com -> Settings -> API -> "Create New Token", then save the
downloaded kaggle.json to  C:\\Users\\<you>\\.kaggle\\kaggle.json  (Mac: ~/.kaggle/kaggle.json).

Already downloaded the zip by hand? Put it at datasets/asl-alphabet.zip and run this script;
it will just unzip it.
"""
import sys
import zipfile

import config

DATASET = "grassknoted/asl-alphabet"
OUT = config.DATASETS_DIR / "asl_alphabet"


def find_train_dir():
    """The zip nests folders (asl_alphabet_train/asl_alphabet_train/A/...). Find the one holding A/."""
    for d in OUT.rglob("A"):
        if d.is_dir() and (d.parent / "B").is_dir() and "test" not in str(d.parent).lower():
            return d.parent
    return None


def main():
    if find_train_dir():
        print(f"Already have the dataset: {find_train_dir()}")
        return
    OUT.mkdir(parents=True, exist_ok=True)

    manual_zip = config.DATASETS_DIR / "asl-alphabet.zip"
    if not manual_zip.exists():
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        except Exception as e:
            sys.exit(f"Kaggle login failed ({e}).\nSee the instructions at the top of this file.")
        print(f"Downloading {DATASET} (~1 GB)...")
        api.dataset_download_files(DATASET, path=str(config.DATASETS_DIR), quiet=False, unzip=False)

    print("Unzipping...")
    with zipfile.ZipFile(manual_zip) as z:
        z.extractall(OUT)
    train = find_train_dir()
    if not train:
        sys.exit(f"Unzipped, but couldn't find the letter folders under {OUT}.")
    print(f"Done. Training images in: {train}")


if __name__ == "__main__":
    main()
