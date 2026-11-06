# ASL word-sign mode

Recognizes isolated ASL signs (whole words, with their motion) from the same MediaPipe landmarks the JoJo mode uses. This is the part of the project aimed at sign-language recognition research, so it's evaluated the strict way: **every test signer is someone the model never saw during training.**

## Data

**ASL Citizen** (Microsoft Research; Desai et al., *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition*, NeurIPS 2023 Datasets and Benchmarks). The full dataset has about 83,000 videos of 2,731 signs from 52 signers. This project uses **100 everyday signs** (HELLO, THANKYOU, HELP, NAME, UNDERSTAND, …; the list is `signs.vocabulary` in `config.yaml`) with the dataset's official split:

| Split | Videos | Signers |
| --- | --- | --- |
| train | 1,491 | 33 |
| val | 378 | 5 |
| test | 1,261 | 11 |

Videos are mostly 640×480 at 30 fps, median 2.4 s. Each sign has 29–45 videos.

### License

ASL Citizen is released for **non-commercial research only** under a Microsoft Research license, and its signers are identifiable people. In this repo:

- The videos, the landmarks extracted from them and any model trained on them stay local: `datasets/`, `data/` and `models/` are git-ignored.
- Only code, accuracy numbers and per-sign accuracy are published. No dataset frames appear in the README, docs or demo.
- There is no public hosted demo built on it.

## Pipeline

```
ASL Citizen zip
  -> scripts/signs/extract_asl_citizen.py   unpack only the chosen signs' videos (+ subset.csv)
  -> scripts/signs/extract_landmarks.py     MediaPipe on every frame (8 processes, ~35 min, resumable)
  -> dyu/signs.py                           per-frame features, trim to the signing span, resample to 32 frames
  -> scripts/signs/train.py                 classifier on the official split, scored on the test signers
```

**Per-frame features** (110 numbers, `dyu/signs.py`):

| Part | Numbers |
| --- | --- |
| 11 upper-body points: nose, eyes, mouth corners, shoulders, elbows, wrists | 22 |
| Each hand: 21 points + a present flag | 2 × 43 |
| Each hand: wrist speed between frames | 2 |

Everything is in **body coordinates**: centred on the shoulder midpoint and divided by shoulder width. Unlike the JoJo features (where each hand is centred on its own wrist), this keeps *where* a hand is relative to the face and body, which is what separates signs like MOTHER (chin) and FATHER (forehead).

**Trimming.** Signers start and end with their hands resting below the frame, so only the frames with a visible hand (±3) are kept. Across all videos a hand is found in 44.5% of frames; on inspection that matches the signing portion, not missed detections.

**Resampling** to 32 frames gives every sign the same length. Training adds left/right mirrored copies, so left-handed signing works too.

## Results

Test set: 1,261 videos from 11 signers no model saw. Chance is 1%.

| Model | Test top-1 | Test top-5 | Notes |
| --- | --- | --- | --- |
| Logistic regression on the flattened 32-frame sequence | **70.1%** | **89.5%** | C picked on the validation signers (73.5% top-1), refit on train + val |

- **Easiest signs:** PLAY, HEARING (93%), WHEN, WORKOUT, WATER (92%).
- **Hardest:** MUSIC (33%), MOVIE (41%), COME, LOVE, HURT, NEED, THANKYOU, POWER (42%). Many of these differ mainly in motion, which a flat linear model only partly captures.

Numbers: `reports/signs/baseline.json`; per sign: `reports/signs/baseline_per_sign.csv`.

## Reproducing

```powershell
# 1. Download ASL Citizen from Microsoft Research and put the zip at datasets/asl_citizen/ASL_Citizen.zip
python scripts/signs/extract_asl_citizen.py   # unpacks only the 100 signs (~2 GB of ~46 GB)
python scripts/signs/extract_landmarks.py     # ~35 min on 8 processes; safe to stop and re-run
python scripts/signs/train.py                 # ~1.5 min
```

To change the vocabulary, edit `signs.vocabulary` in `config.yaml` and re-run all three.

## Next

1. A GRU sequence model over the 32 frames, to model motion and order directly.
2. A live word-sign mode in `app.py`: buffer recent frames, detect when a sign starts and ends, show the top-5 predictions.
3. A larger vocabulary (300, 1,000, all 2,731 signs) to show how accuracy scales.
