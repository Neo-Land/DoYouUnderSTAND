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
  -> scripts/signs/train.py                 logistic regression or GRU, official split, scored on the test signers
  -> dyu/sign_live.py + app.py --mode signs live: cut the webcam stream into signs, classify each
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
| Logistic regression on the flattened 32-frame sequence | 70.1% | 89.5% | C picked on the validation signers |
| **GRU sequence model** | **84.4%** ± 0.1% | **97.1%** | mean of 3 seeds; epochs picked by early stopping on the validation signers |
| GRU, live pipeline (sign cut out by the app's segmenter) | 83.9% | 96.2% | a sign was detected in 98.7% of test videos |

Both models are chosen on the 5 validation signers only, refit on train + val, and scored once on test. The GRU's three seeds land within 0.3 points of each other, so the gain over the baseline (+14 points top-1) isn't a lucky run.

**Why the GRU helps.** The baseline flattens the 32 frames into one long vector, so it learns *where* the hands are at each moment but not how they move. The GRU reads the frames in order and keeps a running memory, so motion and its direction become usable. The signs that gained most were the ones the baseline found hardest, which differ mainly in movement: MUSIC 33% → 67%, LOVE 42% → 83%, COME 42% → 83%.

**Still hardest for the GRU:** GOOD (42%, down from 75% with the baseline; 7 of its 12 test videos are read as THANKYOU), SCHOOL (45%), POWER, THANKYOU (50%), WHAT (54%). GOOD and THANKYOU are a near pair in ASL (both start at the chin and move out), which a 100-sign vocabulary puts side by side.

**Live pipeline.** In the app nobody cuts the clip for the model. `scripts/signs/eval_live.py` replays every test video frame by frame through the same segmenter and recognizer the app uses: the sign is found in 98.7% of videos, and accuracy drops by only 0.5 points against perfectly cut clips.

Numbers: `reports/signs/logreg.json`, `gru.json`, `live.json`; per sign: `reports/signs/*_per_sign.csv`.

## Live sign mode

```powershell
python app.py --mode signs            # webcam; --video <clip> works too, --debug shows the skeleton
```

Sit so your head and shoulders are in frame. Start with your hands down, sign one word, then lower your hands: the top 5 guesses appear on the right and the best one is added to a transcript at the bottom (`c` clears it). It needs `models/sign_gru.pt` (or `sign_logreg.joblib`) from training, which stays local.

How the stream is cut into signs (`dyu/sign_live.py`):

- A frame counts as **signing** when a hand is visible and its wrist is above a rest line, 1.0 shoulder widths below the shoulders. Measured on ASL Citizen, wrists stay within 0.87 shoulder widths below the shoulders in 99% of signing frames, while hands resting on a desk or in a lap sit lower.
- 3 signing frames in a row start a sign (a few frames before are kept so the start isn't clipped); 0.4 s without signing ends it. Signs shorter than 0.3 s are ignored as twitches; longer than 4 s are cut.
- The sign's frames go through the same `sequence()` as training: trimmed to the frames with a visible hand, resampled to 32 frames.

## Reproducing

```powershell
# 1. Download ASL Citizen from Microsoft Research and put the zip at datasets/asl_citizen/ASL_Citizen.zip
python scripts/signs/extract_asl_citizen.py   # unpacks only the 100 signs (~2 GB of ~46 GB)
python scripts/signs/extract_landmarks.py     # ~35 min on 8 processes; safe to stop and re-run
python scripts/signs/train.py --model logreg  # baseline, ~2.5 min
python scripts/signs/train.py --model gru     # GRU, 3 seeds, ~25 min on CPU (needs torch)
python scripts/signs/eval_live.py             # the live pipeline on the test signers
```

To change the vocabulary, edit `signs.vocabulary` in `config.yaml` and re-run all three.

## Next

1. A larger vocabulary (300, 1,000, all 2,731 signs) to show how accuracy scales.
2. Continuous signing (several signs without lowering the hands in between), which needs a model trained on sign boundaries.
