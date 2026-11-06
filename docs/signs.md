# ASL word-sign mode

Recognizes isolated ASL signs (whole words, with their motion) from the same MediaPipe landmarks the JoJo mode uses. This is the part of the project aimed at sign-language recognition research, so it's evaluated the strict way: **every test signer is someone the model never saw during training.**

## Data

**ASL Citizen** (Microsoft Research; Desai et al. 2023, *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition*, arXiv:2304.05934; full citation [below](#citation)). It is a crowdsourced dataset of about 84,000 videos of 2,731 signs from 52 signers recording themselves with everyday webcams, collected with each contributor's consent. The authors frame isolated sign recognition as **dictionary retrieval**: someone signs in front of an ordinary camera and gets back a ranked list of the closest signs, which is what this project's live mode does with its top-5 list. This project uses **100 everyday signs** (HELLO, THANKYOU, HELP, NAME, UNDERSTAND, …; the list is `signs.vocabulary` in `config.yaml`) with the dataset's official split:

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
| GRU, live pipeline (sign cut out by the app's segmenter) | 84.1% | 96.3% | a sign was detected in 98.7% of test videos |

**How this compares with the paper.** The ASL Citizen authors' baselines reach about 62% top-1 on the full 2,731-sign vocabulary. These results are on a 100-sign subset, a much easier task (27 times fewer signs to choose between), so they are **not comparable** and don't claim to beat those baselines. Scaling the same pipeline to the full vocabulary is the way to make a like-for-like comparison.

Both models are chosen on the 5 validation signers only, refit on train + val, and scored once on test. The GRU's three seeds land within 0.3 points of each other, so the gain over the baseline (+14 points top-1) isn't a lucky run.

**Why the GRU helps.** The baseline flattens the 32 frames into one long vector, so it learns *where* the hands are at each moment but not how they move. The GRU reads the frames in order and keeps a running memory, so motion and its direction become usable. The signs that gained most were the ones the baseline found hardest, which differ mainly in movement: MUSIC 33% → 67%, LOVE 42% → 83%, COME 42% → 83%.

**Still hardest for the GRU:** GOOD (42%, down from 75% with the baseline; 7 of its 12 test videos are read as THANKYOU), SCHOOL (45%), POWER, THANKYOU (50%), WHAT (54%). GOOD and THANKYOU are a near pair in ASL (both start at the chin and move out), which a 100-sign vocabulary puts side by side.

**Live pipeline.** In the app nobody cuts the clip for the model. `scripts/signs/eval_live.py` replays every test video frame by frame through the same segmenter and recognizer the app uses: the sign is found in 98.7% of videos, and accuracy drops by only 0.5 points against perfectly cut clips.

Numbers: `reports/signs/logreg.json`, `gru.json`, `live.json`; per sign: `reports/signs/*_per_sign.csv`.

## Live sign mode

```powershell
python app.py --mode signs            # or python app.py and press a; --video <clip> works too, --debug shows the skeleton
```

**Sit close**: your head and chest should fill most of the frame, like the signers in ASL Citizen (their shoulders span about half the frame height; the app says "Move closer" below 0.40). From the JoJo distance your hands are a third of that size, and signs made in the same place with different handshapes get confused (WATER / MOTHER, SICK / LEARN). Start with your hands down, sign one word, then lower your hands: the top 5 guesses appear on the right and the best one is added to a transcript at the bottom (`c` clears it). It needs `models/sign_gru.pt` (or `sign_logreg.joblib`) from training, which stays local.

**Learn ASL** (`python app.py --mode practice`, or `p` in the menu) turns the same pipeline into practice: the app names a word, you sign it, and it grades the attempt. Green means the word was the model's top guess, yellow means it was in the top 5 (with its rank), red shows what the sign looked like instead. A correct sign moves on to the next word; `n` skips. Words come easiest first, ordered by the model's per-sign accuracy on the test signers (`reports/signs/gru_per_sign.csv`), so beginners get early successes. The app shows only the word: per the dataset license it never displays ASL Citizen videos, and it points to public dictionaries (Lifeprint, HandSpeak) for how a sign looks. Scoring logic: `dyu/practice.py`, unit-tested.

How the stream is cut into signs (`dyu/sign_live.py`):

- A frame counts as **signing** when a hand is visible and its wrist is above a rest line, 1.0 shoulder widths below the shoulders. Measured on ASL Citizen, wrists stay within 0.87 shoulder widths below the shoulders in 99% of signing frames, while hands resting on a desk or in a lap sit lower.
- 3 signing frames in a row start a sign (a few frames before are kept so the start isn't clipped); 0.4 s without signing ends it. Signs shorter than 0.3 s are ignored as twitches; longer than 4 s are cut.
- The sign is trimmed to the frames with a hand above the rest line, then goes through the same `sequence()` as training (resampled to 32 frames). Scores are averaged over the sign and its mirror image. On the test signers both steps are neutral (84.1% live top-1 vs 83.9% without); they're there for live clips, which can include resting hands.

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

## Citation

The dataset's authors ask that work using ASL Citizen cite their paper:

```bibtex
@article{desai2023asl,
  title={ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition},
  author={Desai, Aashaka and Berger, Lauren and Minakov, Fyodor O and Milan, Vanessa and Singh, Chinmay and Pumphrey, Kriston and Ladner, Richard E and Daum{\'e} III, Hal and Lu, Alex X and Caselli, Naomi and Bragg, Danielle},
  journal={arXiv preprint arXiv:2304.05934},
  year={2023}
}
```

Dataset and license: Microsoft Research's ASL Citizen page (download via the Microsoft Download Center); the authors' code is at github.com/microsoft/ASL-citizen-code.
