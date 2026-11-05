# Do You UnderSTAND — context for Claude

## What this is
A webcam pose-recognition app. The user strikes a JoJo's Bizarre Adventure pose; the app
recognizes it and summons that character's Stand behind them (image composited behind the
person, menacing "ゴゴゴ" text, sound effect). A second mode recognizes ASL signs from hand
landmarks. The name is a pun: understanding ASL + JoJo Stands.

Purpose: portfolio project for a Microsoft SWE Intern – AI/ML & LLM application (sign-language
team, summer 2027). Shipping a working demo fast matters more than polish. The ASL mode is the
part most relevant to the role, so it should end up solid, not a token feature.

## Owner / environment
- Neo, CS junior at Montclair State. Concise, action-oriented answers preferred.
- Main dev machine: Windows desktop, Python 3.11. No webcam on the desktop yet; poses may be
  recorded as videos on a MacBook and processed on the desktop with `record.py --video`.

## Stack
- Python 3.11, MediaPipe Tasks API (Pose Landmarker + Hand Landmarker, 0.10.35 in .venv).
  Model `.task` files download into `models/` on first run.
- OpenCV, Pillow, pygame; pandas + scikit-learn (logistic regression vs random forest, best
  held-out score wins); PyYAML config; pytest (CI runs it via `.github/workflows/tests.yml`).
- `asl/` is the OLD scaffold (legacy `mp.solutions`, MediaPipe 0.10.14 + numpy<2). It does not
  run in the current .venv; port it to the Tasks API / `dyu` before using it. Exception:
  `asl/extract_asl_citizen.py` needs no MediaPipe and works.

## Layout
- `config.yaml`       — camera, extraction, recognition, effects settings, the six Stands
- `dyu/landmarks.py`  — MediaPipe Tasks wrapper (`Detector`): body + hands per frame, person mask
- `dyu/features.py`   — raw row layout, featurize (hips/torso normalization, hands per wrist), flip
- `dyu/clips.py`, `dyu/smoothing.py`, `dyu/effects.py`, `dyu/config.py`
- `scripts/extract_landmarks.py` — `data/videos/<pose>_s<session>.*` → `data/landmarks.csv`
- `scripts/train.py`  — leave-one-session-out eval → `models/pose_classifier.joblib`, `reports/`
- `app.py`            — live demo (webcam or `--video`)
- `tests/`            — pytest unit tests for features/clips/smoothing
- `reports/`          — committed metrics.json + confusion matrix from the latest training run
- `docs/DEVLOG.md`    — dated log of experiments and before/after numbers
- `asl/`              — legacy ASL fingerspelling code + ASL Citizen subset extraction (see Stack)
- `scripts/train_asl.ps1` — legacy fingerspelling pipeline (runs asl/; needs porting)
- `dyu/signs.py`     — sign features: body-relative hands + upper body per frame, trim, resample
- `scripts/extract_signs.py` — ASL Citizen subset videos -> per-frame landmarks in data/asl_signs/
- `scripts/train_signs.py`   — word-sign classifier on the official split (unseen test signers)
- `assets/stands/`, `assets/sounds/` — user-supplied, gitignored
- `data/`, `models/`, `datasets/` — generated or downloaded; gitignored

## Poses (label → Stand)
jotaro → Star Platinum · giorno → Gold Experience · josuke → Crazy Diamond ·
bucciarati → Sticky Fingers · diego → Scary Monsters · kira → Killer Queen ·
plus `none` (normal movement, not posing) and `dismiss` (both arms up in a Y: sends the Stand away).
Clips are named `<pose>_s<session>`: s1–s3 phone, s4 MacBook webcam, s5 dismiss only.
Hold a pose ~1.5 s to summon; the Stand stays until dismiss is held ~1 s (dyu/smoothing.py).

## Commands (run from the repo root, venv active)
- Setup (Windows): `./setup.ps1`  (Mac/Linux: `./setup.sh`)
- Extract: `python scripts/extract_landmarks.py`  (prints per-clip frames and hand rates)
- Train: `python scripts/train.py`
- Run: `python app.py` or `python app.py --video data/videos/jotaro_s1.mp4 --debug`
- Tests: `python -m pytest -q`

## Conventions
- Evaluation is leave-one-session-out; never report random-frame splits (near-duplicate frames).
- Hands are assigned by distance to the pose wrists, not MediaPipe's handedness label.
  Training adds left/right flipped copies, so the camera mirror setting doesn't matter.
- If the feature layout in `dyu/features.py` changes, re-run extraction; old CSVs/models are invalid.
- Log every experiment in `docs/DEVLOG.md` and keep the README results section current.
- Don't commit videos, `data/*.csv`, `models/`, or Stand art/sounds (all gitignored).

## ASL Citizen license rules (Microsoft Research License — must follow)
Dataset lives in `datasets/asl_citizen/` (gitignored). Non-commercial research use only.
- NEVER commit or publish: the videos, frames/clips of signers, landmarks extracted from the
  videos, or models trained on it. Keep all of it under `datasets/`, `data/` or `models/`.
- Signers are identifiable people: no dataset frames in the demo video, README or slides.
- No public hosted demo built on it. Local runs + a demo video of Neo only.
- OK to publish: code, accuracy numbers/charts, the write-up, and a citation of the paper.
- Delete the data when the project's research is done.

## Roadmap
### Part 1 — JoJo poses
1. [x] Record 6 poses + none, 3 sessions each (data/videos)
2. [ ] Raise accuracy: 81.1% → 91.2% / macro-F1 0.927 with the wrist-crop hand fallback
       (2026-11-04). Weak spot now: `none` → Bucciarati (131 frames, mostly none_s2)
3. [x] Stand effects: art in assets/stands/, bottom fade, per-Stand placement in config.yaml,
       person mask from the selfie segmenter on a body crop (Pose Landmarker's float mask crashes
       on 0.10.35). App ~20 fps at 1280x720 (models on a thread pool). Placement tuned for 16:9 with
       `python scripts/render_stand_previews.py --landscape`; confirm on a live webcam
4. [x] Hold to summon / stay until dismissed (2026-11-05): 0 false summons, 24/24 pose clips summon
       (median 1.8 s), dismiss 2/2. Tune with scripts/eval_live.py + scripts/tune_smoothing.py

### Part 2 — ASL
5. [ ] Fingerspelling baseline: Kaggle ASL Alphabet via scripts/train_asl.ps1, then add a few
       hundred of Neo's own webcam samples per letter (dataset is one signer)
6. [ ] Word signs on ASL Citizen, 100 signs (asl/config.py ASL_WORD_SIGNS). Done: landmarks for all
       3,130 videos (scripts/extract_signs.py -> data/asl_signs/), features in dyu/signs.py, baseline
       logistic regression 70.1% top-1 / 89.5% top-5 on unseen test signers (scripts/train_signs.py).
       Next: GRU sequence model, then a bigger vocabulary
7. [ ] Add word-sign mode to app.py alongside fingerspelling

### Ship
8. [ ] Record a 60–90s demo video + README write-up for the application
