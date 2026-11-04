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
- Python 3.11, MediaPipe 0.10.14 (legacy `mp.solutions` API: pose, hands, selfie_segmentation)
- OpenCV for capture/display, Pillow for text rendering, pygame for sound
- scikit-learn RandomForest on normalized landmark features, saved with joblib
- Versions are pinned in requirements.txt (mediapipe 0.10.14 needs numpy < 2). Don't bump
  mediapipe without checking that `mp.solutions` still exists in the new version.

## Layout
- `src/config.py`     — paths, the six poses and their Stand assets, thresholds
- `src/landmarks.py`  — MediaPipe wrapper + feature extraction/normalization (pose & hands)
- `src/record.py`     — capture labeled samples from webcam or a video file → `data/<mode>.csv`
- `src/train.py`      — train a classifier → `models/<mode>.joblib`
- `src/effects.py`    — Stand compositing, menacing text, sound
- `src/app.py`        — live demo (`--mode pose` or `--mode asl`)
- `assets/stands/`    — Stand PNGs (transparent background), user-supplied, not committed
- `assets/sounds/`    — sound effects, user-supplied, not committed
- `data/`, `models/`  — generated; not committed

## Poses (label → Stand)
jotaro → Star Platinum · giorno → Gold Experience · josuke → Crazy Diamond ·
bucciarati → Sticky Fingers · diego → Scary Monsters · kira → Killer Queen ·
plus `neutral` (standing normally / not posing) — always record this class.

## Commands (run from the repo root, venv active)
- Setup (Windows): `./setup.ps1`  (Mac/Linux: `./setup.sh`)
- Record: `python src/record.py --label jotaro` (SPACE toggles capture, Q quits)
- Record from video: `python src/record.py --label jotaro --video clips/jotaro.mov`
- Train: `python src/train.py --mode pose`
- Run: `python src/app.py --mode pose`
- ASL: same commands with `--mode asl` and letter labels (e.g. `--label A`)

## Conventions
- Frames from the webcam are mirrored (selfie view) before processing, in both record and app.
  Keep that consistent or the classifier sees left/right swapped.
- Feature vectors must stay identical between record.py and app.py — always go through
  `landmarks.Extractor`. If the feature layout changes, old CSVs/models are invalid: re-record.
- Don't commit copyrighted images or audio (assets are gitignored).

## Roadmap
1. [ ] Record 6 poses + neutral (~200+ samples each, a few angles/distances)
2. [ ] Train, check per-class accuracy, tune STABLE_FRAMES / threshold
3. [ ] Polish effects (Stand fade-in, positioning per pose)
4. [ ] ASL mode: start with static fingerspelling letters (A–Y minus J/Z), then motion signs
       with a sequence model (e.g. LSTM over landmark windows)
5. [ ] Record a 60–90s demo video + README write-up for the application
