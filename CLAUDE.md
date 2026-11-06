# Do You UnderSTAND — context for Claude

## What this is
A webcam recognition app with two modes. **JoJo poses:** the user holds a JoJo's Bizarre Adventure
pose and that character's Stand appears behind them (composited behind the person, ゴゴゴ text,
sound) until they hold the dismiss pose. **ASL word signs:** recognizes isolated ASL signs, trained
and tested on Microsoft Research's ASL Citizen dataset (signer-independent split). The name is a
pun: understanding ASL + JoJo Stands.

Purpose: portfolio project for a Microsoft SWE Intern – AI/ML & LLM application (sign-language
team, summer 2027). The ASL mode is the part most relevant to the role, so it should end up solid.

## Owner / environment
- Neo, CS junior at Montclair State. Concise, action-oriented answers preferred.
- Main dev machine: Windows desktop, Python 3.11, no webcam. Live testing happens on a MacBook
  (clone in ~/DoYouUnderSTAND; data/landmarks.csv is copied over by hand, then trained there).
  New clips are recorded on the Mac and processed on the desktop.

## Stack
- Python 3.11, MediaPipe Tasks API 0.10.35 (Pose Landmarker, Hand Landmarker, selfie segmenter).
  Model files download into `models/` on first run.
- OpenCV, Pillow, pygame; pandas + scikit-learn; PyYAML config; pytest + ruff (both in CI).
- `dyu` is an installable package (`pyproject.toml`; setup runs `pip install -e .`), so scripts
  import it directly. Shared logic lives in `dyu/`; scripts are thin command-line wrappers.

## Layout
- `app.py`            — live demo: `--mode poses` (JoJo, default) or `--mode signs` (ASL)
- `config.yaml`       — every setting: camera, extraction, recognition, effects, Stands, `signs:`
- `dyu/`              — library; `dyu/__init__.py` has the module map
  - `landmarks.py` Detector (MediaPipe, wrist-crop fallback, person mask on a body crop, thread pool)
  - `features.py` JoJo features, mirroring, framing check · `pose_model.py` classifiers + feature matrices
  - `smoothing.py` hold-to-summon / dismiss · `live_eval.py` replay clips through the live logic
  - `effects.py` Stand rendering · `clips.py`, `config.py`
  - `signs.py` ASL sign features · `sign_model.py` GRU (torch) · `sign_live.py` live segmentation + recognizer
- `scripts/poses/`    — extract_landmarks, train, eval_live, tune_smoothing, render_stand_previews
- `scripts/signs/`    — extract_asl_citizen, extract_landmarks, train (--model logreg|gru), eval_live
- `tests/`            — pytest unit tests
- `reports/poses/`, `reports/signs/` — committed results (numbers and charts only)
- `docs/`             — `poses.md`, `signs.md` (detail for each mode), `DEVLOG.md` (experiment log)
- `assets/`, `data/`, `datasets/`, `models/` — local only, gitignored

## Poses (label → Stand)
jotaro → Star Platinum · giorno → Gold Experience · josuke → Crazy Diamond ·
bucciarati → Sticky Fingers · diego → Scary Monsters · kira → Killer Queen ·
plus `none` (normal movement) and `dismiss` (both arms up in a Y: sends the Stand away).
Clips are named `<label>_s<session>`: s1–s3 phone, s4 MacBook webcam, s5 dismiss only.

## Commands (repo root, venv active)
- Setup: `./setup.ps1` (Windows) or `./setup.sh` (Mac/Linux)
- JoJo: `python scripts/poses/extract_landmarks.py`, `python scripts/poses/train.py`, `python app.py`
- JoJo live metrics / tuning: `python scripts/poses/eval_live.py`, `python scripts/poses/tune_smoothing.py`
- ASL: `python scripts/signs/extract_asl_citizen.py`, `python scripts/signs/extract_landmarks.py`,
  `python scripts/signs/train.py --model gru`, `python scripts/signs/eval_live.py`, `python app.py --mode signs`
- Checks: `python -m pytest -q` and `python -m ruff check .`

## Conventions
- JoJo evaluation is leave-one-session-out; ASL uses the official split (unseen test signers).
  Never report random-frame splits.
- Hands are assigned by distance to the pose wrists, not MediaPipe's handedness label. Training adds
  left/right mirrored copies.
- If a feature layout changes (`dyu/features.py`, `dyu/signs.py`), re-run extraction/training.
- Shared logic goes in `dyu/`, not in scripts; scripts don't import each other.
- Keep `ruff check .` and the tests passing. Every script starts with a docstring: what it reads,
  writes, and how to run it.
- Log every experiment in `docs/DEVLOG.md` (newest first) and keep README / docs results current.
- Don't commit videos, `data/`, `models/`, or Stand art/sounds (all gitignored).

## ASL Citizen license rules (Microsoft Research License — must follow)
Dataset lives in `datasets/asl_citizen/` (gitignored). Non-commercial research use only.
- NEVER commit or publish: the videos, frames/clips of signers, landmarks extracted from the
  videos, or models trained on it. Keep all of it under `datasets/`, `data/` or `models/`.
- Signers are identifiable people: no dataset frames in the demo video, README or slides.
- No public hosted demo built on it. Local runs + a demo video of Neo only.
- OK to publish: code, accuracy numbers/charts, the write-up, and a citation of the paper.
- Delete the data when the project's research is done.

## Roadmap
### Part 1 — JoJo poses (done)
- [x] 6 poses + none + dismiss, 5 sessions; 91.1% / macro-F1 0.921 leave-one-session-out
- [x] Stand effects, ~20 fps at 1280x720, keep-in-frame, framing check
- [x] Hold to summon / stay until dismissed: 0 false summons, 24/24 summon, dismiss 2/2

### Part 2 — ASL word signs
- [x] Landmarks for 100 ASL Citizen signs (3,130 videos); baseline 70.1% top-1 / 89.5% top-5 on
      unseen test signers
- [x] GRU sequence model: 84.4% ± 0.1% top-1 / 97.1% top-5 on unseen signers (3 seeds)
- [x] Live word-sign mode (`app.py --mode signs`): sign detected in 98.7% of test videos, 83.9% top-1
- [ ] Test sign mode live on the Mac (copy models/sign_gru.pt over privately; never commit it)
- [ ] Larger vocabulary (300 / 1,000 / all 2,731 signs)
- [ ] Optional: fingerspelling (the old Kaggle-based scaffold was removed; rebuild on dyu if wanted)

### Ship
- [ ] 60–90 s demo video + application write-up
