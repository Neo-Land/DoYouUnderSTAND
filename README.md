# Do You UnderSTAND

Strike a JoJo pose in front of your webcam and that character's Stand appears behind you, with ゴゴゴ menacing text and a sound effect. Under the hood it's a real-time body + hand pose classifier built on MediaPipe landmarks, evaluated on held-out recording sessions. Next up: an ASL mode that recognizes isolated signs using the same pipeline.

| Pose | User | Stand |
| --- | --- | --- |
| The Direct Point | Jotaro Kujo | Star Platinum |
| The High-Fashion Collar Grab | Giorno Giovanna | Gold Experience |
| The S-Curve Backbend | Josuke Higashikata | Crazy Diamond |
| The Floating Wrist-Cross | Bruno Bucciarati | Sticky Fingers |
| The Bestial Low Crouch | Diego Brando | Scary Monsters |
| The L-Shaped Hand Frame | Yoshikage Kira | Killer Queen |

**How to use it:** hold a pose for about 1.5 seconds and its Stand appears. It stays while you move around, until you hold the **dismiss pose** (both arms up in a "Y") for about a second. Holding a different Stand's pose switches Stands, and the Stand also leaves if you're out of view for 5 seconds.

## Progress

| Part | Status |
| --- | --- |
| Record 6 poses + `none` + `dismiss` (3 phone sessions + 2 webcam sessions) | Done |
| Pose classifier, leave-one-session-out | Done: 91.1% accuracy, macro-F1 0.921 over 5 sessions (see [Results](#results)) |
| Stand effects: art behind the user, bottom fade, per-Stand placement, kept on screen, ゴゴゴ text | Done; ~20 fps at 1280×720 |
| Hold to summon, stay until dismissed | Done: 0 false summons during normal movement on held-out sessions |
| Tested live on a MacBook webcam | Done |
| ASL mode (word signs, ASL Citizen) | Next |

## How it works

```
video/webcam frame
  -> MediaPipe Pose Landmarker (33 body points) + Hand Landmarker (21 points per hand), run in parallel
  -> selfie segmenter on a square crop around the body (person mask), in parallel with the wrist-crop hand pass
  -> framing check: shoulders, hips and knees in view, otherwise "Step back" and no prediction
  -> features: body centred on hips and scaled by torso length; hands centred on wrist, scaled by hand size
  -> classifier (logistic regression / random forest, picked by held-out score)
  -> summon/dismiss logic: a pose must keep winning a vote over recent frames for 1.5 s; the Stand stays until dismissed
  -> render: Stand image (bottom faded out, kept on screen) drawn into the frame, then the person mask puts you back in front of it
```

Design choices worth knowing:
- **Videos are only used for training.** At runtime the app loads a small model file and reads the live webcam.
- **Leave-one-session-out evaluation.** Neighbouring frames are near-duplicates, so a random split inflates accuracy. Each recording session is held out in turn.
- **Mirror augmentation.** Every training frame is also flipped left/right, so it works whether or not the camera mirrors the image.
- **Hands matched to wrists**, not to MediaPipe's left/right label, which flips with mirroring.
- **Wrist-crop fallback for missed hands.** In full-body shots a hand is only a few dozen pixels wide, so the full-frame hand detector often misses it. For any wrist the pose model found but no hand was matched to, the app crops a square around that hand, enlarges it to 256 px and runs a second hand detector on it. A detected hand is rejected if its wrist is far from the pose wrist.
- **Framing check.** Every training frame shows at least head to knees. A close-up upper body is unlike anything the model has seen, and it would still confidently name a pose (it called every head-to-waist frame Giorno), so those frames aren't classified.

## Results

8 classes (6 poses + `none` + `dismiss`), 3,931 frames from 30 clips, one person, 5 recording sessions: 3 on a phone and 2 on a MacBook webcam. Scores are leave-one-session-out: each session is held out in turn and predicted by a model trained on the others.

| Version | Sessions | Accuracy | Macro-F1 | Notes |
| --- | --- | --- | --- | --- |
| Baseline | 3 (phone) | 81.1% | 0.822 | Kira recall 0.49: 167 Kira frames read as Bucciarati |
| + wrist-crop hand fallback | 3 (phone) | 91.2% | 0.927 | Kira recall 0.98; logistic regression |
| + webcam clips and `dismiss` | 5 | **91.1%** | **0.921** | random forest; the webcam session alone: 90% |

The third row is a harder test (more sessions, a second camera, an eighth class), so it isn't directly comparable with the second. A model trained only on the phone clips scored 72–83% on the webcam session, so recording on the actual webcam mattered. Jotaro (F1 1.00), Diego (0.99) and Josuke (0.97) are strongest; Giorno (recall 0.72, mostly read as `none`) and `none` itself (0.85) are weakest. The dismiss pose is recognized 98% of the time and is never mistaken for a Stand pose. Experiment notes: [docs/DEVLOG.md](docs/DEVLOG.md).

### Live behaviour

Per-frame accuracy isn't what a user sees: a Stand only appears after a pose wins a vote over recent frames for long enough. `scripts/eval_live.py` replays every frame of every clip through the app's detector, framing check and summon/dismiss logic, each session scored by a model trained on the others.

| Logic | Stand showing during normal movement | Correct Stand while posing | Pose clips that summon | Time to summon | Dismiss |
| --- | --- | --- | --- | --- | --- |
| Vote + linger (threshold 0.6, 7 of 10 frames, 1.5 s linger), 3 sessions | 42% | 94% | 17 / 18 | 0.2 s | none |
| Vote + linger, tuned (0.4, 18 of 20, 0.3 s), 3 sessions | 25% | 96% | 18 / 18 | 0.6 s | none |
| **Hold to summon, stay until dismissed** (0.4, 9 of 15, 1.5 s hold), 5 sessions | **0%** (0 false summons) | 85% | **24 / 24** | 1.8 s | 2 / 2, 1.4 s |

A wrong Stand never appeared in any clip. "Correct Stand while posing" is lower in the last row because the first 1.5 s of every pose now count as not yet showing, and a few older phone clips (Giorno s1/s3, Josuke s1) take 6–8 s to summon. Settings were picked on the same clips they're scored on, so expect live numbers to be a bit worse.

## Setup (Windows, Python 3.11)

```powershell
cd DoYouUnderSTAND
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again. The MediaPipe model files (~30 MB) download automatically the first time.

## 1. Record clips

Put videos in `data/videos/`, named `<pose>_s<session>`:

```
jotaro_s1.mp4  jotaro_s2.mp4  jotaro_s3.mp4
giorno_s1.mov  ...
none_s1.mp4    none_s2.mp4    none_s3.mp4     <- normal movement, so Stands don't trigger randomly
dismiss_s4.mp4 dismiss_s5.mp4                 <- the dismiss pose (both arms up in a "Y")
```

- Whole body in frame, camera at chest height, 2-3 m away, 720p is plenty.
- 10-15 s per clip: step into the pose quickly, hold it with small natural variations, step out.
- Different session = different day, outfit, lighting or room. That's what makes the model robust.

## 2. Extract landmarks

```powershell
python scripts/extract_landmarks.py
```

Writes `data/landmarks.csv` and prints how many frames each clip contributed and in what share of them each hand was found (`hands L __% R __%`). A clip flagged "check framing" had too many frames where no body was found.

## 3. Train

```powershell
python scripts/train.py
```

Prints accuracy and macro-F1 for each held-out session, a per-class report, and saves `reports/confusion_matrix.png` and `models/pose_classifier.joblib`. If two poses are confused, exaggerate the difference and re-record them.

## 4. Run

```powershell
python app.py                                    # webcam
python app.py --video data/videos/jotaro_s1.mp4  # no webcam yet? test on a clip
python app.py --camera 1 --no-sound --debug
```

Keys: `q` quit, `d` skeleton overlay, `m` mute.

Stand back until you're in view at least from head to knees, like the training clips. If your shoulders, hips or knees are out of frame, the app doesn't classify and shows "Step back" instead: a close-up upper body is unlike anything the model trained on, and it would confidently name a pose anyway (it called every head-to-waist frame Giorno).

On a Mac, `--camera 1` picks another camera if your iPhone (Continuity Camera) opens instead of the built-in one.

Frames taller than `extraction.max_height` (720 px) are shrunk first, the same as during extraction, so phone clips (1080×1920) run at normal speed and the models see what they were trained on.

## Assets (not included)

Stand images and sounds aren't part of this repo; they stay local and are git-ignored.

- `assets/stands/<pose>.png`: one image per Stand. Transparent PNGs look best; others show as rectangles. The bottom 25% of each image fades out, so art cropped at the legs doesn't end in a hard edge.
- `assets/sounds/<pose>.wav`: optional per-Stand sound (WAV, OGG or MP3; update the path in `config.yaml` if not .wav).
- `assets/sounds/menacing.wav`: optional fallback sound for any Stand without its own.

Missing files never crash the app: no image shows a labelled placeholder, no sound plays silence.

## Tuning (`config.yaml`)

- Stand summons too easily, or by accident: raise `hold_seconds`, `min_votes` or `threshold`.
- Stand is slow to appear: lower `hold_seconds` or `min_votes`.
- Dismissing is too slow or too easy: `dismiss_seconds`. Leaving the frame removes the Stand after `absent_seconds`.
- Find settings from data instead of guessing: `python scripts/eval_live.py` once (caches every frame's landmarks), then `python scripts/tune_smoothing.py` to grid-search `threshold` / `window` / `min_votes` / `hold_seconds` on held-out sessions, keeping only settings that summon every pose clip and dismiss every dismiss clip.
- Stand in the wrong spot or size: `stand_scale`, `stand_offset_x`, `stand_offset_y` (globally, or per Stand with `scale`, `offset_x`, `offset_y`). Units are torso lengths; + x = screen right, − y = up.
- Stand cut off at the top or sides when you stand close to the camera: `keep_in_frame: true` (default) shrinks a Stand to at most `stand_max_height` (95%) of the frame height and moves it fully on screen.
- Bottom fade: `stand_fade_bottom` (0.25 = bottom quarter fades out, 0 = off), or per Stand with `fade_bottom`.
- Check placement without a webcam: `python scripts/render_stand_previews.py --landscape` saves one still per Stand from its `_s2` clip, padded to 16:9 like a webcam frame, to `reports/stand_previews/` (git-ignored, since it contains the art). Leave out `--landscape` to see the raw portrait clip.

## Tests

```powershell
pytest -q
```

## Speed

Measured on the recorded clips padded to 1280×720 (a webcam-sized frame), 16-core desktop CPU, everything on:

| | Before | After |
| --- | --- | --- |
| Person mask | 85 ms (whole frame; mostly missed a full-body person) | 13 ms (crop around the body) |
| Drawing the Stand behind you | 48 ms (whole frame) | ~10 ms (Stand's area only) |
| Whole frame | ~180–200 ms (~5 fps) | ~49 ms (**~20 fps**) |

The models run on a small thread pool: MediaPipe releases Python's lock, so pose and hands, and then the two wrist crops and the mask, run at the same time. Landmarks are bit-for-bit identical to running them one by one.

## Limitations

- Trained on one person: expect lower accuracy on other people until more people record clips.
- Static poses only; motion isn't modeled yet (that comes with the ASL mode's temporal model).
- The camera sees in 2D, so poses that differ mostly in depth are hard to tell apart.

## Roadmap

- ASL mode: 100 isolated signs from Microsoft Research's ASL Citizen dataset, signer-independent evaluation, GRU temporal model, top-5 predictions in the app.
- FastAPI inference endpoint with latency measurements.

JoJo's Bizarre Adventure characters and Stands belong to Hirohiko Araki / Shueisha. This is a non-commercial fan project.
