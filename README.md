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

## How it works

```
video/webcam frame
  -> MediaPipe Pose Landmarker (33 body points + person mask) + Hand Landmarker (21 points per hand)
  -> features: body centred on hips and scaled by torso length; hands centred on wrist, scaled by hand size
  -> classifier (logistic regression / random forest, picked by held-out score)
  -> smoothing: majority vote over 10 frames + confidence threshold + 1.5 s linger
  -> render: Stand drawn into the frame, then the person mask puts you back in front of it
```

Design choices worth knowing:
- **Videos are only used for training.** At runtime the app loads a small model file and reads the live webcam.
- **Leave-one-session-out evaluation.** Neighbouring frames are near-duplicates, so a random split inflates accuracy. Each recording session is held out in turn.
- **Mirror augmentation.** Every training frame is also flipped left/right, so it works whether or not the camera mirrors the image.
- **Hands matched to wrists**, not to MediaPipe's left/right label, which flips with mirroring.
- **Wrist-crop fallback for missed hands.** In full-body shots a hand is only a few dozen pixels wide, so the full-frame hand detector often misses it. For any wrist the pose model found but no hand was matched to, the app crops a square around that hand, enlarges it to 256 px and runs a second hand detector on it. A detected hand is rejected if its wrist is far from the pose wrist.

## Results

7 classes (6 poses + `none`), 2,735 frames from 21 clips, one person, 3 recording sessions. Scores are leave-one-session-out: each session is held out in turn and predicted by a model trained on the other two.

| Version | Accuracy | Macro-F1 | Kira recall | Kira → Bucciarati | none → Bucciarati |
| --- | --- | --- | --- | --- | --- |
| Baseline | 81.1% | 0.822 | 0.49 | 167 frames | 137 frames |
| + wrist-crop hand fallback | **91.2%** | **0.927** | 0.98 | 7 frames | 131 frames |

Both rows use logistic regression, which beat the random forest each time. Hands found in Kira and Bucciarati clips went from as low as 0–50% per clip to 99–100%. Diego, Jotaro and Josuke are at F1 1.00. The weakest class is now `none` (recall 0.74): normal movement still gets read as Bucciarati, mostly in one session. Experiment notes: [docs/DEVLOG.md](docs/DEVLOG.md).

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

## Assets (not included)

Stand images and sounds aren't part of this repo; they stay local and are git-ignored.

- `assets/stands/<pose>.png`: one image per Stand. Transparent PNGs look best; others show as rectangles.
- `assets/sounds/<pose>.wav`: optional per-Stand sound (WAV, OGG or MP3; update the path in `config.yaml` if not .wav).
- `assets/sounds/menacing.wav`: optional fallback sound for any Stand without its own.

Missing files never crash the app: no image shows a labelled placeholder, no sound plays silence.

## Tuning (`config.yaml`)

- Stand flickers or triggers too easily: raise `threshold` or `min_votes`.
- Stand is slow to appear: lower `min_votes`.
- Stand in the wrong spot or size: `stand_scale`, `stand_offset_x`, `stand_offset_y` (globally, or per Stand with `scale`, `offset_x`, `offset_y`).

## Tests

```powershell
pytest -q
```

## Limitations

- Trained on one person: expect lower accuracy on other people until more people record clips.
- Static poses only; motion isn't modeled yet (that comes with the ASL mode's temporal model).
- The camera sees in 2D, so poses that differ mostly in depth are hard to tell apart.

## Roadmap

- ASL mode: 20 isolated signs from Microsoft Research's ASL Citizen dataset, signer-independent evaluation, GRU temporal model, top-5 predictions in the app.
- FastAPI inference endpoint with latency measurements.

JoJo's Bizarre Adventure characters and Stands belong to Hirohiko Araki / Shueisha. This is a non-commercial fan project.
