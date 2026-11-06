# JoJo pose mode

Strike one of six JoJo poses in front of a webcam: hold it for about 1.5 seconds and that character's Stand appears behind you, with ゴゴゴ text and a sound. The Stand stays while you move around, until you hold the **dismiss pose** (both arms up in a "Y") for about a second. Holding a different Stand's pose switches Stands, and the Stand also leaves if you're out of view for 5 seconds.

| Pose | User | Stand | Label |
| --- | --- | --- | --- |
| The Direct Point | Jotaro Kujo | Star Platinum | `jotaro` |
| The High-Fashion Collar Grab | Giorno Giovanna | Gold Experience | `giorno` |
| The S-Curve Backbend | Josuke Higashikata | Crazy Diamond | `josuke` |
| The Floating Wrist-Cross | Bruno Bucciarati | Sticky Fingers | `bucciarati` |
| The Bestial Low Crouch | Diego Brando | Scary Monsters | `diego` |
| The L-Shaped Hand Frame | Yoshikage Kira | Killer Queen | `kira` |
| Normal movement | | (none) | `none` |
| Both arms up in a "Y" | | dismisses the Stand | `dismiss` |

## Contents

- [Running it](#running-it)
- [How it works](#how-it-works)
- [Results](#results)
- [Training your own model](#training-your-own-model)
- [Tuning](#tuning)
- [Limitations](#limitations)

## Running it

```powershell
python app.py --mode poses                       # webcam, straight into JoJo mode (python app.py opens the menu)
python app.py --video data/videos/jotaro_s1.mp4  # no webcam? play a recorded clip
python app.py --camera 1 --no-sound --debug      # another camera, muted, skeleton + per-frame guess shown
```

Keys: `s` sound on/off, `d` skeleton overlay, `m` back to the menu, `Tab` next mode, `q` quit. With `--debug` the bottom left shows the classifier's guess for the current frame and its confidence, which is the first thing to look at when a pose isn't recognized.

- **Stand back** until you're in view at least from head to knees, like the training clips. Otherwise the app shows "Step back" and doesn't classify (see [framing check](#how-it-works)).
- **On a Mac**, `--camera 1` picks another camera if your iPhone (Continuity Camera) opens instead of the built-in one.
- **No trained model?** The app still runs and shows the skeleton, which is a handy setup check.

**Assets (not included).** Stand images and sounds are fan material and stay local (git-ignored):

- `assets/stands/<label>.png`: one image per Stand. Transparent PNGs look best; others show as rectangles.
- `assets/sounds/<label>.wav`: per-Stand sound; `assets/sounds/menacing.wav` plays for any Stand without its own. `python scripts/make_sounds.py` synthesizes an original set (a low rumble with an impact, a different pitch per Stand); add `--callout` to have your computer's text-to-speech voice shout each Stand's name over it ("Star Platinum!"), or `--recordings <folder>` to use your own recorded `<label>.wav` shouts. Or replace them with clips you own. Paths are in `config.yaml`.

Missing files never crash the app: a missing image shows a labelled placeholder, a missing sound plays silence.

## How it works

```
webcam frame (shrunk to 720 px tall if larger)
  -> MediaPipe Pose Landmarker (33 body points) + Hand Landmarker (21 points per hand), run in parallel
  -> wrist-crop pass for missed hands, in parallel with the person mask (selfie segmenter on a crop around the body)
  -> framing check: shoulders, hips and knees in view, otherwise "Step back" and no prediction
  -> features: body centred on the hips and scaled by torso length; hands centred on the wrist, scaled by hand size
  -> classifier (logistic regression or random forest, whichever scored better on held-out sessions)
  -> hold-to-summon / dismiss logic (dyu/smoothing.py)
  -> render: Stand image behind the person (mask), bottom faded out, kept on screen; ゴゴゴ text; sound
```

| Step | Code |
| --- | --- |
| Detection, wrist crops, person mask | `dyu/landmarks.py` (`Detector`) |
| Features, mirroring, framing check | `dyu/features.py` |
| Classifier choice and training features | `dyu/pose_model.py`, `scripts/poses/train.py` |
| Summon / dismiss | `dyu/smoothing.py` (`PoseSmoother`) |
| Rendering | `dyu/effects.py` (`StandRenderer`) |
| The live loop | `app.py` |

Design choices worth knowing:

- **Leave-one-session-out evaluation.** Neighbouring frames are near-duplicates, so a random train/test split would let the model memorize the test set. Each recording session (a different day, outfit, room or camera) is held out in turn.
- **Mirror augmentation.** Every training frame is also flipped left/right, so it works whether or not the camera mirrors the image, and on either side.
- **Hands matched to wrists**, not to MediaPipe's left/right label, which flips with mirroring.
- **Wrist-crop fallback.** In full-body shots a hand is only a few dozen pixels wide and the full-frame hand detector often misses it. For any wrist with no matched hand, a square around it is enlarged to 256 px and searched again; a hand whose wrist lands far from the pose wrist is rejected.
- **Framing check.** Every training frame shows at least head to knees. A close-up upper body is unlike anything the model has seen, and it would still name a pose with high confidence (it called every head-to-waist frame Giorno), so those frames aren't classified.
- **Hold to summon, stay until dismissed.** A pose must keep winning a vote over the last frames for `hold_seconds`; the Stand then stays through normal movement. Short misreads during normal movement never last long enough to summon anything.
- **Person mask on a body crop.** The selfie segmenter is built for close-ups; run on a whole wide frame it took 85 ms and mostly missed a full-body person. On a square crop around the body it takes 13 ms and matches portrait masks at IoU 0.92–0.97. (The Pose Landmarker's own mask is a float image, and reading float masks aborts the process on mediapipe 0.10.35 / Windows.)

## Results

All numbers are leave-one-session-out; the full history is in [DEVLOG.md](DEVLOG.md).

**Classifier.** 8 classes, 3,931 frames from 30 clips, one person, 5 sessions (s1–s3 on a phone, s4 on a MacBook webcam, s5 dismiss only).

| Version | Sessions | Accuracy | Macro-F1 | Notes |
| --- | --- | --- | --- | --- |
| Baseline | 3 (phone) | 81.1% | 0.822 | Kira recall 0.49: 167 Kira frames read as Bucciarati (hands missed) |
| + wrist-crop hand fallback | 3 (phone) | 91.2% | 0.927 | Kira recall 0.98 |
| + webcam session and `dismiss` | 5 | **91.1%** | **0.921** | random forest; held-out webcam session 90% |

The last row is a harder test (a second camera and an eighth class), so it isn't directly comparable with the one above. A model trained only on the phone clips scored 72–83% on the webcam session, so recording on the actual webcam mattered. Strongest: Jotaro (F1 1.00), Diego (0.99), Josuke (0.97). Weakest: Giorno (recall 0.72, mostly read as `none`) and `none` (0.85). The dismiss pose is recognized 98% of the time and never mistaken for a Stand pose. Current numbers: `reports/poses/metrics.json` and `reports/poses/confusion_matrix.png`.

**Live behaviour.** What a user sees depends on the summon logic, not just per-frame accuracy. `scripts/poses/eval_live.py` replays every frame of every clip through the app's detector, framing check and summon/dismiss logic, each session scored by a model trained on the others.

| Logic | Stand during normal movement | Correct Stand while posing | Pose clips that summon | Time to summon | Dismiss |
| --- | --- | --- | --- | --- | --- |
| Vote + 1.5 s linger (3 sessions) | 42% | 94% | 17 / 18 | 0.2 s | n/a |
| Vote + linger, tuned (3 sessions) | 25% | 96% | 18 / 18 | 0.6 s | n/a |
| **Hold to summon, stay until dismissed** (5 sessions) | **0%** (0 false summons) | 85% | **24 / 24** | 1.8 s | 2 / 2, 1.4 s |

A wrong Stand never appeared in any clip. "Correct Stand while posing" is lower in the last row because the first 1.5 s of every pose count as not yet showing, and a few older phone clips (Giorno s1/s3, Josuke s1) take 6–8 s to summon. Settings were chosen on the same clips they're scored on, so expect live numbers to be a bit worse.

**Speed.** Measured with the app's own pipeline on 1280×720 frames (webcam size) on a 16-core desktop CPU, Jotaro clip (one hand hidden in a pocket):

| Version | Detection | Classify + draw | Whole frame |
| --- | --- | --- | --- |
| Before any speed work (logistic-regression model) | ~180 ms, mask on the whole frame | | ~5 fps |
| Body-crop mask, parallel models, draw only the Stand's area (logistic regression) | ~35 ms | ~10 ms | ~20 fps |
| After the switch to a random-forest model (better accuracy with the webcam session) | 44 ms | 66 ms | 9.1 fps |
| Random forest on one core per frame | 44 ms | 41 ms | 11.8 fps |
| + forest prediction without scikit-learn's per-tree overhead (identical output) | 44 ms | 30 ms | 13.6 fps |
| + mask every other frame, failed hand search retried every 3rd frame | 35 ms | 30 ms | **15.4 fps** |

The random forest won on held-out accuracy once the webcam session and the dismiss pose were added, but it's slower per frame: scikit-learn spreads its 300 trees across all cores and re-checks each tree on every call, which costs far more than the work for a single frame (39 ms). The app now predicts on one core and calls each tree's compiled routine directly (3 ms, the same probabilities to the last digit, unit-tested). The two live-only shortcuts change nothing the classifier sees during training: the person mask only affects drawing, and a hand that's hidden on purpose is just searched for less often. `python app.py --profile` prints the same breakdown on any machine.

## Training your own model

**1. Record clips** into `data/videos/`, named `<label>_s<session>`:

```
jotaro_s1.mp4  jotaro_s2.mp4  jotaro_s3.mp4
giorno_s1.mov  ...
none_s1.mp4    none_s2.mp4    none_s3.mp4     <- normal movement, so Stands don't trigger randomly
dismiss_s4.mp4 dismiss_s5.mp4                 <- the dismiss pose
```

- Head to at least the knees in frame, camera at chest height, 2–3 m away; 720p is plenty.
- 10–15 s per clip: step into the pose, hold it with small natural variations, step out. Trim the end if you reach for the camera.
- A new session = a different day, outfit, lighting, room or camera. That's what makes the model robust, and what the evaluation measures. Record on the camera you'll demo with.

**2. Extract landmarks**: `python scripts/poses/extract_landmarks.py` writes `data/landmarks.csv` and prints, per clip, the frames kept and how often each hand was found (`hands L __% R __%`). A clip flagged "check framing" had too many frames with no body.

**3. Train**: `python scripts/poses/train.py` prints accuracy and macro-F1 per held-out session and a per-class report, and saves `reports/poses/` and `models/pose_classifier.joblib`. If two poses get confused, exaggerate the difference and re-record them.

**4. Check live behaviour**: `python scripts/poses/eval_live.py` (the first run caches every frame's landmarks in `data/live_frames.csv`; later runs only process new clips).

## Tuning

All in `config.yaml`.

| Problem | Setting |
| --- | --- |
| A Stand summons too easily, or by accident | raise `recognition.hold_seconds`, `min_votes` or `threshold` |
| A Stand is slow to appear | lower `hold_seconds` or `min_votes` |
| Dismissing is too slow or too easy | `dismiss_seconds` (leaving the frame removes the Stand after `absent_seconds`) |
| Stand in the wrong spot or size | `effects.stand_scale`, `stand_offset_x`, `stand_offset_y`, or per Stand `scale`, `offset_x`, `offset_y` (torso lengths; + x = screen right, − y = up) |
| Stand cut off when you stand close | `keep_in_frame` (default on) caps a Stand at `stand_max_height` of the frame and moves it on screen |
| Hard edge where the art is cropped | `stand_fade_bottom` (0.25 = bottom quarter fades out), or per Stand `fade_bottom` |

- **Pick recognition settings from data:** run `python scripts/poses/eval_live.py` once, then `python scripts/poses/tune_smoothing.py`. It grid-searches `threshold` / `window` / `min_votes` / `hold_seconds` on held-out sessions and only keeps settings that summon every pose clip and dismiss every dismiss clip.
- **Check placement without a webcam:** `python scripts/poses/render_stand_previews.py --landscape` saves one still per Stand from its `_s2` clip, padded to 16:9, to `reports/poses/stand_previews/` (git-ignored, since it contains the art).

## Limitations

- Trained on one person: expect lower accuracy on other people until more people record clips.
- Static poses only; motion isn't modelled.
- The camera sees in 2D, so poses that differ mostly in depth are hard to tell apart.
