# Dev log

Experiments on the JoJo pose classifier, newest first. All scores are leave-one-session-out
(3 sessions, 2,735 frames, 7 classes, one person) unless noted.

Paths in entries are as they were on that date; see the 2026-11-05 restructure entry for the mapping.

## 2026-11-05: Repo restructure (no behaviour change)

Cleanup for readability; no model, setting or result changed.

- **Removed** the old `asl/` scaffold (legacy `mp.solutions` fingerspelling app, Kaggle download and
  import scripts) and `scripts/train_asl.ps1`. None of it ran in the current environment. It's in
  git history (commit 17af818) if fingerspelling is rebuilt.
- **Shared code moved into the package.** `dyu/pose_model.py` (classifiers + feature matrices, from
  `scripts/train.py`) and `dyu/live_eval.py` (replay logic, from `scripts/eval_live.py`), so scripts no
  longer import each other. `dyu` is installable (`pyproject.toml`), which removed the `sys.path`
  workarounds from every script and test.
- **Grouped by mode**, with paths moved as follows:

| Before | After |
| --- | --- |
| `scripts/extract_landmarks.py`, `train.py`, `eval_live.py`, `tune_smoothing.py`, `render_stand_previews.py` | `scripts/poses/…` (same names) |
| `asl/extract_asl_citizen.py` | `scripts/signs/extract_asl_citizen.py` |
| `scripts/extract_signs.py`, `scripts/train_signs.py` | `scripts/signs/extract_landmarks.py`, `scripts/signs/train.py` |
| `reports/metrics.json`, `confusion_matrix.png`, `live_eval.csv`, `smoothing_grid.csv` | `reports/poses/…` |
| `reports/signs_baseline.json`, `signs_baseline_per_sign.csv` | `reports/signs/baseline.json`, `baseline_per_sign.csv` |
| `asl/config.py` `ASL_WORD_SIGNS` and paths | `config.yaml` `signs:` section |
| `models/sign_baseline.joblib` | `models/sign_classifier.joblib` |

- **Docs:** README cut to an overview (results at a glance, quick start, structure, licensing);
  detail moved to `docs/poses.md` and `docs/signs.md`.
- **Checks:** `ruff` lint added to CI next to pytest; fixed what it found (unused imports, ambiguous
  names, `zip()` without `strict=`). `setup.ps1` falls back to `python` when the `py` launcher is
  missing.

## 2026-11-05: ASL word signs, step 1–2: landmarks and a baseline

**Data.** 100 everyday signs from ASL Citizen (list in `asl/config.py`, videos pulled from the zip
by `asl/extract_asl_citizen.py`): 3,130 videos, 52 signers, official split: train 1,491 videos / 33
signers, val 378 / 5, test 1,261 / 11. Videos are mostly 640×480 at 30 fps, median 2.4 s.

**Landmarks** (`scripts/extract_signs.py`). The JoJo app's Detector on every frame, 8 processes, each
reusing one detector with a 10 s timestamp gap between videos. 241,518 frames in 35 min. Body found
in 100% of frames; a hand in 44.5%. That isn't missed detections: checked on a median video (MORE),
hands are found exactly while the signer signs, and they're below the frame before and after.
Saved per video in `data/asl_signs/` (git-ignored: landmarks derived from ASL Citizen must not be
published).

**Features** (`dyu/signs.py`). Per frame: 11 upper-body points and both hands (21 points each) in
body coordinates (centred on the shoulders, divided by shoulder width), hand-present flags, wrist
speed: 110 numbers. Unlike the JoJo features, hands are NOT centred on their own wrist, so location
relative to the face is kept. Each video is trimmed to the frames with a visible hand (±3) and
resampled to 32 frames. Training adds mirrored copies. 5 unit tests.

**Baseline** (`scripts/train_signs.py`). Logistic regression on the flattened 32×110 sequence,
standardized. C picked on val (0.003–0.1; 0.1 best at 73.5% top-1, the top of the range tried), then
refit on train + val and scored once on test.

| | Top-1 | Top-5 |
| --- | --- | --- |
| Chance | 1.0% | 5.0% |
| Val (5 signers) | 73.5% | 88.4% |
| **Test (11 unseen signers)** | **70.1%** | **89.5%** |

Hardest on test: MUSIC 33%, MOVIE1 41%, COME, LOVE, HURT, NEED, THANKYOU, POWER 42%. Easiest:
PLAY 93%, HEARING 93%, WHEN, WORKOUT, WATER 92%.

**Next.** A GRU over the 32-frame sequence (motion and order matter, which a flat linear model only
partly captures); also try a larger C since 0.1 was the edge of the grid. Then a live word-sign mode.

## 2026-11-05: Webcam session, dismiss pose, hold to summon / stay until dismissed

**Data.** Session s4 is the MacBook webcam: one clip per pose plus `none_s4` (everyday movement,
25 s). The new `dismiss` class is both arms up in a "Y", recorded twice (s4, and s5 in another
room). Clips were trimmed (last 2.5 s: reaching for the camera; first 2 s for jotaro, kira and diego).
Now 8 classes, 3,931 frames, 30 clips, 5 sessions.

**Model** (leave-one-session-out):

| | Sessions | Best model | Accuracy | Macro-F1 | Held-out webcam session |
| --- | --- | --- | --- | --- | --- |
| Before | 3 | logistic regression | 91.2% | 0.927 | n/a |
| + 5 webcam clips | 4 | random forest | 88.9% | 0.900 | 83% (LR 72%) |
| + kira/diego webcam clips | 4 | random forest | 91.4% | 0.924 | 87% |
| + dismiss (s4, s5) | 5 | random forest | 91.1% | 0.921 | s4 90%, s5 96% |

A model trained on the phone sessions alone scored 72–83% on the webcam: different camera, room and
distance. Random forest now beats logistic regression. Dismiss: recall 0.98, never confused with a
Stand pose; 50 `none` frames are read as dismiss (stretching), which only matters while a Stand is up.
Weakest: Giorno (recall 0.72, 126 frames read as `none`, mostly phone sessions s1/s3) and `none`
(84 frames read as Bucciarati).

**New logic** (`dyu/smoothing.py`, rewritten). A pose "wins" while it holds `min_votes` of the
last `window` votes (frames under `threshold` confidence vote `none`). A Stand pose that keeps
winning for `hold_seconds` summons its Stand (or switches to it); the Stand then stays through
normal movement until `dismiss` keeps winning for `dismiss_seconds`, or nobody is framed for
`absent_seconds`. The app shows a hold bar while a pose is building up. 6 new unit tests.

**Measured** with `eval_live.py` (now reports false summons and dismiss time; dismiss clips start
with a Stand showing). First try with the old vote settings (18 of 20, threshold 0.4, hold 1.5 s):
0 false summons, but slow — josuke_s1 took 6.7 s even at 100% frame accuracy, and bucciarati_s3
never summoned, because random-forest confidences are less extreme than logistic regression's.
`tune_smoothing.py` searched 192 combinations; 33 summon all 24 pose clips, dismiss both dismiss
clips and make 0 false summons. Chosen: threshold 0.4, 9 of 15 frames, hold 1.5 s (the user asked
for 1–2 s; hold 1.0 s scored slightly higher but is less safe against accidental summons live).

| | Vote + linger (3 sessions) | Hold + dismiss (5 sessions) |
| --- | --- | --- |
| Stand during normal movement | 25% | 0% (0 false summons in 1.6 min) |
| Pose clips that summon | 18 / 18 | 24 / 24 |
| Median time to summon | 0.6 s | 1.8 s |
| Correct Stand while posing | 96% | 85% (first 1.5 s count as not showing) |
| Dismiss | none | 2 / 2, median 1.4 s |

**Still weak.** Giorno s1/s3 take 6–8 s and Josuke s1 about 6 s to summon. Settings were picked on
the same clips they're scored on. Not yet tried live with the new logic.

## 2026-11-05: First webcam test; "Step back" framing guard

**Problem.** First live run on the MacBook: Gold Experience appeared immediately, whatever the pose.
The camera was close (and the iPhone's Continuity Camera kept zooming in), so only the upper body
was in view. In every training frame, shoulders, hips and knees are all inside the frame (100% for
every class; ankles often aren't). A head-to-waist view is unlike anything the classifier has seen,
and logistic regression still names a class with high confidence. Reproduced by cropping clips to
head-to-waist: none_s1, kira_s2, jotaro_s2 and bucciarati_s2 were all called **Giorno on 100% of
frames**.

**Change.** `dyu.features.body_in_frame()` checks that shoulders, hips and knees lie inside the
frame (2% margin). `app.py` only classifies frames that pass and otherwise shows "Step back: show
yourself at least head to knees". `eval_live.py` applies the same check. Unit-tested.

**Effect.** 99.9% of training frames and of the live-replay frames pass (3 and 5 `none` frames
fail), so live metrics are unchanged (25% false Stand time, 96% on-time, 18/18 triggered). All
cropped close-up frames are now skipped instead of summoning Gold Experience.

**Also from this test:** `app.py --video` now shrinks frames to 720 px tall like extraction (phone
clips ran at ~7 fps at 1080×1920; now ~15 fps), and `setup.sh` is executable on macOS.

**Webcam result.** The guard works live: Gold Experience is recognized at 0.99 when posed, and the mask puts it behind the user. But standing closer than in the clips makes the torso, and so every Stand, much larger: the Stands' heads left the top of the frame. A portrait camera wouldn't help (a narrower frame, not a taller one). Instead `dyu.effects.fit_in_frame()` caps a Stand at 95% of the frame height and shifts it fully on screen (`effects.keep_in_frame`, `stand_max_height`). Checked by cropping the s2 clips to the webcam's framing (head near the top, cut just below the knees): before, five of six Stands lost their heads; after, all six are fully visible. Unit-tested.

**Still open.** Some poses don't get the right Stand live. Not yet known which, or what they're confused with. The guard catches the obvious failure, but the
model has still only seen one person, one room and full-body framing from about 2–3 m away.

## 2026-11-04: App speed 5 → 20 fps; Stand placement for webcam frames

**Speed problem.** Timing the app's per-frame work on the clips padded to 1280×720 (webcam size)
gave ~180–200 ms per frame, about 5 fps. Breakdown at that size: pose 14 ms, hands 12 ms, each
wrist crop 20 ms, person mask **85 ms**, drawing **48 ms**. Worse, the selfie segmenter run on a
whole wide frame mostly missed a full-body person (only a blob around the head), so the Stand
would have been drawn over the user live.

**Changes** (`dyu/landmarks.py`, `dyu/effects.py`):
- The segmenter runs on a square crop around the body, built from the pose landmarks with a 25%
  margin and scaled to its 256×256 input, and the mask is pasted back. 13 ms, and it matches the
  known-good portrait masks at IoU 0.92–0.97.
- The Stand-behind-person blend and the mask blur run only on the Stands' area, not the whole
  frame: 48 → ~10 ms.
- Pose and hands, and then the two wrist crops and the mask, run on a 3-thread pool (MediaPipe
  releases the GIL; each crop has its own HandLandmarker). The landmarks are identical to the
  sequential version: 0 differing frames on kira_s2 and jotaro_s2 against the cached run.

**Result:** ~49 ms per frame at 1280×720, about **20 fps** (portrait clips ~22 fps).

**Placement.** `render_stand_previews.py --landscape` pads the portrait clips to 16:9 so placement
can be judged at webcam proportions. Padding stretches the edge pixels: flat black got marked as
"person" by the segmenter (hiding the Stand), and mirroring duplicated the person and confused
the pose detector. Changes (torso lengths):

| Stand | Before (scale, x, y) | After |
| --- | --- | --- |
| josuke | 3.0, 0.5, -0.4 | 3.0, 0.6, -0.3 (head was touching the top) |
| bucciarati | 3.4, 0.6, -0.1 | 3.4, 0.85, -0.1 (was mostly hidden behind the user) |
| diego | 2.0, 0.8, 0.2 | 2.8, 1.1, -0.2 (crouching shortens the torso, so it was tiny and low) |

Jotaro, Giorno and Kira are unchanged. All six now sit fully in frame, beside and behind the user.
Still to do: confirm on a real webcam, where distance from the camera and the room will differ.

## 2026-11-04: Fewer false triggers by tuning the smoother

**Problem.** Per-frame scores don't show what a user sees. `scripts/eval_live.py` now replays every
frame of every clip (7,555 frames) through the app's detector and `PoseSmoother`, each session
scored by a model trained on the other two. With the old settings (threshold 0.6, 7 of 10 frames,
1.5 s linger), a Stand showed during 42% of normal movement (`none_s1` 13%, `none_s2` 85%,
`none_s3` 28%), and `giorno_s2` never triggered.

**Diagnosis.**
- `none_s1`'s wrong frames are low-confidence (median 0.65), and the 1.5 s linger turned short
  bursts into long false Stands (15 wrong frames → 92 frames of Giorno on screen).
- `none_s2` is me texting with both hands at chest height. The held-out model has never seen texting
  as `none` and calls it Bucciarati with confidence 1.00. No threshold can filter that out.
- `giorno_s2` is classified correctly but at confidence ~0.42, below the 0.6 threshold, so it never
  collected votes.

**Search.** `scripts/tune_smoothing.py` caches landmarks once and grid-searches 1,200 combinations:
logistic-regression C (0.01–1) × threshold (0.4–0.8) × window (10–30) × vote share (60–90%) ×
linger (0.3–1.5 s). C made no difference. Weighting `none` ×3 or ×10 in training helped by only
1–4 points, and ×10 broke `giorno_s2` again, so the model is unchanged.

**Result** (`config.yaml` recognition: threshold 0.4, window 20, min_votes 18, linger 0.3 s):

| | Before | After |
| --- | --- | --- |
| Stand showing during normal movement | 42% | 25% |
| none_s1 / s2 / s3 | 13% / 85% / 28% | 0% / 61% / 12% |
| Correct Stand while posing (held part) | 94% | 96% |
| Pose clips that trigger | 17 / 18 | 18 / 18 |
| Wrong Stand during a pose clip | 0% | 0% |
| Median time to trigger | 0.2 s | 0.6 s |

**Caveat.** The settings were chosen on the same 21 clips they're scored on, so expect live numbers
to be somewhat worse. The model the app ships is trained on all sessions, so texting itself is now
known `none`, but other everyday movements it hasn't seen will behave like texting did here.

**Still weak.**
- Confident false triggers on unseen everyday movement. Next: record more varied `none` clips
  (texting, phone call, arms crossed, touching the head or cap, walking in and out of frame).
- `giorno_s2` triggers late (1.7 s) and is shown for only 35% of the hold: low confidence in that
  session. More Giorno clips would help.

## 2026-11-04: Stand art, bottom fade, per-Stand placement, person-mask fix

**Art.** Six transparent Stand images in `assets/stands/` (git-ignored fan art, never committed).
All have an alpha channel; 27–60% of pixels are fully transparent. Jotaro (11%) and Kira (8%)
have opaque pixels on the image border because the art is cut off at the legs.

**Bottom fade** (`dyu/effects.py`). After resizing, a Stand's alpha is multiplied by a linear ramp
over the bottom 25% of the image, reaching 0 on the last row. Configurable as
`effects.stand_fade_bottom` (0 = off), with a per-Stand `fade_bottom` override. It hides the
cut-off legs and makes the Stand look like it's emerging. Unit-tested.

**Per-Stand placement** (`config.yaml`). Starting values, in torso lengths:

| Stand | scale | offset_x | offset_y | Image |
| --- | --- | --- | --- | --- |
| jotaro | 3.0 | 0.45 | -0.45 | waist-up |
| josuke | 3.0 | 0.5 | -0.4 | waist-up |
| kira | 3.0 | 0.5 | -0.4 | waist-up |
| giorno | 3.4 | 0.6 | -0.1 | full body |
| bucciarati | 3.4 | 0.6 | -0.1 | full body |
| diego | 2.0 | 0.8 | 0.2 | wide dinosaur, faces left |

**Person-mask fix** (`dyu/landmarks.py`). `app.py` crashed on the first frame with a body: on
mediapipe 0.10.35 / Windows, reading the Pose Landmarker's float32 segmentation mask
(`Image.numpy_view`) fails an internal check and aborts the process. The mask now comes from
MediaPipe's selfie segmenter as a uint8 category mask (0 = person), which reads safely. It costs
about 30 ms per frame, only when the mask is on (app and previews, not extraction).

**Visual check.** `scripts/render_stand_previews.py` renders one still per Stand from its `_s2`
clip into `reports/stand_previews/` (git-ignored). All six composite behind the user and fade
out at the bottom. On these portrait phone clips (405 px wide after resizing), Josuke's,
Bucciarati's and Diego's Stands are partly cut off at the right edge, and Diego's dinosaur sits low and small
because the user's torso is short in a crouch. A landscape webcam frame has more room, so final
tuning should happen live.

**Numbers.** The classifier and landmarks are unchanged, so accuracy stays at 91.2% and macro-F1
at 0.927. Tests: 13 passed (11 existing + 2 for the fade).

## 2026-11-04: Wrist-crop fallback for missed hands

**Problem.** Baseline was 81.1% accuracy, macro-F1 0.822. Kira had recall 0.49: 167 of 332 Kira
frames were predicted as Bucciarati. Bucciarati precision was 0.46, partly because 137 `none`
frames were also called Bucciarati. Hand detection rates showed why: in full-body shots the
hands are tiny and the full-frame Hand Landmarker missed them. Kira's second hand was found in
0% of s2 frames and 29% of s3. Without both hands, Kira's L-shaped finger frame is invisible
and the body alone looks like Bucciarati.

**Change** (`dyu/landmarks.py`, `scripts/extract_landmarks.py`):
- A second Hand Landmarker (IMAGE mode, 1 hand, detection and presence confidence 0.3).
- After the normal pass, for each side with no hand and pose wrist visibility >= 0.3: a square
  crop centred on the pose index point, side = max(48 px, 0.9 x torso length), clipped to the
  frame, enlarged to 256 px with cubic interpolation, and run through the second detector.
- Points are mapped back to the full frame. A hand whose wrist is more than 0.5 x torso length
  from the pose wrist is rejected (it's probably the other hand).
- Extraction now prints the share of frames with each hand found, per clip.

**Hand detection, L / R share of frames:**

| Clip | Before | After |
| --- | --- | --- |
| kira_s1 | 100% / 100% | 100% / 100% |
| kira_s2 | 100% / 0% | 100% / 99% |
| kira_s3 | 100% / 29% | 100% / 100% |
| bucciarati_s1 | 100% / 100% | 100% / 100% |
| bucciarati_s2 | 0% / 98% | 100% / 100% |
| bucciarati_s3 | 100% / 50% | 100% / 100% |

All other pose clips are now 100% / 100%, except Jotaro's left hand at 0% (it's in a pocket,
as expected). The `none` clips are 75–100%.

**Results** (logistic regression, best of LR vs RF both times):

| | Before | After |
| --- | --- | --- |
| Accuracy | 81.1% | 91.2% |
| Macro-F1 | 0.822 | 0.927 |
| Per session (s1 / s2 / s3) | 0.93 / 0.66 / 0.85 | 0.99 / 0.83 / 0.92 |
| Kira recall | 0.49 | 0.98 |
| Bucciarati precision | 0.46 | 0.69 |
| Kira → Bucciarati | 167 | 7 |
| none → Bucciarati | 137 | 131 |

**Still weak.**
- `none` recall is 0.74. Normal movement is still read as Bucciarati (131 frames) and Giorno
  (54). Of the 131, 94 come from `none_s2`, so one session's "normal movement" looks like
  Bucciarati's floating wrist-cross. Next: look at what `none_s2` contains, record more varied
  `none` clips, and check whether the app's confidence threshold and voting window already hide
  these frames live.
- Session s2 is still the weakest fold (0.83).
- The extra detector costs time per frame when a hand is missed. That doesn't matter for offline
  extraction, but the live app's frame rate hasn't been measured yet.
- Still one person; no cross-person evaluation.
