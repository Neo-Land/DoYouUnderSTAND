# Dev log

Experiments on the JoJo pose classifier, newest first. All scores are leave-one-session-out
(3 sessions, 2,735 frames, 7 classes, one person) unless noted.

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
