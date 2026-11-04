# Dev log

Experiments on the JoJo pose classifier, newest first. All scores are leave-one-session-out
(3 sessions, 2,735 frames, 7 classes, one person) unless noted.

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
