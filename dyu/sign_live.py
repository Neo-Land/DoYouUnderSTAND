"""Live word-sign recognition: find where each sign starts and ends in a webcam stream, then classify it.

The model is trained on clips that contain exactly one sign. A webcam gives a continuous stream, so
SignSegmenter cuts it into signs:

- A frame counts as *signing* when a hand is visible and its wrist is above a rest line, REST_LINE
  shoulder widths below the shoulders. In ASL Citizen, wrists stay within 0.87 shoulder widths below
  the shoulders in 99% of signing frames, while hands resting on a desk or in a lap sit lower.
- `start_frames` signing frames in a row start a sign; `end_seconds` without signing ends it.
- A sign shorter than `min_seconds` is ignored (a hand twitch); one longer than `max_seconds` is cut.

SignRecognizer then turns the sign's frames into a sequence exactly as in training
(dyu.signs.sequence) and returns the top-k signs with probabilities.
"""
import re
from collections import deque
from pathlib import Path

import numpy as np

from .features import RAW_COLUMNS, raw_to_row
from .signs import mirror_sequence, sequence

REST_LINE = 1.0     # shoulder widths below the shoulder line; lower wrists count as resting
_C = {c: i for i, c in enumerate(RAW_COLUMNS)}


def is_signing(row):
    """True if a raw landmark row shows a hand raised into the signing space."""
    if row is None or np.isnan(row[0]):
        return False
    sh_y = (row[_C["pose_y11"]] + row[_C["pose_y12"]]) / 2
    width = float(np.hypot(row[_C["pose_x11"]] - row[_C["pose_x12"]], row[_C["pose_y11"]] - row[_C["pose_y12"]]))
    if width < 1e-6:
        return False
    for side in ("lh", "rh"):
        if row[_C[f"{side}_present"]] > 0.5 and (row[_C[f"{side}_y0"]] - sh_y) / width < REST_LINE:
            return True
    return False


class SignSegmenter:
    """Feed one frame at a time; returns the frames of a sign when one finishes, else None."""

    def __init__(self, start_frames=3, end_seconds=0.4, min_seconds=0.3, max_seconds=4.0, pre_frames=4):
        self.start_frames = start_frames
        self.end_seconds = end_seconds
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.recent = deque(maxlen=pre_frames + start_frames)  # frames just before a sign starts
        self.reset()

    def reset(self):
        self.frames, self.started, self.last_active = [], None, None
        self.streak = 0

    @property
    def active(self):
        """True while a sign is being recorded."""
        return self.started is not None

    def update(self, row, now):
        """`row` is a raw landmark row (or None when no body was found); `now` is the time in seconds."""
        signing = is_signing(row)
        row = row if row is not None else np.full(len(RAW_COLUMNS), np.nan, np.float32)
        if not self.active:
            self.recent.append(row)
            self.streak = self.streak + 1 if signing else 0
            if self.streak >= self.start_frames:
                self.started, self.last_active = now, now
                self.frames = list(self.recent)        # include the lead-in so the start isn't clipped
            return None

        self.frames.append(row)
        if signing:
            self.last_active = now
        ended = now - self.last_active >= self.end_seconds
        too_long = now - self.started >= self.max_seconds
        if not (ended or too_long):
            return None
        sign, duration = self.frames, self.last_active - self.started
        self.reset()
        self.recent.clear()
        return np.asarray(sign, np.float32) if duration >= self.min_seconds else None


def display_name(gloss):
    """ASL Citizen gloss -> readable word: 'WANT1' -> 'WANT', 'THANKYOU' -> 'THANK YOU'."""
    word = re.sub(r"\d+$", "", gloss)
    return {"THANKYOU": "THANK YOU", "NOTMIND": "NOT MIND"}.get(word, word)


class SignRecognizer:
    """Loads the trained sign model (GRU if available, else the logistic-regression baseline)."""

    def __init__(self, models_dir, rest_trim=True, mirror_average=True):
        """rest_trim: cut the sign to the frames above the rest line before classifying (live clips can
        include resting hands that dataset clips never show). mirror_average: average the scores of
        the sign and its mirror image."""
        self.rest_trim = rest_trim
        self.mirror_average = mirror_average
        models_dir = Path(models_dir)
        gru, logreg = models_dir / "sign_gru.pt", models_dir / "sign_logreg.joblib"
        if gru.exists():
            try:
                from .sign_model import SignClassifier

                self.model, self.name = SignClassifier.load(gru), "GRU"
            except ImportError:  # torch not installed
                gru = None
        if not (gru and gru.exists()):
            if not logreg.exists():
                raise FileNotFoundError(f"No sign model in {models_dir} (train one with scripts/signs/train.py)")
            import joblib

            bundle = joblib.load(logreg)
            self.model, self.name = bundle["model"], "logistic regression"
        self.classes = np.asarray(self.model.classes_)

    def top(self, frames, k=5):
        """Raw frames of one sign -> [(word, probability), ...] best first."""
        if self.rest_trim:
            frames = rest_trimmed(frames)
        seqs = [sequence(frames)]
        if self.mirror_average:
            seqs.append(mirror_sequence(seqs[0]))
        batch = np.stack(seqs)
        flat_model = self.name != "GRU"
        proba = self.model.predict_proba(batch.reshape(len(batch), -1) if flat_model else batch).mean(axis=0)
        best = np.argsort(-proba)[:k]
        return [(display_name(self.classes[i]), float(proba[i])) for i in best]


def rest_trimmed(frames, margin=2):
    """Keep only the stretch where a hand is above the rest line (plus `margin` frames each side).
    Dataset clips show hands only while signing; live clips can also show them resting in a lap."""
    frames = np.asarray(frames)
    idx = np.flatnonzero([is_signing(None if np.isnan(r[0]) else r) for r in frames])
    if len(idx) == 0:
        return frames
    return frames[max(0, idx[0] - margin):idx[-1] + 1 + margin]


# Signers in ASL Citizen sit close to their webcam: shoulder width is 0.46-0.65 of the frame height for
# 90% of videos (median 0.56). Much smaller and the hands get too small for reliable finger landmarks,
# which shows up as confusing signs made in the same place with different handshapes.
MIN_SHOULDER_WIDTH = 0.40


def shoulder_width(row):
    """Shoulder width in frame heights (raw rows store x, y in frame-height units), or None."""
    if row is None or np.isnan(row[0]):
        return None
    return float(np.hypot(row[_C["pose_x11"]] - row[_C["pose_x12"]], row[_C["pose_y11"]] - row[_C["pose_y12"]]))


def frame_row(result):
    """Detector FrameResult -> raw landmark row (NaN row if no body), as saved for training."""
    if result.pose is None:
        return None
    return raw_to_row(result.pose, result.left_hand, result.right_hand)
