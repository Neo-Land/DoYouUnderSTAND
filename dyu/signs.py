"""Sign (word) features: per-frame landmarks -> a fixed-length sequence for a sign classifier.

Input per frame is the same raw row as the JoJo pipeline (dyu.features.RAW_COLUMNS: 33 pose points
with visibility, plus both hands as 21 x/y points and a present flag), as saved by
scripts/signs/extract_landmarks.py. Frames where no body was found are NaN rows.

Per-frame sign features (FRAME_DIM = 110):
  upper body  11 pose points (nose, eyes, mouth corners, shoulders, elbows, wrists)       22
  hands       per hand: 21 points in body coordinates + a present flag                 2 x 43
  hand speed  per hand: frame-to-frame movement of the wrist in body coordinates         2
Body coordinates: centred on the shoulder midpoint and divided by shoulder width. Unlike the JoJo
features (hands centred on their own wrist), this keeps WHERE a hand is relative to the face and
body, which separates signs like MOTHER (chin) and FATHER (forehead).

Sequence: only the stretch where a hand is visible (plus a little margin) is kept, since signers
rest their hands out of frame before and after signing, and it is resampled to SEQ_LEN frames.
"""
import numpy as np

from .features import N_HAND, N_POSE

SEQ_LEN = 32
UPPER = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]   # nose, eyes, mouth L/R, shoulders, elbows, wrists
L_SH, R_SH = 11, 12
L_WR, R_WR = 15, 16
_POSE_W = N_POSE * 3
_HAND_W = 1 + N_HAND * 2
FRAME_DIM = len(UPPER) * 2 + 2 * (N_HAND * 2 + 1) + 2

# Left/right pairs inside UPPER (by position in the UPPER list), swapped when mirroring
_UPPER_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
_UPPER_PERM = np.arange(len(UPPER))
for _a, _b in _UPPER_PAIRS:
    _UPPER_PERM[_a], _UPPER_PERM[_b] = _b, _a


def _split_raw(raw):
    """(T, RAW) rows -> pose (T, 33, 3), hands (T, 2, 21, 2), present (T, 2)."""
    raw = np.asarray(raw, np.float32)
    pose = raw[:, :_POSE_W].reshape(-1, N_POSE, 3)
    hands, present = [], []
    for k in range(2):
        off = _POSE_W + k * _HAND_W
        present.append(raw[:, off])
        hands.append(raw[:, off + 1:off + _HAND_W].reshape(-1, N_HAND, 2))
    return pose, np.stack(hands, 1), np.stack(present, 1)


def frame_features(raw):
    """(T, RAW) raw rows -> (T, FRAME_DIM) features. NaN (no body) rows stay NaN."""
    pose, hands, present = _split_raw(raw)
    xy = pose[:, :, :2]
    centre = (xy[:, L_SH] + xy[:, R_SH]) / 2
    width = np.linalg.norm(xy[:, L_SH] - xy[:, R_SH], axis=1)
    width = np.where(width > 1e-6, width, 1.0)[:, None, None]
    body = (xy[:, UPPER] - centre[:, None]) / width
    hand = (hands - centre[:, None, None]) / width[:, None]
    present = (present > 0.5).astype(np.float32)
    hand = hand * present[:, :, None, None]                       # missing hand -> zeros
    wrist = hand[:, :, 0]                                          # (T, 2, 2)
    speed = np.zeros((len(raw), 2), np.float32)
    if len(raw) > 1:
        step = np.linalg.norm(np.diff(wrist, axis=0), axis=2)
        both = present[1:] * present[:-1]
        speed[1:] = step * both
    parts = [body.reshape(len(raw), -1)]
    for k in range(2):
        parts += [hand[:, k].reshape(len(raw), -1), present[:, k:k + 1]]
    parts.append(speed)
    return np.concatenate(parts, axis=1).astype(np.float32)


def signing_span(raw, margin=3):
    """(start, end) frame range where a hand is visible, padded by `margin`; whole clip if none."""
    _, _, present = _split_raw(raw)
    any_hand = np.nan_to_num(present).max(axis=1) > 0.5
    idx = np.flatnonzero(any_hand)
    if len(idx) == 0:
        return 0, len(raw)
    return max(0, idx[0] - margin), min(len(raw), idx[-1] + 1 + margin)


def resample(feats, n=SEQ_LEN):
    """(T, D) -> (n, D) by linear interpolation over time, after filling NaN rows from neighbours."""
    feats = np.asarray(feats, np.float32)
    ok = ~np.isnan(feats).any(axis=1)
    if not ok.any():
        return np.zeros((n, feats.shape[1]), np.float32)
    t = np.flatnonzero(ok)
    good = feats[ok]
    src = np.linspace(0, len(feats) - 1, n)
    out = np.empty((n, feats.shape[1]), np.float32)
    for d in range(feats.shape[1]):
        out[:, d] = np.interp(src, t, good[:, d])
    return out


def sequence(raw, n=SEQ_LEN):
    """Raw frames of one video -> (n, FRAME_DIM): trimmed to the signing span and resampled."""
    a, b = signing_span(raw)
    return resample(frame_features(np.asarray(raw)[a:b]), n)


def mirror_sequence(seq):
    """Left/right mirror image of a (T, FRAME_DIM) sequence: negate x, swap sides and hands."""
    seq = np.asarray(seq, np.float32).copy()
    T = len(seq)
    nb = len(UPPER) * 2
    body = seq[:, :nb].reshape(T, len(UPPER), 2)[:, _UPPER_PERM]
    body[:, :, 0] *= -1
    h = N_HAND * 2 + 1
    hands = [seq[:, nb + k * h: nb + (k + 1) * h].copy() for k in range(2)]
    for hk in hands:
        pts = hk[:, :-1].reshape(T, N_HAND, 2)
        pts[:, :, 0] *= -1
        hk[:, :-1] = pts.reshape(T, -1)
    speed = seq[:, -2:][:, ::-1]
    return np.concatenate([body.reshape(T, -1), hands[1], hands[0], speed], axis=1)


VOCAB_SIZES = (100, 300)


def vocabulary(signs_cfg, size=100):
    """The sign glosses for a vocabulary size: 100 = the core set the app uses (`vocabulary` in
    config.yaml), 300 = the core set plus `extra_vocabulary`."""
    core = list(signs_cfg["vocabulary"])
    if size == 100:
        return core
    if size == 300:
        return core + list(signs_cfg.get("extra_vocabulary", []))
    raise ValueError(f"vocabulary size must be one of {VOCAB_SIZES}, not {size}")


def vocab_suffix(size):
    """File-name suffix for a vocabulary size: '' for the core 100 (unchanged names), '_300' otherwise."""
    return "" if size == 100 else f"_{size}"
