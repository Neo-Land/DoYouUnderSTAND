"""Raw landmark storage format, feature engineering, and mirror augmentation.

Raw frame (what extraction saves, so features can change without re-extracting videos):
  pose:       (33, 3)  x, y, visibility  -- x and y in "frame-height units" (pixels / frame height),
                                            so geometry isn't distorted by the aspect ratio
  left_hand:  (21, 2)  or None           -- the hand nearest the pose's LEFT wrist
  right_hand: (21, 2)  or None           -- the hand nearest the pose's RIGHT wrist

Features (what the classifier sees), 185 numbers:
  body:  33 points centred on the mid-hip and scaled by torso length       (66)
  vis:   33 visibility scores                                              (33)
  hands: per hand, 21 points centred on the wrist, scaled by hand size,
         plus a "hand present" flag                                    (2 x 43)

Centering + scaling makes the model ignore where you stand and how far you are
from the camera. Only the *shape* of the pose is left.
"""
import numpy as np

N_POSE = 33
N_HAND = 21

# MediaPipe pose landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

# Left/right landmark pairs, swapped when mirroring
POSE_FLIP_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]
_POSE_PERM = np.arange(N_POSE)
for _a, _b in POSE_FLIP_PAIRS:
    _POSE_PERM[_a], _POSE_PERM[_b] = _b, _a

RAW_COLUMNS = (
    [f"pose_{c}{i}" for i in range(N_POSE) for c in ("x", "y", "v")]
    + ["lh_present"] + [f"lh_{c}{i}" for i in range(N_HAND) for c in ("x", "y")]
    + ["rh_present"] + [f"rh_{c}{i}" for i in range(N_HAND) for c in ("x", "y")]
)
FEATURE_DIM = N_POSE * 2 + N_POSE + 2 * (N_HAND * 2 + 1)


def raw_to_row(pose, left_hand, right_hand):
    """Flatten one raw frame into a CSV row (len(RAW_COLUMNS) numbers)."""
    parts = [np.asarray(pose, dtype=np.float32).reshape(-1)]
    for hand in (left_hand, right_hand):
        if hand is None:
            parts += [np.zeros(1, np.float32), np.zeros(N_HAND * 2, np.float32)]
        else:
            parts += [np.ones(1, np.float32), np.asarray(hand, dtype=np.float32).reshape(-1)]
    return np.concatenate(parts)


def row_to_raw(row):
    """Inverse of raw_to_row."""
    row = np.asarray(row, dtype=np.float32)
    pose = row[: N_POSE * 3].reshape(N_POSE, 3)
    i = N_POSE * 3
    hands = []
    for _ in range(2):
        present = row[i] > 0.5
        hand = row[i + 1 : i + 1 + N_HAND * 2].reshape(N_HAND, 2)
        hands.append(hand.copy() if present else None)
        i += 1 + N_HAND * 2
    return pose.copy(), hands[0], hands[1]


def _safe_norm(v):
    n = float(np.linalg.norm(v))
    return n if n > 1e-6 else 1.0


def featurize(pose, left_hand, right_hand):
    """Raw frame -> fixed-length feature vector (FEATURE_DIM,)."""
    xy = pose[:, :2]
    vis = pose[:, 2]
    hip = (xy[L_HIP] + xy[R_HIP]) / 2
    shoulder = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2
    torso = _safe_norm(shoulder - hip)

    parts = [((xy - hip) / torso).reshape(-1), vis]
    for hand in (left_hand, right_hand):
        if hand is None:
            parts += [np.zeros(N_HAND * 2, np.float32), np.zeros(1, np.float32)]
        else:
            size = _safe_norm(hand[9] - hand[0])  # wrist -> middle-finger knuckle
            parts += [((hand - hand[0]) / size).reshape(-1), np.ones(1, np.float32)]
    return np.concatenate(parts).astype(np.float32)


L_KNEE, R_KNEE = 25, 26
# Landmarks that were inside the frame in every training frame. If any is outside, the person is
# framed tighter than anything the classifier has seen (e.g. a close laptop camera showing only the
# upper body) and its prediction is meaningless, so the app doesn't classify that frame.
FRAMING_POINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE)


def body_in_frame(pose, aspect, margin=0.02):
    """True if shoulders, hips and knees all lie inside the frame.

    `pose` is a raw (33, 3) frame in frame-height units, so x runs 0..aspect (width / height)
    and y runs 0..1. `margin` allows points this far past the edge (MediaPipe's estimates for
    joints right at the border can land slightly outside).
    """
    pts = np.asarray(pose)[list(FRAMING_POINTS), :2]
    x_ok = (pts[:, 0] >= -margin) & (pts[:, 0] <= aspect + margin)
    y_ok = (pts[:, 1] >= -margin) & (pts[:, 1] <= 1 + margin)
    return bool(np.all(x_ok & y_ok))


def flip_raw(pose, left_hand, right_hand):
    """Mirror a raw frame horizontally.

    Training on both the original and mirrored frames makes the model work whether
    your camera mirrors the image or not, and whichever side you strike a pose on.
    """
    p = pose.copy()
    p[:, 0] = -p[:, 0]
    p = p[_POSE_PERM]

    def flip_hand(h):
        if h is None:
            return None
        h = h.copy()
        h[:, 0] = -h[:, 0]
        return h

    # After mirroring, the old left hand is now on the right side.
    return p, flip_hand(right_hand), flip_hand(left_hand)


def assign_hands(pose_xy, hands):
    """Match detected hands to the pose's left/right wrists by distance.

    MediaPipe's own Left/Right label depends on whether the image is mirrored,
    so matching to the body is more reliable. Returns (left_hand, right_hand).
    """
    if not hands:
        return None, None
    lw, rw = pose_xy[L_WRIST], pose_xy[R_WRIST]
    if len(hands) == 1:
        h = hands[0]
        return (h, None) if np.linalg.norm(h[0] - lw) <= np.linalg.norm(h[0] - rw) else (None, h)
    a, b = hands[0], hands[1]
    straight = np.linalg.norm(a[0] - lw) + np.linalg.norm(b[0] - rw)
    swapped = np.linalg.norm(b[0] - lw) + np.linalg.norm(a[0] - rw)
    return (a, b) if straight <= swapped else (b, a)
