
import numpy as np
import pytest

from dyu.clips import parse_clip_name
from dyu.features import (
    FEATURE_DIM,
    RAW_COLUMNS,
    assign_hands,
    featurize,
    flip_raw,
    raw_to_row,
    row_to_raw,
)
from dyu.smoothing import PoseSmoother

rng = np.random.default_rng(0)


def random_frame(hands=(True, True)):
    pose = np.column_stack([rng.uniform(0.2, 1.5, 33), rng.uniform(0.1, 0.9, 33), rng.uniform(0, 1, 33)]).astype(np.float32)
    lh = rng.uniform(0, 1, (21, 2)).astype(np.float32) if hands[0] else None
    rh = rng.uniform(0, 1, (21, 2)).astype(np.float32) if hands[1] else None
    return pose, lh, rh


def test_row_roundtrip():
    for hands in [(True, True), (True, False), (False, False)]:
        pose, lh, rh = random_frame(hands)
        row = raw_to_row(pose, lh, rh)
        assert len(row) == len(RAW_COLUMNS)
        p2, l2, r2 = row_to_raw(row)
        assert np.allclose(pose, p2)
        assert (l2 is None) == (lh is None) and (r2 is None) == (rh is None)


def test_feature_shape():
    assert featurize(*random_frame()).shape == (FEATURE_DIM,)
    assert featurize(*random_frame((False, False))).shape == (FEATURE_DIM,)


def test_features_ignore_position_and_distance():
    pose, lh, rh = random_frame()
    f1 = featurize(pose, lh, rh)
    moved = pose.copy()
    moved[:, :2] = moved[:, :2] * 0.5 + 0.3  # farther away and shifted
    lh2, rh2 = lh * 0.5 + 0.3, rh * 0.5 + 0.3
    assert np.allclose(f1, featurize(moved, lh2, rh2), atol=1e-4)


def test_flip_twice_is_identity():
    pose, lh, rh = random_frame((True, False))
    p, left, right = flip_raw(*flip_raw(pose, lh, rh))
    assert np.allclose(p, pose) and np.allclose(left, lh) and right is None


def test_flip_swaps_hands():
    pose, lh, rh = random_frame((True, False))
    _, left, right = flip_raw(pose, lh, rh)
    assert left is None and right is not None


def test_assign_hands_by_wrist():
    pose = np.zeros((33, 2), np.float32)
    pose[15] = [0.2, 0.5]  # left wrist
    pose[16] = [0.8, 0.5]  # right wrist
    near_right = np.full((21, 2), 0.8, np.float32)
    near_right[0] = [0.79, 0.5]
    near_left = np.full((21, 2), 0.2, np.float32)
    near_left[0] = [0.21, 0.5]
    left, right = assign_hands(pose, [near_right, near_left])
    assert left is near_left and right is near_right
    left, right = assign_hands(pose, [near_right])
    assert left is None and right is near_right


@pytest.mark.parametrize("name,expected", [
    ("jotaro_s1.mp4", ("jotaro", "s1")),
    ("Kira_S02.MOV", ("kira", "s2")),
    ("none_s3.mov", ("none", "s3")),
    ("jotaro.mp4", None),
])
def test_parse_clip_name(name, expected):
    assert parse_clip_name(name) == expected


def feed(s, label, prob, t0, seconds, fps=10):
    """Send `label` for `seconds` at `fps`; return (last output, time after)."""
    out, t = None, t0
    for _ in range(int(round(seconds * fps))):
        out = s.update(label, prob, t)
        t += 1 / fps
    return out, t


def smoother():
    return PoseSmoother(window=5, min_votes=4, threshold=0.5, hold_seconds=1.0,
                        dismiss_seconds=0.5, absent_seconds=2.0)


def test_smoother_needs_the_pose_held_before_summoning():
    s = smoother()
    out, t = feed(s, "jotaro", 0.9, 0.0, 1.0)      # winner after 4 frames, held ~0.6 s: not yet
    assert out is None
    out, t = feed(s, "jotaro", 0.9, t, 0.6)
    assert out == "jotaro"


def test_short_bursts_never_summon():
    s = smoother()
    t = 0.0
    for _ in range(5):                               # 0.8 s of a pose, then normal movement
        out, t = feed(s, "giorno", 0.9, t, 0.8)
        out, t = feed(s, "none", 0.9, t, 0.8)
        assert out is None


def test_stand_stays_through_normal_movement_until_dismissed():
    s = smoother()
    _, t = feed(s, "kira", 0.9, 0.0, 2.0)
    out, t = feed(s, "none", 0.9, t, 5.0)            # moving around normally: Stand stays
    assert out == "kira"
    out, t = feed(s, "kira", 0.2, t, 1.0)            # low confidence: stays too
    assert out == "kira"
    out, t = feed(s, "dismiss", 0.9, t, 0.6)         # dismiss winning for < 0.5 s: still there
    assert out == "kira"
    out, t = feed(s, "dismiss", 0.9, t, 0.5)
    assert out is None


def test_holding_another_pose_switches_stand():
    s = smoother()
    _, t = feed(s, "kira", 0.9, 0.0, 2.0)
    out, t = feed(s, "diego", 0.9, t, 2.0)
    assert out == "diego"


def test_stand_leaves_when_nobody_is_in_frame():
    s = smoother()
    _, t = feed(s, "josuke", 0.9, 0.0, 2.0)
    out, t = feed(s, None, 0.0, t, 1.5)              # briefly out of frame: stays
    assert out == "josuke"
    out, t = feed(s, None, 0.0, t, 1.0)              # gone longer than absent_seconds
    assert out is None


def test_dismiss_alone_never_shows_a_stand():
    s = smoother()
    out, _ = feed(s, "dismiss", 0.9, 0.0, 3.0)
    assert out is None
