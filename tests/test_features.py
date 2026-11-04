import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dyu.clips import parse_clip_name  # noqa: E402
from dyu.features import (  # noqa: E402
    FEATURE_DIM, RAW_COLUMNS, assign_hands, featurize, flip_raw, raw_to_row, row_to_raw,
)
from dyu.smoothing import PoseSmoother  # noqa: E402

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
    p, l, r = flip_raw(*flip_raw(pose, lh, rh))
    assert np.allclose(p, pose) and np.allclose(l, lh) and r is None


def test_flip_swaps_hands():
    pose, lh, rh = random_frame((True, False))
    _, l, r = flip_raw(pose, lh, rh)
    assert l is None and r is not None


def test_assign_hands_by_wrist():
    pose = np.zeros((33, 2), np.float32)
    pose[15] = [0.2, 0.5]  # left wrist
    pose[16] = [0.8, 0.5]  # right wrist
    near_right = np.full((21, 2), 0.8, np.float32); near_right[0] = [0.79, 0.5]
    near_left = np.full((21, 2), 0.2, np.float32); near_left[0] = [0.21, 0.5]
    l, r = assign_hands(pose, [near_right, near_left])
    assert l is near_left and r is near_right
    l, r = assign_hands(pose, [near_right])
    assert l is None and r is near_right


@pytest.mark.parametrize("name,expected", [
    ("jotaro_s1.mp4", ("jotaro", "s1")),
    ("Kira_S02.MOV", ("kira", "s2")),
    ("none_s3.mov", ("none", "s3")),
    ("jotaro.mp4", None),
])
def test_parse_clip_name(name, expected):
    assert parse_clip_name(name) == expected


def test_smoother_activates_lingers_and_switches():
    s = PoseSmoother(window=5, min_votes=3, threshold=0.6, linger_seconds=1.0)
    t = 0.0
    for _ in range(2):
        assert s.update("jotaro", 0.9, t) is None
        t += 0.1
    assert s.update("jotaro", 0.9, t) == "jotaro"
    for _ in range(5):            # low confidence -> votes "none", but still lingering
        t += 0.1
        assert s.update("jotaro", 0.3, t) == "jotaro"
    t += 1.5
    assert s.update(None, 0.0, t) is None
    for _ in range(3):
        t += 0.1
        out = s.update("kira", 0.95, t)
    assert out == "kira"
