import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dyu.features import N_POSE, RAW_COLUMNS, raw_to_row  # noqa: E402
from dyu.signs import (FRAME_DIM, SEQ_LEN, frame_features, mirror_sequence, resample,  # noqa: E402
                       sequence, signing_span)

rng = np.random.default_rng(1)


def clip(frames=40, hands_from=10, hands_to=30):
    rows = []
    for t in range(frames):
        pose = np.column_stack([rng.uniform(0.3, 1.0, N_POSE), rng.uniform(0.2, 0.9, N_POSE),
                                np.ones(N_POSE)]).astype(np.float32)
        pose[11, :2], pose[12, :2] = (0.8, 0.5), (0.5, 0.5)            # shoulders 0.3 apart
        hand = rng.uniform(0.3, 0.9, (21, 2)).astype(np.float32)
        on = hands_from <= t < hands_to
        rows.append(raw_to_row(pose, hand if on else None, hand if on else None))
    return np.array(rows, np.float32)


def test_shapes():
    raw = clip()
    assert frame_features(raw).shape == (len(raw), FRAME_DIM)
    assert sequence(raw).shape == (SEQ_LEN, FRAME_DIM)


def test_signing_span_follows_visible_hands():
    assert signing_span(clip(), margin=2) == (8, 32)
    assert signing_span(clip(hands_from=99, hands_to=99)) == (0, 40)   # no hands: whole clip


def test_resample_fills_missing_frames():
    f = np.arange(10, dtype=np.float32)[:, None].repeat(3, 1)
    f[4] = np.nan
    out = resample(f, 19)
    assert out.shape == (19, 3) and not np.isnan(out).any()
    assert np.isclose(out[0, 0], 0) and np.isclose(out[-1, 0], 9)


def test_mirror_is_its_own_inverse_and_swaps_hands():
    seq = sequence(clip())
    m = mirror_sequence(seq)
    assert np.allclose(mirror_sequence(m), seq)
    assert not np.allclose(m, seq)


def test_location_relative_to_shoulders():
    raw = clip(frames=1, hands_from=0, hands_to=1)
    f = frame_features(raw)[0]
    # shoulders at x 0.8 / 0.5 -> centred and divided by width 0.3: +-0.5
    xs = f[0:22:2]
    assert np.isclose(sorted([xs[5], xs[6]])[0], -0.5, atol=1e-5)
    assert len(RAW_COLUMNS) == raw.shape[1]
