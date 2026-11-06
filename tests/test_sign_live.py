import numpy as np

from dyu.features import N_POSE, raw_to_row
from dyu.sign_live import SignSegmenter, display_name, is_signing

FPS = 30


def frame(hand_drop=None):
    """One raw row: shoulders at y=0.5, 0.2 apart. `hand_drop` = right wrist height below the shoulder
    line in shoulder widths (None = no hand visible)."""
    pose = np.zeros((N_POSE, 3), np.float32)
    pose[:, 2] = 1.0
    pose[11, :2], pose[12, :2] = (0.6, 0.5), (0.4, 0.5)
    hand = None
    if hand_drop is not None:
        hand = np.full((21, 2), 0.5, np.float32)
        hand[:, 1] = 0.5 + hand_drop * 0.2
    return raw_to_row(pose, None, hand)


def run(seg, frames, t0=0.0):
    out, t = [], t0
    for f in frames:
        sign = seg.update(f, t)
        if sign is not None:
            out.append(sign)
        t += 1 / FPS
    return out, t


def test_is_signing_uses_the_rest_line():
    assert is_signing(frame(0.3))           # hand at chest height
    assert not is_signing(frame(1.5))       # hand resting low (desk, lap)
    assert not is_signing(frame(None))      # no hand
    assert not is_signing(None)             # no body


def test_resting_hands_never_start_a_sign():
    signs, _ = run(SignSegmenter(), [frame(1.5)] * 120)
    assert signs == []


def test_raise_sign_lower_gives_one_sign_with_lead_in():
    frames = [frame(None)] * 20 + [frame(0.3)] * 30 + [frame(None)] * 20
    signs, _ = run(SignSegmenter(), frames)
    assert len(signs) == 1
    assert 30 <= len(signs[0]) <= 30 + 4 + 3 + 15       # the sign, a few lead-in frames, the end gap


def test_a_twitch_is_ignored():
    frames = [frame(None)] * 10 + [frame(0.3)] * 5 + [frame(None)] * 30   # 0.17 s < min_seconds
    signs, _ = run(SignSegmenter(), frames)
    assert signs == []


def test_a_very_long_sign_is_cut():
    signs, _ = run(SignSegmenter(max_seconds=2.0), [frame(0.3)] * (5 * FPS))
    assert len(signs) >= 2


def test_display_names():
    assert display_name("WANT1") == "WANT"
    assert display_name("THANKYOU") == "THANK YOU"
    assert display_name("HELLO") == "HELLO"
