import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("cv2")
from dyu.effects import fade_bottom  # noqa: E402


def sprite(h=40, w=10, alpha=200):
    img = np.full((h, w, 4), 120, np.uint8)
    img[:, :, 3] = alpha
    return img


def test_fade_bottom_clears_last_row_and_keeps_top():
    img = sprite()
    out = fade_bottom(img, 0.25)
    assert np.all(out[-1, :, 3] == 0)                         # bottom row fully transparent
    assert np.array_equal(out[:30], img[:30])                 # top 75% untouched (alpha and colour)
    col = out[30:, 0, 3].astype(int)
    assert np.all(np.diff(col) <= 0) and col[0] > 0           # ramps down monotonically
    assert np.array_equal(out[:, :, :3], img[:, :, :3])       # colour channels never change
    assert img[-1, 0, 3] == 200                               # input is not modified


def test_fade_bottom_off():
    img = sprite()
    assert fade_bottom(img, 0) is img
