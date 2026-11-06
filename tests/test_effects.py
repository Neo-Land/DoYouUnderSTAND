
import numpy as np
import pytest

pytest.importorskip("cv2")
from dyu.effects import fade_bottom


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


def test_fit_in_frame_shrinks_tall_stand_and_keeps_top_on_screen():
    from dyu.effects import fit_in_frame
    # 720p frame, user close to the camera: a 1200 px tall Stand starting above the frame
    x, y, w, h = fit_in_frame(900, -300, 600, 1200, 1280, 720, max_height=0.95)
    assert h == int(720 * 0.95) and abs(w / h - 0.5) < 0.01   # capped height, aspect kept
    assert y >= 0                                                # head on screen
    assert 0 <= x and x + w <= 1280                              # not cut off at the sides


def test_fit_in_frame_leaves_a_stand_that_already_fits_alone():
    from dyu.effects import fit_in_frame
    assert fit_in_frame(700, 50, 300, 500, 1280, 720) == (700, 50, 300, 500)
