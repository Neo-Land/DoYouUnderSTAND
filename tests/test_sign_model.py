import numpy as np
import pytest

from dyu.sign_model import _time_warp, augment
from dyu.signs import FRAME_DIM, SEQ_LEN

rng = np.random.default_rng(0)


def test_time_warp_keeps_shape_and_endpoints():
    x = np.cumsum(rng.normal(size=(SEQ_LEN, FRAME_DIM)), axis=0).astype(np.float32)
    y = _time_warp(x, np.random.default_rng(1))
    assert y.shape == x.shape
    assert np.allclose(y[0], x[0]) and np.allclose(y[-1], x[-1])


def test_augment_keeps_batch_shape():
    batch = rng.normal(size=(4, SEQ_LEN, FRAME_DIM)).astype(np.float32)
    assert augment(batch, np.random.default_rng(2)).shape == batch.shape


def test_gru_learns_an_easy_problem_and_reloads(tmp_path):
    pytest.importorskip("torch")
    from dyu.sign_model import GRUConfig, SignClassifier

    X = rng.normal(size=(90, SEQ_LEN, FRAME_DIM)).astype(np.float32)
    y = np.array(["A", "B", "C"] * 30)
    # Feature 1 is the nose's y position: mirroring (used in training) flips x but keeps y, so the
    # classes stay separable after augmentation.
    X[y == "A", :, 1] += 3
    X[y == "B", :, 1] -= 3
    m = SignClassifier(GRUConfig(max_epochs=25, patience=25, noise=0.0)).fit(X[:60], y[:60], log=None)
    assert m.top_k(X[60:], y[60:], 1) > 0.6          # well above chance (1/3)

    path = tmp_path / "gru.pt"
    m.save(path)
    again = SignClassifier.load(path)
    assert np.allclose(m.predict_proba(X[:5]), again.predict_proba(X[:5]), atol=1e-6)
