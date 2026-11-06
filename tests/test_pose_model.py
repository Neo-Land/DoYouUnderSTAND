import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from dyu.pose_model import fast_predict_proba  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 12)).astype(np.float32)
y = np.where(X[:, 0] + X[:, 1] > 0, "a", np.where(X[:, 2] > 0.5, "b", "c"))


def test_fast_forest_proba_matches_scikit_learn_exactly():
    m = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)
    probe = rng.normal(size=(30, 12)).astype(np.float32)
    assert np.allclose(fast_predict_proba(m, probe), m.predict_proba(probe), atol=1e-12)


def test_other_models_fall_back_to_their_own_predict_proba():
    m = LogisticRegression(max_iter=500).fit(X, y)
    assert np.allclose(fast_predict_proba(m, X[:5]), m.predict_proba(X[:5]))
