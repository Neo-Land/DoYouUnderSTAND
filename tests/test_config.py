"""config.yaml sanity checks: catch values YAML would silently misread."""
from dyu.config import load_config


def test_sign_vocabulary_is_plain_strings():
    # Bare YES / NO / ON / OFF are booleans in YAML; the vocabulary must stay a list of gloss strings.
    vocab = load_config()["signs"]["vocabulary"]
    assert all(isinstance(g, str) for g in vocab)
    assert len(vocab) == len(set(vocab))
    assert "YES" in vocab and "NO" in vocab


def test_every_stand_has_display_info():
    for label, stand in load_config()["stands"].items():
        assert {"pose", "user", "stand", "image"} <= set(stand), label
