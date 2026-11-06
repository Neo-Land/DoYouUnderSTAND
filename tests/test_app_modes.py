"""Mode switching in app.py. Needs MediaPipe (the modes import the detector module), so CI skips it."""
import argparse
import copy

import pytest

pytest.importorskip("mediapipe")

import app  # noqa: E402
from dyu.config import load_config  # noqa: E402


def make_app(tmp_path, mode="menu", sign_models=None):
    cfg = copy.deepcopy(load_config())
    if sign_models is not None:
        cfg["signs"]["models_dir"] = str(sign_models)
    args = argparse.Namespace(mode=mode, debug=False, no_sound=True)
    return app.App(args, cfg)


def test_starts_in_the_menu_and_quits_on_q(tmp_path):
    a = make_app(tmp_path)
    assert a.current == "menu"
    assert a.handle_key(ord("q")) is False


def test_asl_without_a_sign_model_stays_in_the_menu_with_a_notice(tmp_path):
    empty = tmp_path / "no_models"
    empty.mkdir()
    a = make_app(tmp_path, sign_models=empty)
    a.handle_key(ord("a"))
    assert a.current == "menu"
    assert "sign model" in a.ctx.notice


def test_m_returns_to_the_menu_and_d_toggles_the_skeleton(tmp_path):
    a = make_app(tmp_path, mode="poses")
    assert a.current == "poses"
    a.handle_key(ord("d"))
    assert a.ctx.debug is True
    a.handle_key(ord("m"))
    assert a.current == "menu"


def test_tab_cycles_from_the_menu_into_the_modes(tmp_path):
    empty = tmp_path / "no_models"
    empty.mkdir()
    a = make_app(tmp_path, sign_models=empty)
    a.handle_key(9)                       # Tab: menu -> poses
    assert a.current == "poses"
    a.handle_key(9)                       # poses -> signs, which has no model here -> back to the menu
    assert a.current == "menu"
