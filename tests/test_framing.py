
import numpy as np

from dyu.features import FRAMING_POINTS, L_HIP, L_KNEE, L_SHOULDER, N_POSE, body_in_frame

ASPECT = 16 / 9  # landscape webcam: x runs 0..1.78 in frame-height units, y runs 0..1


def standing_pose():
    """A person standing in the middle of a landscape frame, head to knees comfortably in view."""
    pose = np.zeros((N_POSE, 3), np.float32)
    pose[:, 0], pose[:, 1], pose[:, 2] = 0.9, 0.5, 1.0
    pose[[11, 12], 1] = 0.30   # shoulders
    pose[[23, 24], 1] = 0.60   # hips
    pose[[25, 26], 1] = 0.80   # knees
    pose[[27, 28], 1] = 1.10   # ankles below the frame edge: allowed, like many training clips
    return pose


def test_full_body_passes():
    assert body_in_frame(standing_pose(), ASPECT)


def test_close_up_fails_when_knees_or_hips_leave_frame():
    for idx in (L_KNEE, L_HIP):
        pose = standing_pose()
        pose[idx, 1] = 1.2  # below the bottom edge, as with a laptop camera close up
        assert not body_in_frame(pose, ASPECT)


def test_sideways_out_of_frame_fails():
    pose = standing_pose()
    pose[L_SHOULDER, 0] = ASPECT + 0.1
    assert not body_in_frame(pose, ASPECT)


def test_margin_allows_points_just_past_the_edge():
    pose = standing_pose()
    pose[list(FRAMING_POINTS), 1] = np.clip(pose[list(FRAMING_POINTS), 1], 0, 1.01)
    pose[L_KNEE, 1] = 1.01
    assert body_in_frame(pose, ASPECT)
