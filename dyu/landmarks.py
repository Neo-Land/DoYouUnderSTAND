"""MediaPipe wrapper: one call per frame gives body + hand landmarks and a person mask.

Uses the MediaPipe Tasks API (Pose Landmarker + Hand Landmarker). The model files
(~30 MB) download automatically into models/ the first time.
"""
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .config import ROOT
from .features import N_HAND, N_POSE, assign_hands

MODEL_DIR = ROOT / "models"
MODEL_URLS = {
    "pose": (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ),
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    ),
}


def ensure_models():
    """Download the MediaPipe model files if they aren't in models/ yet."""
    MODEL_DIR.mkdir(exist_ok=True)
    paths = {}
    for key, (name, url) in MODEL_URLS.items():
        path = MODEL_DIR / name
        if not path.exists():
            print(f"Downloading {name} (one-time)...")
            urllib.request.urlretrieve(url, path)
        paths[key] = path
    return paths


@dataclass
class FrameResult:
    pose: Optional[np.ndarray] = None          # (33, 3) x, y in frame-height units + visibility
    left_hand: Optional[np.ndarray] = None     # (21, 2) frame-height units
    right_hand: Optional[np.ndarray] = None
    pose_px: Optional[np.ndarray] = None       # (33, 2) pixel coordinates, for drawing/placement
    hands_px: List[np.ndarray] = field(default_factory=list)
    mask: Optional[np.ndarray] = None          # (H, W) float 0..1, 1 = person


class Detector:
    def __init__(self, segmentation=False, num_hands=2):
        paths = ensure_models()
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(paths["pose"])),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            output_segmentation_masks=segmentation,
        )
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(paths["hand"])),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self.pose = vision.PoseLandmarker.create_from_options(pose_opts)
        self.hands = vision.HandLandmarker.create_from_options(hand_opts)
        self._last_ts = -1

    def process(self, frame_bgr, timestamp_ms):
        """Run both models on one BGR frame. Timestamps must increase (VIDEO mode)."""
        ts = int(timestamp_ms)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_res = self.pose.detect_for_video(image, ts)
        hand_res = self.hands.detect_for_video(image, ts)

        out = FrameResult()
        if pose_res.segmentation_masks:
            out.mask = np.squeeze(pose_res.segmentation_masks[0].numpy_view()).astype(np.float32)

        hands_hu = []
        for hand in hand_res.hand_landmarks or []:
            pts = np.array([[lm.x * w / h, lm.y] for lm in hand], dtype=np.float32)
            if pts.shape == (N_HAND, 2):
                hands_hu.append(pts)
                out.hands_px.append(np.array([[lm.x * w, lm.y * h] for lm in hand], dtype=np.float32))

        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks[0]
            pose = np.array(
                [[lm.x * w / h, lm.y, lm.visibility if lm.visibility is not None else 1.0] for lm in lms],
                dtype=np.float32,
            )
            if pose.shape == (N_POSE, 3):
                out.pose = pose
                out.pose_px = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
                out.left_hand, out.right_hand = assign_hands(pose[:, :2], hands_hu)
        return out

    def close(self):
        self.pose.close()
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# Skeleton lines for the debug overlay
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (0, 11), (0, 12),
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]


def draw_landmarks(frame, result):
    """Draw the skeleton and hands on the frame (debug view)."""
    if result.pose_px is not None:
        pts = result.pose_px.astype(int)
        for a, b in POSE_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 255, 200), 2, cv2.LINE_AA)
        for p in pts[11:]:
            cv2.circle(frame, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
    for hand in result.hands_px:
        pts = hand.astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 120, 0), 1, cv2.LINE_AA)
    return frame
