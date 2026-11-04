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
    "segmenter": (
        "selfie_segmenter.tflite",
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter/float16/latest/selfie_segmenter.tflite",
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
        )
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(paths["hand"])),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        # Second hand model for zoomed-in wrist crops. In full-body shots hands are only a few
        # dozen pixels wide and the full-frame detector often misses them; enlarging the area
        # around a wrist the pose model already found recovers most of those misses.
        crop_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(paths["hand"])),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        self.pose = vision.PoseLandmarker.create_from_options(pose_opts)
        self.hands = vision.HandLandmarker.create_from_options(hand_opts)
        self.hand_crop = vision.HandLandmarker.create_from_options(crop_opts)
        # Person mask from the selfie segmenter as a uint8 category mask. The Pose Landmarker's own
        # mask is float32, and reading float masks (Image.numpy_view) aborts the process on
        # mediapipe 0.10.35 / Windows. Category values: 0 = person, 255 = background.
        self.segmenter = None
        if segmentation:
            seg_opts = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(paths["segmenter"])),
                running_mode=vision.RunningMode.VIDEO,
                output_category_mask=True,
                output_confidence_masks=False,
            )
            self.segmenter = vision.ImageSegmenter.create_from_options(seg_opts)
        self._last_ts = -1
        self.crop_recoveries = 0  # how many hands the crop pass found (for stats)

    CROP_SIZE = 256  # pixels the wrist crop is enlarged to

    def _detect_hand_in_crop(self, rgb, wrist, index, torso_px):
        """Look for one hand around a wrist. Returns (21, 2) pixel coords in the full frame, or None."""
        H, W = rgb.shape[:2]
        center = index  # the pose "index" point sits at the knuckles, roughly mid-hand
        side = max(48.0, 0.9 * torso_px)
        x0, y0 = int(max(0, center[0] - side / 2)), int(max(0, center[1] - side / 2))
        x1, y1 = int(min(W, center[0] + side / 2)), int(min(H, center[1] + side / 2))
        cw, ch = x1 - x0, y1 - y0
        if cw < 16 or ch < 16:
            return None
        crop = rgb[y0:y1, x0:x1]
        scale = self.CROP_SIZE / max(cw, ch)
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        res = self.hand_crop.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
        if not res.hand_landmarks:
            return None
        pts = np.array([[x0 + lm.x * cw, y0 + lm.y * ch] for lm in res.hand_landmarks[0]], dtype=np.float32)
        if pts.shape != (N_HAND, 2):
            return None
        if np.linalg.norm(pts[0] - wrist) > 0.5 * torso_px:  # found some other hand; reject
            return None
        return pts

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
        if self.segmenter is not None:
            cat = self.segmenter.segment_for_video(image, ts).category_mask
            if cat is not None:
                out.mask = (np.squeeze(cat.numpy_view()) == 0).astype(np.float32)

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
                self._recover_missing_hands(rgb, out)
        return out

    def _recover_missing_hands(self, rgb, out):
        """Second pass: for each wrist with no hand assigned, search a zoomed crop around it."""
        px, h = out.pose_px, rgb.shape[0]
        torso = float(np.linalg.norm((px[11] + px[12]) / 2 - (px[23] + px[24]) / 2))
        if torso < 10:
            return
        for side, wrist_i, index_i in (("left_hand", 15, 19), ("right_hand", 16, 20)):
            if getattr(out, side) is not None or out.pose[wrist_i, 2] < 0.3:
                continue
            hand_px = self._detect_hand_in_crop(rgb, px[wrist_i], px[index_i], torso)
            if hand_px is not None:
                setattr(out, side, hand_px / h)  # pixels -> frame-height units
                out.hands_px.append(hand_px)
                self.crop_recoveries += 1

    def close(self):
        self.pose.close()
        self.hands.close()
        self.hand_crop.close()
        if self.segmenter is not None:
            self.segmenter.close()

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
