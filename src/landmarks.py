"""MediaPipe wrapper and feature extraction.

Every feature vector used for training or prediction must come from Extractor.features(),
so record.py and app.py always agree on the layout.
"""
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---- Pose: 33 landmarks -> 66 normalized (x, y) + 33 visibility = 99 features
POSE_DIM = 33 * 2 + 33
# Left/right landmark pairs, used to mirror samples for augmentation
POSE_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                     (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

# ---- Hands: 2 slots x 21 landmarks x (x, y, z) = 126 features (empty slot = zeros)
HAND_DIM = 21 * 3
HANDS_DIM = 2 * HAND_DIM

FEATURE_DIM = {"pose": POSE_DIM, "asl": HANDS_DIM}


def pose_features(pose_landmarks):
    """Center on the hip midpoint and scale by torso length -> position/size invariant."""
    lm = np.array([[p.x, p.y, p.visibility] for p in pose_landmarks.landmark], dtype=np.float32)
    hips = (lm[23, :2] + lm[24, :2]) / 2
    shoulders = (lm[11, :2] + lm[12, :2]) / 2
    scale = float(np.linalg.norm(shoulders - hips)) or 1e-6
    xy = (lm[:, :2] - hips) / scale
    return np.concatenate([xy.flatten(), lm[:, 2]])


def mirror_pose_features(vec):
    """Return the left/right mirror image of a pose feature vector."""
    xy = vec[:66].reshape(33, 2).copy()
    vis = vec[66:].copy()
    xy[:, 0] *= -1
    for a, b in POSE_MIRROR_PAIRS:
        xy[[a, b]] = xy[[b, a]]
        vis[[a, b]] = vis[[b, a]]
    return np.concatenate([xy.flatten(), vis])


def hand_features(hand_landmarks):
    """Center on the wrist and scale by the farthest landmark."""
    lm = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    lm -= lm[0]
    scale = float(np.max(np.linalg.norm(lm[:, :2], axis=1))) or 1e-6
    return (lm / scale).flatten()


def mirror_hand_features(vec):
    """Mirror image of a hands vector: negate x and swap the left/right hand slots.
    Lets a model trained on right-handed signing also read left-handed signing."""
    hands = vec.reshape(2, 21, 3).copy()
    hands[:, :, 0] *= -1
    return hands[::-1].flatten()


MIRROR = {"pose": mirror_pose_features, "asl": mirror_hand_features}


class Extractor:
    """Runs MediaPipe on BGR frames and returns (features or None, raw results)."""

    def __init__(self, mode: str, static: bool = False):
        """static=True for unrelated still images (datasets); False for video/webcam."""
        self.mode = mode
        if mode == "pose":
            self.model = mp_pose.Pose(static_image_mode=static, model_complexity=1,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
        elif mode == "asl":
            self.model = mp_hands.Hands(static_image_mode=static, max_num_hands=2,
                                        min_detection_confidence=0.5 if static else 0.6,
                                        min_tracking_confidence=0.5)
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.model.process(rgb)
        return self._features(results), results

    def _features(self, results):
        if self.mode == "pose":
            if not results.pose_landmarks:
                return None
            return pose_features(results.pose_landmarks)
        if not results.multi_hand_landmarks:
            return None
        vec = np.zeros(HANDS_DIM, dtype=np.float32)
        for hand, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            slot = 0 if handed.classification[0].label == "Left" else 1
            vec[slot * HAND_DIM:(slot + 1) * HAND_DIM] = hand_features(hand)
        return vec

    def draw(self, frame_bgr, results):
        if self.mode == "pose" and results.pose_landmarks:
            mp_draw.draw_landmarks(frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        elif self.mode == "asl" and results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame_bgr, hand, mp_hands.HAND_CONNECTIONS)

    def anchor_x(self, results, frame_width):
        """Horizontal pixel position of the user's body center (for placing the Stand)."""
        if self.mode == "pose" and results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            return int((lm[11].x + lm[12].x) / 2 * frame_width)
        return frame_width // 2

    def close(self):
        self.model.close()
