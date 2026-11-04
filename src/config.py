"""Paths, poses, and tuning knobs. Edit this file to add poses or tweak behavior."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASETS_DIR = ROOT / "datasets"   # downloaded public datasets (gitignored)
MODELS_DIR = ROOT / "models"
STANDS_DIR = ROOT / "assets" / "stands"
SOUNDS_DIR = ROOT / "assets" / "sounds"

# label -> display info. `offset_x` shifts the Stand sideways relative to the user
# (fraction of the Stand's width; negative = to the user's left on screen).
POSES = {
    "jotaro":     {"user": "Jotaro Kujo",        "stand": "Star Platinum",   "asset": "star_platinum",   "offset_x": 0.30},
    "giorno":     {"user": "Giorno Giovanna",    "stand": "Gold Experience", "asset": "gold_experience", "offset_x": 0.30},
    "josuke":     {"user": "Josuke Higashikata", "stand": "Crazy Diamond",   "asset": "crazy_diamond",   "offset_x": 0.30},
    "bucciarati": {"user": "Bruno Bucciarati",   "stand": "Sticky Fingers",  "asset": "sticky_fingers",  "offset_x": -0.30},
    "diego":      {"user": "Diego Brando",       "stand": "Scary Monsters",  "asset": "scary_monsters",  "offset_x": -0.30},
    "kira":       {"user": "Yoshikage Kira",     "stand": "Killer Queen",    "asset": "killer_queen",    "offset_x": 0.30},
}
NEUTRAL_LABEL = "neutral"

# Recognition
CONFIDENCE_THRESHOLD = 0.65   # min classifier probability to count a frame
STABLE_FRAMES = 8             # consecutive matching frames before a Stand triggers
HOLD_SECONDS = 3.0            # how long the Stand stays after you drop the pose
FADE_SECONDS = 0.4            # Stand fade-in time

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MIRROR_CAMERA = True          # selfie view; keep the same in record.py and app.py

MODES = ("pose", "asl")


def data_path(mode: str) -> Path:
    return DATA_DIR / f"{mode}.csv"


def model_path(mode: str) -> Path:
    return MODELS_DIR / f"{mode}.joblib"
