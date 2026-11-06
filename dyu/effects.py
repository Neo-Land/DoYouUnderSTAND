"""Visual and audio effects: the Stand behind you, the ゴゴゴ text, and sound."""
import math
import os
from pathlib import Path

import cv2
import numpy as np

from .config import resolve
from .features import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

# ---------------------------------------------------------------- images

def load_rgba(path):
    """Load an image as BGRA. Images without transparency get a fully opaque alpha."""
    path = Path(path)
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def fade_bottom(img, fraction):
    """Return a copy of BGRA `img` whose alpha ramps linearly to 0 over the bottom `fraction` of rows.

    The first faded row keeps (almost) full alpha and the last row becomes fully transparent, so art
    cropped at the legs dissolves instead of ending in a hard edge. fraction <= 0 returns `img` as is.
    """
    h = img.shape[0]
    n = min(h, int(round(h * float(fraction))))
    if n <= 0:
        return img
    out = img.copy()
    ramp = np.linspace(1.0, 0.0, n + 1, dtype=np.float32)[1:]  # excludes 1.0, ends at exactly 0
    a = out[h - n:, :, 3].astype(np.float32) * ramp[:, None]
    out[h - n:, :, 3] = np.round(a).astype(np.uint8)
    return out


def fit_in_frame(x, y, w, h, frame_w, frame_h, max_height=0.95):
    """Shrink and shift a sprite box (top-left x, y, size w x h) so the Stand stays on screen.

    Placement is in torso lengths, so when the user stands close to the camera the Stand can grow
    taller than the frame and its head leaves the top. This caps the height at `max_height` of the
    frame (shrinking around the box's centre, keeping its aspect ratio), then moves the box down so
    its top is on screen and sideways so it isn't cut off at the left or right edge. The bottom may
    still run off the frame: that's where the image fades out anyway.
    """
    limit = max(20, int(frame_h * max_height))
    if h > limit:
        s = limit / h
        cx, cy = x + w / 2, y + h / 2
        w, h = max(20, int(w * s)), limit
        x, y = int(cx - w / 2), int(cy - h / 2)
    y = max(y, 0)
    if w <= frame_w:
        x = min(max(x, 0), frame_w - w)
    return int(x), int(y), int(w), int(h)


def blend_rgba(dst, src, x, y, opacity=1.0):
    """Alpha-blend BGRA `src` onto BGR `dst` in place, top-left corner at (x, y). Clips at edges."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1 or opacity <= 0:
        return dst
    s = src[y0 - y : y1 - y, x0 - x : x1 - x]
    a = (s[:, :, 3:4].astype(np.float32) / 255.0) * opacity
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (roi * (1 - a) + s[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    return dst


# ---------------------------------------------------------------- menacing text

FONT_CANDIDATES = [
    "C:/Windows/Fonts/YuGothB.ttc",
    "C:/Windows/Fonts/meiryob.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


class MenacingText:
    """Purple ゴ glyphs bobbing around the user, JoJo manga style."""

    def __init__(self, char="ゴ", color=(170, 50, 220), outline=(25, 0, 40)):
        from PIL import ImageFont  # noqa: F401  (fail early if Pillow is missing)

        self.font_path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
        self.char = char if self.font_path else "GO"
        if not self.font_path:
            print("No Japanese font found; using 'GO' for the menacing text.")
        self.color = color
        self.outline = outline
        self._cache = {}

    def _glyph(self, size):
        size = max(16, int(size) // 8 * 8)  # bucket sizes so the cache stays small
        if size in self._cache:
            return self._cache[size]
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(self.font_path, size) if self.font_path else ImageFont.load_default(size=size)
        stroke = max(2, size // 12)
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        left, top, right, bottom = probe.textbbox((0, 0), self.char, font=font, stroke_width=stroke)
        img = Image.new("RGBA", (right - left + 4, bottom - top + 4), (0, 0, 0, 0))
        ImageDraw.Draw(img).text(
            (2 - left, 2 - top), self.char, font=font, fill=self.color + (255,),
            stroke_width=stroke, stroke_fill=self.outline + (255,),
        )
        rgba = np.array(img)
        glyph = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        self._cache[size] = glyph
        return glyph

    def draw(self, frame, center_x, center_y, torso_px, opacity, t):
        if opacity <= 0.01 or torso_px <= 1:
            return frame
        # Columns of glyphs on both sides of the user, each bobbing on its own phase.
        layout = [(-1.6, -0.9, 1.0), (-1.9, -0.1, 0.8), (-1.5, 0.7, 0.9),
                  (1.6, -0.7, 1.0), (1.9, 0.1, 0.8), (1.5, 0.9, 0.9)]
        for i, (dx, dy, s) in enumerate(layout):
            g = self._glyph(torso_px * 0.45 * s)
            bob = math.sin(t * 3.0 + i * 1.3) * torso_px * 0.06
            x = int(center_x + dx * torso_px - g.shape[1] / 2)
            y = int(center_y + dy * torso_px + bob - g.shape[0] / 2)
            blend_rgba(frame, g, x, y, opacity)
        return frame


# ---------------------------------------------------------------- sound

class SoundPlayer:
    """Plays a sound when a Stand appears. Missing files or audio problems never crash the app."""

    def __init__(self, sound_map, default_path=None, volume=0.8, enabled=True):
        self.ok = False
        self.muted = not enabled
        self.sounds = {}
        self.default = None
        if not enabled:
            return
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame

            pygame.mixer.init()
            self._pygame = pygame
            self.ok = True
        except Exception as e:  # no audio device, pygame missing, etc.
            print(f"Sound disabled ({e}).")
            return
        for label, path in sound_map.items():
            snd = self._load(path, volume)
            if snd is not None:
                self.sounds[label] = snd
        if default_path:
            self.default = self._load(default_path, volume)

    def _load(self, path, volume):
        if not path:
            return None
        p = resolve(path)
        if not p.exists():
            return None
        try:
            snd = self._pygame.mixer.Sound(str(p))
            snd.set_volume(volume)
            return snd
        except Exception as e:
            print(f"Couldn't load sound {p.name}: {e}")
            return None

    def play(self, label):
        if self.muted or not self.ok:
            return
        snd = self.sounds.get(label) or self.default
        if snd is not None:
            snd.play()

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted


# ---------------------------------------------------------------- stand renderer

class StandRenderer:
    """Fades Stands in/out and composites them BEHIND the user using the person mask."""

    def __init__(self, stands_cfg, effects_cfg):
        self.cfg = effects_cfg
        self.stands = {}
        for label, s in stands_cfg.items():
            img = load_rgba(resolve(s["image"])) if s.get("image") else None
            if img is None:
                print(f"[{label}] no image at {s.get('image')}; a placeholder will be shown.")
            self.stands[label] = {"img": img, "cfg": s, "alpha": 0.0}
        self.fade = max(1e-3, float(effects_cfg.get("fade_seconds", 0.35)))
        self.menacing = MenacingText(effects_cfg.get("menacing_char", "ゴ")) if effects_cfg.get("menacing_text", True) else None
        self.anchor = None  # smoothed (shoulder_x, shoulder_y, torso_px)

    def update(self, active_label, dt):
        step = dt / self.fade
        for label, s in self.stands.items():
            target = 1.0 if label == active_label else 0.0
            s["alpha"] = min(1.0, s["alpha"] + step) if target > s["alpha"] else max(0.0, s["alpha"] - step)

    def _update_anchor(self, pose_px):
        if pose_px is None:
            return
        sh = (pose_px[L_SHOULDER] + pose_px[R_SHOULDER]) / 2
        hip = (pose_px[L_HIP] + pose_px[R_HIP]) / 2
        torso = float(np.linalg.norm(sh - hip))
        new = np.array([sh[0], sh[1], torso], dtype=np.float32)
        self.anchor = new if self.anchor is None else 0.7 * self.anchor + 0.3 * new  # smooth jitter

    @staticmethod
    def _placeholder(label, w, h):
        img = np.zeros((h, w, 4), np.uint8)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (220, 80, 200, 200), 6)
        cv2.putText(img, label.upper(), (10, h // 2), cv2.FONT_HERSHEY_DUPLEX, w / 220, (255, 255, 255, 255), 2)
        return img

    def draw(self, frame, result, t):
        self._update_anchor(result.pose_px)
        visible = [(k, s) for k, s in self.stands.items() if s["alpha"] > 0.01]
        if not visible or self.anchor is None:
            return frame

        cx, cy, torso = self.anchor
        layer = frame.copy()
        boxes = []
        for label, s in visible:
            scfg = s["cfg"]
            scale = float(scfg.get("scale", self.cfg.get("stand_scale", 2.4)))
            off_x = float(scfg.get("offset_x", self.cfg.get("stand_offset_x", 0.7)))
            off_y = float(scfg.get("offset_y", self.cfg.get("stand_offset_y", -0.15)))
            h = max(20, int(torso * scale))
            img = s["img"]
            if img is None:
                img = self._placeholder(label, int(h * 0.6), h)
            w = max(20, int(img.shape[1] * h / img.shape[0]))
            bob = math.sin(t * 2.0) * torso * 0.04  # Stands float
            x = int(cx + off_x * torso - w / 2)
            y = int(cy + off_y * torso + bob - h / 2)
            if self.cfg.get("keep_in_frame", True):
                x, y, w, h = fit_in_frame(x, y, w, h, frame.shape[1], frame.shape[0],
                                          float(self.cfg.get("stand_max_height", 0.95)))
            sprite = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            sprite = fade_bottom(sprite, float(scfg.get("fade_bottom", self.cfg.get("stand_fade_bottom", 0.25))))
            blend_rgba(layer, sprite, x, y, s["alpha"])
            boxes.append((x, y, x + w, y + h))

        # Put the person back on top of the Stand: out = person where mask=1, Stand layer elsewhere.
        # Only the Stands' area can differ from the camera frame, so blend just that region.
        out = layer
        if result.mask is not None and result.mask.shape[:2] == frame.shape[:2] and boxes:
            H, W = frame.shape[:2]
            pad = 8  # room for the blur
            x0 = max(0, min(b[0] for b in boxes) - pad)
            y0 = max(0, min(b[1] for b in boxes) - pad)
            x1 = min(W, max(b[2] for b in boxes) + pad)
            y1 = min(H, max(b[3] for b in boxes) + pad)
            if x0 < x1 and y0 < y1:
                m = cv2.GaussianBlur(result.mask[y0:y1, x0:x1], (0, 0), 2.0)[:, :, None]
                person, stand = frame[y0:y1, x0:x1].astype(np.float32), layer[y0:y1, x0:x1].astype(np.float32)
                out[y0:y1, x0:x1] = (person * m + stand * (1 - m)).astype(np.uint8)

        if self.menacing:
            top_alpha = max(s["alpha"] for _, s in visible)
            self.menacing.draw(out, cx, cy + torso * 0.5, torso, top_alpha, t)
        return out
