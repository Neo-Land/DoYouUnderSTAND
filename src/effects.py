"""Visual and audio effects: Stand compositing, menacing text, sound."""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config

# Fonts that can render ゴ (Windows, then macOS). Falls back to Latin text if none exist.
JP_FONTS = [
    "C:/Windows/Fonts/YuGothB.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/meiryob.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]
MENACING_PURPLE = (190, 60, 220)  # RGB


def _font(size):
    for path in JP_FONTS:
        try:
            return ImageFont.truetype(path, size), True
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size), False
    except TypeError:  # very old Pillow
        return ImageFont.load_default(), False


def render_text_rgba(text, size, fill=(255, 255, 255), stroke=(0, 0, 0), stroke_width=None):
    """Render text to a BGRA numpy image with an outline."""
    font, _ = _font(size)
    sw = stroke_width if stroke_width is not None else max(2, size // 14)
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = tmp.textbbox((0, 0), text, font=font, stroke_width=sw)
    img = Image.new("RGBA", (r - l + 4, b - t + 4), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((2 - l, 2 - t), text, font=font, fill=fill + (255,),
                             stroke_width=sw, stroke_fill=stroke + (255,))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def paste_rgba(frame, overlay, x, y, alpha=1.0, under_mask=None):
    """Alpha-blend a BGRA overlay onto a BGR frame at (x, y), clipped to the frame.
    If under_mask (HxW float, 1 = person) is given, the overlay is hidden behind the person."""
    h, w = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + ow, w), min(y + oh, h)
    if x1 <= x0 or y1 <= y0:
        return frame
    ov = overlay[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (ov[..., 3:4].astype(np.float32) / 255.0) * alpha
    if under_mask is not None:
        a *= (1.0 - under_mask[y0:y1, x0:x1, None])
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (ov[..., :3] * a + roi * (1 - a)).astype(np.uint8)
    return frame


@lru_cache(maxsize=None)
def load_stand(asset, frame_h):
    """Load assets/stands/<asset>.png scaled to ~95% of the frame height. None if missing."""
    path = config.STANDS_DIR / f"{asset}.png"
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[effects] missing Stand image: {path}")
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    target_h = int(frame_h * 0.95)
    scale = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), target_h), interpolation=cv2.INTER_AREA)


def draw_stand(frame, label, center_x, person_mask, alpha):
    info = config.POSES.get(label)
    if not info:
        return frame
    stand = load_stand(info["asset"], frame.shape[0])
    if stand is not None:
        sw = stand.shape[1]
        x = int(center_x + info["offset_x"] * sw - sw / 2)
        paste_rgba(frame, stand, x, 0, alpha=alpha * 0.92, under_mask=person_mask)
    return frame


@lru_cache(maxsize=None)
def _menacing_glyph(size):
    _, has_jp = _font(size)
    return render_text_rgba("ゴ" if has_jp else "GO", size, fill=MENACING_PURPLE)


def draw_menacing(frame, t, alpha=1.0):
    """ゴゴゴ glyphs down both edges, wobbling a little."""
    h, w = frame.shape[:2]
    glyph = _menacing_glyph(max(40, h // 9))
    gh = glyph.shape[0]
    for i, (fx, fy) in enumerate([(0.04, 0.15), (0.10, 0.38), (0.05, 0.62),
                                  (0.86, 0.20), (0.80, 0.45), (0.87, 0.70)]):
        dx = int(6 * math.sin(t * 9 + i * 1.7))
        dy = int(4 * math.cos(t * 7 + i))
        paste_rgba(frame, glyph, int(fx * w) + dx, int(fy * h) - gh // 2 + dy, alpha=alpha)
    return frame


@lru_cache(maxsize=None)
def _name_banner(text, size):
    return render_text_rgba(text, size, fill=(255, 230, 90))


def draw_stand_name(frame, label, alpha=1.0):
    info = config.POSES.get(label)
    if not info:
        return frame
    h, w = frame.shape[:2]
    _, has_jp = _font(10)
    text = f"「{info['stand']}」" if has_jp else info["stand"]
    banner = _name_banner(text, max(28, h // 14))
    paste_rgba(frame, banner, (w - banner.shape[1]) // 2, h - banner.shape[0] - 20, alpha=alpha)
    return frame


class Sound:
    """Plays assets/sounds/<asset>.wav|.ogg, falling back to menacing.wav. Silent if unavailable."""

    def __init__(self):
        self.ok = False
        self.cache = {}
        try:
            import pygame
            pygame.mixer.init()
            self.pygame = pygame
            self.ok = True
        except Exception as e:  # no audio device, pygame missing, etc.
            print(f"[effects] sound disabled: {e}")

    def _load(self, name):
        if name not in self.cache:
            self.cache[name] = None
            for ext in (".wav", ".ogg", ".mp3"):
                p = config.SOUNDS_DIR / f"{name}{ext}"
                if p.exists():
                    self.cache[name] = self.pygame.mixer.Sound(str(p))
                    break
        return self.cache[name]

    def play(self, label):
        if not self.ok:
            return
        info = config.POSES.get(label, {})
        snd = self._load(info.get("asset", "")) or self._load("menacing")
        if snd:
            snd.play()
