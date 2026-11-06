"""Small drawing helpers shared by the app's modes (OpenCV, BGR colours).

OpenCV's built-in fonts only draw ASCII, so on-screen text must stay ASCII (no em dashes or dots).
"""
import cv2

FONT = cv2.FONT_HERSHEY_DUPLEX

# Palette (BGR)
WHITE = (255, 255, 255)
PURPLE = (230, 120, 255)     # the app's accent (ゴゴゴ purple)
DIM_PURPLE = (120, 90, 150)
INFO = (120, 200, 255)       # hints: "step back", "move closer"
RED = (40, 40, 230)
GREEN = (90, 200, 90)
AMBER = (40, 190, 250)
PANEL = (25, 18, 30)


_OUTLINE = [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)]


def put_text(img, text, org, scale=0.7, color=WHITE, thick=2):
    """Text with a dark outline, readable on any camera background.

    The outline is the same text drawn in black at small offsets. (A single thicker black stroke
    would be wider: OpenCV spaces glyphs by stroke thickness, so its last letters stick out.)
    """
    x, y = org
    for dx, dy in _OUTLINE:
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def text_width(text, scale, thick=2):
    (w, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    return w


def put_centered(img, text, y, scale=0.8, color=WHITE, thick=2):
    put_text(img, text, ((img.shape[1] - text_width(text, scale, thick)) // 2, y), scale, color, thick)


def panel(img, x0, y0, x1, y1, color=PANEL, alpha=0.6):
    """A translucent rectangle (blends in place)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x0 >= x1 or y0 >= y1:
        return img
    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
    return img


def bar(img, x, y, width, frac, color=PURPLE, height=12):
    """A progress bar: dark track, coloured fill."""
    cv2.rectangle(img, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + int(width * max(0.0, min(1.0, frac))), y + height), color, -1)
