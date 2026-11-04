"""Parsing training clip file names like `jotaro_s1.mp4` into (label, session)."""
import re
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
_CLIP_RE = re.compile(r"^(?P<label>.+?)_s(?P<session>\d+)$", re.IGNORECASE)


def parse_clip_name(path):
    """`jotaro_s1.mp4` -> ("jotaro", "s1"). Returns None if the name doesn't match the pattern."""
    stem = Path(path).stem
    m = _CLIP_RE.match(stem)
    if not m:
        return None
    return m.group("label").lower(), f"s{int(m.group('session'))}"


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS
