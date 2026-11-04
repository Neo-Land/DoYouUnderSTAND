"""Loads config.yaml and resolves paths relative to the project root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p):
    """Turn a config path (relative to the project root) into an absolute Path."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
