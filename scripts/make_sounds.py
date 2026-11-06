"""Synthesize the app's sound effects: an original "menacing" rumble per Stand, made from scratch.

Usage:
    python scripts/make_sounds.py            # writes assets/sounds/<label>.wav and menacing.wav

The JoJo mode plays assets/sounds/<label>.wav when a Stand appears (menacing.wav for any Stand
without its own). These are generated here from sine waves and noise, so they're free to use and
share; to use clips you own instead, drop them in with the same names (paths are in config.yaml).

Each sound (about 1.6 s): a low drone that swells in with a slow tremolo (the "rumble"), a filtered
noise swell, and a soft impact at the moment the Stand appears. Each Stand gets its own pitch,
tremolo rate and brightness so they're distinguishable.
"""
import wave

import numpy as np

from dyu.config import ROOT, load_config, resolve

RATE = 44100

# label -> (base pitch Hz, tremolo Hz, brightness 0..1)
VOICES = {
    "jotaro": (55.0, 6.0, 0.55),
    "giorno": (65.4, 5.0, 0.75),
    "josuke": (61.7, 7.0, 0.65),
    "bucciarati": (49.0, 4.5, 0.45),
    "diego": (41.2, 8.0, 0.35),
    "kira": (58.3, 3.5, 0.60),
}
DEFAULT = (52.0, 5.5, 0.5)


def envelope(n, attack, release):
    """Linear fade in over `attack` seconds and out over `release` seconds."""
    env = np.ones(n)
    a, r = int(attack * RATE), int(release * RATE)
    env[:a] = np.linspace(0, 1, a)
    env[n - r:] *= np.linspace(1, 0, r)
    return env


def lowpass(x, alpha):
    """One-pole low-pass filter: smaller alpha = darker."""
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += alpha * (v - acc)
        y[i] = acc
    return y


def synth(base, tremolo, bright, seconds=1.6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    n = len(t)
    # drone: base + fifth + octave, slightly detuned, with a slow tremolo
    drone = (np.sin(2 * np.pi * base * t) + 0.6 * np.sin(2 * np.pi * base * 1.5 * t + 0.3)
             + 0.35 * np.sin(2 * np.pi * base * 2.01 * t + 1.1))
    drone *= 0.55 + 0.45 * np.sin(2 * np.pi * tremolo * t) ** 2
    drone *= envelope(n, 0.35, 0.5)
    # noise swell, darker or brighter per Stand
    noise = lowpass(rng.normal(size=n), 0.02 + 0.08 * bright)
    noise /= np.abs(noise).max() + 1e-9
    noise *= np.clip(t / 0.9, 0, 1) ** 2 * envelope(n, 0.05, 0.4)
    # soft impact at ~0.12 s: a pitch-dropping thump
    k = t - 0.12
    hit = np.where(k > 0, np.sin(2 * np.pi * (90 - 40 * np.clip(k / 0.25, 0, 1)) * k) * np.exp(-k * 9), 0.0)
    mix = 0.55 * drone + 0.35 * noise + 0.8 * hit
    mix /= np.abs(mix).max() + 1e-9
    return (0.85 * mix * 32767).astype(np.int16)


def write_wav(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())


def main():
    cfg = load_config()
    written = []
    for i, (label, stand) in enumerate(cfg["stands"].items()):
        path = resolve(stand.get("sound") or f"assets/sounds/{label}.wav")
        write_wav(path, synth(*VOICES.get(label, DEFAULT), seed=i))
        written.append(path)
    default = resolve(cfg["effects"].get("default_sound", "assets/sounds/menacing.wav"))
    write_wav(default, synth(*DEFAULT, seed=99))
    written.append(default)
    for p in written:
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
