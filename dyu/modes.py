"""The app's screens. Each mode turns one camera frame (plus its landmarks) into the picture to show.

    MenuMode      start screen: pick a mode
    PosesMode     JoJo: hold a pose to summon its Stand, arms up in a Y to dismiss it
    SignsMode     ASL: sign a word, see the top 5 guesses and a transcript
    PracticeMode  Learn ASL: the app names a word, you sign it, you get graded

Every mode has the same small interface, used by the loop in app.py:
    enter()                       called when the mode becomes active (reset its state)
    process(frame, res, now, dt)  draw this frame; return the image to show
    on_key(key)                   handle a key the app didn't handle globally
    hints                         the key help shown in the footer
    needs_mask                    whether the person mask is needed (JoJo only; it costs ~13 ms a frame)

Heavy resources (the pose model, the sign model) live in AppContext and load the first time a mode
needs them, so the menu opens instantly and a missing model only disables that one mode.
"""
import time

import cv2
import joblib
import numpy as np

from .config import resolve
from .effects import SoundPlayer, StandRenderer
from .features import FEATURE_DIM, body_in_frame, featurize
from .landmarks import draw_landmarks
from .pose_model import fast_predict_proba
from .practice import CLOSE, CORRECT, PracticeSession, easiest_first
from .sign_live import MIN_SHOULDER_WIDTH, SignRecognizer, SignSegmenter, display_name, frame_row, shoulder_width
from .smoothing import PoseSmoother
from .ui import AMBER, DIM_PURPLE, GREEN, INFO, PURPLE, RED, WHITE, bar, panel, put_centered, put_text


class AppContext:
    """Settings plus lazily loaded models, shared by all modes."""

    def __init__(self, cfg, debug=False, sound=True):
        self.cfg = cfg
        self.debug = debug
        self.sound_enabled = sound
        self.notice, self.notice_until = "", 0.0
        self._poses = None
        self._recognizer = None

    def notify(self, text, seconds=4.0):
        """A message the menu shows for a few seconds (e.g. why a mode couldn't open)."""
        self.notice, self.notice_until = text, time.perf_counter() + seconds

    def poses(self):
        """(classifier bundle or None, smoother, renderer, sound) for JoJo mode, loaded once."""
        if self._poses is None:
            cfg, fx, stands = self.cfg, self.cfg["effects"], self.cfg["stands"]
            path = resolve(cfg["training"]["model_path"])
            bundle = None
            if path.exists():
                bundle = joblib.load(path)
                if bundle.get("feature_dim") != FEATURE_DIM:
                    print("The saved pose model uses different features; re-run scripts/poses/train.py.")
                    bundle = None
                else:
                    # Training uses every core (n_jobs=-1), but for one frame at a time the coordination
                    # costs more than the work: 39 ms vs 16 ms per frame for the random forest.
                    try:
                        bundle["model"].set_params(n_jobs=1)
                    except ValueError:       # a model without that setting (e.g. logistic regression)
                        pass
            else:
                print(f"No trained pose model at {path}; JoJo mode shows landmarks only.")
            sound = SoundPlayer({k: v.get("sound") for k, v in stands.items()}, fx.get("default_sound"),
                                fx.get("volume", 0.8), enabled=fx.get("sound", True) and self.sound_enabled)
            self._poses = (bundle, PoseSmoother.from_config(cfg["recognition"]), StandRenderer(stands, fx), sound)
        return self._poses

    def recognizer(self):
        """The sign recognizer, or None if no sign model is available (with a notice explaining why)."""
        if self._recognizer is None:
            try:
                self._recognizer = SignRecognizer(resolve(self.cfg["signs"]["models_dir"]))
                print(f"Loaded {self._recognizer.name} sign model: {len(self._recognizer.classes)} signs")
            except FileNotFoundError:
                self.notify("ASL modes need a sign model in models/ (see docs/signs.md)")
                return None
        return self._recognizer


class Mode:
    name = ""
    hints = ""
    needs_mask = False

    def __init__(self, ctx):
        self.ctx = ctx

    def enter(self):
        pass

    def process(self, frame, res, now, dt):
        return frame

    def on_key(self, key):
        pass


# ---------------------------------------------------------------- menu

class MenuMode(Mode):
    name = "Menu"
    hints = "[j] JoJo  [a] ASL signs  [p] Learn ASL  [q] quit"
    CARDS = [("J", "JoJo poses", "Hold a JoJo pose to summon its Stand. Arms up in a Y to dismiss."),
             ("A", "ASL signs", "Sign a word and see the top 5 guesses, live."),
             ("P", "Learn ASL", "Practice: the app names a word, you sign it, you get a score.")]

    def process(self, frame, res, now, dt):
        out = frame
        h, w = out.shape[:2]
        panel(out, 0, 0, w, h, alpha=0.55)
        put_centered(out, "Do You UnderSTAND", int(h * 0.2), 1.6, PURPLE, 3)
        put_centered(out, "JoJo poses and ASL signs, from your webcam", int(h * 0.2) + 42, 0.7)
        card_w, gap = min(360, (w - 80) // 3), 20
        total = 3 * card_w + 2 * gap
        x = (w - total) // 2
        y0, y1 = int(h * 0.36), int(h * 0.36) + 170
        for key, title, blurb in self.CARDS:
            panel(out, x, y0, x + card_w, y1, color=(45, 30, 55), alpha=0.75)
            cv2.rectangle(out, (x, y0), (x + card_w, y1), DIM_PURPLE, 1)
            put_text(out, f"[{key}]", (x + 16, y0 + 40), 1.0, PURPLE)
            put_text(out, title, (x + 76, y0 + 40), 0.8)
            for i, line in enumerate(_wrap(blurb, 30)):
                put_text(out, line, (x + 16, y0 + 80 + 26 * i), 0.55, (220, 220, 220), 1)
            x += card_w + gap
        put_centered(out, "Press a key to start. [m] comes back here from any mode, [Tab] switches modes.",
                     y1 + 50, 0.55, (220, 220, 220), 1)
        if self.ctx.notice and now < self.ctx.notice_until:
            put_centered(out, self.ctx.notice, y1 + 90, 0.65, AMBER)
        return out


# ---------------------------------------------------------------- JoJo poses

class PosesMode(Mode):
    name = "JoJo poses"
    hints = "[s] sound  [d] skeleton  [m] menu  [Tab] next mode  [q] quit"
    needs_mask = True

    def enter(self):
        self.bundle, self.smoother, self.renderer, self.sound = self.ctx.poses()
        self.smoother.reset()
        self.prev_active = None
        self.label, self.prob = None, 0.0

    def on_key(self, key):
        if key == ord("s"):
            self.sound.toggle_mute()

    def process(self, frame, res, now, dt):
        stands = self.ctx.cfg["stands"]
        self.label, self.prob = None, 0.0
        # Only classify when the body is framed like the training clips (shoulders, hips and knees in
        # view): a close-up upper body is unlike any training frame, and the model would still name a pose.
        framed = res.pose is not None and body_in_frame(res.pose, frame.shape[1] / frame.shape[0])
        if self.bundle is not None and framed:
            proba = fast_predict_proba(self.bundle["model"], [featurize(res.pose, res.left_hand, res.right_hand)])[0]
            i = int(np.argmax(proba))
            self.label, self.prob = self.bundle["labels"][i], float(proba[i])
        active = self.smoother.update(self.label, self.prob, now)
        if active and active != self.prev_active:
            self.sound.play(active)
        self.prev_active = active

        self.renderer.update(active, dt)
        out = self.renderer.draw(frame, res, now)
        if self.ctx.debug or self.bundle is None:
            draw_landmarks(out, res)

        if active:
            s = stands.get(active, {})
            put_text(out, f"{s.get('user', active)}  -  {s.get('stand', '')}", (20, 45), 1.0, PURPLE)
            put_text(out, s.get("pose", ""), (20, 80), 0.6)
        elif res.pose is None:
            put_text(out, "Step into frame (whole body)", (20, 45), 0.8, INFO)
        elif not framed:
            put_text(out, "Step back: show yourself at least head to knees", (20, 45), 0.8, INFO)
        holding, frac = self.smoother.progress(now)       # how long the pose still has to be held
        if holding is not None and frac > 0:
            name = "Dismissing" if holding == self.smoother.dismiss else stands.get(holding, {}).get("stand", holding)
            y = 115 if active else 80
            bar(out, 20, y, 260, frac)
            put_text(out, name, (290, y + 12), 0.5)
        if self.ctx.debug and self.label:
            put_text(out, f"frame guess: {self.label} {self.prob:.2f}", (20, out.shape[0] - 50), 0.6)
        if self.sound.muted:
            put_text(out, "(muted)", (out.shape[1] - 110, out.shape[0] - 20), 0.5)
        return out


# ---------------------------------------------------------------- ASL, shared

class _SignBase(Mode):
    """Shared by the two ASL modes: cut the stream into signs, check distance, show status."""

    def enter(self):
        self.recognizer = self.ctx.recognizer()
        self.segmenter = SignSegmenter()

    def sign_from(self, res, now):
        """Feed one frame; return (finished sign frames or None, shoulder width or None)."""
        row = frame_row(res)
        return self.segmenter.update(row, now), shoulder_width(row)

    def status(self, out, res, width, idle_text):
        if res.pose is None:
            put_text(out, "Step into frame (head and shoulders)", (20, 45), 0.8, INFO)
        elif width is not None and width < MIN_SHOULDER_WIDTH:
            # Signers in the training data sit close; far away, hands are too small for finger detail.
            put_text(out, "Move closer: head and chest should fill the frame", (20, 45), 0.8, INFO)
        elif self.segmenter.active:
            cv2.circle(out, (32, 38), 10, RED, -1)
            put_text(out, "Signing...", (52, 47), 0.9, RED)
        else:
            put_text(out, idle_text, (20, 45), 0.8)


def draw_top5(out, guesses, title, highlight=None):
    """The top-5 list with confidence bars at the top right. `highlight` = word to mark (practice target)."""
    x0 = out.shape[1] - 330
    put_text(out, title, (x0, 45), 0.7, PURPLE)
    for i, (word, p) in enumerate(guesses):
        y = 80 + 34 * i
        color = GREEN if word == highlight else (PURPLE if i == 0 else DIM_PURPLE)
        cv2.rectangle(out, (x0, y - 16), (x0 + int(200 * p), y + 4), color, -1)
        put_text(out, f"{word}  {p:.0%}", (x0 + 6, y), 0.75 if i == 0 else 0.6)


class SignsMode(_SignBase):
    name = "ASL signs"
    hints = "[c] clear  [d] skeleton  [m] menu  [Tab] next mode  [q] quit"

    def enter(self):
        super().enter()
        self.guesses, self.transcript = [], []

    def on_key(self, key):
        if key == ord("c"):
            self.guesses, self.transcript = [], []

    def process(self, frame, res, now, dt):
        out = frame
        sign, width = self.sign_from(res, now)
        if sign is not None:
            self.guesses = self.recognizer.top(sign, k=5)
            self.transcript = (self.transcript + [self.guesses[0][0]])[-8:]
        if self.ctx.debug:
            draw_landmarks(out, res)
        self.status(out, res, width, "Sign a word, then lower your hands")
        if self.guesses:
            draw_top5(out, self.guesses, "Last sign")
        if self.transcript:
            put_text(out, "  ".join(self.transcript), (20, out.shape[0] - 55), 0.9)
        return out


# ---------------------------------------------------------------- Learn ASL (practice)

class PracticeMode(_SignBase):
    name = "Learn ASL"
    hints = "[n] next word  [d] skeleton  [m] menu  [Tab] next mode  [q] quit"
    FEEDBACK_SECONDS = 2.5      # how long a grade stays on screen
    ADVANCE_AFTER = 1.8         # after a correct sign, move to the next word this much later

    def __init__(self, ctx):
        super().__init__(ctx)
        cfg = ctx.cfg["signs"]
        self.session = PracticeSession(easiest_first(resolve(cfg["reports_dir"]) / "gru_per_sign.csv",
                                                     list(cfg["vocabulary"]), display_name))

    def enter(self):
        super().enter()
        self.guesses, self.grade, self.grade_time = [], None, 0.0

    def on_key(self, key):
        if key == ord("n"):
            self.session.next()
            self.guesses, self.grade = [], None

    def process(self, frame, res, now, dt):
        out = frame
        h, w = out.shape[:2]
        sign, width = self.sign_from(res, now)
        if sign is not None:
            self.guesses = self.recognizer.top(sign, k=5)
            self.grade, self.grade_time = self.session.grade(self.guesses), now
        if (self.grade is not None and self.grade.outcome == CORRECT and now - self.grade_time > self.ADVANCE_AFTER
                and self.grade.target == self.session.target):
            self.session.next()
            self.guesses = []
        if self.ctx.debug:
            draw_landmarks(out, res)

        self.status(out, res, width, "Sign it, then lower your hands")

        # the word to sign, top centre
        s = self.session
        panel(out, w // 2 - 250, 62, w // 2 + 250, 176, alpha=0.65)
        put_centered(out, f"Word {s.position} of {len(s.words)}", 88, 0.55, (220, 220, 220), 1)
        put_centered(out, f"Sign:  {s.target}", 132, 1.3, WHITE, 3)
        put_centered(out, "Unsure? Look it up on lifeprint.com or handspeak.com", 162, 0.45, (200, 200, 200), 1)

        # feedback banner
        if self.grade is not None and now - self.grade_time < self.FEEDBACK_SECONDS:
            g = self.grade
            if g.outcome == CORRECT:
                text, color = f"Correct!  {g.target}", GREEN
            elif g.outcome == CLOSE:
                text, color = f"Close: {g.target} was guess #{g.rank} (top guess: {g.guess})", AMBER
            else:
                text, color = f"Not quite: that looked like {g.guess}. Try again or press [n]", RED
            panel(out, 0, h - 150, w, h - 100, alpha=0.6)
            put_centered(out, text, h - 115, 0.8, color)
        if self.guesses:
            draw_top5(out, self.guesses, "Your sign looked like", highlight=s.target)

        # score and help, bottom left
        acc = f"  ({s.accuracy:.0%})" if s.attempts else ""
        put_text(out, f"Score {s.correct}/{s.attempts}{acc}   streak {s.streak}   best {s.best_streak}",
                 (20, h - 55), 0.7)
        return out


def _wrap(text, width):
    """Split text into lines of at most `width` characters (on spaces)."""
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])
