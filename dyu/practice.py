"""Learn ASL practice mode: the app names a word, the user signs it, and gets graded.

Pure logic with no camera or drawing, so it is unit-tested. The app's PracticeMode (dyu/modes.py)
feeds it the recognizer's top-5 guesses for each sign the user makes.

Words come easiest first, ordered by the sign model's per-sign accuracy on the test signers
(reports/signs/gru_per_sign.csv), so a beginner gets early successes before the hard signs.
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path

CORRECT, CLOSE, WRONG = "correct", "close", "wrong"


@dataclass
class Grade:
    """The verdict on one attempt."""
    outcome: str            # CORRECT (top guess), CLOSE (in the top 5), WRONG
    target: str             # the word that was asked for
    guess: str              # the model's top guess
    rank: int | None        # 1-5 if the target was in the top 5, else None


@dataclass
class PracticeSession:
    """Keeps the word list, the current target, and the score."""
    words: list[str]
    index: int = 0
    attempts: int = 0
    correct: int = 0
    close: int = 0
    streak: int = 0
    best_streak: int = 0
    history: list[Grade] = field(default_factory=list)

    @property
    def target(self):
        return self.words[self.index % len(self.words)]

    @property
    def position(self):
        """1-based position of the current word in the list."""
        return self.index % len(self.words) + 1

    def grade(self, guesses):
        """Grade one attempt. `guesses` is [(word, probability), ...], best first."""
        words = [w for w, _ in guesses]
        rank = words.index(self.target) + 1 if self.target in words else None
        outcome = CORRECT if rank == 1 else CLOSE if rank else WRONG
        self.attempts += 1
        if outcome == CORRECT:
            self.correct += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            self.streak = 0
            if outcome == CLOSE:
                self.close += 1
        result = Grade(outcome, self.target, words[0] if words else "", rank)
        self.history.append(result)
        return result

    def next(self):
        """Move on to the next word (wraps around after the last)."""
        self.index += 1
        return self.target

    @property
    def accuracy(self):
        return self.correct / self.attempts if self.attempts else 0.0


def easiest_first(per_sign_csv, vocabulary, display):
    """Display names of `vocabulary`, sorted by descending test accuracy (unknown signs go last).

    `display` turns a dataset gloss into the word the recognizer reports (e.g. 'WANT1' -> 'WANT').
    """
    accuracy = {}
    path = Path(per_sign_csv)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                accuracy[row["gloss"]] = float(row["accuracy"])
    order = sorted(vocabulary, key=lambda g: (-accuracy.get(g, -1.0), vocabulary.index(g)))
    return [display(g) for g in order]
