"""Turns noisy per-frame predictions into a stable "which Stand is active" decision."""
from collections import Counter, deque


class PoseSmoother:
    """Hold a pose to summon its Stand; the Stand stays until you hold the dismiss pose.

    Each frame votes for its predicted label if the classifier's confidence is >= `threshold`,
    otherwise for `none`. The "winner" is the label holding at least `min_votes` of the last
    `window` votes (if none does, there is no winner). Then:

    - Summon: a Stand pose that stays the winner for `hold_seconds` summons its Stand. If another
      Stand is showing, it switches to the new one.
    - Stay: the active Stand stays through normal movement (`none`), brief flickers and
      low-confidence frames.
    - Dismiss: `dismiss_label` staying the winner for `dismiss_seconds` removes the Stand.
    - Absent: if no usable frame arrives (nobody in frame, or not framed head to knees: the caller
      passes label=None) for `absent_seconds`, the Stand goes away, so it doesn't linger in an
      empty room.

    Times are in seconds, from the `now` passed to update(), so behaviour doesn't depend on fps.
    """

    def __init__(self, window=20, min_votes=18, threshold=0.4, hold_seconds=1.5, dismiss_seconds=1.0,
                 absent_seconds=5.0, none_label="none", dismiss_label="dismiss"):
        self.window = window
        self.min_votes = min_votes
        self.threshold = threshold
        self.hold = hold_seconds
        self.dismiss_hold = dismiss_seconds
        self.absent = absent_seconds
        self.none = none_label
        self.dismiss = dismiss_label
        self.reset()

    @classmethod
    def from_config(cls, rec):
        """Build from the config.yaml `recognition` section."""
        keys = ("window", "min_votes", "threshold", "hold_seconds", "dismiss_seconds", "absent_seconds")
        return cls(**{k: rec[k] for k in keys if k in rec})

    def reset(self):
        self.history = deque(maxlen=self.window)
        self.current = None
        self.candidate, self.since = None, 0.0   # current winner and when it started winning
        self.last_seen = None                     # last time a usable frame arrived

    def progress(self, now):
        """(label, 0..1): the pose being held and how far along its hold is, for a HUD bar."""
        if self.candidate in (None, self.none) or self.candidate == self.current:
            return None, 0.0
        need = self.dismiss_hold if self.candidate == self.dismiss else self.hold
        if self.candidate == self.dismiss and self.current is None:
            return None, 0.0
        return self.candidate, min(1.0, (now - self.since) / max(need, 1e-6))

    def update(self, label, prob, now):
        if label is not None:
            self.last_seen = now
        elif self.current is not None and self.last_seen is not None and now - self.last_seen > self.absent:
            self.current = None

        vote = label if (label is not None and prob >= self.threshold) else self.none
        self.history.append(vote)
        top, count = Counter(self.history).most_common(1)[0]
        winner = top if count >= self.min_votes else None
        if winner != self.candidate:
            self.candidate, self.since = winner, now

        held = now - self.since
        if winner == self.dismiss:
            if self.current is not None and held >= self.dismiss_hold:
                self.current = None
        elif winner not in (None, self.none) and winner != self.current and held >= self.hold:
            self.current = winner
        return self.current
