"""Turns noisy per-frame predictions into a stable "which Stand is active" decision."""
from collections import Counter, deque


class PoseSmoother:
    """Majority vote over the last `window` frames, with a confidence threshold and linger time.

    - A frame votes for its predicted label only if confidence >= threshold; otherwise it votes "none".
    - A Stand activates when it wins at least `min_votes` of the last `window` frames.
    - Once active, it stays until `linger_seconds` pass without it winning (so it doesn't
      flicker off the moment you shift your weight).
    - A different Stand winning switches immediately.
    """

    def __init__(self, window=10, min_votes=7, threshold=0.6, linger_seconds=1.5, none_label="none"):
        self.window = window
        self.min_votes = min_votes
        self.threshold = threshold
        self.linger = linger_seconds
        self.none = none_label
        self.history = deque(maxlen=window)
        self.current = None
        self.last_seen = 0.0

    def update(self, label, prob, now):
        vote = label if (label is not None and prob >= self.threshold) else self.none
        self.history.append(vote)
        top, count = Counter(self.history).most_common(1)[0]
        if top != self.none and count >= self.min_votes:
            self.current = top
            self.last_seen = now
        elif self.current is not None and now - self.last_seen > self.linger:
            self.current = None
        return self.current

    def reset(self):
        self.history.clear()
        self.current = None
