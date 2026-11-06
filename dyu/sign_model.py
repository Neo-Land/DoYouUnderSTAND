"""GRU sequence classifier for ASL word signs (PyTorch, CPU-friendly).

Input: sign sequences from dyu.signs.sequence(), shape (SEQ_LEN, FRAME_DIM). The model reads the
frames in order, which lets it use motion and its direction (where a plain classifier on the
flattened sequence only sees positions).

    frame features -> linear + LayerNorm + GELU -> 2-layer bidirectional GRU
                   -> mean and max over time -> dropout -> linear -> one score per sign

Training (train_gru) standardizes features with the training set's mean/std, augments every batch
(random left/right mirror, small noise, slight scaling, slight time warp), and uses label smoothing
and weight decay. Early stopping picks the epoch with the best validation top-1.

torch is imported lazily, so the rest of dyu (and its tests) works without PyTorch installed.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .signs import mirror_sequence


@dataclass
class GRUConfig:
    hidden: int = 128          # GRU units per direction
    layers: int = 2
    proj: int = 128            # size of the per-frame projection before the GRU
    dropout: float = 0.3
    lr: float = 2e-3
    weight_decay: float = 1e-3
    label_smoothing: float = 0.1
    batch_size: int = 64
    max_epochs: int = 120
    patience: int = 20         # stop after this many epochs without a better val top-1
    noise: float = 0.02        # std of Gaussian noise added to standardized features
    seed: int = 0


def _build(n_in, n_classes, cfg):
    import torch
    import torch.nn as nn

    class SignGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(n_in, cfg.proj), nn.LayerNorm(cfg.proj), nn.GELU())
            self.gru = nn.GRU(cfg.proj, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                              bidirectional=True, dropout=cfg.dropout if cfg.layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(4 * cfg.hidden, n_classes))

        def forward(self, x):                        # x: (batch, time, features)
            h, _ = self.gru(self.proj(x))            # (batch, time, 2 * hidden)
            pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1)
            return self.head(pooled)

    return SignGRU()


def _time_warp(x, rng, max_shift=0.15):
    """Resample a (T, D) sequence on a slightly uneven time grid: signs are performed at different speeds."""
    T = len(x)
    knots = np.sort(np.concatenate([[0, T - 1], rng.uniform(0, T - 1, 2)]))
    jitter = rng.uniform(-max_shift, max_shift, 2) * T
    warped = knots.copy()
    warped[1:3] = np.clip(knots[1:3] + jitter, 0, T - 1)
    warped = np.sort(warped)
    src = np.interp(np.arange(T), knots, warped)          # where each output frame samples from
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo)[:, None]
    return (x[lo] * (1 - w) + x[hi] * w).astype(x.dtype)  # linear interpolation, all features at once


def augment(batch, rng):
    """Augment a batch of raw (unstandardized) sequences, shape (B, T, D): mirror, time warp, scale."""
    out = np.empty_like(batch)
    for i, seq in enumerate(batch):
        s = mirror_sequence(seq) if rng.random() < 0.5 else seq
        s = _time_warp(s, rng)
        s = s * rng.uniform(0.9, 1.1)                # slightly bigger / smaller signing space
        out[i] = s
    return out


class SignClassifier:
    """Fit / predict wrapper around the GRU, with feature standardization and class names."""

    def __init__(self, cfg=None):
        self.cfg = cfg or GRUConfig()
        self.net = None
        self.classes_ = None
        self.mean_ = self.std_ = None
        self.history = []

    # ---------------------------------------------------------------- training
    def fit(self, X, y, X_val=None, y_val=None, epochs=None, log=print):
        """Train on (N, T, D) sequences. With validation data, early-stop on val top-1; with `epochs`,
        train exactly that many epochs (used for the final refit on train + val)."""
        import torch

        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.classes_ = np.array(sorted(set(y)))
        index = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([index[c] for c in y])
        both = np.concatenate([X, np.stack([mirror_sequence(s) for s in X])])
        self.mean_ = both.reshape(-1, X.shape[2]).mean(0)
        self.std_ = both.reshape(-1, X.shape[2]).std(0) + 1e-6

        self.net = _build(X.shape[2], len(self.classes_), cfg)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        n_epochs = epochs or cfg.max_epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=n_epochs * int(np.ceil(len(X) / cfg.batch_size)))
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        best, best_state, since_best = -1.0, None, 0
        self.history = []
        for epoch in range(1, n_epochs + 1):
            self.net.train()
            order = rng.permutation(len(X))
            total = 0.0
            for start in range(0, len(X), cfg.batch_size):
                idx = order[start:start + cfg.batch_size]
                xb = augment(X[idx], rng)
                xb = (xb - self.mean_) / self.std_
                xb = xb + rng.normal(0, cfg.noise, xb.shape)
                logits = self.net(torch.as_tensor(xb, dtype=torch.float32))
                loss = loss_fn(logits, torch.as_tensor(yi[idx]))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
            row = {"epoch": epoch, "loss": total / len(X)}
            if X_val is not None:
                row["val_top1"] = self.top_k(X_val, y_val, 1)
                if row["val_top1"] > best:
                    best, since_best = row["val_top1"], 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                    row["best"] = True
                else:
                    since_best += 1
            self.history.append(row)
            if log and (epoch % 10 == 0 or row.get("best")):
                log("    epoch %3d  loss %.3f%s" % (epoch, row["loss"],
                    f"  val top-1 {row['val_top1']:.1%}" + (" *" if row.get("best") else "") if "val_top1" in row else ""))
            if X_val is not None and since_best >= cfg.patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @property
    def best_epoch(self):
        marked = [r["epoch"] for r in self.history if r.get("best")]
        return marked[-1] if marked else len(self.history)

    # ---------------------------------------------------------------- prediction
    def predict_proba(self, X):
        import torch

        self.net.eval()
        with torch.no_grad():
            x = torch.as_tensor((np.asarray(X) - self.mean_) / self.std_, dtype=torch.float32)
            return torch.softmax(self.net(x), dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def top_k(self, X, y, k):
        top = self.classes_[np.argsort(-self.predict_proba(X), axis=1)[:, :k]]
        return float(np.mean([t in row for t, row in zip(y, top, strict=True)]))

    # ---------------------------------------------------------------- save / load
    def save(self, path):
        import torch

        torch.save({"state": self.net.state_dict(), "config": asdict(self.cfg), "classes": list(self.classes_),
                    "mean": self.mean_, "std": self.std_, "n_in": len(self.mean_)}, path)

    @classmethod
    def load(cls, path):
        import torch

        blob = torch.load(path, weights_only=False)
        obj = cls(GRUConfig(**blob["config"]))
        obj.classes_ = np.array(blob["classes"])
        obj.mean_, obj.std_ = blob["mean"], blob["std"]
        obj.net = _build(blob["n_in"], len(obj.classes_), obj.cfg)
        obj.net.load_state_dict(blob["state"])
        obj.net.eval()
        return obj
