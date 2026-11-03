"""
PPMSP instance: n vehicles, horizon of T slots; vehicle i must charge p_i slots inside its window
[s_i, e_i] (0-based, inclusive) at power w_i. Powers are stored in kW and as integers in units of
`UNIT` per kW (resolution delta = 1 / UNIT kW), so that slot loads have exact discrete laws.
"""

from dataclasses import dataclass, field

import numpy as np

UNIT = 10  # 1 unit = 0.1 kW (all benchmark powers are multiples of 0.1 kW)


@dataclass
class Instance:
    s: np.ndarray            # (n,) first slot, 0-based
    e: np.ndarray            # (n,) last slot, 0-based inclusive
    p: np.ndarray            # (n,) duration in slots
    w: np.ndarray            # (n,) power in kW
    T: int
    name: str = ""
    family: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.s = np.asarray(self.s, np.int64)
        self.e = np.asarray(self.e, np.int64)
        self.p = np.asarray(self.p, np.int64)
        self.wu = np.rint(np.asarray(self.w, float) * UNIT).astype(np.int64)
        self.w = self.wu / UNIT
        self.n = int(len(self.s))
        self.T = int(self.T)
        if (self.p < 1).any() or (self.p > self.e - self.s + 1).any() or (self.s < 0).any() or (self.e >= self.T).any():
            raise ValueError(f"invalid instance {self.name}: need 1 <= p_i <= |window| and windows inside [0, T)")
        self.win = [np.arange(self.s[i], self.e[i] + 1) for i in range(self.n)]
        avail = [[] for _ in range(self.T)]
        for i in range(self.n):
            for t in range(self.s[i], self.e[i] + 1):
                avail[t].append(i)
        self.avail = avail

    # --- basic quantities ---
    @property
    def w_max(self):
        return float(self.w.max())

    @property
    def n_vars(self):
        return int((self.e - self.s + 1).sum())

    def loads(self, x):
        """Slot loads in units (int64) of a 0/1 schedule x (n x T)."""
        return (self.wu[:, None] * np.asarray(x, np.int64)).sum(axis=0)

    def peak(self, x):
        return float(self.loads(x).max()) / UNIT

    def is_feasible(self, x):
        x = np.asarray(x)
        if x.shape != (self.n, self.T) or not np.isin(x, (0, 1)).all():
            return False
        for i in range(self.n):
            if x[i].sum() != self.p[i] or x[i, :self.s[i]].any() or x[i, self.e[i] + 1:].any():
                return False
        return True

    def features(self):
        L = self.e - self.s + 1
        pw = np.unique(self.wu)
        return {"n": self.n, "T": self.T, "K": int(len(pw)), "mean_window": float(L.mean()),
                "mean_p": float(self.p.mean()), "load_factor": float((self.p / L).mean()),
                "w_max": self.w_max, "energy": float((self.w * self.p).sum()),
                "n_vars": self.n_vars}

    def to_dict(self):
        return {"s": self.s.tolist(), "e": self.e.tolist(), "p": self.p.tolist(), "w": self.w.tolist(),
                "T": self.T, "name": self.name, "family": self.family, "meta": self.meta}

    @staticmethod
    def from_dict(d):
        return Instance(np.array(d["s"]), np.array(d["e"]), np.array(d["p"]), np.array(d["w"]), d["T"],
                        d.get("name", ""), d.get("family", ""), d.get("meta", {}))
