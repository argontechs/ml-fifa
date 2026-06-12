"""Isotonic W/D/L calibration: makes '70% confident' mean 'wins 70% of the time'."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class WDLCalibrator:
    def fit(self, probs, outcomes) -> "WDLCalibrator":
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes)
        self.curves_ = []
        for k in range(3):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
            iso.fit(probs[:, k], (outcomes == k).astype(float))
            self.curves_.append((iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()))
        return self

    def transform(self, probs):
        probs = np.atleast_2d(np.asarray(probs, dtype=float))
        cols = [np.interp(probs[:, k], *self.curves_[k]) for k in range(3)]
        out = np.column_stack(cols)
        return out / out.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict:
        return {"curves": self.curves_}

    @classmethod
    def from_dict(cls, d) -> "WDLCalibrator":
        obj = cls()
        obj.curves_ = [tuple(c) for c in d["curves"]]
        return obj
