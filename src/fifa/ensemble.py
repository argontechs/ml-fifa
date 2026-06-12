"""Blend DC + GBM scoreline matrices; the single prediction entry point."""
from __future__ import annotations

import pandas as pd

from . import matrix as mx


class Predictor:
    def __init__(self, dc, gbm, fb, rho: float = -0.05, w_dc: float = 0.5):
        self.dc, self.gbm, self.fb = dc, gbm, fb
        self.rho, self.w_dc = rho, w_dc

    def matrix_from_lambdas(self, dc_l, gbm_l):
        m1 = mx.score_matrix(dc_l[0], dc_l[1], self.rho)
        m2 = mx.score_matrix(gbm_l[0], gbm_l[1], self.rho)
        m = self.w_dc * m1 + (1.0 - self.w_dc) * m2
        return m / m.sum()

    def matrix_for(self, home: str, away: str, date: pd.Timestamp,
                   tournament: str, neutral: bool, country: str | None = None):
        dc_l = self.dc.predict_lambdas(home, away, neutral)
        X = self.fb.features_for(home, away, date, tournament, neutral, country=country)
        lh, la = self.gbm.predict_lambdas(X)
        return self.matrix_from_lambdas(dc_l, (float(lh[0]), float(la[0])))
