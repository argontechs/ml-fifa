"""Blend DC + GBM scoreline matrices; the single prediction entry point."""
from __future__ import annotations

import pandas as pd

from . import matrix as mx


# How much to trust the bookmaker consensus over the model, scaled by book depth.
# A 40+ book World Cup slate is the sharpest signal in football → trust it 0.70 (the
# literature puts the sharp-market weight in the 0.6–0.8 band). A thin 1–2 book market
# is noisier → fall back toward an even split. Reversible: tune the two endpoints.
DEEP_BOOK, THIN_BOOK = 8, 2
W_DEEP, W_THIN = 0.70, 0.50


def market_weight(n_books: int) -> float:
    if n_books >= DEEP_BOOK:
        return W_DEEP
    if n_books <= THIN_BOOK:
        return W_THIN
    frac = (n_books - THIN_BOOK) / (DEEP_BOOK - THIN_BOOK)
    return W_THIN + (W_DEEP - W_THIN) * frac


class Predictor:
    def __init__(self, dc, gbm, fb, rho: float = -0.05, w_dc: float = 0.5,
                 calibrator=None, book=None):
        self.dc, self.gbm, self.fb = dc, gbm, fb
        self.rho, self.w_dc = rho, w_dc
        self.calibrator = calibrator
        self.book = book or {}

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
        m = self.matrix_from_lambdas(dc_l, (float(lh[0]), float(la[0])))
        if self.calibrator is not None:
            m = mx.rescale_wdl(m, tuple(self.calibrator.transform(mx.wdl(m))[0]))
        market = self.book.get((home, away))
        if market is not None:
            book_p = market[:3]
            n_books = market[3] if len(market) > 3 else DEEP_BOOK  # back-compat
            w = market_weight(n_books)
            model_p = mx.wdl(m)
            target = tuple(w * bp + (1 - w) * op for bp, op in zip(book_p, model_p))
            m = mx.rescale_wdl(m, target)
        return m
