import pytest

from fifa import ensemble


def test_market_weight_deep_books_trust_070():
    # the full WC slate carries 40+ books → sharpest signal → full trust
    assert ensemble.market_weight(42) == pytest.approx(0.70)
    assert ensemble.market_weight(8) == pytest.approx(0.70)  # at the deep threshold


def test_market_weight_thin_books_fall_back():
    # a 1–2 book market is noisy → fall back toward an even split with the model
    assert ensemble.market_weight(1) == pytest.approx(0.50)
    assert ensemble.market_weight(2) == pytest.approx(0.50)


def test_market_weight_ramps_monotonically():
    ws = [ensemble.market_weight(n) for n in range(1, 10)]
    assert ws == sorted(ws)  # non-decreasing in book depth
    assert all(0.50 <= w <= 0.70 for w in ws)
    assert ensemble.market_weight(5) == pytest.approx(0.50 + 0.20 * (5 - 2) / (8 - 2))


def test_predictor_blend_uses_depth_aware_weight(monkeypatch):
    # a deep-book market pull should move the W/D/L further than a thin one
    import numpy as np

    from fifa import matrix as mx

    class _Stub:
        def predict_lambdas(self, *a, **k):
            return (1.4, 1.0)

    class _GBM:
        def predict_lambdas(self, X):
            return np.array([1.4]), np.array([1.0])

    class _FB:
        def features_for(self, *a, **k):
            return None

    base = ensemble.Predictor(_Stub(), _GBM(), _FB(), rho=-0.05, w_dc=0.5)
    market = (0.20, 0.30, 0.50)  # market disagrees with the model (model favors home)
    import pandas as pd
    ts = pd.Timestamp("2026-06-20")

    base.book = {("A", "B"): (*market, 42)}  # deep
    deep = mx.wdl(base.matrix_for("A", "B", ts, "wc", True))
    base.book = {("A", "B"): (*market, 1)}   # thin
    thin = mx.wdl(base.matrix_for("A", "B", ts, "wc", True))

    # deep books pull the away prob closer to the market's 0.50 than thin books do
    assert abs(deep[2] - 0.50) < abs(thin[2] - 0.50)
