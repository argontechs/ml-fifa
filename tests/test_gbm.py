import numpy as np
import pandas as pd

from fifa import features
from fifa.gbm import GoalModel

RNG = np.random.default_rng(11)


def _synthetic(n=4000):
    """Goal counts driven by elo_diff — the model must learn the monotonic link."""
    elo_diff = RNG.uniform(-400, 400, n)
    lam_h = np.exp(0.1 + elo_diff / 600)
    lam_a = np.exp(0.1 - elo_diff / 600)
    X = pd.DataFrame(
        {
            c: RNG.uniform(0, 1, n)
            for c in features.FeatureBuilder.COLUMNS
            if c not in ("elo_diff", "confed_home", "confed_away")
        }
    )
    X["elo_diff"] = elo_diff
    X["confed_home"] = pd.Categorical(["UEFA"] * n, categories=features.CONFED_LEVELS)
    X["confed_away"] = pd.Categorical(["CAF"] * n, categories=features.CONFED_LEVELS)
    X = X[features.FeatureBuilder.COLUMNS]
    dates = pd.Series(pd.Timestamp("2024-01-01"), index=range(n))
    return X, RNG.poisson(lam_h), RNG.poisson(lam_a), dates


def test_learns_monotonic_elo_relationship():
    X, yh, ya, dates = _synthetic()
    gm = GoalModel().fit(X, yh, ya, dates, ref_date=pd.Timestamp("2024-06-01"))
    strong = X.iloc[[0]].assign(elo_diff=300.0)
    weak = X.iloc[[0]].assign(elo_diff=-300.0)
    lh_s, la_s = gm.predict_lambdas(strong)
    lh_w, la_w = gm.predict_lambdas(weak)
    assert lh_s[0] > lh_w[0] and la_s[0] < la_w[0]
    assert (lh_s > 0).all() and (la_s > 0).all()


def test_lambdas_clipped_to_sane_range():
    X, yh, ya, dates = _synthetic(500)
    gm = GoalModel().fit(X, yh, ya, dates, ref_date=pd.Timestamp("2024-06-01"))
    lh, la = gm.predict_lambdas(X)
    assert lh.max() <= 8.0 and lh.min() >= 0.05
