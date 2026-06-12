"""Twin LightGBM Poisson regressors for (lambda_home, lambda_away)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

TIME_DECAY = 0.0003          # per day (tuned 2026-06-12, val RPS sweep)
FRIENDLY_WEIGHT = 0.5

_PARAMS = dict(
    objective="poisson",
    n_estimators=600,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)


class GoalModel:
    def __init__(self, **overrides):
        self.params = {**_PARAMS, **overrides}

    def fit(self, X: pd.DataFrame, y_home, y_away, dates: pd.Series,
            ref_date: pd.Timestamp) -> "GoalModel":
        days = (ref_date - pd.to_datetime(dates)).dt.days.to_numpy().clip(min=0)
        w = np.exp(-TIME_DECAY * days)
        w = w * np.where(X["k_tier"].to_numpy() == 20.0, FRIENDLY_WEIGHT, 1.0)
        self.home_ = LGBMRegressor(**self.params).fit(X, y_home, sample_weight=w)
        self.away_ = LGBMRegressor(**self.params).fit(X, y_away, sample_weight=w)
        return self

    def predict_lambdas(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lh = np.clip(self.home_.predict(X), 0.05, 8.0)
        la = np.clip(self.away_.predict(X), 0.05, 8.0)
        return lh, la
