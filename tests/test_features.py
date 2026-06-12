import numpy as np
import pandas as pd
import pytest

from fifa import elo, features


def _df():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-11", "2020-02-01", "2020-03-01"]),
            "home_team": ["A", "B", "A", "A"],
            "away_team": ["B", "A", "C", "B"],
            "home_score": [3, 1, 0, 2],
            "away_score": [0, 1, 2, 1],
            "tournament": ["Friendly"] * 4,
            "country": ["A", "B", "A", "A"],
            "neutral": [False, False, True, False],
        }
    )
    out, _ = elo.compute_elo(df)
    return out


def test_features_shapes_and_basics():
    fb = features.FeatureBuilder()
    X, y_home, y_away = fb.fit_transform(_df())
    assert len(X) == 4 and list(y_home) == [3, 1, 0, 2]
    # Match 0: nobody has played → neutral priors
    assert X.loc[0, "played_home"] == 0
    assert X.loc[0, "rest_home"] == features.REST_CAP
    assert X.loc[0, "winrate_home"] == pytest.approx(0.5)
    # Match 1 (B v A, 10 days later): both played once
    assert X.loc[1, "played_home"] == 1
    assert X.loc[1, "rest_home"] == 10
    # A won match 0 → A's winrate before match 1 is 1.0 (A is away side here)
    assert X.loc[1, "winrate_away"] == pytest.approx(1.0)
    # elo_diff includes the +100 home-advantage term on non-neutral rows
    df = _df()
    expected_diff = df.loc[0, "elo_home_pre"] - df.loc[0, "elo_away_pre"] + 100
    assert X.loc[0, "elo_diff"] == pytest.approx(expected_diff)


def test_leakage_guard_features_ignore_own_result():
    df1, df2 = _df(), _df()
    df2.loc[3, "home_score"] = 9  # mutate final match result
    X1, _, _ = features.FeatureBuilder().fit_transform(df1)
    X2, _, _ = features.FeatureBuilder().fit_transform(df2)
    pd.testing.assert_series_equal(X1.loc[3], X2.loc[3])  # its own features unchanged


def test_features_for_future_fixture_matches_training_columns():
    fb = features.FeatureBuilder()
    X, _, _ = fb.fit_transform(_df())
    row = fb.features_for("A", "C", pd.Timestamp("2020-04-01"), "FIFA World Cup", neutral=False)
    assert list(row.columns) == list(X.columns)
    assert row.loc[0, "k_tier"] == 60


def test_momentum_and_wc_experience():
    n = 30
    df = pd.DataFrame(
        {
            "date": pd.to_datetime("2018-01-01") + pd.to_timedelta(np.arange(n) * 40, unit="D"),
            "home_team": ["A"] * n,
            "away_team": ["B"] * n,
            "home_score": [2] * n,
            "away_score": [0] * n,
            "tournament": ["FIFA World Cup"] * 3 + ["Friendly"] * (n - 3),
            "country": ["X"] * n,
            "neutral": [True] * n,
        }
    )
    out, _ = elo.compute_elo(df)
    X, _, _ = features.FeatureBuilder().fit_transform(out)
    last = X.iloc[-1]
    assert last["elo_trend1y_home"] > 0  # A keeps winning → momentum up over past year
    assert last["elo_trend1y_away"] < 0  # B keeps losing → momentum down
    assert last["wc_exp_home"] == 3  # career World Cup finals matches before this game
