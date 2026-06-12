import numpy as np
import pandas as pd

from fifa.dixon_coles import DixonColes

RNG = np.random.default_rng(7)
TEAMS = {"Strong": 2.0, "Mid": 1.2, "Weak": 0.7}  # true scoring rates vs average


def _synthetic(n=3000):
    names = list(TEAMS)
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for i in range(n):
        h, a = RNG.choice(names, 2, replace=False)
        lam_h = TEAMS[h] / np.sqrt(TEAMS[a]) * 1.25  # home boost
        lam_a = TEAMS[a] / np.sqrt(TEAMS[h])
        rows.append(
            {
                "date": base_date + pd.Timedelta(days=int(i / 6)),
                "home_team": h,
                "away_team": a,
                "home_score": RNG.poisson(lam_h),
                "away_score": RNG.poisson(lam_a),
                "tournament": "Friendly",
                "neutral": False,
            }
        )
    return pd.DataFrame(rows)


def test_fit_recovers_strength_ordering_and_home_adv():
    df = _synthetic()
    dc = DixonColes(xi=0.0)  # no decay for the synthetic test
    dc.fit(df, ref_date=df["date"].max())
    lh_sw, la_sw = dc.predict_lambdas("Strong", "Weak", neutral=True)
    lh_ws, la_ws = dc.predict_lambdas("Weak", "Strong", neutral=True)
    assert lh_sw > la_sw and la_ws > lh_ws  # strength ordering
    assert dc.home_adv_ > 0.05  # home advantage exists
    lh_home, _ = dc.predict_lambdas("Strong", "Weak", neutral=False)
    assert lh_home > lh_sw  # home flag raises lambda


def test_unknown_team_falls_back_to_average():
    df = _synthetic(500)
    dc = DixonColes(xi=0.0)
    dc.fit(df, ref_date=df["date"].max())
    lh, la = dc.predict_lambdas("Atlantis", "Strong", neutral=True)
    assert 0.1 < lh < 3.0 and 0.1 < la < 4.0  # finite, sane fallback
