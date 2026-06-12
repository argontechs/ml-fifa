import pytest

from fifa import elo


def test_expected_equal_neutral_is_half():
    assert elo.expected(1500, 1500, neutral=True) == pytest.approx(0.5)


def test_expected_home_advantage_100():
    # +100 Elo home edge → We = 1/(1+10^-0.25) ≈ 0.640
    assert elo.expected(1500, 1500, neutral=False) == pytest.approx(0.640065, abs=1e-5)


def test_goal_multiplier_table():
    assert [elo.goal_multiplier(m) for m in (0, 1, 2, 3, 4, 5)] == [1.0, 1.0, 1.5, 1.75, 1.875, 2.0]


def test_tournament_k_tiers():
    assert elo.tournament_k("FIFA World Cup") == 60
    assert elo.tournament_k("Copa América") == 50
    assert elo.tournament_k("UEFA Euro qualification") == 40  # 'qualification' beats Euro
    assert elo.tournament_k("FIFA World Cup qualification") == 40
    assert elo.tournament_k("UEFA Nations League") == 40
    assert elo.tournament_k("Friendly") == 20
    assert elo.tournament_k("King's Cup") == 30  # unknown minor → 30


def test_update_rule_hand_example():
    # Equal teams, neutral, home wins 2-0 in a qualifier: ΔR = 40 · 1.5 · (1−0.5) = 30
    we = elo.expected(1500, 1500, neutral=True)
    assert 40 * elo.goal_multiplier(2) * (1.0 - we) == pytest.approx(30.0)


def test_2022_final_reproduces_published_change():
    # eloratings.net/2022_results.tsv line 968:
    #   2022 12 18  AR FR  3 3  WC QA  -6  2144 2081  (ratings are POST-match)
    # Zero-sum ⇒ pre-match: Argentina 2150, France 2075. Neutral venue (Qatar),
    # 3-3 after ET → W=0.5, G=1.0, K=60. Published change: −6.
    we = elo.expected(2150, 2075, neutral=True)
    delta = 60 * elo.goal_multiplier(0) * (0.5 - we)
    assert round(delta) == -6


import pandas as pd  # noqa: E402


def _mini_df():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
            "home_team": ["A", "A", "B"],
            "away_team": ["B", "C", "C"],
            "home_score": [2, 0, 1],
            "away_score": [0, 0, 1],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "neutral": [True, False, True],
        }
    )


def test_compute_elo_chronology_and_zero_sum():
    out, ratings = elo.compute_elo(_mini_df())
    # Match 1: A 2-0 B neutral friendly: ΔR = 20·1.5·(1−0.5) = +15
    assert out.loc[0, "elo_home_pre"] == 1500 and out.loc[0, "elo_away_pre"] == 1500
    assert out.loc[0, "elo_home_post"] == pytest.approx(1515)
    assert out.loc[0, "elo_away_post"] == pytest.approx(1485)
    # Match 2 uses A's UPDATED rating as pre-match (no leakage of later matches)
    assert out.loc[1, "elo_home_pre"] == pytest.approx(1515)
    # Zero-sum: total rating mass conserved
    assert sum(ratings.values()) == pytest.approx(1500 * 3)


def test_compute_elo_pre_never_includes_own_match():
    df = _mini_df()
    out1, _ = elo.compute_elo(df)
    df2 = df.copy()
    df2.loc[2, "home_score"] = 9  # changing last match must not change its OWN pre-Elo
    out2, _ = elo.compute_elo(df2)
    assert out1.loc[2, "elo_home_pre"] == out2.loc[2, "elo_home_pre"]
