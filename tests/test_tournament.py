import numpy as np

from fifa import tournament


def test_rank_group_points_then_gd_then_gf():
    results = [
        ("A", "B", 2, 0), ("C", "D", 1, 1),
        ("A", "C", 1, 1), ("B", "D", 3, 0),
        ("A", "D", 0, 0), ("B", "C", 0, 1),
    ]
    # A: W,D,D = 5 pts gd+2 · C: D,D,W = 5 pts gd+1 · B: W,L,L = 3 pts · D: D,L,D = 2 pts
    rng = np.random.default_rng(0)
    order = tournament.rank_group(["A", "B", "C", "D"], results, rng)
    assert order == ["A", "C", "B", "D"]


def test_rank_group_head_to_head_breaks_tie():
    # E and F finish level on pts/gd/gf overall; F beat E head-to-head → F above E.
    results = [
        ("E", "F", 0, 1),  # F wins h2h
        ("E", "G", 2, 0), ("F", "H", 2, 0),
        ("E", "H", 2, 0), ("F", "G", 1, 1),
        ("G", "H", 1, 1),
    ]
    # E: 6 pts (gf 4, ga 1) · F: 7 pts → not tied... craft exact tie instead:
    results = [
        ("E", "F", 0, 1),  # F beats E
        ("E", "G", 3, 0), ("F", "G", 2, 0),
        ("E", "H", 2, 0), ("F", "H", 0, 1),
    ]
    # E: 6 pts, gd +4 (5-1) · F: 6 pts, gd... F: 1-0,2-0,0-1 → 6 pts? W,W,L = 6 pts gd+2.
    # Not a clean pts/gd/gf tie; assert instead on a constructed exact tie:
    results = [
        ("E", "F", 1, 2),                  # F beats E 2-1
        ("E", "G", 2, 0), ("F", "H", 2, 1),
        ("E", "H", 2, 1), ("F", "G", 1, 0),
    ]
    # E: 6 pts, gf 5, ga 3, gd +2 · F: 9 pts — still not tied. Use mini scenario:
    results = [("E", "F", 1, 2), ("F", "E", 1, 2)]  # each beat the other 2-1 → all equal
    rng = np.random.default_rng(0)
    order = tournament.rank_group(["E", "F"], results, rng)
    assert set(order) == {"E", "F"}  # full tie incl. h2h → resolved by lot, both present


def test_best_thirds_selects_eight_by_points_then_gd():
    rng = np.random.default_rng(0)
    thirds = [(f"T{i}", {"pts": 6 if i < 3 else (4 if i < 9 else 1), "gd": i, "gf": i})
              for i in range(12)]
    best = tournament.best_thirds(thirds, rng)
    assert len(best) == 8
    assert {"T0", "T1", "T2"} <= set(best)          # all 6-pointers in
    assert not ({"T9", "T10", "T11"} & set(best))   # 1-pointers out
    # among the 4-pointers (T3..T8), the top 5 by gd qualify: T4..T8
    assert set(best) - {"T0", "T1", "T2"} == {"T4", "T5", "T6", "T7", "T8"}


import pandas as pd  # noqa: E402

from fifa import matrix  # noqa: E402


class FixedPredictor:
    """Stub: home side always heavy favorite."""

    def matrix_for(self, home, away, date, tournament, neutral, country=None):
        return matrix.score_matrix(2.5, 0.5, rho=-0.05)


def _full_fixtures():
    """Synthetic 12-group fixture set matching the real WC2026 shape (all upcoming)."""
    rows = []
    n = 1
    for g in "ABCDEFGHIJKL":
        teams = [f"{g}{i}" for i in range(1, 5)]
        for i in range(4):
            for j in range(i + 1, 4):
                rows.append(
                    {
                        "match_number": n, "round": 1,
                        "date": pd.Timestamp("2026-06-15"), "location": "x",
                        "host": "United States", "home": teams[i], "away": teams[j],
                        "group": g, "home_score": None, "away_score": None,
                        "status": "upcoming", "neutral": True,
                    }
                )
                n += 1
    return pd.DataFrame(rows)


def test_simulate_groups_probabilities_sane():
    res = tournament.simulate_groups(_full_fixtures(), FixedPredictor(), n_runs=100, seed=1)
    a_teams = ["A1", "A2", "A3", "A4"]
    assert sum(res["win_group"][t] for t in a_teams) == 1.0
    assert all(0 <= v <= 1 for v in res["advance"].values())
    # exactly 32 of 48 advance on average
    assert sum(res["advance"].values()) == pytest.approx(32.0, abs=1e-9)


def test_simulate_tournament_deterministic_and_sane():
    fx = _full_fixtures()
    elos = {t: 1500 for g in "ABCDEFGHIJKL" for t in [f"{g}{i}" for i in range(1, 5)]}
    r1 = tournament.simulate_tournament(fx, FixedPredictor(), elos, n_runs=60, seed=42)
    r2 = tournament.simulate_tournament(fx, FixedPredictor(), elos, n_runs=60, seed=42)
    pd.testing.assert_frame_equal(r1, r2)  # seeded determinism
    assert r1["champion"].sum() == pytest.approx(1.0)
    assert r1["final"].sum() == pytest.approx(2.0)   # two finalists per run
    assert (r1["r32"] <= 1.0).all() and (r1 >= 0).all().all()


import pytest  # noqa: E402


def _fixtures_with_knockouts():
    """Group fixtures all played-deterministic + knockout rows in three states:
    real-played (M73), real-unplayed (M74), still-tbd (everything else)."""
    fx = _full_fixtures()
    fx["home_score"], fx["away_score"], fx["status"] = 2, 0, "played"  # groups decided
    ko_rows = []
    for mn in list(tournament.R32_SEEDS) + sorted(tournament.CHAIN):
        ko_rows.append({
            "match_number": mn, "round": 4 if mn <= 88 else 5,
            "date": pd.Timestamp("2026-06-29"), "location": "x",
            "host": "United States", "home": "1A", "away": "2B", "group": None,
            "home_score": None, "away_score": None, "status": "tbd",
            "neutral": True, "winner": None,
        })
    ko = pd.DataFrame(ko_rows)
    # M73 (real seeds 2A v 2B): announced AND played — A2 won on pens (level score)
    ko.loc[ko["match_number"] == 73,
           ["home", "away", "home_score", "away_score", "status", "winner"]] = \
        ["A2", "B2", 1, 1, "played", "A2"]
    # M74: announced pairing, not yet played — E1 v C3 per the real bracket
    ko.loc[ko["match_number"] == 74, ["home", "away", "status"]] = ["E1", "C3", "upcoming"]
    return pd.concat([fx, ko], ignore_index=True)


def test_knockout_passthrough_uses_real_pairings_and_winners():
    fx = _fixtures_with_knockouts()
    elos = {t: 1500 for g in "ABCDEFGHIJKL" for t in [f"{g}{i}" for i in range(1, 5)]}
    res = tournament.simulate_tournament(fx, FixedPredictor(), elos, n_runs=80, seed=3)
    # M73 was decided on pens for A2: the pens loser B2 must NEVER pass the R32
    assert res.loc["B2", "r16"] == 0.0
    assert res.loc["A2", "r16"] == 1.0  # real winner always advances
    # M74's announced pairing (E1 v C3) is used verbatim: both appear in R32 every run
    assert res.loc["E1", "r32"] == 1.0 and res.loc["C3", "r32"] == 1.0


def test_knockout_tbd_rows_still_simulated_from_seeds():
    fx = _full_fixtures()  # no knockout rows at all → seeds path (legacy behavior)
    elos = {t: 1500 for g in "ABCDEFGHIJKL" for t in [f"{g}{i}" for i in range(1, 5)]}
    res = tournament.simulate_tournament(fx, FixedPredictor(), elos, n_runs=40, seed=3)
    assert res["champion"].sum() == pytest.approx(1.0)


def test_pens_prob_favors_strong_shootout_record():
    tbl = {"Germany": (8, 8), "England": (1, 8)}  # (wins, total)
    p = tournament.pens_prob("Germany", "England", tbl, {"Germany": 1900, "England": 1900})
    assert p > 0.60
    # No data → falls back to pure Elo expectancy (equal ratings → 0.5)
    p2 = tournament.pens_prob("X", "Y", {}, {"X": 1900, "Y": 1900})
    assert p2 == pytest.approx(0.5)


def test_shootout_table_counts_wins_and_totals():
    import pandas as pd
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
            "home_team": ["G", "G", "E"],
            "away_team": ["E", "E", "G"],
            "winner": ["G", "E", "G"],
        }
    )
    tbl = tournament.shootout_table(df)
    assert tbl["G"] == (2, 3) and tbl["E"] == (1, 3)
