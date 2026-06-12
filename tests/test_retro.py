import pandas as pd

from fifa import ledger, matrix, retro


class StubPredictor:
    def matrix_for(self, home, away, date, tournament, neutral, country=None):
        return matrix.score_matrix(2.0, 0.6, rho=-0.05)


def _fx():
    return pd.DataFrame(
        {
            "match_number": [1, 2, 3],
            "round": [1, 1, 1],
            "date": pd.to_datetime(["2026-06-11 19:00", "2026-06-12 02:00", "2026-06-13 22:00"]),
            "host": ["Mexico", "Mexico", "United States"],
            "home": ["Mexico", "South Korea", "Brazil"],
            "away": ["South Africa", "Czech Republic", "Morocco"],
            "group": ["A", "A", "C"],
            "home_score": [2, 2, None],
            "away_score": [0, 1, None],
            "status": ["played", "played", "upcoming"],
            "neutral": [False, True, True],
        }
    )


def test_backfill_only_missing_played_matches(tmp_path):
    path = tmp_path / "ledger.jsonl"
    cutoffs = []

    def stub_trainer(cutoff):
        cutoffs.append(cutoff)
        return StubPredictor()

    n = retro.backfill(_fx(), path, trainer=stub_trainer)
    assert n == 2  # the two played matches; the upcoming one is NOT backfilled
    book = ledger.load(path)
    assert book[1]["retro"] is True and book[2]["retro"] is True
    assert 3 not in book
    # one trainer fit per distinct matchday, cutoff = that day at midnight
    assert cutoffs == [pd.Timestamp("2026-06-11"), pd.Timestamp("2026-06-12")]


def test_backfill_never_overwrites_frozen_picks(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.record([{"match_number": 1, "home": "Mexico", "away": "South Africa",
                    "kickoff": "x", "predicted": [9, 9], "p": [1, 0, 0], "tier": "LOCK"}], path)
    n = retro.backfill(_fx(), path, trainer=lambda c: StubPredictor())
    assert n == 1  # only match 2 backfilled
    assert ledger.load(path)[1]["predicted"] == [9, 9]  # frozen pick untouched


def test_tracker_carries_retro_flag(tmp_path):
    path = tmp_path / "ledger.jsonl"
    retro.backfill(_fx(), path, trainer=lambda c: StubPredictor())
    rows, tally = ledger.tracker(ledger.load(path), _fx())
    assert all(r["retro"] for r in rows)
    assert tally["n"] == 2
