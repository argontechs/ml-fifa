import pytest

from fifa import odds

SAMPLE = [
    {
        "home_team": "France",
        "away_team": "Senegal",
        "bookmakers": [
            {"key": "b1", "markets": [{"key": "h2h", "outcomes": [
                {"name": "France", "price": 1.60},
                {"name": "Senegal", "price": 6.0},
                {"name": "Draw", "price": 4.0},
            ]}]},
            {"key": "b2", "markets": [{"key": "h2h", "outcomes": [
                {"name": "France", "price": 1.66},
                {"name": "Senegal", "price": 5.8},
                {"name": "Draw", "price": 3.9},
            ]}]},
        ],
    }
]


def test_implied_probs_devigged_and_keyed():
    book = odds.parse_feed(SAMPLE)
    ph, pdr, pa, n = book[("France", "Senegal")]
    assert ph + pdr + pa == pytest.approx(1.0)  # devigged probs sum to 1
    assert n == 2  # both bookmakers in the sample backed the consensus
    assert 0.55 < ph < 0.68  # ~1/1.63 devigged
    assert pdr < ph and pa < ph


def test_missing_fixture_returns_none():
    assert odds.parse_feed([]).get(("X", "Y")) is None


def test_fifa_names_normalized():
    feed = [{
        "home_team": "Korea Republic", "away_team": "Czechia",
        "bookmakers": [{"key": "b", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Korea Republic", "price": 2.0},
            {"name": "Czechia", "price": 3.0},
            {"name": "Draw", "price": 3.5},
        ]}]}],
    }]
    book = odds.parse_feed(feed)
    assert ("South Korea", "Czech Republic") in book
