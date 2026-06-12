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
    p = book[("France", "Senegal")]
    assert sum(p) == pytest.approx(1.0)
    assert 0.55 < p[0] < 0.68  # ~1/1.63 devigged
    assert p[1] < p[0] and p[2] < p[0]


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
