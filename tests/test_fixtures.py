from pathlib import Path

import pandas as pd

import pytest

from fifa import fixtures, livescores

SAMPLE = Path(__file__).parent / "fixtures_sample.json"


@pytest.fixture(autouse=True)
def _no_espn(monkeypatch):
    """Unit tests must never hit the live ESPN scoreboard."""
    monkeypatch.setattr(livescores, "fetch_scoreboard", lambda dates: {})


def test_parse_feed_statuses_and_names(monkeypatch):
    monkeypatch.setattr(fixtures.data, "download", lambda url, dest, **kw: SAMPLE)
    fx = fixtures.load_fixtures()
    assert list(fx["status"]) == ["played", "upcoming", "tbd"]
    assert fx.loc[0, "home_score"] == 2
    assert fx.loc[1, "home"] == "Brazil"
    assert fx.loc[0, "group"] == "A"
    assert pd.api.types.is_datetime64_any_dtype(fx["date"])
    assert fx["date"].dt.tz is None  # naive UTC, consistent with results.csv dates


def test_neutrality_and_host_nations(monkeypatch):
    monkeypatch.setattr(fixtures.data, "download", lambda url, dest, **kw: SAMPLE)
    fx = fixtures.load_fixtures()
    assert bool(fx.loc[0, "neutral"]) is False  # Mexico at Mexico City Stadium
    assert fx.loc[0, "host"] == "Mexico"
    assert bool(fx.loc[1, "neutral"]) is True  # Brazil in New York
    assert fx.loc[1, "host"] == "United States"


def test_host_listed_as_away_gets_swapped(monkeypatch, tmp_path):
    import json
    f = tmp_path / "feed.json"
    f.write_text(json.dumps([{
        "MatchNumber": 50, "RoundNumber": 2, "DateUtc": "2026-06-18 22:00:00Z",
        "Location": "BC Place Vancouver", "HomeTeam": "Qatar", "AwayTeam": "Canada",
        "Group": "Group B", "HomeTeamScore": 1, "AwayTeamScore": 2, "Winner": "Canada",
    }]))
    monkeypatch.setattr(fixtures.data, "download", lambda url, dest, **kw: f)
    fx = fixtures.load_fixtures()
    r = fx.iloc[0]
    assert r["home"] == "Canada" and r["away"] == "Qatar"      # host swapped to home
    assert r["home_score"] == 2 and r["away_score"] == 1       # scores follow the swap
    assert bool(r["neutral"]) is False                          # host has home advantage
