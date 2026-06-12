from pathlib import Path

import pandas as pd

from fifa import fixtures

SAMPLE = Path(__file__).parent / "fixtures_sample.json"


def test_parse_feed_statuses_and_names(monkeypatch):
    monkeypatch.setattr(fixtures.data, "download", lambda url, dest, **kw: SAMPLE)
    fx = fixtures.load_fixtures()
    assert list(fx["status"]) == ["played", "upcoming", "tbd"]
    assert fx.loc[0, "home_score"] == 2
    assert fx.loc[1, "home"] == "Brazil"
    assert fx.loc[0, "group"] == "A"
    assert pd.api.types.is_datetime64_any_dtype(fx["date"])


def test_neutrality_and_host_nations(monkeypatch):
    monkeypatch.setattr(fixtures.data, "download", lambda url, dest, **kw: SAMPLE)
    fx = fixtures.load_fixtures()
    assert bool(fx.loc[0, "neutral"]) is False  # Mexico at Mexico City Stadium
    assert fx.loc[0, "host"] == "Mexico"
    assert bool(fx.loc[1, "neutral"]) is True  # Brazil in New York
    assert fx.loc[1, "host"] == "United States"
