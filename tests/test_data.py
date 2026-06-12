import time

import pytest

from fifa import data


class FakeResp:
    def __init__(self, content=b"x,y\n1,2\n", fail=False):
        self.content = content
        self._fail = fail

    def raise_for_status(self):
        if self._fail:
            import requests

            raise requests.HTTPError("boom")


def test_download_writes_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        assert "User-Agent" in headers  # feed 403s without browser UA
        return FakeResp()

    monkeypatch.setattr(data.requests, "get", fake_get)
    dest = tmp_path / "f.csv"
    p1 = data.download("http://u", dest)
    p2 = data.download("http://u", dest)  # fresh cache → no second call
    assert p1 == p2 == dest and dest.read_bytes() == b"x,y\n1,2\n"
    assert len(calls) == 1


def test_download_uses_stale_cache_on_network_error(tmp_path, monkeypatch):
    dest = tmp_path / "f.csv"
    dest.write_bytes(b"old")
    old = time.time() - 9999 * 3600
    import os

    os.utime(dest, (old, old))

    def fake_get(url, headers=None, timeout=None):
        import requests

        raise requests.ConnectionError("offline")

    monkeypatch.setattr(data.requests, "get", fake_get)
    assert data.download("http://u", dest).read_bytes() == b"old"


def test_download_raises_without_cache(tmp_path, monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        import requests

        raise requests.ConnectionError("offline")

    monkeypatch.setattr(data.requests, "get", fake_get)
    with pytest.raises(RuntimeError, match="no cache"):
        data.download("http://u", tmp_path / "missing.csv")


SAMPLE_CSV = """date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
1994-06-17,Germany,Bolivia,1,0,FIFA World Cup,Chicago,United States,TRUE
1992-03-25,Yugoslavia,Netherlands,2,0,Friendly,Amsterdam,Netherlands,FALSE
2026-06-13,United States,Paraguay,NA,NA,FIFA World Cup,Los Angeles,United States,FALSE
"""


def test_load_results_splits_and_maps(tmp_path, monkeypatch):
    f = tmp_path / "results.csv"
    f.write_text(SAMPLE_CSV)
    monkeypatch.setattr(data, "download", lambda url, dest, **kw: f)
    played, upcoming = data.load_results()
    assert len(played) == 2 and len(upcoming) == 1
    assert played["home_score"].dtype.kind == "i"
    assert "Serbia" in set(played["home_team"])  # Yugoslavia mapped
    assert list(played["date"]) == sorted(played["date"])  # chronological
    assert bool(played.iloc[1]["neutral"]) is True  # 1994 row is second after sort
    assert upcoming.iloc[0]["home_team"] == "United States"


def test_normalize_team_known_aliases():
    assert data.normalize_team("Korea Republic") == "South Korea"
    assert data.normalize_team("Czechia") == "Czech Republic"
    assert data.normalize_team("Türkiye") == "Turkey"
    assert data.normalize_team("France") == "France"  # identity passthrough


def test_assert_known_raises_with_offenders():
    known = {"France", "Brazil"}
    with pytest.raises(ValueError, match="Atlantis"):
        data.assert_known(["France", "Atlantis"], known)


def test_seed_tokens_are_placeholders_not_teams():
    for tok in ("1A", "2L", "3CDFGH", "To be announced"):
        assert data.is_placeholder(tok)
    for name in ("France", "3rd Place", "A1"):
        assert not data.is_placeholder(name)
    data.assert_known(["France", "1A", "3ABCDF"], {"France"})  # must not raise
