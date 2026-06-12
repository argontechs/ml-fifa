import pandas as pd

from fifa import dashboard
from sentiment import match_window as mw


def test_trigraph_covers_all_48_teams():
    missing = sorted(set(dashboard.FLAGS) - set(mw.TRIGRAPH))
    assert missing == []
    assert all(len(c) == 3 and c.isupper() for c in mw.TRIGRAPH.values())
    # spot-check the non-obvious ones (verified against Wikipedia FIFA code list)
    assert mw.TRIGRAPH["Switzerland"] == "SUI"
    assert mw.TRIGRAPH["South Africa"] == "RSA"
    assert mw.TRIGRAPH["Saudi Arabia"] == "KSA"
    assert mw.TRIGRAPH["DR Congo"] == "COD"


def _fx(now):
    return pd.DataFrame(
        {
            "match_number": [1, 2, 3],
            "home": ["Mexico", "France", "Brazil"],
            "away": ["South Africa", "Senegal", "Morocco"],
            "date": [now - pd.Timedelta(minutes=100),   # in play
                     now + pd.Timedelta(minutes=10),    # about to start
                     now + pd.Timedelta(hours=4)],      # far future
            "status": ["played", "upcoming", "upcoming"],
        }
    )


def test_active_matches_window():
    now = pd.Timestamp("2026-06-13 20:00")
    fx = _fx(now)
    active = mw.active_matches(fx, now)
    assert [m["key"] for m in active] == [1, 2]
    tbd = fx.assign(status=["tbd"] * 3)
    assert mw.active_matches(tbd, now) == []


def test_keywords_include_names_aliases_and_hashtags():
    kw = mw.keywords_for({"key": 1, "home": "Mexico", "away": "South Africa"})
    assert "mexico" in kw["home"]
    assert "south africa" in kw["away"]
    assert {"#mexrsa", "#rsamex"} <= kw["both"]
    # dataset aliases route too: Korea Republic → South Korea
    kw2 = mw.keywords_for({"key": 2, "home": "South Korea", "away": "Turkey"})
    assert "korea republic" in kw2["home"]
    assert "türkiye" in kw2["away"]
