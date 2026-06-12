import pandas as pd

from fifa import fixtures, livescores

ESPN_SAMPLE = {
    "events": [
        {"competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"displayName": "Canada"}, "score": "1"},
                {"homeAway": "away", "team": {"displayName": "Bosnia-Herzegovina"}, "score": "1"},
            ],
            "status": {"type": {"completed": True, "state": "post"}},
        }]},
        {"competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"displayName": "United States"}, "score": "2"},
                {"homeAway": "away", "team": {"displayName": "Paraguay"}, "score": "0"},
            ],
            "status": {"type": {"completed": False, "state": "in"}},
        }]},
    ]
}


def test_parse_espn_names_normalized_and_states():
    book = livescores.parse_scoreboard(ESPN_SAMPLE)
    k = frozenset({"Canada", "Bosnia and Herzegovina"})  # hyphen alias mapped
    assert book[k]["final"] is True and book[k]["scores"]["Canada"] == 1
    k2 = frozenset({"United States", "Paraguay"})
    assert book[k2]["final"] is False and book[k2]["live"] is True


def _fx_row(**kw):
    base = {"match_number": 3, "round": 1, "date": pd.Timestamp("2026-06-12 19:00"),
            "location": "x", "host": "Canada", "home": "Canada",
            "away": "Bosnia and Herzegovina", "group": "B", "home_score": None,
            "away_score": None, "status": "upcoming", "winner": None, "neutral": False}
    base.update(kw)
    return pd.DataFrame([base])


def test_merge_fills_final_score_and_flips_status():
    fx = _fx_row()
    book = livescores.parse_scoreboard(ESPN_SAMPLE)
    out = livescores.merge_scores(fx, book, now=pd.Timestamp("2026-06-12 21:00"))
    r = out.iloc[0]
    assert r["status"] == "played" and r["home_score"] == 1 and r["away_score"] == 1


def test_merge_respects_our_orientation():
    # our row lists Bosnia as home (hypothetical swap) — scores must follow OUR columns
    fx = _fx_row(home="Bosnia and Herzegovina", away="Canada", host="x", neutral=True)
    book = livescores.parse_scoreboard(ESPN_SAMPLE)
    out = livescores.merge_scores(fx, book, now=pd.Timestamp("2026-06-12 21:00"))
    r = out.iloc[0]
    assert r["home_score"] == 1 and r["away_score"] == 1 and r["status"] == "played"


def test_merge_live_scores_fill_but_status_stays_upcoming():
    fx = _fx_row(match_number=4, home="United States", away="Paraguay",
                 date=pd.Timestamp("2026-06-13 01:00"))
    book = livescores.parse_scoreboard(ESPN_SAMPLE)
    out = livescores.merge_scores(fx, book, now=pd.Timestamp("2026-06-13 01:30"))
    r = out.iloc[0]
    assert r["home_score"] == 2 and r["away_score"] == 0
    assert r["status"] == "upcoming"  # partial scores must never count as played


def test_merge_never_touches_future_or_already_scored_rows():
    future = _fx_row(date=pd.Timestamp("2026-06-20 19:00"))
    out = livescores.merge_scores(future, livescores.parse_scoreboard(ESPN_SAMPLE),
                                  now=pd.Timestamp("2026-06-12 21:00"))
    assert out.iloc[0]["status"] == "upcoming" and pd.isna(out.iloc[0]["home_score"])
