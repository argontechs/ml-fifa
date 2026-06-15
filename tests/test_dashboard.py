from fifa import dashboard


def _view():
    return {
        "generated_at": "2026-06-12 14:00 UTC",
        "card": {
            "n": 2544, "wdl_acc": 0.598, "rps": 0.167, "exact_rate": 0.132,
            "baseline11_rate": 0.101, "top5_rate": 0.534, "lock_n": 699, "lock_acc": 0.845,
            "draw_recall": 0.017, "logloss": 0.88, "brier": 0.55,
        },
        "matches": [
            {
                "home": "France", "away": "Senegal", "when": "Jun 16", "comp": "Group I",
                "p": (0.58, 0.24, 0.18), "tier": "STRONG", "score": (1, 0),
                "top5": [((1, 0), 0.13), ((2, 0), 0.11)],
                "ctx_home": {"elo": 2130, "trend1y": 42, "last5": [1, 1, 0.5, 1, 0]},
                "ctx_away": {"elo": 1820, "trend1y": -12, "last5": [1, 0.5, 1, 1, 1]},
                "market": True,
            }
        ],
        "tracker_rows": [
            {"match_number": 1, "home": "Mexico", "away": "South Africa",
             "predicted": (2, 0), "actual": (2, 0), "tier": "LOCK",
             "p": (0.70, 0.18, 0.12), "outcome": True, "exact": True},
        ],
        "tracker_tally": {"n": 1, "outcome_hits": 1, "exact_hits": 1, "lock_n": 1,
                          "lock_hits": 1, "retro_n": 0, "retro_outcome_hits": 0,
                          "retro_exact_hits": 0},
        "standings": {
            "A": [
                {"team": "Mexico", "p": 1, "w": 1, "d": 0, "l": 0, "gf": 2, "ga": 0, "pts": 3},
                {"team": "South Africa", "p": 1, "w": 0, "d": 0, "l": 1, "gf": 0, "ga": 2, "pts": 0},
            ],
        },
        "advance": {"Mexico": 0.99, "France": 0.91},
        "champions": {"Argentina": 0.21, "Spain": 0.149},
    }


def test_render_full_view():
    html = dashboard.render(_view())
    for needle in (
        "France", "Senegal", "STRONG",
        "FRANCE", "58%", "to win",            # outcome-call hero (not the exact score)
        "likeliest 1–0", "low-confidence",    # exact score demoted to a labelled chip
        "1X · France or draw · 82%",          # double-chance chip w/ plain-English gloss
        "flagcdn.com/w40/fr.png",            # flags
        "ELO 2130", "↑42", "↓12",            # context + momentum arrows
        "Track record", "✓",                  # honesty tracker
        "Sample so far", "avg stated confidence", "for fun only",  # honest reframe
        "Exact (fun)",                         # demoted exact-score column
        "Group A", "Argentina", "21.0%",       # standings + champions
        "market-blended", "Honest numbers",
        ">+2<", ">-2<",                          # goal-difference columns
    ):
        assert needle in html, needle
    # the exact score must NOT be the visual hero any more
    assert "1<span class=\"dash\">–</span>0" not in html


def test_match_card_modal_draw_shows_caveat():
    v = _view()
    v["matches"][0]["p"] = (0.30, 0.42, 0.28)  # draw is the modal outcome
    html = dashboard.render(v)
    assert "DRAW" in html and "most likely" in html
    assert "model rarely calls draws" in html


def test_draw_blindspot_ticker_only_when_a_draw_was_missed():
    v = _view()
    # one frozen miss that ended level → the blind-spot ticker must appear
    v["tracker_rows"] = [
        {"match_number": 7, "home": "Brazil", "away": "Morocco", "predicted": (1, 0),
         "actual": (1, 1), "tier": "STRONG", "p": (0.55, 0.27, 0.18),
         "outcome": False, "exact": False},
    ]
    v["tracker_tally"] = {"n": 1, "outcome_hits": 0, "exact_hits": 0, "lock_n": 0,
                          "lock_hits": 0, "retro_n": 0, "retro_outcome_hits": 0,
                          "retro_exact_hits": 0}
    assert "Known blind spot — draws" in dashboard.render(v)
    # no draw among misses → ticker must self-suppress (never a stale claim)
    v["tracker_rows"][0]["actual"] = (0, 2)
    v["tracker_rows"][0]["outcome"] = True
    v["tracker_tally"]["outcome_hits"] = 1
    assert "Known blind spot — draws" not in dashboard.render(v)


def test_render_past_page():
    v = _view()
    v["tracker_rows"] = [
        {"match_number": 1, "home": "Mexico", "away": "South Africa", "when": "Jun 11",
         "predicted": (2, 0), "actual": (2, 0), "tier": "LOCK", "p": (0.7, 0.2, 0.1),
         "retro": True, "outcome": True, "exact": True},
        {"match_number": 2, "home": "South Korea", "away": "Czech Republic", "when": "Jun 12",
         "predicted": (1, 1), "actual": (2, 1), "tier": "LEAN", "p": (0.3, 0.4, 0.3),
         "retro": True, "outcome": False, "exact": False},
    ]
    v["tracker_tally"] = {"n": 2, "outcome_hits": 1, "exact_hits": 1, "lock_n": 1,
                          "lock_hits": 1, "retro_n": 2, "retro_outcome_hits": 2,
                          "retro_exact_hits": 1}
    html = dashboard.render_past(v)
    for needle in ("Past matches", "retro", "predicted", "full time",
                   "hero hit", "hero miss", "Mexico", "Czech Republic", "1/2"):
        assert needle in html, needle


def test_nav_present_on_both_pages():
    html_home = dashboard.render(_view())
    assert 'href="past.html"' in html_home
    v = _view()
    html_past = dashboard.render_past(v)
    assert 'href="index.html"' in html_past


def test_render_leaderboard_groups_and_third_race():
    v = _view()
    v["standings"] = {
        g: [{"team": t, "p": 1, "w": 1, "d": 0, "l": 0, "gf": 2, "ga": 0, "pts": 3}
            for t in (f"T{g}1", f"T{g}2", f"T{g}3", f"T{g}4")]
        for g in "ABCDEFGHIJKL"
    }
    v["standings"]["A"][0]["team"] = "Mexico"
    html = dashboard.render_leaderboard(v)
    for needle in ("Leaderboard", "Group A", "Group L", "Third-place race", "Mexico",
                   "leaderboard.html", "top 8 qualify"):
        assert needle in html, needle


def test_track_record_figures_come_from_card_not_hardcoded():
    # the long-run / exact / baseline numbers must mirror the live backtest card,
    # never a stale literal that can contradict the header (review HIGH finding)
    v = _view()
    v["card"] = {**v["card"], "wdl_acc": 0.611, "exact_rate": 0.137, "baseline11_rate": 0.099}
    html = dashboard.render(v)
    assert "61.1% model long-run" in html      # from card['wdl_acc'], not "59.7%"
    assert "~14%" in html                       # ~card['exact_rate'] rounded
    assert "always-1-1 baseline 9.9%" in html   # from card['baseline11_rate']
    assert "59.7% model long-run" not in html   # the old hardcoded literal is gone
    assert "gap is noise" not in html           # softened overclaim


def test_draw_blindspot_handles_none_recall():
    # a backtest test set with zero draws sets draw_recall=None; the ticker must not
    # crash and must not fabricate a false "0.0%" recall (review MEDIUM/LOW finding)
    v = _view()
    v["card"] = {**v["card"], "draw_recall": None}
    v["tracker_rows"] = [
        {"match_number": 7, "home": "Brazil", "away": "Morocco", "predicted": (1, 0),
         "actual": (1, 1), "tier": "STRONG", "p": (0.55, 0.27, 0.18),
         "outcome": False, "exact": False},
    ]
    v["tracker_tally"] = {"n": 1, "outcome_hits": 0, "exact_hits": 0, "lock_n": 0,
                          "lock_hits": 0, "retro_n": 0, "retro_outcome_hits": 0,
                          "retro_exact_hits": 0}
    html = dashboard.render(v)  # must not raise
    assert "Known blind spot — draws" in html
    assert "0.0%" not in html  # no fabricated recall when it is undefined


def test_ou_chip_on_match_card():
    v = _view()
    v["matches"][0]["p_over25"] = 0.62
    html = dashboard.render(v)
    assert "O2.5 62%" in html
    v["matches"][0]["p_over25"] = 0.38
    assert "U2.5 62%" in dashboard.render(v)


def test_render_empty_tracker_and_matches():
    v = _view()
    v["matches"] = []
    v["tracker_rows"], v["tracker_tally"] = [], {"n": 0, "outcome_hits": 0, "exact_hits": 0,
                                                 "lock_n": 0, "lock_hits": 0, "retro_n": 0,
                                                 "retro_outcome_hits": 0, "retro_exact_hits": 0}
    html = dashboard.render(v)
    assert "No upcoming fixtures" in html
    assert "No completed predictions yet" in html


def test_flag_has_onerror_fallback_and_decorative_alt():
    # a flagcdn hiccup must collapse the flag, not show a broken-image glyph + alt text
    html = dashboard._flag("South Africa", 18)
    assert "flagcdn.com/w20/za.png" in html
    assert 'onerror=' in html and "display='none'" in html
    assert 'alt=""' in html  # decorative — the team name is always printed adjacent
    assert dashboard._flag("Atlantis", 18) == ""  # unknown team → no element at all
