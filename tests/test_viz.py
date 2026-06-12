import pandas as pd

from sentiment import viz


def _tl():
    return pd.DataFrame({
        "bucket": [0, 1, 2],
        "home_mean": [0.5, 0.2, 0.6], "home_n": [4, 2, 5], "home_ewma": [0.5, 0.35, 0.5],
        "away_mean": [-0.1, -0.4, 0.0], "away_n": [3, 4, 1], "away_ewma": [-0.1, -0.3, -0.2],
        "both_mean": [0.1, 0.0, 0.2], "both_n": [1, 1, 2], "both_ewma": [0.1, 0.05, 0.1],
    })


def _events():
    return pd.DataFrame({"ts": [70.0], "team": ["Mexico"], "kind": ["goal"], "detail": ["1-0"]})


def test_figure_traces_and_goal_marker():
    fig = viz.figure_for(_tl(), _events(), "Mexico", "South Africa", bucket_s=60)
    names = [t.name for t in fig.data]
    assert "Mexico" in names and "South Africa" in names and "volume" in names
    assert len(fig.layout.shapes) == 1  # one goal vline
    assert "⚽" in fig.layout.annotations[0].text


def test_match_options_active_first():
    now = pd.Timestamp("2026-06-13 20:00")
    fx = pd.DataFrame({
        "match_number": [4, 9],
        "home": ["Brazil", "Germany"], "away": ["Morocco", "Curaçao"],
        "date": [now - pd.Timedelta(minutes=30), now + pd.Timedelta(hours=20)],
        "status": ["upcoming", "upcoming"],
    })
    opts = viz.match_options(fx, now)
    assert opts[0]["value"] == 4 and "LIVE" in opts[0]["label"]
    assert opts[1]["value"] == 9 and "LIVE" not in opts[1]["label"]


def test_merge_options_appends_db_matches_with_data():
    feed_opts = [{"label": "A v B · LIVE", "value": 4}]
    db_rows = [(1, "Mexico", "South Africa"), (4, "A", "B")]  # 4 already present
    merged = viz.merge_options(feed_opts, db_rows)
    assert [o["value"] for o in merged] == [4, 1]
    assert "Mexico v South Africa" in merged[1]["label"]
