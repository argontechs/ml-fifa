import pandas as pd

from sentiment import aggregate, db, events


def _seed(conn):
    # minute 0: two positive home posts; minute 1: one negative away; minute 2: both-bucket
    rows = [
        (0.0, "home", "great start", 0.8),
        (10.0, "home", "love this", 0.6),
        (65.0, "away", "poor defending", -0.5),
        (130.0, "both", "what a game", 0.2),
        (135.0, "home", "unscored yet", None),
    ]
    for ts, side, text, score in rows:
        db.insert_post(conn, ts, "bsky", 1, side, text)
    frame = db.posts_frame(conn, 1)
    ids = list(range(1, len(frame) + 1))
    scores = [r[3] for r in rows]
    db.set_scores(conn, [i for i, s in zip(ids, scores) if s is not None],
                  [s for s in scores if s is not None])


def test_timeline_buckets_and_ewma(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    _seed(conn)
    tl = aggregate.timeline(conn, 1, bucket_s=60)
    assert len(tl) == 3  # minutes 0, 1, 2
    m0 = tl.iloc[0]
    assert m0["home_mean"] == 0.7  # (0.8 + 0.6) / 2
    assert m0["home_n"] == 2
    assert tl.iloc[1]["away_mean"] == -0.5
    assert "home_ewma" in tl.columns and tl["home_ewma"].notna().all()


def test_tallies_and_recent(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    _seed(conn)
    t = aggregate.tallies(conn, 1)
    assert t["total_posts"] == 5
    assert t["scored"] == 4
    recent = aggregate.recent_posts(conn, 1, n=2)
    assert len(recent) == 2 and recent.iloc[0]["ts"] >= recent.iloc[1]["ts"]


def test_goal_event_detection_from_fixture_frames(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    fx0 = pd.DataFrame({"match_number": [1], "home": ["Mexico"], "away": ["South Africa"],
                        "home_score": [0], "away_score": [0], "status": ["played"],
                        "date": [pd.Timestamp("2026-06-13 20:00")]})
    events.poll_once(conn, fx0, active_keys={1})
    fx1 = fx0.assign(home_score=[1])
    events.poll_once(conn, fx1, active_keys={1})
    events.poll_once(conn, fx1, active_keys={1})  # same score → no new event
    ev = db.events_frame(conn, 1)
    goal_rows = ev[ev["detail"] != "0-0"]  # initial upsert may record baseline
    assert len(ev[ev["team"] == "Mexico"]) == 1
