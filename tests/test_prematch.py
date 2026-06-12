import pandas as pd

from sentiment import db, prematch
from tests.test_scorer import stub_model


def _setup(tmp_path, kickoff):
    conn = db.connect(tmp_path / "s.db")
    k_epoch = pd.Timestamp(kickoff, tz="UTC").timestamp()
    # pre-match posts (before kickoff) + one in-match post (after kickoff)
    db.insert_post(conn, k_epoch - 3600, "bsky", 1, "home", "goal machine mexico tonight")
    db.insert_post(conn, k_epoch - 1800, "bsky", 1, "away", "nervous about this one")
    db.insert_post(conn, k_epoch + 600, "bsky", 1, "home", "goal!")  # must be EXCLUDED
    from sentiment import scorer
    scorer.run_once(conn, stub_model)
    return conn


def _fx(kickoff):
    return pd.DataFrame({
        "match_number": [1, 2],
        "home": ["Mexico", "France"], "away": ["South Africa", "Senegal"],
        "date": [pd.Timestamp(kickoff), pd.Timestamp(kickoff) + pd.Timedelta(hours=30)],
        "status": ["upcoming", "upcoming"],
    })


def test_snapshot_frozen_at_kickoff_prematch_posts_only(tmp_path):
    kickoff = "2026-06-13 19:00"
    conn = _setup(tmp_path, kickoff)
    path = tmp_path / "prematch.jsonl"
    now = pd.Timestamp(kickoff) + pd.Timedelta(minutes=5)  # just after kickoff
    n = prematch.record_kickoffs(conn, _fx(kickoff), path, now)
    assert n == 1  # match 1 frozen; match 2 is 30h away
    book = prematch.load(path)
    rec = book[1]
    assert rec["n_posts"] == 2          # the in-match post is excluded
    assert rec["mood_home"] > 0 and rec["mood_away"] < 0
    # idempotent — second sweep records nothing new
    assert prematch.record_kickoffs(conn, _fx(kickoff), path, now) == 0


def test_late_poller_still_freezes_snapshot(tmp_path):
    # poller was down across kickoff and comes back 6h later — snapshot must still
    # freeze (content is identical: only posts with ts <= kickoff are counted)
    kickoff = "2026-06-13 19:00"
    conn = _setup(tmp_path, kickoff)
    path = tmp_path / "prematch.jsonl"
    now = pd.Timestamp(kickoff) + pd.Timedelta(hours=6)
    assert prematch.record_kickoffs(conn, _fx(kickoff), path, now) == 1
    assert prematch.load(path)[1]["n_posts"] == 2  # in-match post still excluded


def test_no_record_before_kickoff(tmp_path):
    kickoff = "2026-06-13 19:00"
    conn = _setup(tmp_path, kickoff)
    path = tmp_path / "prematch.jsonl"
    now = pd.Timestamp(kickoff) - pd.Timedelta(minutes=10)
    assert prematch.record_kickoffs(conn, _fx(kickoff), path, now) == 0
