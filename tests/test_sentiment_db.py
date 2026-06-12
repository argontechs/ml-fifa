from sentiment import db


def _conn(tmp_path):
    return db.connect(tmp_path / "s.db")


def test_schema_idempotent_and_wal(tmp_path):
    conn = _conn(tmp_path)
    conn2 = db.connect(tmp_path / "s.db")  # re-init must not fail
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()
    conn2.close()


def test_post_lifecycle_unscored_to_scored(tmp_path):
    conn = _conn(tmp_path)
    db.insert_post(conn, ts=1000.0, source="bsky", match_key=1, side="home", text="vamos méxico")
    db.insert_post(conn, ts=1001.0, source="bsky", match_key=1, side="away", text="bafana!")
    batch = db.unscored_batch(conn)
    assert len(batch) == 2
    db.set_scores(conn, [batch[0][0]], [0.8])
    assert len(db.unscored_batch(conn)) == 1
    frame = db.posts_frame(conn, 1)
    assert len(frame) == 2 and frame["score"].notna().sum() == 1


def test_score_change_events_only_on_delta(tmp_path):
    conn = _conn(tmp_path)
    db.upsert_match(conn, 1, "Mexico", "South Africa", "2026-06-11 19:00", "0-0")
    assert db.record_score_change(conn, 1, "Mexico", "1-0") is True
    assert db.record_score_change(conn, 1, "Mexico", "1-0") is False  # no delta
    events = db.events_frame(conn, 1)
    assert len(events) == 1 and events.iloc[0]["team"] == "Mexico"
