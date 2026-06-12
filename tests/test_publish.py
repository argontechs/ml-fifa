from sentiment import db, publish
from tests.test_scorer import stub_model


def test_publish_renders_collected_matches(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    db.upsert_match(conn, 1, "Mexico", "South Africa", "2026-06-11", "2-0")
    for i, text in enumerate(("goal!", "ref blew it", "goal again")):
        db.insert_post(conn, 60.0 * i, "bsky", 1, "home", text)
    from sentiment import scorer
    scorer.run_once(conn, stub_model)
    html = publish.render(conn, "2026-06-12 19:00 MYT")
    for needle in ("Mexico vs South Africa", "plotly", "sentiment.html", "snapshot"):
        assert needle in html, needle


def test_publish_empty_state(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    html = publish.render(conn, "now")
    assert "No sentiment data collected yet" in html
