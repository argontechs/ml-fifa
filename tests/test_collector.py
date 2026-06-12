import asyncio
import json

from sentiment import collector, db


def _msg(text):
    return json.dumps({
        "did": "did:plc:x", "kind": "commit",
        "commit": {"operation": "create", "collection": "app.bsky.feed.post",
                   "record": {"text": text, "createdAt": "2026-06-13T20:00:00Z"}},
    })


async def _fake_source(messages):
    for m in messages:
        yield m


KW = {"home": {"mexico", "el tri"}, "away": {"south africa", "bafana"}, "both": {"#mexrsa"}}
MATCH = {"key": 1, "home": "Mexico", "away": "South Africa"}


def test_route_sides():
    assert collector.route("VAMOS MEXICO!!", KW) == "home"
    assert collector.route("bafana bafana looking sharp", KW) == "away"
    assert collector.route("mexico vs south africa is on #worldcup", KW) == "both"
    assert collector.route("watching #MexRsa at the pub", KW) == "both"
    assert collector.route("unrelated cooking post", KW) is None


def test_consume_inserts_routed_posts(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    messages = [
        _msg("Vamos Mexico!"),
        _msg("Bafana will shock them"),
        _msg("#MEXRSA kicking off"),
        json.dumps({"kind": "commit", "commit": {"operation": "delete"}}),  # non-post
        "{{{not json",                                                       # malformed
    ]
    asyncio.run(collector.consume(_fake_source(messages), conn,
                                  windows_fn=lambda: [(MATCH, KW)]))
    frame = db.posts_frame(conn, 1)
    assert len(frame) == 3
    assert sorted(frame["side"]) == ["away", "both", "home"]


def test_post_matching_two_matches_lands_in_both(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    kw2 = {"home": {"france"}, "away": {"senegal"}, "both": {"#frasen"}}
    match2 = {"key": 2, "home": "France", "away": "Senegal"}
    msgs = [_msg("Mexico and France both playing today!")]
    asyncio.run(collector.consume(_fake_source(msgs), conn,
                                  windows_fn=lambda: [(MATCH, KW), (match2, kw2)]))
    assert len(db.posts_frame(conn, 1)) == 1
    assert len(db.posts_frame(conn, 2)) == 1
