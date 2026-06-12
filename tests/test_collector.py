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


def test_route_word_boundaries_and_compounds():
    kw_usa = {"home": {"usa", "united states"}, "away": {"paraguay"}, "both": {"#usapar"}}
    assert collector.route("thousands of fans in the stadium", kw_usa) is None  # 'usa' inside
    assert collector.route("refusal to celebrate? USAge is odd", kw_usa) is None
    assert collector.route("USA! USA! USA!", kw_usa) == "home"
    assert collector.route("Driving through New Mexico tonight", KW) is None  # compound blocker
    assert collector.route("mexico city is buzzing", KW) == "home"
    kw_jor = {"home": {"jordan"}, "away": {"austria"}, "both": {"#joraut"}}
    assert collector.route("new air jordan drop today", kw_jor) is None
    assert collector.route("Michael Jordan was the GOAT", kw_jor) is None
    assert collector.route("Jordan defending deep in this half", kw_jor) == "home"


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


def test_cursor_tracking_and_resume_url():
    holder = {}
    msgs = [
        json.dumps({"did": "x", "kind": "commit", "time_us": 111,
                    "commit": {"operation": "create", "collection": "app.bsky.feed.post",
                               "record": {"text": "Vamos Mexico!"}}}),
        json.dumps({"did": "x", "kind": "commit", "time_us": 222,
                    "commit": {"operation": "create", "collection": "app.bsky.feed.post",
                               "record": {"text": "irrelevant"}}}),
    ]
    import sqlite3
    conn = sqlite3.connect(":memory:")
    from sentiment import db as sdb
    conn.executescript(sdb._SCHEMA)
    asyncio.run(collector.consume(_fake_source(msgs), conn,
                                  windows_fn=lambda: [(MATCH, KW)], cursor=holder))
    assert holder["time_us"] == 222  # tracked even for non-matching posts
    url = collector._cursor_url("wss://example/subscribe?wantedCollections=x", holder)
    assert url.endswith("&cursor=222")
    assert collector._cursor_url("wss://example/sub", {}) == "wss://example/sub"
