import asyncio
from pathlib import Path

from sentiment import aggregate, collector, db, scorer
from tests.test_scorer import stub_model

SAMPLE = Path(__file__).parent / "replay_sample.jsonl"
KW = {"home": {"mexico", "el tri", "méxico"}, "away": {"south africa", "bafana"},
      "both": {"#mexrsa", "#rsamex"}}
MATCH = {"key": 1, "home": "Mexico", "away": "South Africa"}


async def _file_source(path):
    for line in path.read_text().splitlines():
        yield line


def test_end_to_end_replay_through_stub_scorer(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    ticker = {"t": 0.0}

    def clock():
        ticker["t"] += 2.0  # 2s apart → spread across minute buckets
        return ticker["t"]

    stored = asyncio.run(collector.consume(_file_source(SAMPLE), conn,
                                           windows_fn=lambda: [(MATCH, KW)], clock=clock))
    assert stored > 100  # most of the 200 messages are match-related
    while scorer.run_once(conn, stub_model):
        pass
    tl = aggregate.timeline(conn, 1)
    assert len(tl) >= 3
    assert tl["home_n"].sum() > 0 and tl["away_n"].sum() > 0
    t = aggregate.tallies(conn, 1)
    assert t["scored"] == t["total_posts"] == stored
