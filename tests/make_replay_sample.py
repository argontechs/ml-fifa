"""Generate tests/replay_sample.jsonl — 200 synthetic Jetstream messages for a fake
Mexico v South Africa match window. Deterministic (seeded). Run:
    .venv/bin/python tests/make_replay_sample.py
"""
import json
import random
from pathlib import Path

random.seed(26)

HOME = [
    "Vamos Mexico! What a start", "El Tri looking dangerous today",
    "Mexico pressing high, love it", "¡Golazo de México!", "Mexico's midfield is bossing this",
]
AWAY = [
    "Bafana Bafana hold strong!", "South Africa defending like lions",
    "Come on South Africa, you can do this", "Bafana counterattack was electric",
    "South Africa unlucky there, ref was harsh",
]
BOTH = [
    "#MEXRSA is such a fun match", "Watching Mexico vs South Africa at the pub #worldcup",
    "This Mexico South Africa game is heating up", "#RSAMEX what a half!",
]
JUNK = [
    "My sourdough starter finally doubled overnight", "New mechanical keyboard day!",
    "Sunset photos from the hike", "{{{not even json",
]


def msg(text):
    return json.dumps({
        "did": "did:plc:fake", "kind": "commit",
        "commit": {"operation": "create", "collection": "app.bsky.feed.post",
                   "record": {"text": text, "createdAt": "2026-06-13T20:00:00Z"}},
    })


lines = []
for i in range(200):
    r = random.random()
    if r < 0.40:
        text = random.choice(HOME)
    elif r < 0.75:
        text = random.choice(AWAY)
    elif r < 0.92:
        text = random.choice(BOTH)
    else:
        raw = random.choice(JUNK)
        lines.append(raw if raw.startswith("{{{") else msg(raw))
        continue
    lines.append(msg(text))

out = Path(__file__).parent / "replay_sample.jsonl"
out.write_text("\n".join(lines) + "\n")
print(f"wrote {len(lines)} messages → {out}")
