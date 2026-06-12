"""Static sentiment.html for the live site: snapshots of collected matches.

The real-time Dash app stays local; this publishes its charts as a static page
whenever update.py runs (honest 'snapshot at T' labeling).
"""
from __future__ import annotations

from fifa.dashboard import _CSS, _FONTS, _nav

from . import aggregate, db, viz


def render(conn, generated_at: str) -> str:
    matches = conn.execute(
        "SELECT m.match_key, m.home, m.away FROM matches m "
        "WHERE EXISTS (SELECT 1 FROM posts p WHERE p.match_key = m.match_key) "
        "ORDER BY m.match_key DESC"
    ).fetchall()
    sections = []
    for key, home, away in matches:
        tl = aggregate.timeline(conn, key)
        if tl.empty:
            continue
        fig = viz.figure_for(tl, db.events_frame(conn, key), home, away)
        t = aggregate.tallies(conn, key)
        frag = fig.to_html(full_html=False, include_plotlyjs="cdn",
                           config={"displaylogo": False})
        sections.append(
            f"<h2>{home} vs {away} <small>({t['total_posts']:,} posts)</small></h2>{frag}"
        )
    body = "\n".join(sections) or (
        "<p class='ctx' style='text-align:left'>No sentiment data collected yet — the "
        "tracker runs locally during live matches and snapshots appear here afterwards.</p>"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WC26 Predictor · Sentiment</title>{_FONTS}<style>{_CSS}</style></head><body>
<header>
<h1>WC26<span class="dot">·</span>Crowd sentiment</h1>
<div class="sub">bluesky mention mood during matches · snapshot generated {generated_at}</div>
{_nav('sentiment')}
<div class="ticker"><b>Reading this honestly:</b> static snapshots of the local real-time
tracker (Bluesky firehose → multilingual sentiment model). Mention mood ≠ verified fan
allegiance; thin bars mean a quiet crowd, shown rather than hidden.</div>
</header>
{body}
<footer><a href="index.html">back to predictions</a></footer>
</body></html>"""
