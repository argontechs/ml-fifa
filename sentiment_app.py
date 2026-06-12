"""Live sentiment dashboard (Plotly Dash, localhost:8050).

usage: .venv/bin/python sentiment_app.py [--port 8050]
Run alongside sentiment_collect.py and sentiment_score.py during a match.
"""
import argparse

import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from fifa import data, fixtures
from sentiment import aggregate, db, viz

DB_PATH = data.DATA_DIR / "sentiment.db"

STYLE = {
    "backgroundColor": viz.BG, "color": viz.INK, "fontFamily": "IBM Plex Mono, monospace",
    "minHeight": "100vh", "padding": "1.5rem 2rem",
}
CARD = {"backgroundColor": viz.PANEL, "border": "1px solid #222b1e", "borderRadius": "6px",
        "padding": ".7rem 1rem", "display": "inline-block", "marginRight": "1rem",
        "minWidth": "9rem"}

app = Dash(__name__, title="WC26 sentiment")
app.layout = html.Div(style=STYLE, children=[
    html.H2("WC26 · LIVE CROWD SENTIMENT", style={"letterSpacing": ".06em"}),
    html.Div("mention mood from the Bluesky firehose — not verified fan allegiance",
             style={"color": "#7d8a74", "fontSize": "12px", "marginBottom": "1rem"}),
    dcc.Dropdown(id="match", options=[], style={"color": "#111", "maxWidth": "480px"}),
    html.Div(id="cards", style={"margin": "1rem 0"}),
    dcc.Graph(id="timeline"),
    html.H4("LATEST POSTS", style={"color": "#7d8a74", "marginTop": "1rem"}),
    html.Div(id="ticker", style={"fontSize": "12.5px", "lineHeight": "1.9"}),
    dcc.Interval(id="tick", interval=3000),
])


@app.callback(Output("match", "options"), Output("match", "value"),
              Input("tick", "n_intervals"), Input("match", "value"))
def refresh_options(_, current):
    fx = fixtures.load_fixtures()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    conn = db.connect(DB_PATH)
    try:
        db_rows = conn.execute("SELECT match_key, home, away FROM matches").fetchall()
    finally:
        conn.close()
    opts = viz.merge_options(viz.match_options(fx, now), db_rows)
    value = current if current in {o["value"] for o in opts} else (
        opts[0]["value"] if opts else None)
    return [{"label": o["label"], "value": o["value"]} for o in opts], value


@app.callback(Output("timeline", "figure"), Output("cards", "children"),
              Output("ticker", "children"),
              Input("tick", "n_intervals"), Input("match", "value"))
def refresh(_, key):
    conn = db.connect(DB_PATH)
    try:
        if key is None:
            return viz.figure_for(None, None, "", ""), [], []
        row = conn.execute(
            "SELECT home, away FROM matches WHERE match_key=?", (key,)).fetchone()
        home, away = row if row else ("home", "away")
        fig = viz.figure_for(aggregate.timeline(conn, key), db.events_frame(conn, key),
                             home, away)
        t = aggregate.tallies(conn, key)
        fmt = lambda v: "—" if v is None else f"{v:+.2f}"  # noqa: E731
        cards = [
            html.Div([html.Div(fmt(t["mood_home"]), style={"fontSize": "1.6rem",
                                                           "color": viz.VOLT}),
                      html.Div(f"{home} mood", style={"fontSize": "10px"})], style=CARD),
            html.Div([html.Div(fmt(t["mood_away"]), style={"fontSize": "1.6rem",
                                                           "color": viz.LOSS}),
                      html.Div(f"{away} mood", style={"fontSize": "10px"})], style=CARD),
            html.Div([html.Div(f"{t['total_posts']:,}", style={"fontSize": "1.6rem"}),
                      html.Div("posts collected", style={"fontSize": "10px"})], style=CARD),
            html.Div([html.Div(f"{t['rate_per_min']}/min", style={"fontSize": "1.6rem"}),
                      html.Div("recent volume", style={"fontSize": "10px"})], style=CARD),
        ]
        ticker = [html.Div(f"[{p.side}] {p.text[:140]}",
                           style={"color": viz.VOLT if (p.score or 0) > 0.15
                                  else viz.LOSS if (p.score or 0) < -0.15 else "#7d8a74"})
                  for p in aggregate.recent_posts(conn, key).itertuples(index=False)]
        return fig, cards, ticker
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    app.run(debug=False, port=args.port)
