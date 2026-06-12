"""Plotly figures and option lists for the live dashboard (pure, testable)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import match_window

VOLT, LOSS, DIM = "#b8f53d", "#f0573f", "#5b6878"
BG, PANEL, INK = "#0a0e0b", "#10160f", "#e9eee4"


def figure_for(tl: pd.DataFrame, ev: pd.DataFrame, home: str, away: str,
               bucket_s: int = 60) -> go.Figure:
    fig = go.Figure()
    if tl is None or tl.empty:
        fig.add_annotation(text="waiting for posts…", showarrow=False,
                           font={"color": DIM, "size": 16})
    else:
        x = tl["bucket"] - tl["bucket"].min()
        total_n = tl["home_n"] + tl["away_n"] + tl["both_n"]
        fig.add_bar(x=x, y=total_n, name="volume", yaxis="y2",
                    marker={"color": "rgba(125,138,116,.25)"}, hoverinfo="skip")
        fig.add_scatter(x=x, y=tl["home_ewma"], name=home, mode="lines",
                        line={"color": VOLT, "width": 3})
        fig.add_scatter(x=x, y=tl["away_ewma"], name=away, mode="lines",
                        line={"color": LOSS, "width": 3})
        fig.add_scatter(x=x, y=tl["both_ewma"], name="match mood", mode="lines",
                        line={"color": DIM, "width": 1.5, "dash": "dash"})
        if ev is not None and not ev.empty:
            for r in ev.itertuples(index=False):
                gx = r.ts // bucket_s - tl["bucket"].min()
                fig.add_vline(x=gx, line={"color": INK, "width": 1, "dash": "dot"})
                fig.add_annotation(x=gx, y=1.0, yref="paper", showarrow=False,
                                   text=f"⚽ {r.team} {r.detail}",
                                   font={"color": INK, "size": 11})
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL, font={"color": INK, "family": "monospace"},
        yaxis={"title": "sentiment", "range": [-1, 1], "zerolinecolor": "#222b1e"},
        yaxis2={"overlaying": "y", "side": "right", "showgrid": False, "title": "posts/min"},
        xaxis={"title": "minutes", "showgrid": False},
        legend={"orientation": "h", "y": 1.08}, margin={"t": 30, "b": 40},
        height=420,
    )
    return fig


def match_options(fx: pd.DataFrame, now: pd.Timestamp) -> list[dict]:
    """Dropdown options: live matches first (flagged), then the next upcoming ones."""
    active_keys = {m["key"] for m in match_window.active_matches(fx, now)}
    rows = fx[fx["status"] != "tbd"].sort_values("date")
    opts = []
    for r in rows.itertuples(index=False):
        key = int(r.match_number)
        live = key in active_keys
        if not live and r.date < now - pd.Timedelta(hours=3):
            continue  # long-finished matches stay reachable via db, not the picker
        label = f"{r.home} v {r.away}" + (" · LIVE" if live else f" · {r.date:%b %d %H:%M}")
        opts.append({"label": label, "value": key, "live": live})
    opts.sort(key=lambda o: not o["live"])
    return [{"label": o["label"], "value": o["value"],
             **({"label": "🔴 " + o["label"]} if o["live"] else {})} for o in opts]


def merge_options(feed_opts: list[dict], db_rows: list[tuple]) -> list[dict]:
    """Ensure every match with collected data stays selectable (replays, finished games)."""
    have = {o["value"] for o in feed_opts}
    extra = [{"label": f"{h} v {a} · collected", "value": int(k)}
             for k, h, a in db_rows if int(k) not in have]
    return feed_opts + extra
