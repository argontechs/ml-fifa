"""players.html renderer: archetype scatter + WC-nation tables, site design language."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fifa import data as fdata
from fifa.dashboard import _CSS, _FONTS, FLAGS

PALETTE = ["#b8f53d", "#f0573f", "#7ec8e2", "#e2cb7e", "#c08ae2", "#6fd6a8",
           "#e29a7e", "#8a93a3", "#f0d957", "#74b0f0", "#d96fb1", "#a8d96f"]


def _nav_html() -> str:
    from fifa.dashboard import _nav

    return _nav("players")


def _scatter(meta, labels, xy, names) -> str:
    fig = go.Figure()
    for c in sorted(names):
        m = labels == c
        sub = meta[m]
        fig.add_scatter(
            x=xy[m, 0], y=xy[m, 1], mode="markers", name=names[c],
            marker={"size": 6, "color": PALETTE[c % len(PALETTE)], "opacity": 0.8},
            text=[f"{r.player} · {r.team}<br>{r.pos} · {r.minutes:.0f} min"
                  f"<br>npg90 {r.npg90:.2f} · ast90 {r.ast90:.2f} · tklw90 {r.tklw90:.2f}"
                  for r in sub.itertuples(index=False)],
            hoverinfo="text+name",
        )
    fig.update_layout(
        paper_bgcolor="#0a0e0b", plot_bgcolor="#10160f",
        font={"color": "#e9eee4", "family": "monospace", "size": 11},
        xaxis={"title": "style PC1", "showgrid": False, "zeroline": False},
        yaxis={"title": "style PC2", "showgrid": False, "zeroline": False},
        legend={"orientation": "h", "y": -0.12}, height=560,
        margin={"t": 10, "l": 50, "r": 10},
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False})


def _flag(team: str, px: int = 18) -> str:
    code = FLAGS.get(team)
    return (f'<img src="https://flagcdn.com/w20/{code}.png" alt="" '
            f'style="width:{px}px;border-radius:1px"> ') if code else ""


def _nation_tables(meta, labels, names) -> str:
    d = meta.copy()
    d["archetype"] = [names[c] for c in labels]
    wc = d[d["nation"].isin(FLAGS)].sort_values(["nation", "minutes"], ascending=[True, False])
    # order nations by champion odds when available
    order = list(wc["nation"].unique())
    sim_path = fdata.DATA_DIR / "sim_results.json"
    if sim_path.exists():
        champs = json.loads(sim_path.read_text())
        order.sort(key=lambda n: -champs.get(n, {}).get("champion", 0))
    blocks = []
    for nation in order:
        sub = wc[wc["nation"] == nation].head(14)
        rows = "".join(
            f"<tr><td>{r.player}</td><td>{r.pos}</td><td>{r.team}</td>"
            f"<td class='num'>{r.minutes:.0f}</td><td class='num'>{r.npg90:.2f}</td>"
            f"<td class='num'>{r.ast90:.2f}</td><td class='num'>{r.tklw90:.2f}</td>"
            f"<td>{r.archetype}</td></tr>"
            for r in sub.itertuples(index=False)
        )
        blocks.append(
            f'<details><summary>{_flag(nation)}<b>{nation}</b> '
            f'<small>({len(sub)} profiled)</small></summary>'
            f'<table><tr><th>Player</th><th>Pos</th><th>Club</th><th class="num">Min</th>'
            f'<th class="num">npG/90</th><th class="num">A/90</th><th class="num">TklW/90</th>'
            f'<th>Archetype</th></tr>{rows}</table></details>'
        )
    return "\n".join(blocks)


def render(meta: pd.DataFrame, labels: np.ndarray, xy: np.ndarray,
           names: dict[int, str], generated_at: str) -> str:
    n_wc = int(meta["nation"].isin(FLAGS).sum())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WC26 Predictor · Players</title>{_FONTS}<style>{_CSS}
details{{margin:.4rem 0;border:1px solid var(--line);border-radius:6px;padding:.5rem .8rem;
background:var(--panel)}} summary{{cursor:pointer}}</style></head><body>
<header>
<h1>WC26<span class="dot">·</span>Players</h1>
<div class="sub">playing-style archetypes · {len(meta):,} players across 11 leagues (latest
full seasons, ≥900 min) · generated {generated_at}</div>
{_nav_html()}
<div class="ticker"><b>Reading this honestly:</b> profiles come from latest full club seasons across the Big-5 European leagues +
Brasileirão, MLS, Saudi Pro League, Liga MX, Eredivisie and Primeira Liga — players outside
these leagues aren't profiled. Teams group by
<b>nationality</b>, not confirmed squads. Stats are position-normalized, so an archetype means
"relative to others in the same role". Market values/injuries: not available from the current
free source — cut rather than guessed.</div>
</header>
<h2>Style map · every dot is a player</h2>
{_scatter(meta, labels, xy, names)}
<h2>WC2026 nations · profiled players</h2>
{_nation_tables(meta, labels, names)}
<footer>Data: FBref via soccerdata · flags by flagcdn.com ·
<a href="index.html">back to predictions</a></footer>
</body></html>"""
