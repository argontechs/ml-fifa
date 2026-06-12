import numpy as np
import pandas as pd

from fifa import dashboard
from players import page


def _meta():
    return pd.DataFrame({
        "player": ["Kylian Mbappé", "Rodri"],
        "team": ["Real Madrid", "Manchester City"],
        "nation": ["France", "Spain"],
        "pos": ["FW", "MF"], "pos_group": ["FW", "MF"],
        "age": ["27", "29"], "minutes": [1800, 2100], "nineties": [20.0, 23.3],
        "npg90": [0.9, 0.05], "ast90": [0.3, 0.1], "sh90": [4.0, 0.8],
        "sot_pct": [45.0, 30.0], "conv": [0.2, 0.05], "crs90": [1.0, 0.3],
        "int90": [0.2, 1.8], "tklw90": [0.3, 1.5], "fld90": [2.0, 1.0],
        "fls90": [0.5, 1.4], "off90": [0.8, 0.0], "card90": [0.1, 0.3],
    })


def test_render_players_page():
    html = page.render(
        meta=_meta(),
        labels=np.array([0, 1]),
        xy=np.array([[1.0, 2.0], [-1.0, -2.0]]),
        names={0: "Penalty-box striker", 1: "Ball-winner"},
        generated_at="2026-06-12 18:00 MYT",
    )
    for needle in ("Penalty-box striker", "Ball-winner", "Kylian Mbappé", "Rodri",
                   "plotly", "France", "flagcdn.com", "nationality", "players.html"):
        assert needle in html, needle


def test_nav_has_three_tabs_everywhere():
    assert 'href="players.html"' in dashboard._nav("home")
    assert 'href="players.html"' in page._nav_html()
