import pandas as pd

from players import fetch


def _mi(rows, cols):
    """Build a frame with FBref-style MultiIndex columns and (league,season,team,player) index."""
    idx = pd.MultiIndex.from_tuples(
        [("Big5", "2526", r["team"], r["player"]) for r in rows],
        names=["league", "season", "team", "player"],
    )
    return pd.DataFrame([[r[c] for c in cols] for r in rows],
                        index=idx, columns=pd.MultiIndex.from_tuples(cols))


STD_COLS = [("nation", ""), ("pos", ""), ("age", ""), ("Playing Time", "Min"),
            ("Playing Time", "90s"), ("Performance", "Gls"), ("Performance", "Ast"),
            ("Performance", "PK"), ("Performance", "CrdY"), ("Performance", "CrdR")]
SHT_COLS = [("Standard", "Sh"), ("Standard", "SoT%"), ("Standard", "G/Sh")]
MSC_COLS = [("Performance", "Crs"), ("Performance", "Int"), ("Performance", "TklW"),
            ("Performance", "Fld"), ("Performance", "Fls"), ("Performance", "Off")]


def _readers():
    base = {"team": "Real Madrid", "player": "Kylian Mbappé"}
    std = _mi([{**base, ("nation", ""): "fr FRA", ("pos", ""): "FW", ("age", ""): "27",
                ("Playing Time", "Min"): 1800, ("Playing Time", "90s"): 20.0,
                ("Performance", "Gls"): 18, ("Performance", "Ast"): 6, ("Performance", "PK"): 4,
                ("Performance", "CrdY"): 3, ("Performance", "CrdR"): 0}], STD_COLS)
    sht = _mi([{**base, ("Standard", "Sh"): 80, ("Standard", "SoT%"): 45.0,
                ("Standard", "G/Sh"): 0.22}], SHT_COLS)
    msc = _mi([{**base, ("Performance", "Crs"): 30, ("Performance", "Int"): 5,
                ("Performance", "TklW"): 8, ("Performance", "Fld"): 40,
                ("Performance", "Fls"): 12, ("Performance", "Off"): 15}], MSC_COLS)
    return {"standard": lambda: std, "shooting": lambda: sht, "misc": lambda: msc}


def test_assemble_canonical_columns_and_per90():
    df = fetch.assemble(_readers())
    assert len(df) == 1
    r = df.iloc[0]
    assert r["player"] == "Kylian Mbappé" and r["team"] == "Real Madrid"
    assert r["npg90"] == (18 - 4) / 20.0
    assert r["sh90"] == 80 / 20.0
    assert r["card90"] == 3 / 20.0
    assert r["sot_pct"] == 45.0 and r["conv"] == 0.22
    assert r["minutes"] == 1800 and r["nineties"] == 20.0
    assert set(fetch.CANONICAL) <= set(df.columns)


def test_dedupe_keeps_bigger_minutes_but_not_other_people():
    df = pd.DataFrame({
        "player": ["Neymar", "Neymar", "Paulinho", "Paulinho"],
        "nation": ["Brazil", "Brazil", "Brazil", "Portugal"],  # same-name different people
        "minutes": [900, 1500, 1000, 1100],
        "team": ["Al-Hilal", "Santos", "X", "Y"],
    })
    out = fetch.dedupe(df)
    assert len(out) == 3
    assert out[out["player"] == "Neymar"].iloc[0]["team"] == "Santos"  # max minutes kept
    assert set(out[out["player"] == "Paulinho"]["nation"]) == {"Brazil", "Portugal"}


def test_nation_bridge_maps_fbref_codes():
    df = fetch.assemble(_readers())
    assert df.iloc[0]["nation"] == "France"  # 'fr FRA' → FRA → France via TRIGRAPH
    assert fetch.nation_of("br BRA") == "Brazil"
    assert fetch.nation_of("xx XXX") is None
    assert fetch.nation_of(None) is None
