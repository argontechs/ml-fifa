import pandas as pd
import pytest

from players import fetch


def _mi(rows, cols):
    """Build a frame with FBref-style MultiIndex columns and (league,season,team,player) index."""
    idx = pd.MultiIndex.from_tuples(
        [("Big5", "2526", r["team"], r["player"]) for r in rows],
        names=["league", "season", "team", "player"],
    )
    return pd.DataFrame([[r[c] for c in cols] for r in rows],
                        index=idx, columns=pd.MultiIndex.from_tuples(cols))


STD_COLS = [("nation", ""), ("pos", ""), ("age", ""), ("born", ""), ("Playing Time", "Min"),
            ("Playing Time", "90s"), ("Performance", "Gls"), ("Performance", "Ast"),
            ("Performance", "PK"), ("Performance", "CrdY"), ("Performance", "CrdR")]
SHT_COLS = [("Standard", "Sh"), ("Standard", "SoT")]
MSC_COLS = [("Performance", "Crs"), ("Performance", "Int"), ("Performance", "TklW"),
            ("Performance", "Fld"), ("Performance", "Fls"), ("Performance", "Off")]


def _readers():
    base = {"team": "Real Madrid", "player": "Kylian Mbappé"}
    std = _mi([{**base, ("nation", ""): "fr FRA", ("pos", ""): "FW", ("age", ""): "27",
                ("born", ""): "1998",
                ("Playing Time", "Min"): 1800, ("Playing Time", "90s"): 20.0,
                ("Performance", "Gls"): 18, ("Performance", "Ast"): 6, ("Performance", "PK"): 4,
                ("Performance", "CrdY"): 3, ("Performance", "CrdR"): 0}], STD_COLS)
    sht = _mi([{**base, ("Standard", "Sh"): 80, ("Standard", "SoT"): 36}], SHT_COLS)
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
    assert r["sot_pct"] == 45.0 and r["conv"] == 0.225  # computed from summed totals
    assert r["minutes"] == 1800 and r["nineties"] == 20.0
    assert set(fetch.CANONICAL) <= set(df.columns)


def _stint(player, born, minutes, team, pos="FW", nation_raw="br BRA", gls=5):
    return {"player": player, "nation_raw": nation_raw, "born": born, "pos": pos,
            "age": "25", "team": team, "minutes": minutes, "nineties": minutes / 90.0,
            "gls": gls, "ast": 2, "pk": 0, "yc": 1, "rc": 0, "sh": 20, "sot": 8,
            "crs": 3, "int_": 4, "tklw": 5, "fld": 6, "fls": 7, "off": 1}


def test_aggregate_sums_stints_of_the_same_identity():
    # Gallagher case: two stints, neither >=900, full season is — must survive summed
    stints = pd.DataFrame([_stint("Mover", "2000", 600, "Club A"),
                           _stint("Mover", "2000", 500, "Club B")])
    out = fetch.aggregate_stints(stints)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["minutes"] == 1100 and r["team"] == "Club A"  # max-minutes stint shown
    assert r["npg90"] == pytest.approx(10 / (1100 / 90))  # goals summed across stints
    assert r["conv"] == pytest.approx(10 / 40)


def test_aggregate_never_merges_different_people():
    # the two Alissons: same name+nation, different birth years AND positions
    stints = pd.DataFrame([_stint("Alisson", "1992", 2340, "Liverpool", pos="GK"),
                           _stint("Alisson", "1993", 1960, "São Paulo", pos="MF")])
    out = fetch.aggregate_stints(stints)
    assert len(out) == 2
    assert set(out["pos"]) == {"GK", "MF"}


def test_aggregate_keeps_unqualified_nations_separate():
    # nation=None used to merge all same-named players across non-WC nations
    stints = pd.DataFrame([_stint("Lookman", "1997", 1400, "Atalanta", nation_raw="ng NGA"),
                           _stint("Lookman", "1990", 800, "Elsewhere", nation_raw="gh GHA")])
    out = fetch.aggregate_stints(stints)
    assert len(out) == 2  # keyed on raw nation string, not the WC-mapped (None) nation


def test_nation_bridge_maps_fbref_codes():
    df = fetch.assemble(_readers())
    assert df.iloc[0]["nation"] == "France"  # 'fr FRA' → FRA → France via TRIGRAPH
    assert fetch.nation_of("br BRA") == "Brazil"
    assert fetch.nation_of("xx XXX") is None
    assert fetch.nation_of(None) is None
