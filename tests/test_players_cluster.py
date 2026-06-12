import numpy as np
import pandas as pd

from players import cluster, features
from players.fetch import CANONICAL

RNG = np.random.default_rng(5)


def _players(n_per=60):
    """Three synthetic styles: finishers, creators, ball-winners (+ GK + low-minutes)."""
    rows = []
    styles = {
        "FW": {"npg90": 0.7, "sh90": 3.5, "ast90": 0.1, "crs90": 0.5, "tklw90": 0.3, "int90": 0.2},
        "MF": {"npg90": 0.1, "sh90": 1.0, "ast90": 0.5, "crs90": 4.0, "tklw90": 0.8, "int90": 0.6},
        "DF": {"npg90": 0.02, "sh90": 0.3, "ast90": 0.05, "crs90": 0.4, "tklw90": 2.5, "int90": 2.2},
    }
    for pos, mus in styles.items():
        for i in range(n_per):
            row = {c: 0.0 for c in CANONICAL}
            for k, mu in mus.items():
                row[k] = max(0, RNG.normal(mu, mu * 0.15 + 0.02))
            row.update({"sot_pct": RNG.normal(35, 5), "conv": RNG.normal(0.1, 0.03),
                        "fld90": RNG.normal(1, 0.2), "fls90": RNG.normal(1, 0.2),
                        "off90": 0.2, "card90": 0.2,
                        "player": f"{pos}{i}", "team": "T", "nation": "France",
                        "pos": pos, "age": "25", "minutes": 2000, "nineties": 22.0})
            rows.append(row)
    rows.append({**rows[0], "player": "GK1", "pos": "GK"})
    rows.append({**rows[0], "player": "Bench", "minutes": 200})
    return pd.DataFrame(rows)


def test_build_matrix_filters_and_scales_by_position():
    df = _players()
    X, meta = features.build_matrix(df)
    assert "GK1" not in set(meta["player"])      # GKs excluded from clustering
    assert "Bench" not in set(meta["player"])    # minutes filter
    assert list(X.columns) == CANONICAL
    assert len(X) == len(meta) == 180
    # position-grouped scaling: each group's tklw90 ~0 mean despite wildly different raw means
    for pos in ("FW", "DF"):
        grp = X[meta["pos_group"].to_numpy() == pos]
        assert abs(grp["tklw90"].mean()) < 0.2


def _one_position_styles(n_per=60):
    """Three style blobs WITHIN one position group — position-grouped scaling preserves
    within-group structure (between-group structure is removed by design)."""
    rows = []
    styles = {
        "fin": {"npg90": 0.7, "sh90": 3.5, "ast90": 0.1, "crs90": 0.5, "tklw90": 0.3, "int90": 0.2},
        "cre": {"npg90": 0.1, "sh90": 1.0, "ast90": 0.5, "crs90": 4.0, "tklw90": 0.8, "int90": 0.6},
        "win": {"npg90": 0.02, "sh90": 0.3, "ast90": 0.05, "crs90": 0.4, "tklw90": 2.5, "int90": 2.2},
    }
    for style, mus in styles.items():
        for i in range(n_per):
            row = {c: 0.0 for c in CANONICAL}
            for k, mu in mus.items():
                row[k] = max(0, RNG.normal(mu, mu * 0.15 + 0.02))
            row.update({"sot_pct": RNG.normal(35, 5), "conv": RNG.normal(0.1, 0.03),
                        "fld90": RNG.normal(1, 0.2), "fls90": RNG.normal(1, 0.2),
                        "off90": 0.2, "card90": 0.2,
                        "player": f"{style}{i}", "team": "T", "nation": "France",
                        "pos": "MF", "age": "25", "minutes": 2000, "nineties": 22.0})
            rows.append(row)
    return pd.DataFrame(rows)


def test_cluster_recovers_styles_and_names_them():
    df = _one_position_styles()
    X, meta = features.build_matrix(df)
    res = cluster.fit(X, k_range=range(3, 7), seed=42)
    assert res["k"] == 3
    # purity: each true style maps overwhelmingly to one cluster
    style = meta["player"].str[:3].to_numpy()
    lab = pd.Series(res["labels"], index=style)
    for s in ("fin", "cre", "win"):
        counts = lab[lab.index == s].value_counts(normalize=True)
        assert counts.iloc[0] > 0.9
    names = cluster.name_clusters(res["centers"], list(X.columns))
    assert len(names) == 3 and all(isinstance(n, str) and n for n in names.values())
    xy = cluster.project2d(X)
    assert xy.shape == (len(X), 2)


def test_unknown_signature_gets_generated_label():
    name = cluster.label_for(("off90", "card90", "fls90"), group="MF")
    assert name  # never blank — generated descriptive fallback is fine


def test_fit_by_group_clusters_each_position_separately():
    frames = []
    for pos in ("DF", "MF", "FW"):
        d = _one_position_styles(n_per=40)
        d["pos"] = pos
        d["player"] = pos + d["player"]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    X, meta = features.build_matrix(df)
    labels, names = cluster.fit_by_group(X, meta, k_range=range(3, 5), seed=42)
    assert len(labels) == len(meta)
    # archetype ids never cross position groups
    by_group = pd.DataFrame({"g": meta["pos_group"], "lab": labels})
    for lab, grp in by_group.groupby("lab")["g"]:
        assert grp.nunique() == 1
    # names carry their role prefix and exist for every label
    assert set(labels) == set(names)
    assert all(n.split(" · ")[0] in ("DF", "MF", "FW") for n in names.values())
