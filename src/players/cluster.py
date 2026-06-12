"""Playing-style archetypes: k-means WITHIN each position group (features are
position-relative z-scores, so cross-position clusters mislead — a goal-scoring
centre-back is not a striker). Role-specific naming tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# per-role: centroid signature overlap (≥2 of top-3 features) → archetype name
GROUP_ARCHETYPES: dict[str, list[tuple[set, str]]] = {
    "FW": [
        ({"npg90", "sh90", "conv"}, "Penalty-box striker"),
        ({"npg90", "sh90", "sot_pct"}, "Penalty-box striker"),
        ({"ast90", "crs90", "fld90"}, "Wide creator"),
        ({"ast90", "crs90", "off90"}, "Wide creator"),
        ({"ast90", "npg90", "fld90"}, "Complete forward"),
        ({"tklw90", "int90", "fls90"}, "Pressing forward"),
        ({"fld90", "fls90", "card90"}, "Physical forward"),
    ],
    "MF": [
        ({"npg90", "sh90", "conv"}, "Goal-scoring midfielder"),
        ({"npg90", "sh90", "sot_pct"}, "Goal-scoring midfielder"),
        ({"ast90", "crs90", "fld90"}, "Creator"),
        ({"ast90", "crs90", "off90"}, "Wide midfielder"),
        ({"tklw90", "int90", "fls90"}, "Ball-winner"),
        ({"tklw90", "int90", "card90"}, "Ball-winner"),
        ({"fls90", "card90", "fld90"}, "Destroyer"),
    ],
    "DF": [
        ({"crs90", "ast90", "off90"}, "Attacking full-back"),
        ({"crs90", "ast90", "fld90"}, "Attacking full-back"),
        ({"npg90", "conv", "sh90"}, "Set-piece threat"),
        ({"npg90", "conv", "sot_pct"}, "Set-piece threat"),
        ({"tklw90", "int90", "fls90"}, "Stopper"),
        ({"tklw90", "int90", "card90"}, "Stopper"),
        ({"int90", "card90", "fls90"}, "Last-line defender"),
    ],
}


def fit(X: pd.DataFrame, k_range=range(6, 13), seed: int = 42) -> dict:
    best = None
    arr = X.to_numpy()
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(arr)
        score = silhouette_score(arr, km.labels_)
        if best is None or score > best["sil"]:
            best = {"k": k, "sil": score, "labels": km.labels_, "centers": km.cluster_centers_}
    return best


def centroid_signature(center: np.ndarray, cols: list[str], top: int = 3) -> tuple[str, ...]:
    order = np.argsort(center)[::-1][:top]
    return tuple(cols[i] for i in order)


def label_for(signature: tuple[str, ...], group: str = "MF") -> str:
    sig = set(signature)
    for proto, name in GROUP_ARCHETYPES.get(group, []):
        if len(sig & proto) >= 2:
            return name
    return "High " + " · ".join(signature[:2]) + " profile"


def name_clusters(centers: np.ndarray, cols: list[str], group: str = "MF") -> dict[int, str]:
    names: dict[int, str] = {}
    for i, c in enumerate(centers):
        base = label_for(centroid_signature(c, cols), group=group)
        if base in names.values():
            base = f"{base} ({centroid_signature(c, cols)[0]})"
        names[i] = base
    return names


def fit_by_group(X: pd.DataFrame, meta: pd.DataFrame, k_range=range(3, 6),
                 seed: int = 42) -> tuple[np.ndarray, dict[int, str]]:
    """Cluster each position group separately. Returns (global labels, label→name)."""
    labels = np.full(len(X), -1, dtype=int)
    names: dict[int, str] = {}
    offset = 0
    for group in ("DF", "MF", "FW"):
        mask = (meta["pos_group"] == group).to_numpy()
        if mask.sum() < max(k_range) * 5:
            continue
        res = fit(X[mask], k_range=k_range, seed=seed)
        local = name_clusters(res["centers"], list(X.columns), group=group)
        labels[mask] = res["labels"] + offset
        for i, name in local.items():
            names[i + offset] = f"{group} · {name}"
        offset += res["k"]
    return labels, names


def project2d(X: pd.DataFrame, seed: int = 42) -> np.ndarray:
    return PCA(n_components=2, random_state=seed).fit_transform(X.to_numpy())
