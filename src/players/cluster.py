"""Playing-style archetypes: k-means with silhouette-chosen k, named centroids, PCA scatter."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# centroid signature (frozenset of its top-3 features) → archetype name
ARCHETYPES: list[tuple[set, str]] = [
    ({"npg90", "sh90", "conv"}, "Penalty-box striker"),
    ({"npg90", "sh90", "sot_pct"}, "Penalty-box striker"),
    ({"npg90", "ast90", "sh90"}, "Complete forward"),
    ({"ast90", "crs90", "fld90"}, "Wide creator"),
    ({"ast90", "crs90", "off90"}, "Wide creator"),
    ({"ast90", "fld90", "npg90"}, "Advanced playmaker"),
    ({"crs90", "tklw90", "int90"}, "Two-way wide player"),
    ({"tklw90", "int90", "fls90"}, "Ball-winner"),
    ({"tklw90", "int90", "card90"}, "Ball-winner"),
    ({"int90", "card90", "fls90"}, "Stopper"),
    ({"int90", "tklw90", "crs90"}, "Defensive wide player"),
    ({"fls90", "card90", "fld90"}, "Battler"),
]


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


def label_for(signature: tuple[str, ...]) -> str:
    sig = set(signature)
    for proto, name in ARCHETYPES:
        if len(sig & proto) >= 2:
            return name
    return "High " + " · ".join(signature[:2]) + " profile"


def name_clusters(centers: np.ndarray, cols: list[str]) -> dict[int, str]:
    names: dict[int, str] = {}
    for i, c in enumerate(centers):
        base = label_for(centroid_signature(c, cols))
        # disambiguate duplicates: append the strongest feature
        if base in names.values():
            base = f"{base} ({centroid_signature(c, cols)[0]})"
        names[i] = base
    return names


def project2d(X: pd.DataFrame, seed: int = 42) -> np.ndarray:
    return PCA(n_components=2, random_state=seed).fit_transform(X.to_numpy())
