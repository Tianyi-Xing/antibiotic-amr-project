from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def classify_four_trajectories(
    panel: pd.DataFrame,
    *,
    window: int = 4,
    resistance_threshold_pp: float = 0.5,
    prescribing_threshold: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for (antibiotic, area_code), group in panel.sort_values("quarter").groupby(["antibiotic", "area_code"]):
        clean = group.dropna(subset=["broad_spectrum_items_per_1000", "resistance_percent"])
        if len(clean) < 2 * window:
            continue
        early = clean.head(window)
        late = clean.tail(window)
        p_change = float(late["broad_spectrum_items_per_1000"].mean() - early["broad_spectrum_items_per_1000"].mean())
        r_change = float(late["resistance_percent"].mean() - early["resistance_percent"].mean())
        p_down = p_change < -abs(prescribing_threshold)
        r_down = r_change <= -abs(resistance_threshold_pp)
        if p_down and r_down:
            label = "prescribing_down_resistance_down"
        elif p_down and not r_down:
            label = "prescribing_down_resistance_not_down"
        elif not p_down and r_down:
            label = "prescribing_not_down_resistance_down"
        else:
            label = "neither_improved"
        rows.append({
            "antibiotic": antibiotic,
            "area_code": area_code,
            "area_name": clean["area_name"].iloc[-1] if "area_name" in clean else area_code,
            "prescribing_change": p_change,
            "resistance_change": r_change,
            "trajectory_class": label,
            "mean_n_tested": float(clean["n_tested"].mean()) if "n_tested" in clean else np.nan,
        })
    return pd.DataFrame(rows)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cost = np.full((len(a) + 1, len(b) + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[-1, -1])


def _standardized_trajectory(group: pd.DataFrame) -> np.ndarray:
    values = group[["broad_spectrum_items_per_1000", "resistance_percent"]].interpolate(limit_direction="both")
    arr = StandardScaler().fit_transform(values).T
    return arr.reshape(-1)


def cluster_trajectories(panel: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, dict[str, float]]:
    groups = []
    keys = []
    for key, group in panel.sort_values("quarter").groupby(["antibiotic", "area_code"]):
        clean = group.dropna(subset=["broad_spectrum_items_per_1000", "resistance_percent"])
        if len(clean) < 8:
            continue
        groups.append(_standardized_trajectory(clean))
        keys.append(key)
    if len(groups) < n_clusters:
        raise ValueError("Insufficient area trajectories for requested cluster count")
    min_len = min(len(x) for x in groups)
    groups = [x[-min_len:] for x in groups]
    n = len(groups)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            value = dtw_distance(groups[i], groups[j])
            distances[i, j] = distances[j, i] = value
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    labels = model.fit_predict(distances)
    result = pd.DataFrame(keys, columns=["antibiotic", "area_code"])
    result["dtw_cluster"] = labels
    metrics = {"silhouette_precomputed": float(silhouette_score(distances, labels, metric="precomputed"))}
    return result, metrics


def compare_with_truth(assignments: pd.DataFrame, truth: pd.DataFrame) -> float | None:
    if "true_class" not in truth or "area_code" not in truth:
        return None
    merged = assignments.merge(truth[["area_code", "true_class"]], on="area_code", how="inner")
    if merged.empty:
        return None
    return float(adjusted_rand_score(merged["true_class"], merged["dtw_cluster"]))
