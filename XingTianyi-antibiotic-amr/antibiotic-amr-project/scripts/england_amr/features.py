from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_slope(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return np.nan
    y = values[mask]
    x = np.arange(len(values))[mask]
    return float(np.polyfit(x, y, 1)[0])


def engineer_features(
    panel: pd.DataFrame,
    *,
    window: int = 4,
    horizon: int = 4,
    improvement_threshold_pp: float = 0.5,
) -> pd.DataFrame:
    required = {"area_code", "antibiotic", "quarter", "broad_spectrum_items_per_1000", "resistance_percent"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel lacks required columns: {sorted(missing)}")

    out = panel.copy()
    out["quarter"] = pd.to_datetime(out["quarter"])
    out = out.sort_values(["antibiotic", "area_code", "quarter"])
    group_keys = ["antibiotic", "area_code"]

    for source, prefix in (
        ("broad_spectrum_items_per_1000", "prescribing"),
        ("resistance_percent", "resistance"),
    ):
        grouped = out.groupby(group_keys, group_keys=False)[source]
        out[f"{prefix}_change_{window}q"] = grouped.diff(window)
        out[f"{prefix}_mean_{window}q"] = grouped.transform(lambda s: s.rolling(window, min_periods=2).mean())
        out[f"{prefix}_sd_{window}q"] = grouped.transform(lambda s: s.rolling(window, min_periods=min(3, window)).std())
        out[f"{prefix}_slope_{window}q"] = grouped.transform(
            lambda s: s.rolling(window, min_periods=min(3, window)).apply(_rolling_slope, raw=True)
        )

    grouped_r = out.groupby(group_keys, group_keys=False)["resistance_percent"]
    grouped_p = out.groupby(group_keys, group_keys=False)["broad_spectrum_items_per_1000"]
    out[f"future_resistance_{horizon}q"] = grouped_r.shift(-horizon)
    out[f"future_prescribing_{horizon}q"] = grouped_p.shift(-horizon)
    out[f"future_resistance_change_{horizon}q"] = out[f"future_resistance_{horizon}q"] - out["resistance_percent"]
    out[f"future_prescribing_change_{horizon}q"] = out[f"future_prescribing_{horizon}q"] - out["broad_spectrum_items_per_1000"]
    out["resistance_improved"] = (
        out[f"future_resistance_change_{horizon}q"] <= -abs(improvement_threshold_pp)
    ).astype("Int64")
    out["prescribing_reduced"] = (out[f"future_prescribing_change_{horizon}q"] < 0).astype("Int64")
    out["eligible_reduction_episode"] = out["prescribing_reduced"].eq(1)

    out["quarter_index"] = out["quarter"].rank(method="dense").astype(int) - 1
    out["calendar_year"] = out["quarter"].dt.year
    out["calendar_quarter"] = out["quarter"].dt.quarter
    return out
