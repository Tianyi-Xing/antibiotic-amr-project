from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .config import Config

LOGGER = logging.getLogger(__name__)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_national_trends(panel: pd.DataFrame, path: Path) -> None:
    grouped = panel.groupby("quarter", as_index=False).agg(
        prescribing=("broad_spectrum_items_per_1000", "mean"),
        resistance=("resistance_percent", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(grouped["quarter"], grouped["prescribing"], marker="o", label="Broad-spectrum prescribing")
    ax1.set_xlabel("Quarter")
    ax1.set_ylabel("Items per 1,000 registered patients")
    ax2 = ax1.twinx()
    ax2.plot(grouped["quarter"], grouped["resistance"], marker="s", linestyle="--", label="Resistance")
    ax2.set_ylabel("Resistance (%)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("National mean prescribing and resistance trends")
    _save(fig, path)


def plot_trajectory_scatter(classes: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, group in classes.groupby("trajectory_class"):
        ax.scatter(group["prescribing_change"], group["resistance_change"], alpha=0.75, label=label)
    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Change in broad-spectrum prescribing")
    ax.set_ylabel("Change in resistance (percentage points)")
    ax.set_title("Four regional response patterns")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_class_counts(classes: pd.DataFrame, path: Path) -> None:
    counts = classes["trajectory_class"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(counts.index, counts.values)
    ax.set_xlabel("Number of areas")
    ax.set_title("Regional trajectory classification")
    _save(fig, path)


def plot_lag_coefficients(results: pd.DataFrame, path: Path) -> None:
    if results.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for antibiotic, group in results.groupby("antibiotic"):
        group = group.sort_values("lag_quarters")
        yerr = np.vstack([
            group["estimate_per_1sd_prescribing"] - group["ci_low"],
            group["ci_high"] - group["estimate_per_1sd_prescribing"],
        ])
        ax.errorbar(
            group["lag_quarters"], group["estimate_per_1sd_prescribing"],
            yerr=yerr, marker="o", capsize=4, label=antibiotic,
        )
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Prescribing lag (quarters)")
    ax.set_ylabel("Resistance change per 1 SD prescribing")
    ax.set_title("Distributed-lag fixed-effects estimates")
    ax.legend()
    _save(fig, path)


def plot_feature_importance(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    top = table.dropna().head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Model importance")
    ax.set_title("Predictive feature importance")
    _save(fig, path)


def plot_calibration(predictions: pd.DataFrame, path: Path) -> None:
    if predictions.empty or predictions["resistance_improved"].nunique() < 2:
        return
    observed, predicted = calibration_curve(
        predictions["resistance_improved"], predictions["predicted_probability"], n_bins=8, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(predicted, observed, marker="o", label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed improvement rate")
    ax.set_title("Temporal holdout calibration")
    ax.legend()
    _save(fig, path)


def plot_scenario(scenario: pd.DataFrame, path: Path) -> None:
    if scenario.empty:
        return
    top = scenario.reindex(scenario["predicted_probability_change"].abs().sort_values(ascending=False).index).head(15)
    top = top.sort_values("predicted_probability_change")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["area_name"], top["predicted_probability_change"])
    ax.set_xlabel("Predicted probability change")
    ax.set_title("Peer-median prescribing scenario (non-causal)")
    _save(fig, path)


def plot_map_if_available(cfg: Config, classes: pd.DataFrame, path: Path) -> None:
    boundary_path = cfg.path("paths.raw_dir") / cfg.get("paths.boundaries_file")
    if not boundary_path.exists():
        return
    try:
        import geopandas as gpd
    except ImportError:
        LOGGER.info("geopandas not installed; map output skipped")
        return
    try:
        geo = gpd.read_file(boundary_path)
        code_candidates = [c for c in geo.columns if c.lower() in {"row_id", "code", "org_id", "area_code"}]
        if not code_candidates:
            code_candidates = [c for c in geo.columns if "code" in c.lower()]
        if not code_candidates:
            return
        geo["area_code"] = geo[code_candidates[0]].astype(str)
        merged = geo.merge(classes, on="area_code", how="left")
        mapping = {
            label: idx for idx, label in enumerate(sorted(classes["trajectory_class"].dropna().unique()))
        }
        merged["class_id"] = merged["trajectory_class"].map(mapping)
        fig, ax = plt.subplots(figsize=(8, 10))
        merged.plot(column="class_id", categorical=True, legend=True, ax=ax, missing_kwds={"label": "No data"})
        ax.set_axis_off()
        ax.set_title("Regional prescribing–resistance trajectories")
        _save(fig, path)
    except Exception as exc:
        LOGGER.warning("Map generation failed: %s", exc)


def run_visualizations(cfg: Config) -> list[Path]:
    panel = pd.read_csv(cfg.path("paths.processed_dir") / "area_quarter_panel.csv", parse_dates=["quarter"])
    table_dir = cfg.path("paths.table_dir")
    figure_dir = cfg.path("paths.figure_dir")
    outputs: list[Path] = []

    classes = pd.read_csv(table_dir / "four_trajectory_classes.csv")
    lag_path = table_dir / "distributed_lag_fixed_effects.csv"
    lags = pd.read_csv(lag_path) if lag_path.exists() and lag_path.stat().st_size > 1 else pd.DataFrame()
    importance_path = table_dir / "model_feature_importance.csv"
    importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
    prediction_path = table_dir / "final_holdout_predictions.csv"
    predictions = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()
    scenario_path = table_dir / "policy_scenario_results.csv"
    scenario = pd.read_csv(scenario_path) if scenario_path.exists() else pd.DataFrame()

    plot_specs = [
        (plot_national_trends, (panel,), figure_dir / "national_trends.png"),
        (plot_trajectory_scatter, (classes,), figure_dir / "trajectory_scatter.png"),
        (plot_class_counts, (classes,), figure_dir / "trajectory_class_counts.png"),
        (plot_lag_coefficients, (lags,), figure_dir / "distributed_lag_coefficients.png"),
        (plot_feature_importance, (importance,), figure_dir / "feature_importance.png"),
        (plot_calibration, (predictions,), figure_dir / "calibration_curve.png"),
        (plot_scenario, (scenario,), figure_dir / "scenario_probability_changes.png"),
    ]
    for function, args, path in plot_specs:
        function(*args, path)
        if path.exists():
            outputs.append(path)
    map_path = figure_dir / "trajectory_map.png"
    plot_map_if_available(cfg, classes, map_path)
    if map_path.exists():
        outputs.append(map_path)
    return outputs
