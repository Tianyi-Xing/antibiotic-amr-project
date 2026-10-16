from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .clustering import classify_four_trajectories, cluster_trajectories, compare_with_truth
from .config import Config
from .features import engineer_features
from .statistics import distributed_lag_fixed_effects, interrupted_time_series
from .utils import write_json


def run_analysis(cfg: Config) -> dict[str, Path]:
    processed = cfg.path("paths.processed_dir")
    table_dir = cfg.path("paths.table_dir")
    interim = cfg.path("paths.interim_dir")
    panel = pd.read_csv(processed / "area_quarter_panel.csv", parse_dates=["quarter"])
    window = int(cfg.get("analysis.change_window_quarters", 4))
    threshold = float(cfg.get("analysis.improvement_threshold_pp", 0.5))
    features = engineer_features(panel, window=window, horizon=window, improvement_threshold_pp=threshold)
    feature_path = processed / "model_features.csv"
    features.to_csv(feature_path, index=False)

    classes = classify_four_trajectories(
        panel,
        window=window,
        resistance_threshold_pp=threshold,
        prescribing_threshold=float(cfg.get("analysis.prescribing_reduction_threshold", 0.0)),
    )
    class_path = table_dir / "four_trajectory_classes.csv"
    classes.to_csv(class_path, index=False)

    cluster_assignments, cluster_metrics = cluster_trajectories(
        panel, n_clusters=int(cfg.get("analysis.cluster_count", 4))
    )
    truth_path = cfg.path("paths.external_dir") / "synthetic_truth.csv"
    if truth_path.exists():
        truth = pd.read_csv(truth_path)
        ari = compare_with_truth(cluster_assignments, truth)
        if ari is not None:
            cluster_metrics["adjusted_rand_vs_synthetic_truth"] = ari
    cluster_path = table_dir / "dtw_cluster_assignments.csv"
    cluster_assignments.to_csv(cluster_path, index=False)
    write_json(cluster_metrics, table_dir / "clustering_metrics.json")

    lags = [int(x) for x in cfg.get("analysis.lag_quarters", [1, 2, 3, 4])]
    lag_results = distributed_lag_fixed_effects(panel, lag_quarters=lags)
    lag_path = table_dir / "distributed_lag_fixed_effects.csv"
    lag_results.to_csv(lag_path, index=False)

    normalized_p = pd.read_csv(interim / "prescribing_normalized.csv", parse_dates=["date"])
    its_table, its_series = interrupted_time_series(
        normalized_p,
        intervention_date=cfg.get("project.intervention_date", "2015-04-01"),
    )
    its_path = table_dir / "interrupted_time_series.csv"
    its_series_path = table_dir / "interrupted_time_series_national_series.csv"
    its_table.to_csv(its_path, index=False)
    its_series.to_csv(its_series_path, index=False)

    summary = {
        "panel_rows": int(len(panel)),
        "areas": int(panel["area_code"].nunique()),
        "antibiotics": sorted(panel["antibiotic"].dropna().unique().tolist()),
        "quarter_min": str(panel["quarter"].min().date()),
        "quarter_max": str(panel["quarter"].max().date()),
        "trajectory_class_counts": classes["trajectory_class"].value_counts().to_dict(),
        "clustering_metrics": cluster_metrics,
        "its_estimable": bool(not its_table.empty),
    }
    write_json(summary, table_dir / "analysis_summary.json")
    return {
        "features": feature_path,
        "classes": class_path,
        "clusters": cluster_path,
        "lags": lag_path,
        "its": its_path,
    }
