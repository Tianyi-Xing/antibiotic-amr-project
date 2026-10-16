from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import Config


def run_scenario(cfg: Config) -> pd.DataFrame:
    model_path = cfg.path("paths.model_dir") / "resistance_improvement_model.joblib"
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    features: list[str] = bundle["features"]
    data = pd.read_csv(cfg.path("paths.processed_dir") / "model_features.csv", parse_dates=["quarter"])
    latest = data.sort_values("quarter").groupby(["antibiotic", "area_code"], as_index=False).tail(1).copy()
    for feature in features:
        if feature not in latest:
            latest[feature] = np.nan

    peer_candidates = [
        c for c in (
            "age65_share", "age80_share", "imd_score", "diabetes_prevalence",
            "ckd_prevalence", "copd_prevalence", "cancer_prevalence",
            "resistance_percent",
        ) if c in latest.columns
    ]
    if peer_candidates:
        peer_matrix = latest[peer_candidates].copy()
        imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(peer_matrix)
        scaled = StandardScaler().fit_transform(imputed)
        n_clusters = min(int(cfg.get("scenario.peer_cluster_count", 5)), max(2, len(latest) // 4))
        latest["peer_cluster"] = KMeans(
            n_clusters=n_clusters,
            random_state=int(cfg.get("project.random_seed", 2026)),
            n_init=20,
        ).fit_predict(scaled)
    else:
        latest["peer_cluster"] = 0

    intervention_feature = str(cfg.get("scenario.intervention_feature", "broad_spectrum_items_per_1000"))
    quantile = float(cfg.get("scenario.target_quantile", 0.5))
    latest["peer_target_prescribing"] = latest.groupby(["antibiotic", "peer_cluster"])[intervention_feature].transform(
        lambda s: s.quantile(quantile)
    )
    latest["observed_probability"] = pipeline.predict_proba(latest[features])[:, 1]

    scenario = latest.copy()
    scenario[intervention_feature] = np.minimum(
        scenario[intervention_feature], scenario["peer_target_prescribing"]
    )
    # Change features that are algebraically tied to the current exposure. Other future
    # trajectories are deliberately left unchanged; this is a predictive sensitivity test.
    if "prescribing_change_4q" in scenario:
        scenario["prescribing_change_4q"] = scenario["prescribing_change_4q"] - (
            latest[intervention_feature] - scenario[intervention_feature]
        )
    scenario["scenario_probability"] = pipeline.predict_proba(scenario[features])[:, 1]

    output = latest[[
        "area_code", "area_name", "antibiotic", "quarter", "peer_cluster",
        intervention_feature, "peer_target_prescribing", "observed_probability",
    ]].copy()
    output["scenario_prescribing"] = scenario[intervention_feature].to_numpy()
    output["scenario_probability"] = scenario["scenario_probability"].to_numpy()
    output["predicted_probability_change"] = output["scenario_probability"] - output["observed_probability"]
    output["scenario_note"] = (
        "Predictive scenario only; the probability difference is not a causal policy effect."
    )
    output = output.sort_values("predicted_probability_change", ascending=False)
    output.to_csv(cfg.path("paths.table_dir") / "policy_scenario_results.csv", index=False)
    return output
