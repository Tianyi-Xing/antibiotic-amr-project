from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .utils import seed_everything


def generate_demo_data(cfg: Config, n_areas: int = 40) -> dict[str, Path]:
    seed = int(cfg.get("project.random_seed", 2026))
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    raw_dir = cfg.path("paths.raw_dir")
    external_dir = cfg.path("paths.external_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(cfg.get("project.start_date"), cfg.get("project.end_date"), freq="MS")
    intervention = pd.Timestamp(cfg.get("project.intervention_date", "2023-04-01"))
    areas = [f"SICBL{i:03d}" for i in range(1, n_areas + 1)]
    names = [f"Demo Sub-ICB {i:02d}" for i in range(1, n_areas + 1)]
    latent_class = np.array([i % 4 for i in range(n_areas)])
    rng.shuffle(latent_class)

    cov_rows = []
    area_cov: dict[str, dict[str, float]] = {}
    for idx, (code, name) in enumerate(zip(areas, names)):
        deprivation = float(np.clip(rng.normal(22 + 4 * (latent_class[idx] in (1, 3)), 6), 5, 45))
        age65 = float(np.clip(rng.normal(0.18 + 0.015 * (latent_class[idx] == 1), 0.025), 0.10, 0.28))
        age80 = float(np.clip(age65 * rng.uniform(0.20, 0.32), 0.02, 0.09))
        diabetes = float(np.clip(0.055 + deprivation * 0.0008 + rng.normal(0, 0.005), 0.035, 0.11))
        ckd = float(np.clip(0.030 + age65 * 0.08 + rng.normal(0, 0.004), 0.02, 0.07))
        copd = float(np.clip(0.010 + deprivation * 0.00045 + rng.normal(0, 0.003), 0.006, 0.04))
        cancer = float(np.clip(0.018 + age65 * 0.06 + rng.normal(0, 0.003), 0.015, 0.05))
        area_cov[code] = {
            "age65_share": age65,
            "age80_share": age80,
            "imd_score": deprivation,
            "diabetes_prevalence": diabetes,
            "ckd_prevalence": ckd,
            "copd_prevalence": copd,
            "cancer_prevalence": cancer,
        }
        for year in sorted(set(dates.year)):
            drift = (year - dates.year.min()) * 0.0005
            cov_rows.append({
                "area_code": code,
                "year": year,
                "age65_share": age65 + drift,
                "age80_share": age80 + drift / 2,
                "imd_score": deprivation,
                "diabetes_prevalence": diabetes + drift,
                "ckd_prevalence": ckd + drift / 2,
                "copd_prevalence": copd,
                "cancer_prevalence": cancer + drift / 2,
            })

    prescribing_rows = []
    resistance_rows = []
    truth_rows = []
    for idx, (code, name) in enumerate(zip(areas, names)):
        cls = int(latent_class[idx])
        list_size = int(rng.integers(70000, 230000))
        baseline_p = rng.normal(3.5, 0.45) + 0.02 * area_cov[code]["imd_score"]
        baseline_r = rng.normal(17.0, 2.0) + 20 * area_cov[code]["diabetes_prevalence"]
        p_values: list[float] = []
        r_values: list[float] = []
        for t, dt in enumerate(dates):
            season = 0.35 * np.cos(2 * np.pi * dt.month / 12)
            post_months = max(0, (dt.year - intervention.year) * 12 + dt.month - intervention.month)
            if cls in (0, 1):
                policy_drop = -0.070 * post_months - (0.80 if dt >= intervention else 0)
            else:
                policy_drop = 0.015 * post_months
            p = baseline_p + season + policy_drop + rng.normal(0, 0.16)
            p = float(max(0.5, p))
            p_values.append(p)

            lag = 6
            exposure_change = 0.0
            if t >= lag:
                exposure_change = p_values[t - lag] - p_values[max(0, t - lag - 6)]
            if cls in (0, 2):
                response = 1.00 * exposure_change
            else:
                response = 0.03 * exposure_change
            secular = -0.030 * t if cls in (0, 2) else 0.018 * t
            r = baseline_r + secular + response + 6 * area_cov[code]["ckd_prevalence"] + rng.normal(0, 0.48)
            r = float(np.clip(r, 3, 45))
            r_values.append(r)

            items = max(0, round(p * list_size / 1000))
            prescribing_rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "row_id": code,
                "row_name": name,
                "broad_spectrum_items": items,
                "total_list_size": list_size,
                "broad_spectrum_items_per_1000": p,
            })

        rolling_r = pd.Series(r_values).rolling(3, min_periods=1).mean().to_numpy()
        for dt, value in zip(dates, rolling_r):
            n_tested = int(max(25, rng.poisson(85 + list_size / 5000)))
            percent_tested = float(np.clip(rng.normal(82, 6), 55, 100))
            for metric, metric_value in (
                ("e-coli_testing_bacteraemiaPercentResistantRollingMonth", value),
                ("e-coli_testing_bacteraemiaNumberTestedRollingMonth", n_tested),
                ("e-coli_testing_bacteraemiaPercentTestedRollingMonth", percent_tested),
            ):
                resistance_rows.append({
                    "parsed_date": dt.strftime("%Y-%m-%d"),
                    "requested_geography": code,
                    "geography_name": name,
                    "requested_metric": metric,
                    "stratum": "co-amoxiclav",
                    "metric_value": metric_value,
                })
        truth_rows.append({
            "area_code": code,
            "area_name": name,
            "true_class": cls,
            "true_class_label": [
                "prescribing_down_resistance_down",
                "prescribing_down_resistance_not_down",
                "prescribing_not_down_resistance_down",
                "neither_improved",
            ][cls],
        })

    prescribing_path = raw_dir / cfg.get("paths.prescribing_file")
    resistance_path = raw_dir / cfg.get("paths.resistance_file")
    covariates_path = external_dir / cfg.get("paths.covariates_file")
    truth_path = external_dir / "synthetic_truth.csv"
    pd.DataFrame(prescribing_rows).to_csv(prescribing_path, index=False)
    pd.DataFrame(resistance_rows).to_csv(resistance_path, index=False)
    pd.DataFrame(cov_rows).to_csv(covariates_path, index=False)
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False)
    return {
        "prescribing": prescribing_path,
        "resistance": resistance_path,
        "covariates": covariates_path,
        "truth": truth_path,
    }
