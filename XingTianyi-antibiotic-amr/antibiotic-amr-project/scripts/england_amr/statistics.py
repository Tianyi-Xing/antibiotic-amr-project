from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def interrupted_time_series(
    monthly: pd.DataFrame,
    *,
    intervention_date: str,
    outcome: str = "broad_spectrum_items_per_1000",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = monthly.copy()
    data["date"] = pd.to_datetime(data["date"])
    if "total_list_size" in data and data["total_list_size"].notna().any():
        national = (
            data.groupby("date", as_index=False)
            .apply(
                lambda g: pd.Series({
                    outcome: np.average(g[outcome], weights=g["total_list_size"].fillna(1)),
                    "weight": g["total_list_size"].sum(),
                }),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
    else:
        national = data.groupby("date", as_index=False)[outcome].mean()
        national["weight"] = 1.0

    intervention = pd.Timestamp(intervention_date)
    if not (national["date"].min() < intervention <= national["date"].max()):
        return pd.DataFrame(), national
    national = national.sort_values("date")
    national["time"] = np.arange(len(national))
    intervention_index = int(national.loc[national["date"] >= intervention, "time"].min())
    national["post"] = (national["date"] >= intervention).astype(int)
    national["time_after"] = (national["time"] - intervention_index).clip(lower=0)
    national["month"] = national["date"].dt.month.astype(str)
    formula = f"{outcome} ~ time + post + time_after + C(month)"
    model = smf.wls(formula, data=national, weights=national["weight"].clip(lower=1)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 12}
    )
    conf = model.conf_int()
    table = pd.DataFrame({
        "term": model.params.index,
        "estimate": model.params.values,
        "std_error": model.bse.values,
        "p_value": model.pvalues.values,
        "ci_low": conf[0].values,
        "ci_high": conf[1].values,
    })
    national["fitted"] = model.predict(national)
    return table, national


def distributed_lag_fixed_effects(
    panel: pd.DataFrame,
    *,
    lag_quarters: list[int],
    exposure: str = "broad_spectrum_items_per_1000",
    outcome: str = "resistance_percent",
) -> pd.DataFrame:
    all_results = []
    for antibiotic, data in panel.groupby("antibiotic"):
        data = data.sort_values(["area_code", "quarter"]).copy()
        data["quarter_fe"] = pd.to_datetime(data["quarter"]).dt.to_period("Q").astype(str)
        sd = data[exposure].std()
        data["exposure_z"] = (data[exposure] - data[exposure].mean()) / (sd if sd and np.isfinite(sd) else 1.0)
        lag_terms = []
        for lag in lag_quarters:
            col = f"exposure_lag_{lag}q"
            data[col] = data.groupby("area_code")["exposure_z"].shift(lag)
            lag_terms.append(col)
        required = [outcome, "area_code", "quarter_fe"] + lag_terms
        model_data = data.dropna(subset=required).copy()
        if len(model_data) < 50 or model_data["area_code"].nunique() < 5:
            continue
        formula = f"{outcome} ~ " + " + ".join(lag_terms) + " + C(area_code) + C(quarter_fe)"
        weights = model_data["n_tested"].clip(lower=1) if "n_tested" in model_data else np.ones(len(model_data))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.wls(formula, data=model_data, weights=weights).fit(
                cov_type="cluster", cov_kwds={"groups": model_data["area_code"]}
            )
        conf = model.conf_int()
        for term in lag_terms:
            if term not in model.params:
                continue
            all_results.append({
                "antibiotic": antibiotic,
                "lag_quarters": int(term.split("_")[-1].replace("q", "")),
                "estimate_per_1sd_prescribing": float(model.params[term]),
                "std_error": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
                "n_observations": int(model.nobs),
                "n_areas": int(model_data["area_code"].nunique()),
                "r_squared": float(model.rsquared),
            })
    return pd.DataFrame(all_results)
