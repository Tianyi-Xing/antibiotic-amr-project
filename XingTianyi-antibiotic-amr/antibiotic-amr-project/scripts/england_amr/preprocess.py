from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config
from .utils import first_existing, quarter_start, safe_numeric

LOGGER = logging.getLogger(__name__)


def _normalize_antibiotic(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "co-amoxiclav": "co-amoxiclav",
        "co amoxiclav": "co-amoxiclav",
        "amoxicillin/clavulanate": "co-amoxiclav",
        "amoxicillin clavulanate": "co-amoxiclav",
        "augmentin": "co-amoxiclav",
        "ciprofloxacin": "ciprofloxacin",
        "3rd generation cephalosporins": "third-generation cephalosporins",
        "third generation cephalosporins": "third-generation cephalosporins",
        "third-generation cephalosporins": "third-generation cephalosporins",
    }
    for key, mapped in aliases.items():
        if key in text:
            return mapped
    return text or "unspecified"


def normalize_prescribing(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "row_id", "row_name", "broad_spectrum_items_per_1000"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prescribing file lacks columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["area_code"] = out["row_id"].astype(str)
    out["area_name"] = out["row_name"].astype(str)
    for column in ("broad_spectrum_items", "total_list_size", "broad_spectrum_items_per_1000"):
        if column in out:
            out[column] = safe_numeric(out[column])
    keep = [
        c for c in (
            "date", "area_code", "area_name", "broad_spectrum_items",
            "total_list_size", "broad_spectrum_items_per_1000",
        ) if c in out
    ]
    out = out[keep].dropna(subset=["date", "area_code", "broad_spectrum_items_per_1000"])
    return out.sort_values(["area_code", "date"]).drop_duplicates(["area_code", "date"], keep="last")


def normalize_resistance(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    date_col = first_existing(columns, ("parsed_date", "date", "reporting_date", "period_end", "period_start", "month"))
    area_col = first_existing(columns, ("requested_geography", "geography_code", "area_code", "geography"))
    area_name_col = first_existing(columns, ("geography_name", "area_name", "requested_geography", "geography"))
    metric_col = first_existing(columns, ("requested_metric", "metric_name", "metric"))
    value_col = first_existing(columns, ("metric_value", "value", "figure", "y_axis", "rate"))
    antibiotic_col = first_existing(columns, ("stratum", "antibiotic", "category", "breakdown", "label"))
    missing = [name for name, col in {"date": date_col, "area": area_col, "metric": metric_col, "value": value_col}.items() if col is None]
    if missing:
        raise ValueError(f"Could not infer resistance columns {missing}; available={columns}")

    long = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "area_code": df[area_col].astype(str),
        "area_name": df[area_name_col].astype(str) if area_name_col else df[area_col].astype(str),
        "metric": df[metric_col].astype(str),
        "value": safe_numeric(df[value_col]),
        "antibiotic": df[antibiotic_col].map(_normalize_antibiotic) if antibiotic_col else "unspecified",
    })
    long = long.dropna(subset=["date", "area_code", "value"])

    def metric_kind(text: str) -> str:
        low = text.lower()
        if "percentresistant" in low or ("resistant" in low and "percent" in low):
            return "resistance_percent"
        if "numbertested" in low or ("tested" in low and "number" in low):
            return "n_tested"
        if "percenttested" in low or ("tested" in low and "percent" in low):
            return "percent_tested"
        return "other"

    long["metric_kind"] = long["metric"].map(metric_kind)
    long = long[long["metric_kind"] != "other"]
    pivot = (
        long.pivot_table(
            index=["date", "area_code", "area_name", "antibiotic"],
            columns="metric_kind",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    return pivot.sort_values(["antibiotic", "area_code", "date"])


def _select_quarterly_snapshots(df: pd.DataFrame, anchor_months: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    out["quarter"] = quarter_start(out["date"])
    anchors = set(int(x) for x in anchor_months)
    anchored = out[out["date"].dt.month.isin(anchors)].copy()
    if anchored.empty:
        # Fall back to the last available observation in each quarter.
        anchored = out.sort_values("date").groupby(
            [c for c in ("area_code", "antibiotic", "quarter") if c in out.columns],
            as_index=False,
        ).tail(1)
    else:
        anchored = anchored.sort_values("date").groupby(
            [c for c in ("area_code", "antibiotic", "quarter") if c in anchored.columns],
            as_index=False,
        ).tail(1)
    return anchored


def merge_panel(
    prescribing: pd.DataFrame,
    resistance: pd.DataFrame,
    covariates: pd.DataFrame | None,
    *,
    anchor_months: list[int],
    primary_antibiotics: list[str],
    minimum_tested: float,
) -> pd.DataFrame:
    p = prescribing.copy()
    p["quarter"] = quarter_start(p["date"])
    p = (
        p.groupby(["area_code", "area_name", "quarter"], as_index=False)
        .agg(
            broad_spectrum_items_per_1000=("broad_spectrum_items_per_1000", "mean"),
            broad_spectrum_items=("broad_spectrum_items", "sum") if "broad_spectrum_items" in p else ("broad_spectrum_items_per_1000", "size"),
            total_list_size=("total_list_size", "mean") if "total_list_size" in p else ("broad_spectrum_items_per_1000", "size"),
        )
    )

    r = _select_quarterly_snapshots(resistance, anchor_months)
    primary = {_normalize_antibiotic(x) for x in primary_antibiotics}
    r = r[r["antibiotic"].isin(primary)].copy()
    if "n_tested" in r:
        r.loc[r["n_tested"] < minimum_tested, "resistance_percent"] = np.nan

    panel = p.merge(
        r[[c for c in ("area_code", "area_name", "quarter", "antibiotic", "resistance_percent", "n_tested", "percent_tested") if c in r]],
        on=["area_code", "quarter"],
        how="inner",
        suffixes=("_prescribing", "_resistance"),
        validate="one_to_many",
    )
    panel["area_name"] = panel.get("area_name_resistance", panel.get("area_name_prescribing", panel["area_code"]))
    panel = panel.drop(columns=[c for c in ("area_name_prescribing", "area_name_resistance") if c in panel])

    if covariates is not None and not covariates.empty:
        cov = covariates.copy()
        cov["area_code"] = cov["area_code"].astype(str)
        cov["year"] = pd.to_numeric(cov["year"], errors="coerce").astype("Int64")
        panel["year"] = panel["quarter"].dt.year.astype("Int64")
        panel = panel.merge(cov, on=["area_code", "year"], how="left", validate="many_to_one")
    else:
        panel["year"] = panel["quarter"].dt.year

    return panel.sort_values(["antibiotic", "area_code", "quarter"]).reset_index(drop=True)


def run_preprocessing(cfg: Config) -> pd.DataFrame:
    raw_dir = cfg.path("paths.raw_dir")
    external_dir = cfg.path("paths.external_dir")
    prescribing = pd.read_csv(raw_dir / cfg.get("paths.prescribing_file"), low_memory=False)
    resistance = pd.read_csv(raw_dir / cfg.get("paths.resistance_file"), low_memory=False)
    cov_path = external_dir / cfg.get("paths.covariates_file")
    covariates = pd.read_csv(cov_path, low_memory=False) if cov_path.exists() else None

    p = normalize_prescribing(prescribing)
    r = normalize_resistance(resistance)
    panel = merge_panel(
        p,
        r,
        covariates,
        anchor_months=cfg.get("analysis.quarterly_anchor_months", [1, 4, 7, 10]),
        primary_antibiotics=cfg.get("analysis.primary_antibiotics", ["co-amoxiclav"]),
        minimum_tested=float(cfg.get("analysis.minimum_tested", 20)),
    )
    interim = cfg.path("paths.interim_dir")
    p.to_csv(interim / "prescribing_normalized.csv", index=False)
    r.to_csv(interim / "resistance_normalized.csv", index=False)
    panel.to_csv(cfg.path("paths.processed_dir") / "area_quarter_panel.csv", index=False)
    LOGGER.info("Built panel: %d rows, %d areas", len(panel), panel["area_code"].nunique())
    return panel
