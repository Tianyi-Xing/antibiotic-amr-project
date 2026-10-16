from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .utils import first_existing, safe_numeric, write_json


@dataclass
class Check:
    dataset: str
    check: str
    passed: bool
    severity: str
    detail: str


class DataValidationError(RuntimeError):
    pass


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def validate_prescribing(path: Path) -> tuple[pd.DataFrame, list[Check]]:
    df = _load_csv(path)
    checks: list[Check] = []
    required = {"date", "row_id", "row_name", "broad_spectrum_items_per_1000"}
    missing = sorted(required.difference(df.columns))
    checks.append(Check("prescribing", "required_columns", not missing, "error", f"missing={missing}"))
    if missing:
        return df, checks

    parsed = pd.to_datetime(df["date"], errors="coerce")
    checks.append(Check("prescribing", "parseable_dates", parsed.notna().mean() >= 0.99, "error", f"valid_share={parsed.notna().mean():.3f}"))
    dup = df.duplicated(["date", "row_id"]).sum()
    checks.append(Check("prescribing", "unique_area_month", dup == 0, "error", f"duplicates={dup}"))
    rate = safe_numeric(df["broad_spectrum_items_per_1000"])
    invalid = ((rate < 0) | (rate > 1000)).sum()
    checks.append(Check("prescribing", "plausible_rate", invalid == 0, "error", f"outside_[0,1000]={invalid}"))
    missing_rate = rate.isna().mean()
    checks.append(Check("prescribing", "rate_missingness", missing_rate <= 0.10, "warning", f"missing_share={missing_rate:.3f}"))
    return df, checks


def validate_resistance(path: Path) -> tuple[pd.DataFrame, list[Check]]:
    df = _load_csv(path)
    checks: list[Check] = []
    columns = list(df.columns)
    date_col = first_existing(columns, ("parsed_date", "date", "reporting_date", "period_end", "month"))
    area_col = first_existing(columns, ("requested_geography", "geography_code", "area_code", "geography"))
    metric_col = first_existing(columns, ("requested_metric", "metric", "metric_name"))
    value_col = first_existing(columns, ("metric_value", "value", "figure", "y_axis"))
    found = {"date": date_col, "area": area_col, "metric": metric_col, "value": value_col}
    missing = [key for key, value in found.items() if value is None]
    checks.append(Check("resistance", "discoverable_core_columns", not missing, "error", f"mapping={found}"))
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        checks.append(Check("resistance", "parseable_dates", dates.notna().mean() >= 0.95, "error", f"valid_share={dates.notna().mean():.3f}"))
    if metric_col:
        metrics = df[metric_col].astype(str)
        required_tokens = ("PercentResistant", "NumberTested")
        for token in required_tokens:
            present = metrics.str.contains(token, case=False, regex=False).any()
            checks.append(Check("resistance", f"metric_{token}", present, "error", f"present={present}"))
    if value_col:
        numeric = safe_numeric(df[value_col])
        checks.append(Check("resistance", "numeric_values", numeric.notna().mean() >= 0.90, "error", f"numeric_share={numeric.notna().mean():.3f}"))
    return df, checks


def validate_covariates(path: Path) -> tuple[pd.DataFrame | None, list[Check]]:
    if not path.exists():
        return None, [Check("covariates", "file_exists", False, "warning", "Optional file not found; models will use available core features.")]
    df = _load_csv(path)
    required = {"area_code", "year"}
    missing = sorted(required.difference(df.columns))
    checks = [Check("covariates", "required_columns", not missing, "error", f"missing={missing}")]
    if not missing:
        dup = df.duplicated(["area_code", "year"]).sum()
        checks.append(Check("covariates", "unique_area_year", dup == 0, "error", f"duplicates={dup}"))
    share_cols = [c for c in ("age65_share", "age80_share") if c in df]
    for col in share_cols:
        values = safe_numeric(df[col])
        invalid = ((values < 0) | (values > 1)).sum()
        checks.append(Check("covariates", f"range_{col}", invalid == 0, "error", f"outside_[0,1]={invalid}"))
    return df, checks


def run_validation(cfg: Config, fail_on_error: bool = True) -> pd.DataFrame:
    raw_dir = cfg.path("paths.raw_dir")
    external_dir = cfg.path("paths.external_dir")
    prescribing_path = raw_dir / cfg.get("paths.prescribing_file")
    resistance_path = raw_dir / cfg.get("paths.resistance_file")
    covariates_path = external_dir / cfg.get("paths.covariates_file")

    all_checks: list[Check] = []
    for function, path in (
        (validate_prescribing, prescribing_path),
        (validate_resistance, resistance_path),
        (validate_covariates, covariates_path),
    ):
        try:
            _, checks = function(path)
            all_checks.extend(checks)
        except FileNotFoundError as exc:
            dataset = path.name
            all_checks.append(Check(dataset, "file_exists", False, "error", str(exc)))

    result = pd.DataFrame([asdict(check) for check in all_checks])
    output = cfg.path("paths.table_dir") / "data_validation_report.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    write_json(result.to_dict(orient="records"), cfg.path("paths.table_dir") / "data_validation_report.json")

    errors = result[(result["severity"] == "error") & (~result["passed"])]
    if fail_on_error and not errors.empty:
        raise DataValidationError(errors.to_string(index=False))
    return result
