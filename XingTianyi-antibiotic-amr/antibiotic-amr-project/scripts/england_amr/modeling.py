from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .config import Config
from .utils import seed_everything, write_json

LOGGER = logging.getLogger(__name__)


def _make_estimator(cfg: Config) -> Any:
    algorithm = str(cfg.get("model.algorithm", "random_forest")).lower()
    seed = int(cfg.get("project.random_seed", 2026))
    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("Install xgboost or set model.algorithm=random_forest") from exc
        return XGBClassifier(
            n_estimators=int(cfg.get("model.n_estimators", 500)),
            max_depth=int(cfg.get("model.max_depth", 5)),
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=int(cfg.get("model.n_estimators", 500)),
        max_depth=int(cfg.get("model.max_depth", 6)),
        min_samples_leaf=int(cfg.get("model.min_samples_leaf", 5)),
        class_weight=cfg.get("model.class_weight", "balanced"),
        random_state=seed,
        n_jobs=-1,
    )


def _safe_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    prediction = (probabilities >= threshold).astype(int)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "decision_threshold": float(threshold),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["average_precision"] = float(average_precision_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
    return metrics


def _prepare_training_data(cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(cfg.path("paths.processed_dir") / "model_features.csv", parse_dates=["quarter"])
    feature_columns = [str(x) for x in cfg.get("model.feature_columns", [])]
    # Missing optional covariates remain explicit missing values and are imputed.
    for column in feature_columns:
        if column not in data:
            data[column] = np.nan
    target = "resistance_improved"
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data[data["eligible_reduction_episode"].astype(str).isin(["True", "1", "true"])].copy()
    data = data.dropna(subset=[target, "quarter", "area_code"])
    data[target] = data[target].astype(int)
    available = [c for c in feature_columns if c in data.columns]
    if not available:
        raise ValueError("No configured feature columns are available")
    if len(data) < 80 or data["area_code"].nunique() < 8:
        raise ValueError(f"Insufficient training data: rows={len(data)}, areas={data['area_code'].nunique()}")
    return data.sort_values("quarter"), available


def _rolling_origin_cutoffs(quarters: list[pd.Timestamp], test_width: int, minimum_train: int = 8) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    unique = sorted(pd.Timestamp(x) for x in quarters)
    cutoffs = []
    for index in range(minimum_train, len(unique) - test_width + 1, test_width):
        train_end = unique[index - 1]
        test_end = unique[min(index + test_width - 1, len(unique) - 1)]
        cutoffs.append((train_end, test_end))
    return cutoffs[-4:]


def _fit_pipeline(cfg: Config) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", _make_estimator(cfg)),
    ])




def _select_threshold(cfg: Config, train: pd.DataFrame, features: list[str]) -> float:
    quarters = sorted(train["quarter"].drop_duplicates())
    if len(quarters) < 6:
        return 0.5
    validation_quarters = quarters[-2:]
    inner_train = train[~train["quarter"].isin(validation_quarters)]
    validation = train[train["quarter"].isin(validation_quarters)]
    if inner_train["resistance_improved"].nunique() < 2 or validation["resistance_improved"].nunique() < 2:
        return 0.5
    candidate_model = _fit_pipeline(cfg)
    candidate_model.fit(inner_train[features], inner_train["resistance_improved"])
    probabilities = candidate_model.predict_proba(validation[features])[:, 1]
    candidates = np.linspace(0.10, 0.90, 33)
    scores = [
        f1_score(validation["resistance_improved"], probabilities >= threshold, zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])

def _extract_importance(pipeline: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    else:
        values = np.full(len(feature_columns), np.nan)
    return pd.DataFrame({"feature": feature_columns, "importance": values}).sort_values("importance", ascending=False)


def _compute_shap(
    pipeline: Pipeline,
    X: pd.DataFrame,
    feature_columns: list[str],
    table_dir: Path,
    figure_dir: Path,
    seed: int,
) -> None:
    try:
        import shap
    except ImportError:
        LOGGER.warning("SHAP is not installed; explanation outputs were skipped")
        return
    if X.empty:
        return
    sample = X.sample(min(300, len(X)), random_state=seed)
    transformed = pipeline.named_steps["imputer"].transform(sample)
    model = pipeline.named_steps["model"]
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        mean_abs = np.abs(values).mean(axis=0)
        pd.DataFrame({"feature": feature_columns, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        ).to_csv(table_dir / "shap_importance.csv", index=False)
        plt.figure(figsize=(9, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            shap.summary_plot(values, transformed, feature_names=feature_columns, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(figure_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        LOGGER.warning("SHAP computation failed: %s", exc)


def run_training(cfg: Config) -> dict[str, Path]:
    seed = int(cfg.get("project.random_seed", 2026))
    seed_everything(seed)
    data, features = _prepare_training_data(cfg)
    table_dir = cfg.path("paths.table_dir")
    model_dir = cfg.path("paths.model_dir")
    figure_dir = cfg.path("paths.figure_dir")
    model_dir.mkdir(parents=True, exist_ok=True)

    test_width = int(cfg.get("model.rolling_test_quarters", 2))
    cutoffs = _rolling_origin_cutoffs(data["quarter"].drop_duplicates().tolist(), test_width=test_width)
    evaluation_rows = []
    prediction_rows = []
    for fold, (train_end, test_end) in enumerate(cutoffs, start=1):
        train = data[data["quarter"] <= train_end]
        test = data[(data["quarter"] > train_end) & (data["quarter"] <= test_end)]
        if train["resistance_improved"].nunique() < 2 or test.empty:
            continue
        threshold = _select_threshold(cfg, train, features)
        pipeline = _fit_pipeline(cfg)
        pipeline.fit(train[features], train["resistance_improved"])
        prob = pipeline.predict_proba(test[features])[:, 1]
        metrics = _safe_metrics(test["resistance_improved"], prob, threshold)
        evaluation_rows.append({
            "fold": fold,
            "train_end": train_end.date(),
            "test_end": test_end.date(),
            "train_rows": len(train),
            "test_rows": len(test),
            "test_positive_rate": test["resistance_improved"].mean(),
            **metrics,
        })
        fold_predictions = test[["area_code", "area_name", "antibiotic", "quarter", "resistance_improved"]].copy()
        fold_predictions["predicted_probability"] = prob
        fold_predictions["decision_threshold"] = threshold
        fold_predictions["predicted_label"] = (prob >= threshold).astype(int)
        fold_predictions["fold"] = fold
        prediction_rows.append(fold_predictions)

    evaluations = pd.DataFrame(evaluation_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    evaluations.to_csv(table_dir / "rolling_origin_metrics.csv", index=False)
    predictions.to_csv(table_dir / "rolling_origin_predictions.csv", index=False)

    unique_quarters = sorted(data["quarter"].drop_duplicates())
    final_test_quarters = int(cfg.get("model.test_quarters", 4))
    cutoff = unique_quarters[-final_test_quarters] if len(unique_quarters) > final_test_quarters else unique_quarters[-1]
    train = data[data["quarter"] < cutoff]
    test = data[data["quarter"] >= cutoff]
    if train["resistance_improved"].nunique() < 2:
        raise ValueError("Final training period contains only one target class")
    decision_threshold = _select_threshold(cfg, train, features)
    pipeline = _fit_pipeline(cfg)
    pipeline.fit(train[features], train["resistance_improved"])
    final_prob = pipeline.predict_proba(test[features])[:, 1] if not test.empty else np.array([])
    final_metrics = _safe_metrics(test["resistance_improved"], final_prob, decision_threshold) if not test.empty else {"decision_threshold": decision_threshold}
    final_metrics.update({
        "train_end": str(train["quarter"].max().date()),
        "test_start": str(test["quarter"].min().date()) if not test.empty else None,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_columns": features,
        "algorithm": cfg.get("model.algorithm", "random_forest"),
    })
    write_json(final_metrics, table_dir / "final_holdout_metrics.json")

    bundle = {
        "pipeline": pipeline,
        "features": features,
        "target": "resistance_improved",
        "horizon_quarters": int(cfg.get("analysis.change_window_quarters", 4)),
        "training_quarter_max": train["quarter"].max(),
        "decision_threshold": decision_threshold,
    }
    model_path = model_dir / "resistance_improvement_model.joblib"
    joblib.dump(bundle, model_path)
    importance = _extract_importance(pipeline, features)
    importance.to_csv(table_dir / "model_feature_importance.csv", index=False)
    _compute_shap(pipeline, test[features] if not test.empty else train[features], features, table_dir, figure_dir, seed)

    if not test.empty:
        holdout = test[["area_code", "area_name", "antibiotic", "quarter", "resistance_improved"]].copy()
        holdout["predicted_probability"] = final_prob
        holdout["decision_threshold"] = decision_threshold
        holdout["predicted_label"] = (final_prob >= decision_threshold).astype(int)
        holdout.to_csv(table_dir / "final_holdout_predictions.csv", index=False)
    return {"model": model_path, "metrics": table_dir / "final_holdout_metrics.json"}
