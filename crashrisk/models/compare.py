from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crashrisk.config import CrashRiskConfig, ensure_columns_exist
from crashrisk.models.splits import chronological_split
from crashrisk.models.train import _make_pipeline, _run_hyperparameter_search


ESG_PREFIXES = ("controversy_",)
ESG_COLUMNS = ("controversy_score",)


def is_esg_feature(column: str) -> bool:
    return column in ESG_COLUMNS or column.startswith(ESG_PREFIXES)


def baseline_feature_columns(config: CrashRiskConfig) -> tuple[str, ...]:
    return tuple(column for column in config.feature_columns if not is_esg_feature(column))


def _positive_probabilities(model, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    positive_index = list(model.named_steps["classifier"].classes_).index(1)
    return model.predict_proba(frame[feature_columns])[:, positive_index]


def _top_bucket_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_bucket_share: float,
) -> dict[str, float | int]:
    if len(y_true) == 0:
        return {
            "top_bucket_size": 0,
            "precision_at_top_bucket": np.nan,
            "crash_capture_at_top_bucket": np.nan,
        }

    top_k = max(1, ceil(len(y_true) * top_bucket_share))
    top_indices = np.argsort(y_prob)[::-1][:top_k]
    positives = int(np.sum(y_true))
    captured = int(np.sum(y_true[top_indices]))
    return {
        "top_bucket_size": int(top_k),
        "precision_at_top_bucket": float(captured / top_k),
        "crash_capture_at_top_bucket": float(captured / positives) if positives else np.nan,
    }


def _evaluate_split(
    model_name: str,
    split_name: str,
    model,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    top_bucket_share: float,
) -> dict[str, float | int | str]:
    y_true = frame[target_col].astype(int).to_numpy()
    y_prob = _positive_probabilities(model, frame, feature_columns)
    metrics = _top_bucket_metrics(y_true, y_prob, top_bucket_share=top_bucket_share)
    roc_auc = roc_auc_score(y_true, y_prob) if frame[target_col].nunique(dropna=True) >= 2 else np.nan
    return {
        "model": model_name,
        "split": split_name,
        "n_rows": int(len(frame)),
        "n_positive": int(np.sum(y_true)),
        "feature_count": int(len(feature_columns)),
        "roc_auc": float(roc_auc) if pd.notna(roc_auc) else np.nan,
        **metrics,
    }


def compare_esg_lift(
    dataset: pd.DataFrame,
    config: CrashRiskConfig | None = None,
    target_col: str = "high_crash_risk",
) -> pd.DataFrame:
    """Compare baseline (no ESG) vs full model (with ESG controversy features)."""
    config = config or CrashRiskConfig()
    full_features = ensure_columns_exist(config.feature_columns, dataset.columns, "model dataset")
    baseline_features = ensure_columns_exist(baseline_feature_columns(config), dataset.columns, "model dataset")
    ensure_columns_exist((target_col, "date"), dataset.columns, "model dataset")

    if not baseline_features:
        raise ValueError("Baseline feature set is empty after removing ESG controversy features")
    if len(baseline_features) == len(full_features):
        raise ValueError("Full feature set does not include ESG controversy features")

    labeled = dataset.dropna(subset=[target_col]).copy()
    labeled[target_col] = labeled[target_col].astype(int)
    if labeled[target_col].nunique() < 2:
        raise ValueError("Need at least two target classes to compare ESG lift")

    splits = chronological_split(
        labeled,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    if splits.train[target_col].nunique() < 2:
        raise ValueError("Training split needs at least two target classes to compare ESG lift")

    feature_sets = {
        "baseline_no_esg": baseline_features,
        "full_with_esg": full_features,
    }
    rows: list[dict] = []
    for model_name, feature_columns in feature_sets.items():
        feature_columns = list(feature_columns)
        model = _make_pipeline(config)
        model.fit(splits.train[feature_columns], splits.train[target_col])
        for split_name, frame in {"validation": splits.validation, "test": splits.test}.items():
            if frame.empty:
                continue
            rows.append(
                _evaluate_split(
                    model_name=model_name,
                    split_name=split_name,
                    model=model,
                    frame=frame,
                    feature_columns=feature_columns,
                    target_col=target_col,
                    top_bucket_share=config.target_top_quantile,
                )
            )

    report = pd.DataFrame(rows)
    deltas: list[dict] = []
    metric_columns = ["roc_auc", "precision_at_top_bucket", "crash_capture_at_top_bucket"]
    for split_name, split_report in report.groupby("split", sort=False):
        baseline = split_report.loc[split_report["model"] == "baseline_no_esg"]
        full = split_report.loc[split_report["model"] == "full_with_esg"]
        if baseline.empty or full.empty:
            continue
        baseline_row = baseline.iloc[0]
        full_row = full.iloc[0]
        delta_row: dict = {
            "model": "full_minus_baseline",
            "split": split_name,
            "n_rows": int(full_row["n_rows"]),
            "n_positive": int(full_row["n_positive"]),
            "feature_count": int(full_row["feature_count"] - baseline_row["feature_count"]),
            "top_bucket_size": int(full_row["top_bucket_size"]),
        }
        for column in metric_columns:
            delta_row[column] = float(full_row[column] - baseline_row[column])
        deltas.append(delta_row)

    if deltas:
        report = pd.concat([report, pd.DataFrame(deltas)], ignore_index=True)

    return report[
        [
            "model",
            "split",
            "n_rows",
            "n_positive",
            "feature_count",
            "top_bucket_size",
            "roc_auc",
            "precision_at_top_bucket",
            "crash_capture_at_top_bucket",
        ]
    ]


def compare_algorithms(
    dataset: pd.DataFrame,
    config: CrashRiskConfig | None = None,
    target_col: str = "high_crash_risk",
    model_types: list[str] | None = None,
    tune: bool = False,
) -> pd.DataFrame:
    """
    Train and evaluate multiple algorithm types on the same chronological splits.

    Compares logistic_regression, random_forest, and gradient_boosting using the
    full feature set. Optionally runs hyperparameter tuning (GridSearchCV +
    TimeSeriesSplit) for each algorithm.

    Returns a DataFrame with one row per (algorithm, split) combination, reporting
    ROC-AUC, Precision@Top-Bucket, and Crash Capture, plus the best CV ROC-AUC
    when tuning is enabled.
    """
    config = config or CrashRiskConfig()
    if model_types is None:
        model_types = ["logistic_regression", "random_forest", "gradient_boosting"]

    feature_columns = ensure_columns_exist(config.feature_columns, dataset.columns, "model dataset")
    ensure_columns_exist((target_col, "date"), dataset.columns, "model dataset")

    labeled = dataset.dropna(subset=[target_col]).copy()
    labeled[target_col] = labeled[target_col].astype(int)
    if labeled[target_col].nunique() < 2:
        raise ValueError("Need at least two target classes to compare algorithms")

    splits = chronological_split(
        labeled,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    if splits.train[target_col].nunique() < 2:
        raise ValueError("Training split needs at least two target classes")

    feature_columns = list(feature_columns)
    n_splits = getattr(config, "n_cv_splits", 5)
    rows: list[dict] = []

    for mt in model_types:
        pipeline = _make_pipeline(config, mt)
        best_cv_auc: float = float("nan")

        if tune:
            pipeline, cv_info = _run_hyperparameter_search(
                pipeline,
                splits.train[feature_columns],
                splits.train[target_col],
                model_type=mt,
                n_splits=n_splits,
            )
            best_cv_auc = float(cv_info.get("best_cv_roc_auc", float("nan")))
        else:
            pipeline.fit(splits.train[feature_columns], splits.train[target_col].astype(int))

        for split_name, frame in {"validation": splits.validation, "test": splits.test}.items():
            if frame.empty or frame[target_col].nunique(dropna=True) < 2:
                continue
            row = _evaluate_split(
                model_name=mt,
                split_name=split_name,
                model=pipeline,
                frame=frame,
                feature_columns=feature_columns,
                target_col=target_col,
                top_bucket_share=config.target_top_quantile,
            )
            row["best_cv_roc_auc"] = best_cv_auc
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    columns = [
        "model",
        "split",
        "n_rows",
        "n_positive",
        "best_cv_roc_auc",
        "roc_auc",
        "precision_at_top_bucket",
        "crash_capture_at_top_bucket",
        "top_bucket_size",
    ]
    return result[[c for c in columns if c in result.columns]]
