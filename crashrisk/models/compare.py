from __future__ import annotations

import json
from math import ceil

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crashrisk.config import CrashRiskConfig, ensure_columns_exist
from crashrisk.models.splits import chronological_split
from crashrisk.models.train import _PARAM_GRIDS, _make_pipeline, _run_hyperparameter_search


ESG_PREFIXES = ("controversy_",)
ESG_COLUMNS = ("controversy_score",)
TEXT_FEATURE_COLUMNS = (
    "negative_esg_controversy_score_0_100",
    "rolling_sentiment_13w",
)


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


def compare_text_signal_lift(
    dataset: pd.DataFrame,
    config: CrashRiskConfig | None = None,
    target_col: str = "high_crash_risk",
    text_features: tuple[str, ...] = TEXT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """
    Compare the standard full ESG model with a text-augmented variant.

    The text columns are intentionally not added to config.feature_columns, so
    the original 20-feature ESG lift table remains backward-compatible.
    """
    config = config or CrashRiskConfig()
    full_features = ensure_columns_exist(config.feature_columns, dataset.columns, "model dataset")
    available_text_features = ensure_columns_exist(text_features, dataset.columns, "model dataset")
    ensure_columns_exist((target_col, "date"), dataset.columns, "model dataset")

    labeled = dataset.dropna(subset=[target_col]).copy()
    labeled[target_col] = labeled[target_col].astype(int)
    if labeled[target_col].nunique() < 2:
        raise ValueError("Need at least two target classes to compare text lift")

    splits = chronological_split(
        labeled,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    if splits.train[target_col].nunique() < 2:
        raise ValueError("Training split needs at least two target classes to compare text lift")

    feature_sets = {
        "full_with_esg": list(full_features),
        "full_with_esg_plus_text": [*full_features, *available_text_features],
    }
    rows: list[dict] = []
    for model_name, feature_columns in feature_sets.items():
        model = _make_pipeline(config)
        model.fit(splits.train[feature_columns], splits.train[target_col])
        for split_name, frame in {"validation": splits.validation, "test": splits.test}.items():
            if frame.empty or frame[target_col].nunique(dropna=True) < 2:
                continue
            row = _evaluate_split(
                model_name=model_name,
                split_name=split_name,
                model=model,
                frame=frame,
                feature_columns=feature_columns,
                target_col=target_col,
                top_bucket_share=config.target_top_quantile,
            )
            row["text_covered_rows"] = int(frame[list(available_text_features)].notna().any(axis=1).sum())
            rows.append(row)

    report = pd.DataFrame(rows)
    deltas: list[dict] = []
    metric_columns = ["roc_auc", "precision_at_top_bucket", "crash_capture_at_top_bucket"]
    for split_name, split_report in report.groupby("split", sort=False):
        baseline = split_report.loc[split_report["model"] == "full_with_esg"]
        augmented = split_report.loc[split_report["model"] == "full_with_esg_plus_text"]
        if baseline.empty or augmented.empty:
            continue
        baseline_row = baseline.iloc[0]
        augmented_row = augmented.iloc[0]
        delta_row: dict = {
            "model": "text_minus_full_esg",
            "split": split_name,
            "n_rows": int(augmented_row["n_rows"]),
            "n_positive": int(augmented_row["n_positive"]),
            "feature_count": int(augmented_row["feature_count"] - baseline_row["feature_count"]),
            "top_bucket_size": int(augmented_row["top_bucket_size"]),
            "text_covered_rows": int(augmented_row.get("text_covered_rows", 0)),
        }
        for column in metric_columns:
            delta_row[column] = float(augmented_row[column] - baseline_row[column])
        deltas.append(delta_row)

    if deltas:
        report = pd.concat([report, pd.DataFrame(deltas)], ignore_index=True)

    columns = [
        "model",
        "split",
        "n_rows",
        "n_positive",
        "feature_count",
        "top_bucket_size",
        "text_covered_rows",
        "roc_auc",
        "precision_at_top_bucket",
        "crash_capture_at_top_bucket",
    ]
    return report[[column for column in columns if column in report.columns]]


def build_hyperparameter_tuning_results(
    dataset: pd.DataFrame,
    config: CrashRiskConfig | None = None,
    target_col: str = "high_crash_risk",
    model_types: list[str] | None = None,
    run_search: bool = False,
) -> pd.DataFrame:
    """Return the searched grids and, when requested, the best CV result."""
    config = config or CrashRiskConfig()
    if model_types is None:
        model_types = ["logistic_regression", "random_forest", "gradient_boosting"]

    feature_columns = ensure_columns_exist(config.feature_columns, dataset.columns, "model dataset")
    ensure_columns_exist((target_col, "date"), dataset.columns, "model dataset")

    rows: list[dict] = []
    labeled = dataset.dropna(subset=[target_col]).copy()
    labeled[target_col] = labeled[target_col].astype(int)
    can_search = run_search and labeled[target_col].nunique() >= 2
    splits = None
    if can_search:
        splits = chronological_split(
            labeled,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
        )
        can_search = splits.train[target_col].nunique() >= 2

    for model_type in model_types:
        grid = _PARAM_GRIDS.get(model_type, {})
        row: dict[str, object] = {
            "model": model_type,
            "status": "not_run",
            "grid_searched": json.dumps(grid, sort_keys=True),
            "best_params": "{}",
            "best_cv_roc_auc": np.nan,
            "cv_roc_auc_std": np.nan,
            "n_cv_splits": getattr(config, "n_cv_splits", 5),
            "n_candidates": int(np.prod([len(values) for values in grid.values()])) if grid else 0,
        }
        if can_search and splits is not None:
            pipeline = _make_pipeline(config, model_type)
            _, cv_info = _run_hyperparameter_search(
                pipeline,
                splits.train[list(feature_columns)],
                splits.train[target_col],
                model_type=model_type,
                n_splits=getattr(config, "n_cv_splits", 5),
            )
            row.update(
                {
                    "status": "ok",
                    "best_params": json.dumps(cv_info.get("best_params", {}), sort_keys=True),
                    "best_cv_roc_auc": cv_info.get("best_cv_roc_auc", np.nan),
                    "cv_roc_auc_std": cv_info.get("cv_roc_auc_std", np.nan),
                    "n_cv_splits": cv_info.get("n_cv_splits", getattr(config, "n_cv_splits", 5)),
                    "n_candidates": cv_info.get("n_candidates", row["n_candidates"]),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def build_test_diagnostics(
    dataset: pd.DataFrame,
    config: CrashRiskConfig | None = None,
    target_col: str = "high_crash_risk",
    n_calibration_bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build confusion-matrix and reliability-curve data on the test split."""
    config = config or CrashRiskConfig()
    feature_columns = ensure_columns_exist(config.feature_columns, dataset.columns, "model dataset")
    ensure_columns_exist((target_col, "date"), dataset.columns, "model dataset")

    labeled = dataset.dropna(subset=[target_col]).copy()
    labeled[target_col] = labeled[target_col].astype(int)
    if labeled[target_col].nunique() < 2:
        empty_confusion = pd.DataFrame(
            columns=["split", "threshold", "threshold_value", "n_rows", "tp", "fp", "tn", "fn"]
        )
        empty_calibration = pd.DataFrame(
            columns=[
                "split",
                "bin",
                "probability_min",
                "probability_max",
                "n_rows",
                "mean_predicted_probability",
                "observed_crash_rate",
            ]
        )
        return empty_confusion, empty_calibration

    splits = chronological_split(
        labeled,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    model = _make_pipeline(config)
    model.fit(splits.train[list(feature_columns)], splits.train[target_col])

    test = splits.test.copy()
    if test.empty or test[target_col].nunique(dropna=True) < 2:
        return pd.DataFrame(), pd.DataFrame()

    y_true = test[target_col].astype(int).to_numpy()
    y_prob = _positive_probabilities(model, test, list(feature_columns))

    confusion_rows = [
        _confusion_row("test", "probability_0_50", 0.5, y_true, (y_prob >= 0.5).astype(int)),
        _confusion_row(
            "test",
            "top_20_percent",
            config.target_top_quantile,
            y_true,
            _top_share_predictions(y_prob, config.target_top_quantile),
        ),
    ]
    calibration = _calibration_rows("test", y_true, y_prob, n_bins=n_calibration_bins)
    return pd.DataFrame(confusion_rows), pd.DataFrame(calibration)


def _top_share_predictions(y_prob: np.ndarray, share: float) -> np.ndarray:
    y_pred = np.zeros(len(y_prob), dtype=int)
    if len(y_prob) == 0:
        return y_pred
    top_k = max(1, ceil(len(y_prob) * share))
    top_indices = np.argsort(y_prob)[::-1][:top_k]
    y_pred[top_indices] = 1
    return y_pred


def _confusion_row(
    split_name: str,
    threshold_name: str,
    threshold_value: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int | str]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    false_positive_rate = fp / (fp + tn) if (fp + tn) else np.nan
    false_negative_rate = fn / (fn + tp) if (fn + tp) else np.nan
    accuracy = (tp + tn) / len(y_true) if len(y_true) else np.nan
    return {
        "split": split_name,
        "threshold": threshold_name,
        "threshold_value": float(threshold_value),
        "n_rows": int(len(y_true)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision) if pd.notna(precision) else np.nan,
        "recall": float(recall) if pd.notna(recall) else np.nan,
        "false_positive_rate": float(false_positive_rate) if pd.notna(false_positive_rate) else np.nan,
        "false_negative_rate": float(false_negative_rate) if pd.notna(false_negative_rate) else np.nan,
        "accuracy": float(accuracy) if pd.notna(accuracy) else np.nan,
    }


def _calibration_rows(split_name: str, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> list[dict]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        rows.append(
            {
                "split": split_name,
                "bin": idx + 1,
                "probability_min": float(left),
                "probability_max": float(right),
                "n_rows": int(mask.sum()),
                "mean_predicted_probability": float(np.mean(y_prob[mask])) if mask.any() else np.nan,
                "observed_crash_rate": float(np.mean(y_true[mask])) if mask.any() else np.nan,
            }
        )
    return rows
