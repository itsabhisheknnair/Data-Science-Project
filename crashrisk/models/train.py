from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crashrisk.config import CrashRiskConfig, ensure_columns_exist
from crashrisk.models.splits import chronological_split


# Hyperparameter grids for each model type. Keys must match pipeline step names.
_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
    },
    "random_forest": {
        "classifier__n_estimators": [100],
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [5, 10],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [100],
        "classifier__max_depth": [2, 3],
        "classifier__learning_rate": [0.05, 0.10],
    },
}


def _make_classifier(model_type: str, random_state: int):
    if model_type == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    # Default: logistic regression
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )


def _make_pipeline(config: CrashRiskConfig, model_type: str | None = None) -> Pipeline:
    """Build a sklearn Pipeline for the given model type."""
    mt = model_type or getattr(config, "model_type", "logistic_regression")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("classifier", _make_classifier(mt, config.random_state)),
        ]
    )


def _evaluate(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
) -> dict[str, float]:
    if frame.empty or frame[target_col].nunique(dropna=True) < 2:
        return {}
    y_true = frame[target_col].astype(int).to_numpy()
    positive_index = list(model.named_steps["classifier"].classes_).index(1)
    y_prob = model.predict_proba(frame[feature_columns])[:, positive_index]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _run_hyperparameter_search(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    n_splits: int,
) -> tuple[Pipeline, dict[str, object]]:
    """
    Run GridSearchCV with TimeSeriesSplit folds.

    Returns the best estimator (already fitted on X, y) and a summary dict
    containing best_params, best_cv_roc_auc, and cv_roc_auc_std.
    """
    param_grid = _PARAM_GRIDS.get(model_type, {})
    if not param_grid:
        pipeline.fit(X, y.astype(int))
        return pipeline, {}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y.astype(int))

    best_idx = search.best_index_
    cv_summary: dict[str, object] = {
        "best_params": dict(search.best_params_),
        "best_cv_roc_auc": float(search.best_score_),
        "cv_roc_auc_std": float(search.cv_results_["std_test_score"][best_idx]),
        "n_cv_splits": n_splits,
        "n_candidates": int(len(search.cv_results_["mean_test_score"])),
    }
    return search.best_estimator_, cv_summary


def _extract_feature_importance(
    pipeline: Pipeline,
    feature_columns: list[str],
) -> dict[str, float]:
    """Return feature importance sorted descending, supporting LR, RF, and GB."""
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    elif hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    else:
        importances = np.ones(len(feature_columns))
    return dict(
        sorted(
            zip(feature_columns, importances, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def train_classifier(
    dataset: pd.DataFrame,
    config: CrashRiskConfig | None = None,
    target_col: str = "high_crash_risk",
    model_type: str | None = None,
    tune: bool = False,
) -> Pipeline:
    """
    Train a crash-risk classifier with optional hyperparameter tuning.

    Parameters
    ----------
    dataset:     Feature panel with target labels.
    config:      CrashRiskConfig (defaults applied if None).
    target_col:  Binary target column name.
    model_type:  One of "logistic_regression", "random_forest", "gradient_boosting".
                 Overrides config.model_type when provided.
    tune:        If True, run GridSearchCV with TimeSeriesSplit before fitting
                 the final model; best hyperparameters are applied to the final fit.

    Returns
    -------
    Fitted Pipeline with extra attributes:
      .feature_columns_     list[str]
      .model_type_          str
      .metrics_             dict with "validation", "test", and optionally
                            "cross_validation" sub-dicts
      .feature_importance_  dict[str, float] sorted by importance
      .trained_rows_        int
      .training_date_range_ tuple[Timestamp, Timestamp]
      .class_balance_       dict[int, int]
    """
    config = config or CrashRiskConfig()
    mt = model_type or getattr(config, "model_type", "logistic_regression")
    feature_columns = ensure_columns_exist(config.feature_columns, dataset.columns, "model dataset")
    ensure_columns_exist((target_col, "date"), dataset.columns, "model dataset")

    labeled = dataset.dropna(subset=[target_col]).copy()
    labeled[target_col] = labeled[target_col].astype(int)
    if labeled[target_col].nunique() < 2:
        raise ValueError("Need at least two target classes to train")

    splits = chronological_split(
        labeled,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    metrics: dict[str, object] = {}
    cv_summary: dict[str, object] = {}

    if splits.train[target_col].nunique() >= 2:
        n_splits = getattr(config, "n_cv_splits", 5)
        eval_pipeline = _make_pipeline(config, mt)

        if tune:
            eval_pipeline, cv_summary = _run_hyperparameter_search(
                eval_pipeline,
                splits.train[feature_columns],
                splits.train[target_col],
                model_type=mt,
                n_splits=n_splits,
            )
        else:
            eval_pipeline.fit(
                splits.train[feature_columns],
                splits.train[target_col].astype(int),
            )

        metrics["validation"] = _evaluate(eval_pipeline, splits.validation, feature_columns, target_col)
        metrics["test"] = _evaluate(eval_pipeline, splits.test, feature_columns, target_col)
        if cv_summary:
            metrics["cross_validation"] = {
                k: v
                for k, v in cv_summary.items()
                if isinstance(v, (int, float))
            }
            metrics["best_params"] = cv_summary.get("best_params", {})

    # Final model trained on all labeled data with (optionally) tuned hyperparameters
    final_model = _make_pipeline(config, mt)
    if tune and cv_summary.get("best_params"):
        final_model.set_params(**cv_summary["best_params"])
    final_model.fit(labeled[feature_columns], labeled[target_col].astype(int))

    final_model.feature_columns_ = feature_columns
    final_model.target_col_ = target_col
    final_model.model_type_ = mt
    final_model.metrics_ = metrics
    final_model.trained_rows_ = int(len(labeled))
    final_model.training_date_range_ = (
        pd.Timestamp(labeled["date"].min()),
        pd.Timestamp(labeled["date"].max()),
    )
    final_model.class_balance_ = {
        int(label): int(count)
        for label, count in labeled[target_col].value_counts().sort_index().items()
    }
    final_model.feature_importance_ = _extract_feature_importance(final_model, feature_columns)
    return final_model
