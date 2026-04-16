from __future__ import annotations

from pathlib import Path

import pandas as pd

from crashrisk.analysis.business import (
    build_weekly_forward_portfolio_returns,
    business_analysis_to_dataframe,
    compute_business_analysis,
    quarter_snapshot_backtest,
)
from crashrisk.analysis.reporting import build_report_artifacts, build_text_analysis_outputs, join_text_signals_to_panel
from crashrisk.config import CrashRiskConfig, RawDataPaths, discover_raw_paths
from crashrisk.features.pipeline import build_feature_panel
from crashrisk.models.compare import (
    build_hyperparameter_tuning_results,
    build_test_diagnostics,
    compare_algorithms,
    compare_esg_lift,
    compare_text_signal_lift,
)
from crashrisk.models.scenarios import make_price_history, make_price_scenarios
from crashrisk.models.score import score_latest
from crashrisk.models.train import train_classifier
from crashrisk.targets import make_targets


def run_mvp(
    raw_paths: RawDataPaths | dict[str, str | Path] | None = None,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    outputs_dir: str | Path = "outputs",
    config: CrashRiskConfig | None = None,
    tune: bool = False,
) -> dict[str, pd.DataFrame | dict]:
    """
    End-to-end crash-risk pipeline.

    Steps
    -----
    1. Load and validate raw data.
    2. Build the weekly feature panel.
    3. Label future crash-risk targets (NCSKEW-based).
    4. Train the primary classifier (with optional hyperparameter tuning).
    5. Compare ESG controversy lift (baseline vs. full model).
    6. Compare algorithm types (LR vs. RF vs. GB).
    7. Score the latest available week.
    8. Build price history and scenario outputs.
    9. Compute portfolio-level business analysis.
    10. Build report-ready data summary, SQL, text-analysis, and figure artifacts.
    11. Write all outputs to disk.

    Parameters
    ----------
    raw_paths:     Explicit file paths (overrides raw_dir when provided).
    raw_dir:       Directory to discover raw CSV/Excel inputs.
    processed_dir: Output directory for intermediate Parquet files.
    outputs_dir:   Output directory for CSV files consumed by the frontend.
    config:        CrashRiskConfig (defaults applied if None).
    tune:          If True, run GridSearchCV hyperparameter tuning on the
                   primary model and write the tuning-results table.

    Returns
    -------
    Dict containing all computed DataFrames and the business analysis dict.
    """
    config = config or CrashRiskConfig()
    paths = RawDataPaths.from_mapping(raw_paths) if raw_paths is not None else discover_raw_paths(raw_dir)
    processed_dir = Path(processed_dir)
    outputs_dir = Path(outputs_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Feature engineering ────────────────────────────────────────────────
    feature_panel = build_feature_panel(paths, config=config)
    text_outputs = build_text_analysis_outputs(paths, config=config)
    feature_panel = join_text_signals_to_panel(feature_panel, text_outputs["weekly"])
    dataset = make_targets(
        feature_panel,
        horizon_weeks=config.target_horizon_weeks,
        top_quantile=config.target_top_quantile,
    )

    # ── Primary model ──────────────────────────────────────────────────────
    model = train_classifier(dataset, config=config, tune=tune)

    # ── ESG lift comparison ────────────────────────────────────────────────
    model_comparison = compare_esg_lift(dataset, config=config)
    text_model_comparison = compare_text_signal_lift(dataset, config=config)

    # ── Algorithm comparison (LR vs RF vs GB) ─────────────────────────────
    algorithm_comparison = compare_algorithms(dataset, config=config, tune=False)
    hyperparameter_tuning_results = build_hyperparameter_tuning_results(
        dataset,
        config=config,
        run_search=tune,
    )
    confusion_matrix, calibration_curve = build_test_diagnostics(dataset, config=config)

    # ── Scoring & scenarios ────────────────────────────────────────────────
    scores = score_latest(model, feature_panel)
    price_history = make_price_history(feature_panel)
    price_scenarios = make_price_scenarios(
        feature_panel,
        scores,
        horizon_weeks=config.target_horizon_weeks,
    )

    # ── Business analysis ──────────────────────────────────────────────────
    business_portfolio_returns = build_weekly_forward_portfolio_returns(feature_panel, model)
    business_analysis = compute_business_analysis(
        feature_panel,
        model=model,
        portfolio_returns=business_portfolio_returns,
    )
    business_analysis_df = business_analysis_to_dataframe(business_analysis)

    # ── Quarter snapshot backtest (out-of-sample pitch) ────────────────────
    quarter_bt = quarter_snapshot_backtest(panel=feature_panel, model=model)

    # ── Feature importance ─────────────────────────────────────────────────
    feature_importance_df = pd.DataFrame(
        [
            {"feature": feature, "importance": importance}
            for feature, importance in model.feature_importance_.items()
        ]
    )

    report_artifacts = build_report_artifacts(
        raw_paths=paths,
        feature_panel=feature_panel,
        dataset=dataset,
        scores=scores,
        price_history=price_history,
        price_scenarios=price_scenarios,
        feature_importance=feature_importance_df,
        model_comparison=model_comparison,
        outputs_dir=outputs_dir,
        config=config,
        text_outputs=text_outputs,
        calibration_curve=calibration_curve,
    )

    # ── Write outputs ──────────────────────────────────────────────────────
    feature_panel.to_parquet(processed_dir / "feature_panel.parquet", index=False)
    dataset.to_parquet(processed_dir / "model_dataset.parquet", index=False)
    scores.to_csv(outputs_dir / "stock_scores.csv", index=False)
    price_history.to_csv(outputs_dir / "price_history.csv", index=False)
    price_scenarios.to_csv(outputs_dir / "price_scenarios.csv", index=False)
    model_comparison.to_csv(outputs_dir / "esg_model_comparison.csv", index=False)
    text_model_comparison.to_csv(outputs_dir / "text_model_comparison.csv", index=False)
    algorithm_comparison.to_csv(outputs_dir / "algorithm_comparison.csv", index=False)
    hyperparameter_tuning_results.to_csv(outputs_dir / "hyperparameter_tuning_results.csv", index=False)
    confusion_matrix.to_csv(outputs_dir / "confusion_matrix.csv", index=False)
    calibration_curve.to_csv(outputs_dir / "calibration_curve.csv", index=False)
    feature_importance_df.to_csv(outputs_dir / "feature_importance.csv", index=False)
    business_analysis_df.to_csv(outputs_dir / "business_analysis.csv", index=False)
    business_portfolio_returns.to_csv(outputs_dir / "business_portfolio_returns.csv", index=False)
    if "error" not in quarter_bt:
        pd.DataFrame(quarter_bt["weekly_series"]).to_csv(
            outputs_dir / "quarter_backtest_returns.csv", index=False
        )
        pd.DataFrame(quarter_bt["excluded_tickers"]).to_csv(
            outputs_dir / "quarter_excluded_stocks.csv", index=False
        )
    report_artifacts["data_summary"].to_csv(outputs_dir / "data_summary.csv", index=False)
    report_artifacts["cleaning_log"].to_csv(outputs_dir / "cleaning_log.csv", index=False)
    report_artifacts["feature_descriptive_stats"].to_csv(outputs_dir / "feature_descriptive_stats.csv", index=False)
    report_artifacts["feature_correlation_matrix"].to_csv(outputs_dir / "feature_correlation_matrix.csv", index=False)
    report_artifacts["sql_summary"].to_csv(outputs_dir / "sql_summary.csv", index=False)
    report_artifacts["textual_analysis"].to_csv(outputs_dir / "textual_analysis.csv", index=False)
    report_artifacts["textual_ticker_summary"].to_csv(outputs_dir / "textual_ticker_summary.csv", index=False)
    report_artifacts["text_coverage"].to_csv(outputs_dir / "text_coverage.csv", index=False)
    report_artifacts["lda_topic_words"].to_csv(outputs_dir / "lda_topic_words.csv", index=False)
    report_artifacts["lda_ticker_topics"].to_csv(outputs_dir / "lda_ticker_topics.csv", index=False)

    return {
        "feature_panel": feature_panel,
        "model_dataset": dataset,
        "model": model,
        "model_comparison": model_comparison,
        "text_model_comparison": text_model_comparison,
        "algorithm_comparison": algorithm_comparison,
        "hyperparameter_tuning_results": hyperparameter_tuning_results,
        "confusion_matrix": confusion_matrix,
        "calibration_curve": calibration_curve,
        "scores": scores,
        "price_history": price_history,
        "price_scenarios": price_scenarios,
        "feature_importance": feature_importance_df,
        "business_analysis": business_analysis,
        "business_analysis_df": business_analysis_df,
        "business_portfolio_returns": business_portfolio_returns,
        "quarter_backtest": quarter_bt,
        **report_artifacts,
    }
