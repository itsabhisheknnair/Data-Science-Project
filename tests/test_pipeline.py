from __future__ import annotations

import pandas as pd

from crashrisk.pipeline import run_mvp


def test_run_mvp_writes_requested_outputs(workspace_tmp_path, synthetic_raw_paths, small_config):
    processed_dir = workspace_tmp_path / "processed"
    outputs_dir = workspace_tmp_path / "outputs"

    result = run_mvp(
        raw_paths=synthetic_raw_paths,
        processed_dir=processed_dir,
        outputs_dir=outputs_dir,
        config=small_config,
    )

    assert (processed_dir / "feature_panel.parquet").exists()
    assert (processed_dir / "model_dataset.parquet").exists()
    assert (outputs_dir / "stock_scores.csv").exists()
    assert (outputs_dir / "price_history.csv").exists()
    assert (outputs_dir / "price_scenarios.csv").exists()
    assert (outputs_dir / "esg_model_comparison.csv").exists()
    assert (outputs_dir / "text_model_comparison.csv").exists()
    assert (outputs_dir / "business_portfolio_returns.csv").exists()
    assert (outputs_dir / "data_summary.csv").exists()
    assert (outputs_dir / "feature_descriptive_stats.csv").exists()
    assert (outputs_dir / "feature_correlation_matrix.csv").exists()
    assert (outputs_dir / "sql_summary.md").exists()
    assert (outputs_dir / "textual_analysis.csv").exists()
    assert (outputs_dir / "textual_ticker_summary.csv").exists()
    assert (outputs_dir / "text_coverage.csv").exists()
    assert (outputs_dir / "lda_topic_words.csv").exists()
    assert (outputs_dir / "lda_ticker_topics.csv").exists()
    assert (outputs_dir / "hyperparameter_tuning_results.csv").exists()
    assert (outputs_dir / "confusion_matrix.csv").exists()
    assert (outputs_dir / "calibration_curve.csv").exists()
    assert (outputs_dir / "figures" / "feature_correlation_heatmap.png").exists()
    assert (outputs_dir / "figures" / "price_time_series.png").exists()
    assert (outputs_dir / "figures" / "probability_calibration.png").exists()
    assert {"ticker", "as_of_date", "crash_probability", "risk_bucket", "top_drivers"}.issubset(
        result["scores"].columns
    )
    assert {"ticker", "date", "adj_close"}.issubset(result["price_history"].columns)
    assert {"ticker", "price_p05", "price_p50", "price_p95"}.issubset(result["price_scenarios"].columns)
    assert {"model", "split", "roc_auc", "precision_at_top_bucket"}.issubset(result["model_comparison"].columns)
    assert {"model", "split", "roc_auc", "text_covered_rows"}.issubset(result["text_model_comparison"].columns)
    assert {"section", "metric", "value", "detail"}.issubset(result["data_summary"].columns)
    assert {"feature", "mean", "std", "min", "max", "null_percent"}.issubset(
        result["feature_descriptive_stats"].columns
    )
    assert {"threshold", "tp", "fp", "tn", "fn"}.issubset(result["confusion_matrix"].columns)
    assert {"bin", "mean_predicted_probability", "observed_crash_rate"}.issubset(
        result["calibration_curve"].columns
    )
    assert {"date", "strategy", "benchmark", "n_holdings", "n_excluded", "excluded_tickers"}.issubset(
        result["business_portfolio_returns"].columns
    )
    written_text_summary = pd.read_csv(outputs_dir / "textual_ticker_summary.csv")
    assert {"status"}.issubset(written_text_summary.columns)
    if written_text_summary.loc[0, "status"] == "ok":
        assert "negative_esg_controversy_score_0_100" in written_text_summary.columns

    written_scores = pd.read_csv(outputs_dir / "stock_scores.csv")
    written_scenarios = pd.read_csv(outputs_dir / "price_scenarios.csv")
    written_comparison = pd.read_csv(outputs_dir / "esg_model_comparison.csv")
    written_portfolio_returns = pd.read_csv(outputs_dir / "business_portfolio_returns.csv")
    assert len(written_scores) == len(result["scores"])
    assert len(written_scenarios) == len(result["price_scenarios"])
    assert len(written_comparison) == len(result["model_comparison"])
    assert len(written_portfolio_returns) == len(result["business_portfolio_returns"])
