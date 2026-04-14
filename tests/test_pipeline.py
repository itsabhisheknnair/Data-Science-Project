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
    assert (outputs_dir / "data_summary.csv").exists()
    assert (outputs_dir / "sql_summary.md").exists()
    assert (outputs_dir / "textual_analysis.csv").exists()
    assert {"ticker", "as_of_date", "crash_probability", "risk_bucket", "top_drivers"}.issubset(
        result["scores"].columns
    )
    assert {"ticker", "date", "adj_close"}.issubset(result["price_history"].columns)
    assert {"ticker", "price_p05", "price_p50", "price_p95"}.issubset(result["price_scenarios"].columns)
    assert {"model", "split", "roc_auc", "precision_at_top_bucket"}.issubset(result["model_comparison"].columns)
    assert {"section", "metric", "value", "detail"}.issubset(result["data_summary"].columns)

    written_scores = pd.read_csv(outputs_dir / "stock_scores.csv")
    written_scenarios = pd.read_csv(outputs_dir / "price_scenarios.csv")
    written_comparison = pd.read_csv(outputs_dir / "esg_model_comparison.csv")
    assert len(written_scores) == len(result["scores"])
    assert len(written_scenarios) == len(result["price_scenarios"])
    assert len(written_comparison) == len(result["model_comparison"])
