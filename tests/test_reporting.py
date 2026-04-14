from __future__ import annotations

import pandas as pd

from crashrisk.analysis.reporting import build_textual_analysis
from crashrisk.pipeline import run_mvp


def test_run_mvp_writes_report_artifacts(workspace_tmp_path, synthetic_raw_paths, small_config):
    processed_dir = workspace_tmp_path / "processed"
    outputs_dir = workspace_tmp_path / "outputs"

    result = run_mvp(
        raw_paths=synthetic_raw_paths,
        processed_dir=processed_dir,
        outputs_dir=outputs_dir,
        config=small_config,
    )

    assert (outputs_dir / "data_summary.csv").exists()
    assert (outputs_dir / "cleaning_log.csv").exists()
    assert (outputs_dir / "sql_summary.csv").exists()
    assert (outputs_dir / "sql_summary.md").exists()
    assert (outputs_dir / "textual_analysis.csv").exists()
    assert (outputs_dir / "fds_report_outline.md").exists()
    assert (outputs_dir / "figures" / "risk_probability_ranking.svg").exists()
    assert (outputs_dir / "figures" / "feature_importance.svg").exists()
    assert (outputs_dir / "figures" / "text_word_cloud.svg").exists()
    assert {"section", "metric", "value", "detail"}.issubset(result["data_summary"].columns)
    assert {"dataset", "check", "value", "detail"}.issubset(result["cleaning_log"].columns)
    assert {"query_name", "query", "result_json"}.issubset(result["sql_summary"].columns)
    assert result["textual_analysis"].loc[0, "status"] == "no_text_file"


def test_textual_analysis_scores_optional_news_text(synthetic_raw_paths, small_config):
    raw_dir = synthetic_raw_paths.controversies.parent
    pd.DataFrame(
        [
            {
                "ticker": "CCC",
                "date": "2020-02-07",
                "source": "Bloomberg",
                "headline": "CCC faces pollution investigation and lawsuit",
                "description": "The company said it will resolve the emissions controversy.",
            },
            {
                "ticker": "AAA",
                "date": "2020-02-07",
                "source": "Bloomberg",
                "headline": "AAA reports improved safety progress",
                "description": "The firm said governance controls improved.",
            },
        ]
    ).to_csv(raw_dir / "news_text.csv", index=False)

    analysis = build_textual_analysis(synthetic_raw_paths, config=small_config)

    assert set(analysis["status"]) == {"ok"}
    assert {"ticker", "date", "text_sentiment_score", "rolling_sentiment_13w"}.issubset(analysis.columns)
    ccc = analysis.loc[analysis["ticker"] == "CCC"].iloc[0]
    aaa = analysis.loc[analysis["ticker"] == "AAA"].iloc[0]
    assert ccc["negative_word_count"] > aaa["negative_word_count"]
    assert ccc["controversy_keyword_count"] > 0
