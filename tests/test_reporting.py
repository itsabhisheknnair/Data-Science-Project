from __future__ import annotations

import pandas as pd

from crashrisk.analysis.reporting import build_text_ticker_summary, build_textual_analysis
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
    assert (outputs_dir / "feature_descriptive_stats.csv").exists()
    assert (outputs_dir / "feature_correlation_matrix.csv").exists()
    assert (outputs_dir / "sql_summary.csv").exists()
    assert (outputs_dir / "sql_summary.md").exists()
    assert (outputs_dir / "textual_analysis.csv").exists()
    assert (outputs_dir / "textual_ticker_summary.csv").exists()
    assert (outputs_dir / "text_coverage.csv").exists()
    assert (outputs_dir / "lda_topic_words.csv").exists()
    assert (outputs_dir / "lda_ticker_topics.csv").exists()
    assert (outputs_dir / "fds_report_outline.md").exists()
    assert (outputs_dir / "figures" / "risk_probability_ranking.svg").exists()
    assert (outputs_dir / "figures" / "feature_importance.svg").exists()
    assert (outputs_dir / "figures" / "text_word_cloud.svg").exists()
    assert (outputs_dir / "figures" / "feature_correlation_heatmap.png").exists()
    assert (outputs_dir / "figures" / "price_time_series.png").exists()
    assert {"section", "metric", "value", "detail"}.issubset(result["data_summary"].columns)
    assert {"dataset", "check", "value", "denominator", "percent", "detail"}.issubset(
        result["cleaning_log"].columns
    )
    assert {"query_name", "query", "result_json"}.issubset(result["sql_summary"].columns)
    assert set(small_config.feature_columns).issubset(set(result["feature_descriptive_stats"]["feature"]))
    assert "zero_or_negative_adj_close" in set(result["cleaning_log"]["check"])
    assert "retained_missing_leverage" in set(result["cleaning_log"]["check"])
    assert result["textual_analysis"].loc[0, "status"] == "no_text_file"
    assert result["textual_ticker_summary"].loc[0, "status"] == "no_text_file"


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
    assert {
        "ticker",
        "date",
        "text_sentiment_score",
        "rolling_sentiment_13w",
        "negative_esg_controversy_score_0_100",
    }.issubset(analysis.columns)
    ccc = analysis.loc[analysis["ticker"] == "CCC"].iloc[0]
    aaa = analysis.loc[analysis["ticker"] == "AAA"].iloc[0]
    assert ccc["negative_word_count"] > aaa["negative_word_count"]
    assert ccc["controversy_keyword_count"] > 0
    assert ccc["negative_esg_controversy_score_0_100"] > aaa["negative_esg_controversy_score_0_100"]


def test_text_ticker_summary_supports_generic_xlsx_columns(synthetic_raw_paths, small_config):
    raw_dir = synthetic_raw_paths.controversies.parent
    pd.DataFrame(
        [
            {
                "Symbol": "CCC",
                "Published Date": "2020-02-07",
                "Provider": "Bloomberg",
                "Story": "CCC faces fraud investigation and pollution lawsuit.",
            },
            {
                "Symbol": "AAA",
                "Published Date": "2020-02-07",
                "Provider": "AltData",
                "Story": "AAA reports clean energy progress and improved governance controls.",
            },
        ]
    ).to_excel(raw_dir / "news_text.xlsx", index=False)

    summary = build_text_ticker_summary(synthetic_raw_paths, config=small_config)

    assert set(summary["status"]) == {"ok"}
    assert {"ticker", "latest_text_date", "negative_esg_controversy_score_0_100", "score_band"}.issubset(summary.columns)
    assert summary.iloc[0]["ticker"] == "CCC"


def test_run_mvp_with_optional_text_writes_coverage_lda_and_text_model(
    workspace_tmp_path,
    synthetic_raw_paths,
    small_config,
):
    raw_dir = synthetic_raw_paths.controversies.parent
    pd.DataFrame(
        [
            {
                "ticker": "CCC",
                "date": "2020-05-01",
                "headline": "CCC fraud investigation and pollution lawsuit expands",
                "description": "Regulators cite misconduct, violation risk, and board oversight failures.",
            },
            {
                "ticker": "AAA",
                "date": "2020-05-01",
                "headline": "AAA reports improved governance controls",
                "description": "The company highlights clean energy progress and safe operations.",
            },
            {
                "ticker": "BBB",
                "date": "2020-08-07",
                "headline": "BBB faces labor controversy review",
                "description": "Analysts flag allegation risk and remediation uncertainty.",
            },
        ]
    ).to_csv(raw_dir / "controversy_text.csv", index=False)

    result = run_mvp(
        raw_paths=synthetic_raw_paths,
        processed_dir=workspace_tmp_path / "processed-text",
        outputs_dir=workspace_tmp_path / "outputs-text",
        config=small_config,
    )

    assert {"negative_esg_controversy_score_0_100", "rolling_sentiment_13w"}.issubset(
        result["feature_panel"].columns
    )
    assert result["text_coverage"].loc[result["text_coverage"]["split"].eq("overall"), "article_count"].iloc[0] == 3
    assert {"topic", "word", "weight"}.issubset(result["lda_topic_words"].columns)
    assert result["lda_topic_words"]["status"].eq("ok").any()
    assert {"ticker", "dominant_topic", "topic_probability"}.issubset(result["lda_ticker_topics"].columns)
    assert {"full_with_esg", "full_with_esg_plus_text", "text_minus_full_esg"}.issubset(
        set(result["text_model_comparison"]["model"])
    )
