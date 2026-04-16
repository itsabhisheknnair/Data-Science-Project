from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from crashrisk.api.main import app


def test_render_api_predicts_from_uploaded_raw_files(workspace_tmp_path, synthetic_raw_paths, monkeypatch):
    api_tmp = Path("data") / "processed" / "api-tmp"
    api_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CRASHRISK_API_TMP_DIR", str(api_tmp))
    client = TestClient(app)
    open_files = []
    try:
        files = {}
        for field in ("prices", "benchmark_prices", "fundamentals", "controversies"):
            path = getattr(synthetic_raw_paths, field)
            handle = path.open("rb")
            open_files.append(handle)
            files[field] = (path.name, handle, "text/csv")

        response = client.post("/predict", files=files, data={"tune": "false"})
    finally:
        for handle in open_files:
            handle.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["score_count"] > 0
    assert {
        "scores",
        "price_history",
        "price_scenarios",
        "model_comparison",
        "text_model_comparison",
        "algorithm_comparison",
        "hyperparameter_tuning_results",
        "confusion_matrix",
        "calibration_curve",
        "feature_descriptive_stats",
        "feature_correlation_matrix",
        "text_coverage",
        "lda_topic_words",
        "lda_ticker_topics",
        "data_summary",
    }.issubset(payload)
    assert {"ticker", "crash_probability", "risk_bucket"}.issubset(payload["scores"][0])
    assert {"ticker", "price_p05", "price_p50", "price_p95"}.issubset(payload["price_scenarios"][0])


def test_render_api_accepts_optional_news_text(workspace_tmp_path, synthetic_raw_paths, monkeypatch):
    api_tmp = Path("data") / "processed" / "api-tmp"
    api_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CRASHRISK_API_TMP_DIR", str(api_tmp))
    client = TestClient(app)
    text_path = workspace_tmp_path / "news_text.csv"
    text_path.write_text(
        "ticker,date,headline,description\n"
        "CCC,2020-02-07,CCC faces pollution investigation,The company plans to resolve the controversy.\n",
        encoding="utf-8",
    )

    open_files = []
    try:
        files = {}
        for field in ("prices", "benchmark_prices", "fundamentals", "controversies"):
            path = getattr(synthetic_raw_paths, field)
            handle = path.open("rb")
            open_files.append(handle)
            files[field] = (path.name, handle, "text/csv")
        text_handle = text_path.open("rb")
        open_files.append(text_handle)
        files["news_text"] = (text_path.name, text_handle, "text/csv")

        response = client.post("/predict", files=files, data={"tune": "false"})
    finally:
        for handle in open_files:
            handle.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["textual_analysis"][0]["status"] == "ok"
    assert payload["textual_ticker_summary"][0]["status"] == "ok"
    assert payload["text_coverage"][0]["status"] == "ok"
    assert payload["lda_topic_words"][0]["status"] == "ok"
    assert payload["text_model_comparison"]
    assert "negative_esg_controversy_score_0_100" in payload["textual_ticker_summary"][0]
