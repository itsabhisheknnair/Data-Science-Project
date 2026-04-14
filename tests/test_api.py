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
    assert {"scores", "price_history", "price_scenarios", "model_comparison"}.issubset(payload)
    assert {"ticker", "crash_probability", "risk_bucket"}.issubset(payload["scores"][0])
    assert {"ticker", "price_p05", "price_p50", "price_p95"}.issubset(payload["price_scenarios"][0])
