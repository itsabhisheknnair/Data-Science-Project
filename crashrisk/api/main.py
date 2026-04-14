from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from crashrisk.pipeline import run_mvp


REQUIRED_UPLOADS = ("prices", "benchmark_prices", "fundamentals", "controversies")
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _cors_origins() -> list[str]:
    raw = os.getenv("CRASHRISK_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _temporary_parent() -> Path:
    raw = os.getenv("CRASHRISK_API_TMP_DIR", "").strip()
    parent = Path(raw) if raw else Path("data/processed/api-tmp")
    parent.mkdir(parents=True, exist_ok=True)
    return parent


@contextmanager
def _work_dir():
    parent = _temporary_parent()
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = parent / f"crashrisk-api-{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


app = FastAPI(
    title="ESG Crash-Risk API",
    version="0.1.0",
    description="Upload Bloomberg-style raw files and receive ESG controversy crash-risk scores.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)
    return clean.to_dict(orient="records")


def _write_upload(upload: UploadFile, destination: Path) -> Path:
    suffix = Path(upload.filename or "").suffix.lower() or ".csv"
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"{upload.filename!r} has unsupported extension {suffix!r}. Use one of: {allowed}.",
        )

    target = destination.with_suffix(suffix)
    target.write_bytes(upload.file.read())
    if target.stat().st_size == 0:
        raise HTTPException(status_code=400, detail=f"{upload.filename!r} is empty.")
    return target


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "ESG Crash-Risk API",
        "health": "/health",
        "schema": "/schema",
        "predict": "/predict",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "inputs": {
            "prices": ["ticker", "date", "adj_close", "volume"],
            "benchmark_prices": ["date", "benchmark_close"],
            "fundamentals": [
                "ticker",
                "period_end",
                "market_cap",
                "shares_outstanding",
                "market_to_book",
                "leverage",
                "roa",
            ],
            "controversies": ["ticker", "date", "sector", "controversy_score"],
        },
        "optional_inputs": {
            "news_text": ["ticker", "date", "headline/title/description/body/text/summary", "source optional"],
            "controversy_text": ["ticker", "date", "headline/title/description/body/text/summary", "source optional"],
        },
        "accepted_extensions": sorted(ALLOWED_EXTENSIONS),
        "response_tables": [
            "scores",
            "price_history",
            "price_scenarios",
            "model_comparison",
            "algorithm_comparison",
            "feature_importance",
            "business_analysis",
            "data_summary",
            "cleaning_log",
            "sql_summary",
            "textual_analysis",
        ],
    }


@app.post("/predict")
def predict(
    prices: UploadFile = File(...),
    benchmark_prices: UploadFile = File(...),
    fundamentals: UploadFile = File(...),
    controversies: UploadFile = File(...),
    news_text: UploadFile | None = File(None),
    controversy_text: UploadFile | None = File(None),
    tune: bool = Form(False),
) -> dict[str, Any]:
    uploads = {
        "prices": prices,
        "benchmark_prices": benchmark_prices,
        "fundamentals": fundamentals,
        "controversies": controversies,
    }

    with _work_dir() as tmp_dir:
        raw_paths = {
            name: _write_upload(upload, tmp_dir / name)
            for name, upload in uploads.items()
        }
        for name, upload in {"news_text": news_text, "controversy_text": controversy_text}.items():
            if upload is not None and upload.filename:
                _write_upload(upload, tmp_dir / name)

        try:
            result = run_mvp(
                raw_paths=raw_paths,
                processed_dir=tmp_dir / "processed",
                outputs_dir=tmp_dir / "outputs",
                tune=tune,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "metadata": {
            "score_count": len(result["scores"]),
            "ticker_count": int(result["scores"]["ticker"].nunique()) if not result["scores"].empty else 0,
            "tune": tune,
        },
        "scores": _records(result["scores"]),
        "price_history": _records(result["price_history"]),
        "price_scenarios": _records(result["price_scenarios"]),
        "model_comparison": _records(result["model_comparison"]),
        "algorithm_comparison": _records(result["algorithm_comparison"]),
        "feature_importance": _records(result["feature_importance"]),
        "business_analysis": _records(result["business_analysis_df"]),
        "data_summary": _records(result["data_summary"]),
        "cleaning_log": _records(result["cleaning_log"]),
        "sql_summary": _records(result["sql_summary"]),
        "textual_analysis": _records(result["textual_analysis"]),
    }
