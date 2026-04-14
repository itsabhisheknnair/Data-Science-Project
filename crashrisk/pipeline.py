from __future__ import annotations

from pathlib import Path

import pandas as pd

from crashrisk.analysis.business import business_analysis_to_dataframe, compute_business_analysis
from crashrisk.config import CrashRiskConfig, RawDataPaths, discover_raw_paths
from crashrisk.features.pipeline import build_feature_panel
from crashrisk.models.compare import compare_algorithms, compare_esg_lift
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
    10. Write all outputs to disk.

    Parameters
    ----------
    raw_paths:     Explicit file paths (overrides raw_dir when provided).
    raw_dir:       Directory to discover raw CSV/Excel inputs.
    processed_dir: Output directory for intermediate Parquet files.
    outputs_dir:   Output directory for CSV files consumed by the frontend.
    config:        CrashRiskConfig (defaults applied if None).
    tune:          If True, run GridSearchCV hyperparameter tuning on the
                   primary model and the algorithm comparison.

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
    dataset = make_targets(
        feature_panel,
        horizon_weeks=config.target_horizon_weeks,
        top_quantile=config.target_top_quantile,
    )

    # ── Primary model ──────────────────────────────────────────────────────
    model = train_classifier(dataset, config=config, tune=tune)

    # ── ESG lift comparison ────────────────────────────────────────────────
    model_comparison = compare_esg_lift(dataset, config=config)

    # ── Algorithm comparison (LR vs RF vs GB) ─────────────────────────────
    algorithm_comparison = compare_algorithms(dataset, config=config, tune=tune)

    # ── Scoring & scenarios ────────────────────────────────────────────────
    scores = score_latest(model, feature_panel)
    price_history = make_price_history(feature_panel)
    price_scenarios = make_price_scenarios(
        feature_panel,
        scores,
        horizon_weeks=config.target_horizon_weeks,
    )

    # ── Business analysis ──────────────────────────────────────────────────
    business_analysis = compute_business_analysis(price_history, scores)
    business_analysis_df = business_analysis_to_dataframe(business_analysis)

    # ── Feature importance ─────────────────────────────────────────────────
    feature_importance_df = pd.DataFrame(
        [
            {"feature": feature, "importance": importance}
            for feature, importance in model.feature_importance_.items()
        ]
    )

    # ── Write outputs ──────────────────────────────────────────────────────
    feature_panel.to_parquet(processed_dir / "feature_panel.parquet", index=False)
    dataset.to_parquet(processed_dir / "model_dataset.parquet", index=False)
    scores.to_csv(outputs_dir / "stock_scores.csv", index=False)
    price_history.to_csv(outputs_dir / "price_history.csv", index=False)
    price_scenarios.to_csv(outputs_dir / "price_scenarios.csv", index=False)
    model_comparison.to_csv(outputs_dir / "esg_model_comparison.csv", index=False)
    algorithm_comparison.to_csv(outputs_dir / "algorithm_comparison.csv", index=False)
    feature_importance_df.to_csv(outputs_dir / "feature_importance.csv", index=False)
    business_analysis_df.to_csv(outputs_dir / "business_analysis.csv", index=False)

    return {
        "feature_panel": feature_panel,
        "model_dataset": dataset,
        "model": model,
        "model_comparison": model_comparison,
        "algorithm_comparison": algorithm_comparison,
        "scores": scores,
        "price_history": price_history,
        "price_scenarios": price_scenarios,
        "feature_importance": feature_importance_df,
        "business_analysis": business_analysis,
        "business_analysis_df": business_analysis_df,
    }
