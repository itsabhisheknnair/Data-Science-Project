from __future__ import annotations

import argparse
from pathlib import Path

from crashrisk.demo_data import write_demo_data
from crashrisk.pipeline import run_mvp


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo data and run the crash-risk backend MVP.")
    parser.add_argument("--raw-dir",       default="data/raw",       help="Directory for generated raw CSV files.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory for backend parquet outputs.")
    parser.add_argument("--outputs-dir",   default="outputs",        help="Directory for frontend score CSV output.")
    parser.add_argument("--weeks",   type=int, default=260, help="Number of weekly demo observations.")
    parser.add_argument("--seed",    type=int, default=7,   help="Random seed for repeatable demo data.")
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run GridSearchCV hyperparameter tuning (slower but more thorough).",
    )
    args = parser.parse_args()

    files = write_demo_data(raw_dir=args.raw_dir, weeks=args.weeks, seed=args.seed)
    result = run_mvp(
        raw_paths=files,
        processed_dir=Path(args.processed_dir),
        outputs_dir=Path(args.outputs_dir),
        tune=args.tune,
    )

    scores = result["scores"]
    algo_df = result["algorithm_comparison"]
    biz = result["business_analysis"]
    outputs = Path(args.outputs_dir)

    print(f"\nWrote demo raw files to      : {args.raw_dir}")
    print(f"Wrote {len(scores)} scores       : {outputs / 'stock_scores.csv'}")
    print(f"Wrote price history          : {outputs / 'price_history.csv'}")
    print(f"Wrote price scenarios        : {outputs / 'price_scenarios.csv'}")
    print(f"Wrote ESG lift report        : {outputs / 'esg_model_comparison.csv'}")
    print(f"Wrote algorithm comparison   : {outputs / 'algorithm_comparison.csv'}")
    print(f"Wrote feature importance     : {outputs / 'feature_importance.csv'}")
    print(f"Wrote business analysis      : {outputs / 'business_analysis.csv'}")
    print(f"Wrote data summary           : {outputs / 'data_summary.csv'}")
    print(f"Wrote cleaning log           : {outputs / 'cleaning_log.csv'}")
    print(f"Wrote SQL summary            : {outputs / 'sql_summary.md'}")
    print(f"Wrote textual analysis       : {outputs / 'textual_analysis.csv'}")
    print(f"Wrote report figures         : {outputs / 'figures'}")
    print(f"Wrote report outline         : {outputs / 'fds_report_outline.md'}")

    # ── Quick summary ──────────────────────────────────────────────────────
    print("\n--- Algorithm comparison (test split) ---")
    test_rows = algo_df[algo_df["split"] == "test"] if not algo_df.empty else algo_df
    for _, row in test_rows.iterrows():
        auc = f"{row['roc_auc']:.3f}" if "roc_auc" in row and row["roc_auc"] == row["roc_auc"] else "N/A"
        print(f"  {row['model']:<25}  ROC-AUC={auc}")

    print("\n--- Business analysis ---")
    if "error" not in biz:
        alpha = biz.get("alpha_annualized")
        print(f"  Alpha (annualized)   : {alpha:.2%}" if isinstance(alpha, float) else "  Alpha: N/A")
        print(f"  Strategy Sharpe      : {biz.get('strategy_sharpe', 'N/A')}")
        print(f"  Max drawdown (strat) : {biz.get('max_drawdown_strategy', 'N/A')}")
        gain = biz.get("economic_gain_annual", 0)
        print(f"  Economic gain / yr   : ${gain:,.0f}" if isinstance(gain, (int, float)) else f"  Economic gain / yr   : {gain}")
        print(f"  Justifies team?      : {biz.get('justifies_team', False)}")

    print("\nOpen the dashboard by serving the project folder:")
    print("  python -m http.server 8000")
    print("  http://localhost:8000/frontend/")


if __name__ == "__main__":
    main()
