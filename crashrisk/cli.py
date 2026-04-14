from __future__ import annotations

import argparse

from crashrisk.pipeline import run_mvp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lean crash-risk backend MVP.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing raw CSV/XLSX exports.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory for parquet outputs.")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory for scoring outputs.")
    args = parser.parse_args()

    result = run_mvp(raw_dir=args.raw_dir, processed_dir=args.processed_dir, outputs_dir=args.outputs_dir)
    scores = result["scores"]
    print(f"Wrote {len(scores)} scores to {args.outputs_dir}/stock_scores.csv")


if __name__ == "__main__":
    main()

