"""
Run this ONCE on your real CSV before deploying, to shrink it under
GitHub's 100MB file size limit.

Usage (from the flight-backend folder):

    python convert_to_parquet.py "C:\\FlightDashboard\\results (1)\\model_predictions_for_powerbi.csv"

This creates model_predictions.parquet in the same folder. Point
FLIGHT_DATA_PATH (in config.py or as an environment variable) at that
new .parquet file instead of the original .csv.
"""
import sys
import os
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_parquet.py <path-to-csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    out_path = os.path.join(os.path.dirname(csv_path), "model_predictions.parquet")

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    print(f"Writing {out_path} ...")
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")

    csv_mb = os.path.getsize(csv_path) / (1024 * 1024)
    parquet_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nDone. {csv_mb:.1f} MB -> {parquet_mb:.1f} MB ({csv_mb/parquet_mb:.1f}x smaller)")
    if parquet_mb > 100:
        print("WARNING: still over 100MB. GitHub will reject a normal push of this file.")
        print("You'll need Git LFS, or host the data file separately (e.g. cloud storage).")
    else:
        print("Under GitHub's 100MB limit, safe to commit and push.")


if __name__ == "__main__":
    main()
