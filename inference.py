"""Minimal inference baseline. Training agent should save model.joblib and metadata.json."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="workspace/artifacts")
    parser.add_argument("--input", required=True, help="CSV or Parquet input")
    parser.add_argument("--output", default="predictions.parquet")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    path = Path(args.input)
    lf = pl.scan_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pl.scan_csv(path)
    features = metadata["feature_columns"]
    # Prediction batches bound memory; adjust batch_rows for available RAM.
    batch_rows = int(metadata.get("inference_batch_rows", 50000))
    offset, frames = 0, []
    while True:
        batch = lf.slice(offset, batch_rows).collect()
        if batch.height == 0:
            break
        X = batch.select(features).to_pandas()
        preds = model.predict(X)
        frames.append(pl.DataFrame({"prediction": preds}))
        offset += batch_rows
    pl.concat(frames).write_parquet(args.output) if frames else pl.DataFrame(
        {"prediction": []}
    ).write_parquet(args.output)


if __name__ == "__main__":
    main()
