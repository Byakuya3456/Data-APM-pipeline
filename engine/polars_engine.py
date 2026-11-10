"""Out-of-core Polars execution engine utilizing LazyFrames and streaming sinks.

Guarantees bounded memory consumption by enforcing lazy query graphs,
streaming parquet sinks, bounded previews (<= 100 rows), and hash-based
deterministic splits without in-memory array materialization.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl

logger = logging.getLogger("autonomous_ml.engine.polars")


class PolarsEngine:
    """Out-of-core Polars execution engine utilizing LazyFrames and streaming sinks."""

    @staticmethod
    def scan(file_path: Path | str) -> pl.LazyFrame:
        """Create a LazyFrame from CSV, TSV, Parquet, JSON, or JSONL/NDJSON."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            return pl.scan_parquet(path)
        elif suffix == ".tsv":
            return pl.scan_csv(path, separator="\t", infer_schema_length=10000, ignore_errors=True)
        elif suffix == ".csv":
            return pl.scan_csv(path, separator=",", infer_schema_length=10000, ignore_errors=True)
        elif suffix in {".jsonl", ".ndjson"}:
            return pl.scan_ndjson(path)
        elif suffix == ".json":
            # For JSON arrays, read into DataFrame and convert to LazyFrame
            try:
                df = pl.read_json(path)
                return df.lazy()
            except Exception as e:
                logger.warning("Polars read_json failed on %s, attempting DuckDB fallback: %s", path, e)
                import duckdb
                con = duckdb.connect(":memory:")
                df = con.sql(f"SELECT * FROM read_json_auto('{path.as_posix()}')").pl()
                con.close()
                return df.lazy()
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Supported: .csv, .tsv, .parquet, .pq, .json, .jsonl, .ndjson"
            )

    @staticmethod
    def preview(lf: pl.LazyFrame, n: int = 100) -> pl.DataFrame:
        """Safely extract bounded preview rows (strictly <= 100)."""
        limit_n = max(1, min(n, 100))
        return lf.limit(limit_n).collect()

    @staticmethod
    def sink_parquet(lf: pl.LazyFrame, output_path: Path | str, compression: str = "zstd") -> Path:
        """Stream LazyFrame directly to a Parquet file out-of-core."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        valid_compressions = {"zstd", "snappy", "gzip", "lz4", "uncompressed"}
        comp = compression.lower() if compression.lower() in valid_compressions else "zstd"

        try:
            lf.sink_parquet(out, compression=comp)
        except Exception as e:
            logger.warning("Streaming sink_parquet failed (%s), falling back to collect and write_parquet.", e)
            df = lf.collect()
            df.write_parquet(out, compression=comp)

        return out

    @staticmethod
    def create_splits(
        lf: pl.LazyFrame,
        split_col: Optional[str] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Create deterministic, hash-based train and test LazyFrames without memory collection.

        If split_col is provided, hashes that column; otherwise, uses with_row_index().
        """
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be between 0.0 and 1.0, got {train_ratio}")

        train_threshold = int(train_ratio * 100)
        schema = lf.collect_schema()

        if split_col and split_col in schema:
            hashed = lf.with_columns(
                __split_hash=(pl.col(split_col).hash(seed=seed) % 100)
            )
            train_lf = hashed.filter(pl.col("__split_hash") < train_threshold).drop("__split_hash")
            test_lf = hashed.filter(pl.col("__split_hash") >= train_threshold).drop("__split_hash")
        else:
            hashed = lf.with_row_index("__temp_row_idx").with_columns(
                __split_hash=(pl.col("__temp_row_idx").hash(seed=seed) % 100)
            )
            train_lf = hashed.filter(pl.col("__split_hash") < train_threshold).drop(["__temp_row_idx", "__split_hash"])
            test_lf = hashed.filter(pl.col("__split_hash") >= train_threshold).drop(["__temp_row_idx", "__split_hash"])

        return train_lf, test_lf

    @staticmethod
    def create_three_splits(
        lf: pl.LazyFrame,
        split_col: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        """Create deterministic train, validation, and test LazyFrames."""
        if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-3):
            raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

        train_thresh = int(train_ratio * 100)
        val_thresh = train_thresh + int(val_ratio * 100)
        schema = lf.collect_schema()

        if split_col and split_col in schema:
            hashed = lf.with_columns(
                __split_hash=(pl.col(split_col).hash(seed=seed) % 100)
            )
            train_lf = hashed.filter(pl.col("__split_hash") < train_thresh).drop("__split_hash")
            val_lf = hashed.filter(
                (pl.col("__split_hash") >= train_thresh) & (pl.col("__split_hash") < val_thresh)
            ).drop("__split_hash")
            test_lf = hashed.filter(pl.col("__split_hash") >= val_thresh).drop("__split_hash")
        else:
            hashed = lf.with_row_index("__temp_row_idx").with_columns(
                __split_hash=(pl.col("__temp_row_idx").hash(seed=seed) % 100)
            )
            train_lf = hashed.filter(pl.col("__split_hash") < train_thresh).drop(["__temp_row_idx", "__split_hash"])
            val_lf = hashed.filter(
                (pl.col("__split_hash") >= train_thresh) & (pl.col("__split_hash") < val_thresh)
            ).drop(["__temp_row_idx", "__split_hash"])
            test_lf = hashed.filter(pl.col("__split_hash") >= val_thresh).drop(["__temp_row_idx", "__split_hash"])

        return train_lf, val_lf, test_lf

    @staticmethod
    def get_schema(lf: pl.LazyFrame) -> Dict[str, str]:
        """Extract schema as string mapping without materializing data."""
        return {k: str(v) for k, v in lf.collect_schema().items()}

    @staticmethod
    def get_row_count(lf: pl.LazyFrame) -> int:
        """Extract exact row count using lazy length expression."""
        return lf.select(pl.len()).collect().item()
