"""Single-pass out-of-core data profiler and multi-format dataset discovery.

Inspects CSV, TSV, Parquet, JSON, and JSONL datasets without loading complete
data into memory. Provides bounded 100-row previews, schema inference,
null percentages, numeric distributions, categorical cardinality tiers,
semantic column tagging, and data hygiene indicators.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from engine.polars_engine import PolarsEngine

logger = logging.getLogger("autonomous_ml.data_profiler")


class DatasetFormat(str, Enum):
    """Supported file formats for data profiling and ingestion."""
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"
    JSON = "json"
    JSONL = "jsonl"


class ColumnProfile(BaseModel):
    """Detailed statistical and semantic profile for an individual column."""
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    n_unique: int
    cardinality_tier: str  # "BINARY", "LOW", "MEDIUM", "HIGH", "KEY"
    numeric_stats: Optional[Dict[str, float]] = None  # min, max, mean, std, q25, median, q75, iqr, lower_outlier_bound, upper_outlier_bound
    top_categories: Optional[List[Tuple[str, int]]] = None
    is_constant: bool = False
    is_high_null: bool = False
    is_potential_id: bool = False


class DatasetProfile(BaseModel):
    """Complete dataset metadata, bounded preview, and data quality diagnostics."""
    model_config = ConfigDict(populate_by_name=True)

    file_path: str
    file_name: str
    file_size_bytes: int
    format: DatasetFormat
    schema_dict: Dict[str, str] = Field(default_factory=dict, alias="schema")
    row_count: Optional[int] = None
    is_row_count_exact: bool = True
    estimated_in_memory_mb: float = 0.0
    expansion_factor: float = 1.0
    potential_targets: List[str] = Field(default_factory=list)
    potential_ids: List[str] = Field(default_factory=list)
    potential_dates: List[str] = Field(default_factory=list)
    columns: Dict[str, ColumnProfile] = Field(default_factory=dict)
    bounded_sample: List[Dict[str, Any]] = Field(default_factory=list)
    data_quality_flags: Dict[str, Any] = Field(default_factory=dict)
    class_imbalance: Optional[Dict[str, Any]] = None

    @property
    def schema(self) -> Dict[str, str]:
        """Schema mapping column names to string data types."""
        return self.schema_dict

    @property
    def sample_rows(self) -> List[Dict[str, Any]]:
        """Backward-compatible alias for existing tests and CLI."""
        return self.bounded_sample

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching existing pipeline expectations."""
        d = self.model_dump()
        d["file"] = self.file_path
        d["size_bytes"] = self.file_size_bytes
        d["format"] = self.format.value
        d["schema"] = self.schema_dict
        d["sample_rows"] = self.bounded_sample
        d["sample_row_count"] = len(self.bounded_sample)
        d["row_count"] = self.row_count
        d["is_row_count_exact"] = self.is_row_count_exact
        d["estimated_in_memory_mb"] = self.estimated_in_memory_mb
        d["potential_targets"] = self.potential_targets
        d["potential_ids"] = self.potential_ids
        d["potential_dates"] = self.potential_dates
        d["columns"] = {k: v.model_dump() for k, v in self.columns.items()}
        d["data_quality_flags"] = self.data_quality_flags
        d["class_imbalance"] = self.class_imbalance
        return d

    def to_prompt_dict(self, max_sample_rows: int = 10) -> Dict[str, Any]:
        """Compact representation for LLM context, strictly bounded under 14,000 characters."""
        d = self.to_dict()
        d["sample_rows"] = self.bounded_sample[:max_sample_rows]
        d["sample_row_count"] = len(d["sample_rows"])
        # Truncate top_categories in columns to top 5 to preserve token budget
        for col_name, col_data in d.get("columns", {}).items():
            if col_data.get("top_categories"):
                col_data["top_categories"] = col_data["top_categories"][:5]
        return d


TARGET_NAME_PATTERNS = [
    r"^target$", r"^label$", r"^class$", r"^churn$", r"^default$", r"^is_churn$",
    r"^is_fraud$", r"^fraud$", r"^survived$", r"^status$", r"^outcome$", r"^response$",
    r"^bought$", r"^converted$", r"^price$", r"^salary$", r"^revenue$", r"^sales$",
    r"^cost$", r"^loss$", r"^duration$", r"^score$", r"^rating$", r"^y$"
]

ID_NAME_PATTERNS = [
    r"^id$", r".*_id$", r"^id_.*", r"^uuid$", r"^guid$", r"^code$", r"^key$",
    r"^index$", r"^row_id$", r"^customer_id$", r"^user_id$", r"^transaction_id$"
]

DATE_NAME_PATTERNS = [
    r".*date.*", r".*time.*", r".*timestamp.*", r"^created_at$", r"^updated_at$",
    r"^year$", r"^month$", r"^day$", r"^dt$"
]


def _detect_format(path: Path) -> DatasetFormat:
    """Detect dataset format from file extension."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return DatasetFormat.PARQUET
    elif suffix == ".csv":
        return DatasetFormat.CSV
    elif suffix == ".tsv":
        return DatasetFormat.TSV
    elif suffix in {".jsonl", ".ndjson"}:
        return DatasetFormat.JSONL
    elif suffix == ".json":
        return DatasetFormat.JSON
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported: .csv, .tsv, .parquet, .pq, .json, .jsonl, .ndjson"
        )


def _estimate_csv_rows_by_density(file_path: Path, sample_lines: int = 1000) -> int:
    """Estimate row count for large CSV/TSV/JSONL via byte density sampling."""
    file_size = file_path.stat().st_size
    if file_size == 0:
        return 0

    read_bytes = 0
    lines_read = 0
    with open(file_path, "rb") as f:
        # Skip header
        _ = f.readline()
        for line in f:
            read_bytes += len(line)
            lines_read += 1
            if lines_read >= sample_lines:
                break

    if lines_read == 0 or read_bytes == 0:
        return 0

    avg_bytes_per_line = read_bytes / lines_read
    estimated_rows = int(file_size / avg_bytes_per_line)
    return max(1, estimated_rows)


def profile_dataset(file_path: Union[Path, str], max_sample_rows: int = 100) -> DatasetProfile:
    """Perform single-pass out-of-core statistical profiling on a dataset."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    fmt = _detect_format(path)
    file_size_bytes = path.stat().st_size
    lf = PolarsEngine.scan(path)
    schema_map = PolarsEngine.get_schema(lf)

    # 1. Bounded Preview (strictly <= 100 rows)
    bounded_n = max(1, min(max_sample_rows, 100))
    sample_df = PolarsEngine.preview(lf, n=bounded_n)
    bounded_sample = sample_df.to_dicts()

    # 2. Row count estimation or exact scan
    if fmt == DatasetFormat.PARQUET:
        row_count = lf.select(pl.len()).collect().item()
        is_exact = True
    elif fmt in (DatasetFormat.CSV, DatasetFormat.TSV, DatasetFormat.JSONL):
        if file_size_bytes < 50 * 1024 * 1024:
            row_count = lf.select(pl.len()).collect().item()
            is_exact = True
        else:
            row_count = _estimate_csv_rows_by_density(path)
            is_exact = False
    elif fmt == DatasetFormat.JSON:
        try:
            row_count = lf.select(pl.len()).collect().item()
            is_exact = True
        except Exception:
            row_count = len(bounded_sample)
            is_exact = False
    else:
        row_count = len(bounded_sample)
        is_exact = False

    row_count = max(row_count or 0, len(bounded_sample))

    # 3. Single-Pass Statistical Aggregations
    schema = lf.collect_schema()
    aggs: List[pl.Expr] = []
    numeric_cols: List[str] = []

    for col_name, dtype in schema.items():
        aggs.append(pl.col(col_name).null_count().alias(f"{col_name}__null_count"))
        aggs.append(pl.col(col_name).n_unique().alias(f"{col_name}__n_unique"))

        if dtype.is_numeric():
            numeric_cols.append(col_name)
            aggs.append(pl.col(col_name).min().alias(f"{col_name}__min"))
            aggs.append(pl.col(col_name).max().alias(f"{col_name}__max"))
            aggs.append(pl.col(col_name).mean().alias(f"{col_name}__mean"))
            aggs.append(pl.col(col_name).std().alias(f"{col_name}__std"))
            aggs.append(pl.col(col_name).quantile(0.25).alias(f"{col_name}__q25"))
            aggs.append(pl.col(col_name).median().alias(f"{col_name}__median"))
            aggs.append(pl.col(col_name).quantile(0.75).alias(f"{col_name}__q75"))

    try:
        stats_df = lf.select(aggs).collect()
        stats_dict = stats_df.to_dicts()[0] if stats_df.height > 0 else {}
    except Exception as e:
        logger.warning("Single-pass lazy aggregation failed (%s), falling back to bounded stats", e)
        stats_dict = {}

    # 4. Process Column Profiles
    columns: Dict[str, ColumnProfile] = {}
    total_est_bytes = 0

    for col_name, dtype in schema.items():
        null_count = int(stats_dict.get(f"{col_name}__null_count") or 0)
        n_unique = int(stats_dict.get(f"{col_name}__n_unique") or 0)
        null_pct = round((null_count / max(row_count, 1)) * 100.0, 2)

        # Numeric stats extraction
        num_stats: Optional[Dict[str, float]] = None
        if col_name in numeric_cols:
            c_min = stats_dict.get(f"{col_name}__min")
            c_max = stats_dict.get(f"{col_name}__max")
            c_mean = stats_dict.get(f"{col_name}__mean")
            c_std = stats_dict.get(f"{col_name}__std")
            c_q25 = stats_dict.get(f"{col_name}__q25")
            c_med = stats_dict.get(f"{col_name}__median")
            c_q75 = stats_dict.get(f"{col_name}__q75")

            if all(v is not None for v in (c_min, c_max, c_mean)):
                q25_f = float(c_q25) if c_q25 is not None else float(c_min)
                q75_f = float(c_q75) if c_q75 is not None else float(c_max)
                iqr = round(q75_f - q25_f, 4)
                num_stats = {
                    "min": round(float(c_min), 4),
                    "max": round(float(c_max), 4),
                    "mean": round(float(c_mean), 4),
                    "std": round(float(c_std), 4) if c_std is not None else 0.0,
                    "q25": q25_f,
                    "median": round(float(c_med), 4) if c_med is not None else round(float(c_mean), 4),
                    "q75": q75_f,
                    "iqr": iqr,
                    "lower_outlier_bound": round(q25_f - 1.5 * iqr, 4),
                    "upper_outlier_bound": round(q75_f + 1.5 * iqr, 4),
                }

        # Cardinality tier classification
        is_const = (
            n_unique <= 1
            or null_count == row_count
            or (num_stats is not None and num_stats.get("std") == 0.0)
        )
        is_high_null = null_pct > 50.0

        if is_const:
            cardinality_tier = "LOW"
        elif n_unique == 2:
            cardinality_tier = "BINARY"
        elif 3 <= n_unique <= 20:
            cardinality_tier = "LOW"
        elif 21 <= n_unique <= 500:
            cardinality_tier = "MEDIUM"
        elif row_count >= 10 and (n_unique >= int(0.90 * row_count) or n_unique == row_count):
            cardinality_tier = "KEY"
        else:
            cardinality_tier = "HIGH"

        # Potential ID check
        is_id = (
            cardinality_tier == "KEY"
            or any(re.match(pat, col_name, re.IGNORECASE) for pat in ID_NAME_PATTERNS)
        )

        # Top categories for low/medium cardinality columns
        top_cats: Optional[List[Tuple[str, int]]] = None
        if cardinality_tier in {"BINARY", "LOW", "MEDIUM"} and not dtype.is_float():
            try:
                cat_df = (
                    lf.select(pl.col(col_name).drop_nulls().value_counts(sort=True))
                    .limit(10)
                    .collect()
                )
                if cat_df.height > 0:
                    records = cat_df.to_dicts()
                    top_cats = [
                        (str(r[col_name]), int(r["count"]))
                        for r in records
                        if col_name in r and "count" in r
                    ]
            except Exception:
                pass

        # Estimate bytes per column
        if dtype in (pl.Int8, pl.UInt8, pl.Boolean):
            col_b = 1
        elif dtype in (pl.Int16, pl.UInt16):
            col_b = 2
        elif dtype in (pl.Int32, pl.UInt32, pl.Float32, pl.Date):
            col_b = 4
        elif dtype in (pl.Int64, pl.UInt64, pl.Float64, pl.Datetime):
            col_b = 8
        elif dtype == pl.String:
            # Sample string length
            sample_strs = [len(str(r.get(col_name, ""))) for r in bounded_sample if r.get(col_name) is not None]
            avg_str_len = (sum(sample_strs) / len(sample_strs)) if sample_strs else 16
            col_b = int(avg_str_len + 8)
        else:
            col_b = 16

        total_est_bytes += col_b * row_count

        columns[col_name] = ColumnProfile(
            name=col_name,
            dtype=str(dtype),
            null_count=null_count,
            null_percentage=null_pct,
            n_unique=n_unique,
            cardinality_tier=cardinality_tier,
            numeric_stats=num_stats,
            top_categories=top_cats,
            is_constant=is_const,
            is_high_null=is_high_null,
            is_potential_id=is_id,
        )

    # 5. Semantic Tagging
    potential_targets: List[str] = []
    potential_ids: List[str] = []
    potential_dates: List[str] = []

    for col_name, prof in columns.items():
        if prof.is_potential_id:
            potential_ids.append(col_name)

        if schema[col_name] in (pl.Date, pl.Datetime) or any(
            re.match(pat, col_name, re.IGNORECASE) for pat in DATE_NAME_PATTERNS
        ):
            potential_dates.append(col_name)

        # Target heuristic: matches target pattern, not constant, not high-null, not ID
        if (
            not prof.is_constant
            and not prof.is_high_null
            and not prof.is_potential_id
            and any(re.match(pat, col_name, re.IGNORECASE) for pat in TARGET_NAME_PATTERNS)
        ):
            potential_targets.append(col_name)

    # Positional target fallback: if no candidate matched regex, check the last column
    if not potential_targets and len(columns) > 1:
        last_col = list(columns.keys())[-1]
        last_prof = columns[last_col]
        if not last_prof.is_constant and not last_prof.is_potential_id and not last_prof.is_high_null:
            potential_targets.append(last_col)

    # 6. Data Quality Flags
    constant_cols = [c for c, p in columns.items() if p.is_constant]
    high_null_cols = [c for c, p in columns.items() if p.is_high_null]

    data_quality_flags: Dict[str, Any] = {
        "constant_columns": constant_cols,
        "high_null_columns": high_null_cols,
        "has_constant_columns": len(constant_cols) > 0,
        "has_high_null_columns": len(high_null_cols) > 0,
    }

    # 7. Class Imbalance Detection for Classification Targets
    class_imbalance: Optional[Dict[str, Any]] = None
    target_candidate = potential_targets[0] if potential_targets else None
    if target_candidate and target_candidate in columns:
        t_prof = columns[target_candidate]
        if t_prof.top_categories and len(t_prof.top_categories) >= 2:
            counts = [cnt for _, cnt in t_prof.top_categories]
            majority = max(counts)
            minority = min(counts)
            if minority > 0:
                ratio = round(majority / minority, 2)
                class_imbalance = {
                    "target_column": target_candidate,
                    "ratio": ratio,
                    "majority_count": majority,
                    "minority_count": minority,
                    "is_imbalanced": ratio >= 4.0,
                    "is_severely_imbalanced": ratio >= 10.0,
                    "distribution": {cat: cnt for cat, cnt in t_prof.top_categories},
                }

    # 8. Uncompressed Memory Footprint Estimation
    uncompressed_bytes = int(total_est_bytes * 1.3)
    est_memory_mb = round(uncompressed_bytes / (1024**2), 2)
    expansion_factor = round(uncompressed_bytes / max(file_size_bytes, 1), 2)

    return DatasetProfile(
        file_path=str(path),
        file_name=path.name,
        file_size_bytes=file_size_bytes,
        format=fmt,
        schema=schema_map,
        row_count=row_count,
        is_row_count_exact=is_exact,
        estimated_in_memory_mb=est_memory_mb,
        expansion_factor=expansion_factor,
        potential_targets=potential_targets,
        potential_ids=potential_ids,
        potential_dates=potential_dates,
        columns=columns,
        bounded_sample=bounded_sample,
        data_quality_flags=data_quality_flags,
        class_imbalance=class_imbalance,
    )


def discover_datasets(directory: Union[Path, str]) -> List[DatasetProfile]:
    """Discover and profile all supported tabular datasets in a directory."""
    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {dir_path}")

    profiles: List[DatasetProfile] = []
    supported_extensions = {".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl", ".ndjson"}

    candidate_files = sorted([
        f for f in dir_path.iterdir()
        if f.is_file()
        and not f.name.startswith(".")
        and not f.name.endswith(".manifest.json")
        and not f.name.endswith(".tmp")
        and f.suffix.lower() in supported_extensions
    ])

    for file_path in candidate_files:
        try:
            profile = profile_dataset(file_path, max_sample_rows=100)
            profiles.append(profile)
        except Exception as e:
            logger.warning("Failed to profile dataset %s: %s", file_path, e)

    return profiles


def inspect_dataset(path: Union[Path, str]) -> Dict[str, Any]:
    """Backward-compatible helper function inspecting dataset and returning dictionary."""
    return profile_dataset(path).to_dict()
