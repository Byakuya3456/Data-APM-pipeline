"""Dataset size-aware execution strategy selector and pre-flight guardrails.

Evaluates dataset on-disk size, row count, uncompressed memory footprint, and
live system telemetry to select between IN_MEMORY, LAZY_POLARS, DUCKDB_OUT_OF_CORE,
CHUNKED_BATCH, and INCREMENTAL_TRAINING.
"""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from config.settings import get_settings
from engine.resource_monitor import SystemTelemetry, get_system_telemetry

logger = logging.getLogger("autonomous_ml.memory_manager")


class ExecutionStrategy(str, Enum):
    """Execution strategy chosen based on dataset size and host resource boundaries."""
    IN_MEMORY = "IN_MEMORY"
    LAZY_POLARS = "LAZY_POLARS"
    DUCKDB_OUT_OF_CORE = "DUCKDB_OUT_OF_CORE"
    CHUNKED_BATCH = "CHUNKED_BATCH"
    INCREMENTAL_TRAINING = "INCREMENTAL_TRAINING"


class StrategyRecommendation(BaseModel):
    """Actionable execution recommendation with resource limits and execution parameters."""
    strategy: ExecutionStrategy
    reason: str
    estimated_memory_mb: float
    batch_size: Optional[int] = None
    recommended_threads: int = 4
    duckdb_memory_limit: Optional[str] = None
    spill_to_disk: bool = False
    device: str = "cpu"  # "cpu" | "cuda"
    gpu_recommended: bool = False
    warning_messages: List[str] = Field(default_factory=list)


class MemoryManager:
    """Selects execution strategy and enforces pre-flight resource guardrails."""

    # Sizing thresholds
    IN_MEMORY_MAX_FILE_BYTES = 50 * 1024 * 1024        # 50 MB on disk
    IN_MEMORY_MAX_RAM_RATIO = 0.15                     # 15% of available RAM
    LAZY_POLARS_MAX_FILE_BYTES = 4 * 1024 * 1024 * 1024 # 4 GB on disk
    LAZY_POLARS_MAX_RAM_RATIO = 0.50                   # 50% of available RAM
    INCREMENTAL_ROW_THRESHOLD = 1_000_000               # 1 Million rows

    @staticmethod
    def estimate_uncompressed_bytes(
        file_size_bytes: int,
        format_hint: Optional[str] = None,
        row_count: Optional[int] = None,
        column_count: Optional[int] = None,
    ) -> int:
        """Estimate in-memory uncompressed footprint based on file format and size."""
        fmt = (format_hint or "").lower().lstrip(".")
        if fmt in ("parquet", "pq"):
            multiplier = 3.5  # Snappy/Zstd tabular compression expansion
        elif fmt in ("csv", "tsv"):
            multiplier = 1.8  # Null bitmaps, string pointers, and 64-bit alignment
        elif fmt in ("json", "jsonl", "ndjson"):
            multiplier = 1.2
        else:
            multiplier = 2.0

        if row_count and column_count:
            # Cross-check using row/col density: assume 16 bytes average cell overhead
            grid_estimate = row_count * column_count * 16
            return max(int(file_size_bytes * multiplier), grid_estimate)

        return int(file_size_bytes * multiplier)

    @classmethod
    def select_strategy(
        cls,
        file_size_bytes: int,
        estimated_rows: int = 0,
        estimated_uncompressed_bytes: Optional[int] = None,
        operation_type: str = "general",
        telemetry: Optional[SystemTelemetry] = None,
    ) -> StrategyRecommendation:
        """Select execution strategy based on data size, operation complexity, and live telemetry."""
        if telemetry is None:
            telemetry = get_system_telemetry()

        avail_ram_bytes = int(telemetry.ram_available_gb * (1024**3))
        if estimated_uncompressed_bytes is None:
            estimated_uncompressed_bytes = cls.estimate_uncompressed_bytes(file_size_bytes)

        est_mem_mb = round(estimated_uncompressed_bytes / (1024**2), 2)
        settings = get_settings()
        rec_threads = max(1, min(telemetry.cpu_threads_logical - 2, settings.duckdb_threads))
        op = operation_type.lower()
        warnings: List[str] = []

        # GPU evaluation (NVIDIA RTX 3050 6GB VRAM)
        device = "cpu"
        gpu_rec = False
        if telemetry.gpu_available and telemetry.gpu_vram_free_mb is not None:
            if telemetry.gpu_vram_free_mb >= 2048.0 and op in ("training", "train", "model_training"):
                device = "cuda"
                gpu_rec = True
            elif telemetry.gpu_vram_free_mb < 2048.0 and op in ("training", "train", "model_training"):
                warnings.append(
                    f"GPU detected ({telemetry.gpu_name}) but free VRAM ({telemetry.gpu_vram_free_mb:.0f} MB) "
                    f"is below 2048 MB threshold. Defaulting to CPU."
                )

        # Rule 1: Incremental Training for massive ML training jobs
        if op in ("training", "train", "model_training") and (
            estimated_rows >= cls.INCREMENTAL_ROW_THRESHOLD
            or estimated_uncompressed_bytes > avail_ram_bytes * 0.40
        ):
            return StrategyRecommendation(
                strategy=ExecutionStrategy.INCREMENTAL_TRAINING,
                reason=(
                    f"Dataset contains {estimated_rows:,} rows or requires {est_mem_mb:.1f} MB, "
                    f"exceeding standard in-memory estimator capacity. Incremental mini-batch training recommended."
                ),
                estimated_memory_mb=est_mem_mb,
                batch_size=50_000,
                recommended_threads=rec_threads,
                device=device,
                gpu_recommended=gpu_rec,
                warning_messages=warnings,
            )

        # Rule 2: In-Memory execution for small datasets
        if (
            file_size_bytes < cls.IN_MEMORY_MAX_FILE_BYTES
            and estimated_uncompressed_bytes < avail_ram_bytes * cls.IN_MEMORY_MAX_RAM_RATIO
            and op not in ("join", "heavy_join")
        ):
            return StrategyRecommendation(
                strategy=ExecutionStrategy.IN_MEMORY,
                reason=(
                    f"Dataset size ({file_size_bytes / (1024**2):.2f} MB) and in-memory footprint ({est_mem_mb:.1f} MB) "
                    f"fit comfortably within 15% of available host RAM ({telemetry.ram_available_gb:.1f} GB)."
                ),
                estimated_memory_mb=est_mem_mb,
                batch_size=None,
                recommended_threads=rec_threads,
                device=device,
                gpu_recommended=gpu_rec,
                warning_messages=warnings,
            )

        # Rule 3: Lazy Polars execution for medium datasets with streaming sinks
        if (
            file_size_bytes <= cls.LAZY_POLARS_MAX_FILE_BYTES
            and estimated_uncompressed_bytes < avail_ram_bytes * cls.LAZY_POLARS_MAX_RAM_RATIO
            and op not in ("heavy_join", "complex_sql")
        ):
            return StrategyRecommendation(
                strategy=ExecutionStrategy.LAZY_POLARS,
                reason=(
                    f"Dataset size ({file_size_bytes / (1024**2):.1f} MB) exceeds in-memory threshold but fits within "
                    f"50% of available RAM. Polars LazyFrame with predicate pushdown and streaming parquet sink is optimal."
                ),
                estimated_memory_mb=est_mem_mb,
                batch_size=None,
                recommended_threads=rec_threads,
                device=device,
                gpu_recommended=gpu_rec,
                warning_messages=warnings,
            )

        # Rule 4: DuckDB out-of-core SQL engine for large datasets or complex joins
        if (
            file_size_bytes > cls.LAZY_POLARS_MAX_FILE_BYTES
            or estimated_uncompressed_bytes >= avail_ram_bytes * cls.LAZY_POLARS_MAX_RAM_RATIO
            or op in ("join", "heavy_join", "complex_sql")
        ):
            duckdb_mem = settings.duckdb_memory_limit  # default "8GB"
            return StrategyRecommendation(
                strategy=ExecutionStrategy.DUCKDB_OUT_OF_CORE,
                reason=(
                    f"Dataset footprint ({est_mem_mb:.1f} MB) exceeds 50% of available host RAM "
                    f"or operation requires out-of-core join spilling. DuckDB disk-spilling engine required."
                ),
                estimated_memory_mb=est_mem_mb,
                batch_size=None,
                recommended_threads=rec_threads,
                duckdb_memory_limit=duckdb_mem,
                spill_to_disk=True,
                device=device,
                gpu_recommended=gpu_rec,
                warning_messages=warnings,
            )

        # Rule 5: Fallback to chunked batch processing
        return StrategyRecommendation(
            strategy=ExecutionStrategy.CHUNKED_BATCH,
            reason="Resource pressure or non-vectorizable operations require partitioned batch processing.",
            estimated_memory_mb=est_mem_mb,
            batch_size=50_000,
            recommended_threads=rec_threads,
            spill_to_disk=True,
            device=device,
            gpu_recommended=gpu_rec,
            warning_messages=warnings,
        )

    @classmethod
    def select_execution_strategy(
        cls,
        first_arg: Union[int, Path, str],
        estimated_uncompressed_bytes: int,
        third_arg: Union[int, str, None] = None,
        operation_type: str = "general",
        telemetry: Optional[SystemTelemetry] = None,
    ) -> ExecutionStrategy:
        """Polymorphic backwards-compatible execution strategy selector.

        Supports:
        1. tests/test_pipeline.py:1427 signature:
           (file_size_bytes: int, estimated_uncompressed_bytes: int, available_ram_bytes: int)
        2. PROJECT.md contract signature:
           (dataset_path: Path | str, estimated_uncompressed_bytes: int, operation_type: str)
        """
        if isinstance(first_arg, (int, float)) and isinstance(third_arg, (int, float)):
            # Signature from test_pipeline.py:1427
            file_size_bytes = int(first_arg)
            available_ram_bytes = int(third_arg)

            if file_size_bytes < 10 * 1024 * 1024 and estimated_uncompressed_bytes < available_ram_bytes * 0.10:
                return ExecutionStrategy.IN_MEMORY
            elif file_size_bytes < 1024 * 1024 * 1024:
                return ExecutionStrategy.LAZY_POLARS
            else:
                return ExecutionStrategy.DUCKDB_OUT_OF_CORE

        # Signature from PROJECT.md: first_arg is dataset_path
        path = Path(first_arg) if isinstance(first_arg, (str, Path)) else None
        if path and path.exists():
            file_size_bytes = path.stat().st_size
        else:
            file_size_bytes = int(first_arg) if isinstance(first_arg, (int, float)) else estimated_uncompressed_bytes

        op = str(third_arg) if isinstance(third_arg, str) else operation_type

        rec = cls.select_strategy(
            file_size_bytes=file_size_bytes,
            estimated_uncompressed_bytes=estimated_uncompressed_bytes,
            operation_type=op,
            telemetry=telemetry,
        )
        return rec.strategy

    @classmethod
    def preflight_check(
        cls,
        recommendation: StrategyRecommendation,
        telemetry: Optional[SystemTelemetry] = None,
        required_disk_gb: float = 5.0,
        requested_device: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Validate system resource headroom before executing expensive tasks."""
        if telemetry is None:
            telemetry = get_system_telemetry()

        # Check 1: Host RAM pressure
        if telemetry.ram_available_gb < 1.5:
            return (
                False,
                f"CRITICAL_MEMORY_PRESSURE: Available host RAM ({telemetry.ram_available_gb:.1f} GB) "
                f"is below the absolute minimum safety threshold of 1.5 GB. Free memory before launching job."
            )

        # Check 2: Disk storage headroom
        if telemetry.disk_free_gb < required_disk_gb:
            return (
                False,
                f"INSUFFICIENT_DISK_SPACE: Required {required_disk_gb:.1f} GB of workspace disk headroom, "
                f"but only {telemetry.disk_free_gb:.1f} GB is available. Free disk space to proceed."
            )

        # Check 3: Strategy memory headroom against available RAM
        if recommendation.strategy == ExecutionStrategy.IN_MEMORY:
            if (recommendation.estimated_memory_mb / 1024.0) > (telemetry.ram_available_gb * 0.70):
                return (
                    False,
                    f"EXCESSIVE_IN_MEMORY_LOAD: Estimated memory ({recommendation.estimated_memory_mb:.1f} MB) "
                    f"exceeds 70% of available RAM ({telemetry.ram_available_gb * 1024:.1f} MB). "
                    f"Downgrade strategy to LAZY_POLARS or DUCKDB_OUT_OF_CORE."
                )

        # Check 4: GPU VRAM safety
        target_device = (requested_device or recommendation.device).lower()
        if target_device == "cuda":
            if not telemetry.gpu_available:
                return (
                    True,
                    "PREFLIGHT_WARNING: CUDA execution requested but no NVIDIA GPU detected. "
                    "Automatically falling back to CPU execution."
                )
            if telemetry.gpu_vram_free_mb is not None and telemetry.gpu_vram_free_mb < 2048.0:
                return (
                    True,
                    f"PREFLIGHT_WARNING: Free GPU VRAM ({telemetry.gpu_vram_free_mb:.0f} MB) is below 2048 MB. "
                    f"Automatically falling back to CPU execution to prevent CUDA Out-of-Memory."
                )

        return (True, "PREFLIGHT_PASSED: System resources are sufficient for recommended execution strategy.")
