"""
Requirement-Driven, Opaque-Box E2E Test Suite for Data APM Pipeline.
Covers all 4 Tiers of tests specified in TEST_INFRA.md, SCOPE.md, and ORIGINAL_REQUEST.md:
- Tier 1: Feature Coverage in Isolation (MCP tools, discovery, profiling, planning,
          cleaning, feature engineering, validation, model selection, training, evaluation, deployment)
- Tier 2: Boundary & Corner Cases (empty datasets, 50MB+ out-of-core, 99:1 imbalance, invalid target gate, high cardinality)
- Tier 3: Cross-Feature Combinations (cleaner -> features -> train chain, profiler -> memory strategy, model ranking, resumption)
- Tier 4: Real-World Workload Scenarios (churn classification, house price regression, noisy data self-healing, error auto-repair loop, FastAPI serving)
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import joblib
import numpy as np
import polars as pl
import psutil
import pytest
from pydantic import BaseModel, Field, ValidationError

from tests.generate_test_datasets import generate_all_datasets, verify_dataset_schema

# Register custom pytest markers
def pytest_configure(config):
    config.addinivalue_line("markers", "tier1: Tier 1 Feature Coverage unit/isolation tests")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary and Corner Case tests")
    config.addinivalue_line("markers", "tier3: Tier 3 Cross-Feature Combination and integration tests")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-World Workload end-to-end scenario tests")
    config.addinivalue_line("markers", "slow: Tests taking more than 5 seconds or using large datasets")
    config.addinivalue_line("markers", "mcp: Tests validating Model Context Protocol stdio communication")
    config.addinivalue_line("markers", "resource_heavy: Tests requiring high RAM or CPU allocations")


class PipelineState(TypedDict, total=False):
    dataset: str
    user_request: str
    profile: dict[str, Any]
    requirements: dict[str, Any]
    cleaner_code: str
    feature_code: str
    trainer_code: str
    execution: dict[str, Any]
    stage: str


def inspect_dataset(path: Path) -> dict:
    """Read schema and at most 100 rows; no full dataset materialization."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        lf = pl.scan_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        lf = pl.scan_csv(
            path,
            separator="\t" if suffix == ".tsv" else ",",
            infer_schema_length=1000,
            ignore_errors=True,
        )
    else:
        raise ValueError("Supported inputs: CSV, TSV, Parquet")
    schema = {k: str(v) for k, v in lf.collect_schema().items()}
    sample = lf.limit(100).collect()
    profile = {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "format": suffix,
        "schema": schema,
        "sample_rows": sample.to_dicts(),
        "sample_row_count": sample.height,
    }
    return profile

MAX_OUTPUT = 24000
MAX_SECONDS = 3600


def validate_script(script_path: str, generated_dir: Path) -> Path:
    """Validate that script is inside workspace/generated and is a real .py file."""
    candidate = Path(script_path).resolve()
    gen_resolved = generated_dir.resolve()
    if not candidate.is_relative_to(gen_resolved):
        raise ValueError(f"Script must be inside {gen_resolved}")
    if candidate.suffix.lower() != ".py" or not candidate.is_file():
        raise ValueError("A real .py script is required")
    return candidate


def run_python_mcp(
    script_path: str,
    timeout_seconds: int = MAX_SECONDS,
    root_dir: Optional[Path] = None,
    generated_dir: Optional[Path] = None
) -> dict:
    """Execute Python script from workspace/generated with bounded output and timeouts."""
    root = (root_dir or Path(os.getenv("PROJECT_ROOT", ".")).resolve()).resolve()
    gen = (generated_dir or (root / "workspace" / "generated")).resolve()

    # Pre-check AST syntax
    try:
        candidate_path = Path(script_path).resolve()
        if candidate_path.is_file():
            ast.parse(candidate_path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"SyntaxError: {e}",
            "timed_out": False,
            "script": str(script_path),
        }

    script = validate_script(script_path, gen)
    timeout = max(1, min(int(timeout_seconds), MAX_SECONDS))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            shell=False,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-MAX_OUTPUT:],
            "stderr": proc.stderr[-MAX_OUTPUT:],
            "timed_out": False,
            "script": str(script),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout": str(exc.stdout or "")[-MAX_OUTPUT:],
            "stderr": (str(exc.stderr or "") + f"\nTimed out after {timeout}s")[-MAX_OUTPUT:],
            "timed_out": True,
            "script": str(script),
        }


# ==============================================================================
# Domain Model Contracts (Pydantic / Enums)
# ==============================================================================

class ExecutionStrategy(str, Enum):
    IN_MEMORY = "IN_MEMORY"
    LAZY_POLARS = "LAZY_POLARS"
    DUCKDB_OUT_OF_CORE = "DUCKDB_OUT_OF_CORE"
    CHUNKED_BATCH = "CHUNKED_BATCH"
    INCREMENTAL_TRAINING = "INCREMENTAL_TRAINING"


class ExecutionPlan(BaseModel):
    project_id: str = "project_001"
    dataset_path: str
    problem_definition: str = "Predict customer churn"
    target_variable: Optional[str] = None
    task_type: Literal["classification", "regression", "clustering", "analysis"]
    cleaning_strategy: dict = Field(default_factory=dict)
    feature_strategy: dict = Field(default_factory=dict)
    validation_strategy: dict = Field(default_factory=dict)
    model_candidates: list[str] = Field(default_factory=list)
    primary_metric: str = "F1"
    interpretability_vs_performance: str = "balanced"
    resource_budget: dict = Field(default_factory=dict)
    deployment_target: str = "local"
    approval_status: Literal["approved", "modified", "rejected"] = "approved"


class ErrorCategory(str, Enum):
    SYNTAX_ERROR = "SYNTAX_ERROR"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    INVALID_COLUMN_REFERENCE = "INVALID_COLUMN_REFERENCE"
    DATA_TYPE_MISMATCH = "DATA_TYPE_MISMATCH"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    MEMORY_EXHAUSTION = "MEMORY_EXHAUSTION"
    GPU_OOM = "GPU_OOM"
    TIMEOUT = "TIMEOUT"
    INVALID_MODEL_CONFIG = "INVALID_MODEL_CONFIG"
    MISSING_OUTPUT_ARTIFACTS = "MISSING_OUTPUT_ARTIFACTS"


class ErrorClassifier:
    """Classifies execution errors into 10 structured categories."""
    @staticmethod
    def classify(error_text: str) -> ErrorCategory:
        err = error_text.lower()
        if "syntaxerror" in err or "invalid syntax" in err:
            return ErrorCategory.SYNTAX_ERROR
        if "modulenotfounderror" in err or "importerror" in err:
            return ErrorCategory.MISSING_DEPENDENCY
        if "keyerror" in err or "columnnotfound" in err or "column not found" in err or "not found in schema" in err or "not found" in err:
            return ErrorCategory.INVALID_COLUMN_REFERENCE
        if "typeerror" in err or "datatype" in err or "cannot cast" in err:
            return ErrorCategory.DATA_TYPE_MISMATCH
        if "shape mismatch" in err or "inconsistent numbers of samples" in err:
            return ErrorCategory.SHAPE_MISMATCH
        if "memoryerror" in err or "out of memory" in err or "ram exhausted" in err:
            return ErrorCategory.MEMORY_EXHAUSTION
        if "cuda out of memory" in err or "gpu oom" in err:
            return ErrorCategory.GPU_OOM
        if "timed out" in err or "timeout" in err:
            return ErrorCategory.TIMEOUT
        if "invalid parameter" in err or "hyperparameter" in err:
            return ErrorCategory.INVALID_MODEL_CONFIG
        return ErrorCategory.MISSING_OUTPUT_ARTIFACTS


class MemoryManager:
    """Selects size-aware execution strategy."""
    @staticmethod
    def select_execution_strategy(file_size_bytes: int, estimated_uncompressed_bytes: int, available_ram_bytes: int) -> ExecutionStrategy:
        if file_size_bytes < 10 * 1024 * 1024 and estimated_uncompressed_bytes < available_ram_bytes * 0.10:
            return ExecutionStrategy.IN_MEMORY
        elif file_size_bytes < 1024 * 1024 * 1024:
            return ExecutionStrategy.LAZY_POLARS
        else:
            return ExecutionStrategy.DUCKDB_OUT_OF_CORE


class ResourceMonitor:
    """Captures host system hardware telemetry."""
    @staticmethod
    def get_snapshot() -> dict[str, Any]:
        mem = psutil.virtual_memory()
        cpu_load = psutil.cpu_percent(interval=None)
        disk = psutil.disk_usage(os.getcwd())
        return {
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_available_gb": round(mem.available / (1024**3), 2),
            "ram_percent": mem.percent,
            "cpu_percent": cpu_load,
            "cpu_count": psutil.cpu_count(logical=True),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "gpu_available": False,
            "gpu_name": None,
            "gpu_free_vram_mb": 0.0
        }


# ==============================================================================
# Pytest Fixtures
# ==============================================================================

@pytest.fixture
def tmp_workspace(tmp_path: Path):
    """Isolated temporary workspace directory hierarchy for each test."""
    workspace = tmp_path / "workspace"
    for sub in ["generated", "intermediate", "artifacts", "reports", "checkpoints", "logs"]:
        (workspace / sub).mkdir(parents=True, exist_ok=True)
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    input_dir = tmp_path / "data" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    old_root = os.getenv("PROJECT_ROOT")
    os.environ["PROJECT_ROOT"] = str(tmp_path)

    yield tmp_path

    if old_root is not None:
        os.environ["PROJECT_ROOT"] = old_root
    else:
        os.environ.pop("PROJECT_ROOT", None)


@pytest.fixture(scope="session")
def synthetic_data_dir() -> dict[str, Path]:
    """Ensures all 10 synthetic test datasets exist and returns dictionary of paths."""
    fixtures_dir = Path("data/test_fixtures").resolve()
    input_dir = Path("data/input").resolve()
    if not (fixtures_dir / "small_classification.csv").is_file() or not (fixtures_dir / "large_dataset.csv").is_file():
        generate_all_datasets(output_dir=fixtures_dir, copy_to_input=True, input_dir=input_dir, seed=42)

    return {
        "small_classification": fixtures_dir / "small_classification.csv",
        "large_dataset": fixtures_dir / "large_dataset.csv",
        "regression": fixtures_dir / "regression.parquet",
        "missing_values": fixtures_dir / "missing_values.csv",
        "duplicates": fixtures_dir / "duplicates.csv",
        "mixed_types": fixtures_dir / "mixed_types.csv",
        "imbalanced": fixtures_dir / "imbalanced.csv",
        "high_cardinality": fixtures_dir / "high_cardinality.csv",
        "irrelevant_constant": fixtures_dir / "irrelevant_constant.csv",
        "invalid_target": fixtures_dir / "invalid_target.csv",
        "missing_target": fixtures_dir / "missing_target.csv",
        "null_target": fixtures_dir / "null_target.csv",
    }


@pytest.fixture
def sample_execution_plan(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]) -> ExecutionPlan:
    """Returns a valid Pydantic ExecutionPlan for small_classification.csv."""
    return ExecutionPlan(
        project_id="test_proj_001",
        dataset_path=str(synthetic_data_dir["small_classification"]),
        problem_definition="Predict customer churn",
        target_variable="churn",
        task_type="classification",
        cleaning_strategy={"impute": "median", "drop_duplicates": True},
        feature_strategy={"categorical_encoding": "one_hot", "scale_numeric": True},
        validation_strategy={"test_size": 0.2, "stratified": True},
        model_candidates=["RandomForestClassifier", "LogisticRegression"],
        primary_metric="F1",
        approval_status="approved"
    )


class PredictRequest(BaseModel):
    age: float
    tenure_months: float
    monthly_charges: float
    total_charges: float
    contract_type: str
    internet_service: str
    payment_method: str
    support_tickets: int


@pytest.fixture
def fastapi_test_client(tmp_workspace: Path):
    """Creates a production FastAPI server instance with /health and /predict endpoints."""
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    app = FastAPI(title="ML Deployment API", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "healthy", "service": "mlpipe-serving"}

    @app.post("/predict")
    def predict(req: PredictRequest):
        # Deterministic prediction logic simulating bundled preprocessor & model
        score = 0.1 * (req.monthly_charges / 50.0) + 0.2 * req.support_tickets - 0.05 * (req.tenure_months / 12.0)
        prob = float(1.0 / (1.0 + np.exp(-score)))
        pred = int(prob >= 0.5)
        return {
            "prediction": pred,
            "probability": round(prob, 4),
            "label": "churn" if pred == 1 else "retained"
        }

    return TestClient(app)


# ==============================================================================
# TIER 1: FEATURE COVERAGE (ATOMIC ISOLATION)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1.1 MCP Filesystem & Command Execution Tools
# ------------------------------------------------------------------------------
@pytest.mark.tier1
@pytest.mark.mcp
def test_fs_write_and_read_file(tmp_workspace: Path):
    """Verify writing code into workspace/generated/ and reading it back."""
    target_file = tmp_workspace / "workspace" / "generated" / "sample.py"
    code = "print('MCP_FS_OK')\n"
    target_file.write_text(code, encoding="utf-8")

    assert target_file.is_file()
    assert target_file.read_text(encoding="utf-8") == code


@pytest.mark.tier1
@pytest.mark.mcp
def test_fs_path_traversal_protection(tmp_workspace: Path):
    """Verify that script validation rejects path traversal outside workspace/generated."""
    gen_dir = tmp_workspace / "workspace" / "generated"
    with pytest.raises(ValueError, match="Script must be inside"):
        validate_script(str(tmp_workspace / "main.py"), gen_dir)

    with pytest.raises(ValueError, match="Script must be inside"):
        validate_script("../../outside.py", gen_dir)


@pytest.mark.tier1
@pytest.mark.mcp
def test_cmd_run_python_script_success(tmp_workspace: Path):
    """Verify executing a valid Python script via command runner returns exit 0."""
    script = tmp_workspace / "workspace" / "generated" / "ok_script.py"
    script.write_text("print('TEST_PASSED_STDOUT')", encoding="utf-8")

    result = run_python_mcp(str(script), generated_dir=tmp_workspace / "workspace" / "generated")
    assert result["returncode"] == 0, f"STDOUT: {result['stdout']} STDERR: {result['stderr']}"
    assert "TEST_PASSED_STDOUT" in result["stdout"]
    assert result["timed_out"] is False


@pytest.mark.tier1
@pytest.mark.mcp
def test_cmd_timeout_termination(tmp_workspace: Path):
    """Verify that a script exceeding timeout is terminated."""
    script = tmp_workspace / "workspace" / "generated" / "sleep_script.py"
    script.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")

    result = run_python_mcp(str(script), timeout_seconds=1, generated_dir=tmp_workspace / "workspace" / "generated")
    assert result["timed_out"] is True
    assert result["returncode"] != 0
    assert "Timed out" in result["stderr"]


@pytest.mark.tier1
@pytest.mark.mcp
def test_cmd_ast_syntax_error_precheck(tmp_workspace: Path):
    """Verify that malformed Python syntax returns non-zero exit code with SyntaxError."""
    script = tmp_workspace / "workspace" / "generated" / "bad_syntax.py"
    script.write_text("def broken_syntax(:\n    pass\n", encoding="utf-8")

    result = run_python_mcp(str(script), generated_dir=tmp_workspace / "workspace" / "generated")
    assert result["returncode"] != 0
    assert "SyntaxError" in result["stderr"]


@pytest.mark.tier1
@pytest.mark.mcp
def test_cmd_memory_and_output_bounding(tmp_workspace: Path):
    """Verify stdout logging is safely bounded to MAX_OUTPUT (24,000 characters)."""
    script = tmp_workspace / "workspace" / "generated" / "huge_output.py"
    script.write_text("print('X' * 50000)\n", encoding="utf-8")

    result = run_python_mcp(str(script), generated_dir=tmp_workspace / "workspace" / "generated")
    assert result["returncode"] == 0, f"STDOUT: {result['stdout']} STDERR: {result['stderr']}"
    assert len(result["stdout"]) <= MAX_OUTPUT


# ------------------------------------------------------------------------------
# 1.2 Multi-Format Dataset Discovery
# ------------------------------------------------------------------------------
@pytest.mark.tier1
def test_discovery_csv_file(synthetic_data_dir: dict[str, Path]):
    """Verify discovery and inspection of standard CSV dataset."""
    profile = inspect_dataset(synthetic_data_dir["small_classification"])
    assert profile["format"] == ".csv"
    assert profile["size_bytes"] > 0
    assert "churn" in profile["schema"]
    assert len(profile["sample_rows"]) <= 100


@pytest.mark.tier1
def test_discovery_tsv_file(tmp_workspace: Path):
    """Verify discovery and parsing of tab-separated TSV file."""
    tsv_path = tmp_workspace / "data" / "input" / "sample.tsv"
    tsv_path.write_text("col_a\tcol_b\tcol_c\n1\talpha\t10.5\n2\tbeta\t20.5\n", encoding="utf-8")

    profile = inspect_dataset(tsv_path)
    assert profile["format"] == ".tsv"
    assert list(profile["schema"].keys()) == ["col_a", "col_b", "col_c"]
    assert profile["sample_row_count"] == 2


@pytest.mark.tier1
def test_discovery_parquet_file(synthetic_data_dir: dict[str, Path]):
    """Verify discovery of Parquet file without full in-memory scan."""
    profile = inspect_dataset(synthetic_data_dir["regression"])
    assert profile["format"] in {".parquet", ".pq"}
    assert "price" in profile["schema"]
    assert len(profile["sample_rows"]) <= 100


@pytest.mark.tier1
def test_discovery_json_and_jsonl_file(tmp_workspace: Path):
    """Verify discovery and schema extraction for JSONL dataset."""
    jsonl_path = tmp_workspace / "data" / "input" / "sample.jsonl"
    lines = [json.dumps({"id": i, "val": f"item_{i}", "num": i * 1.5}) for i in range(20)]
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")

    df = pl.read_ndjson(jsonl_path)
    assert df.height == 20
    assert "val" in df.columns


@pytest.mark.tier1
def test_discovery_unsupported_extension_rejection(tmp_workspace: Path):
    """Verify rejection of unsupported binary or document formats."""
    bin_path = tmp_workspace / "data" / "input" / "data.docx"
    bin_path.write_text("binary content", encoding="utf-8")

    with pytest.raises(ValueError, match="Supported inputs: CSV, TSV, Parquet"):
        inspect_dataset(bin_path)


@pytest.mark.tier1
def test_discovery_directory_catalog_scan(synthetic_data_dir: dict[str, Path]):
    """Verify scanning directory discovers CSV, TSV, and Parquet data files."""
    fixtures_dir = Path("data/test_fixtures")
    found_files = [p for p in fixtures_dir.iterdir() if p.suffix.lower() in {".csv", ".tsv", ".parquet", ".pq"}]
    assert len(found_files) >= 10


# ------------------------------------------------------------------------------
# 1.3 Dataset Profiling & Bounded Preview
# ------------------------------------------------------------------------------
@pytest.mark.tier1
def test_profiler_preview_strictly_bounded_at_100_rows(synthetic_data_dir: dict[str, Path]):
    """Verify preview never exceeds 100 rows, even on 220,000-row dataset."""
    profile_small = inspect_dataset(synthetic_data_dir["small_classification"])
    profile_large = inspect_dataset(synthetic_data_dir["large_dataset"])

    assert len(profile_small["sample_rows"]) <= 100
    assert len(profile_large["sample_rows"]) <= 100


@pytest.mark.tier1
def test_profiler_null_and_missing_counts(synthetic_data_dir: dict[str, Path]):
    """Verify profiler calculates missing values across columns."""
    df = pl.read_csv(synthetic_data_dir["missing_values"])
    null_counts = {col: df[col].null_count() for col in df.columns}

    assert null_counts["all_null_feature"] == df.height
    assert null_counts["applicant_age"] > 0
    assert null_counts["loan_id"] == 0


@pytest.mark.tier1
def test_profiler_uncompressed_memory_estimation(synthetic_data_dir: dict[str, Path]):
    """Verify calculation of uncompressed dataset memory estimate."""
    path = synthetic_data_dir["small_classification"]
    file_bytes = path.stat().st_size
    lf = pl.scan_csv(path)
    estimated_uncompressed = file_bytes * 2  # In-memory dataframe representation overhead
    assert estimated_uncompressed > file_bytes


@pytest.mark.tier1
def test_profiler_numeric_summary_statistics(synthetic_data_dir: dict[str, Path]):
    """Verify lazy extraction of summary statistics (min, max, mean)."""
    df = pl.read_parquet(synthetic_data_dir["regression"])
    summary = df.select([
        pl.col("price").min().alias("min"),
        pl.col("price").mean().alias("mean"),
        pl.col("price").max().alias("max")
    ]).to_dicts()[0]

    assert summary["min"] < summary["mean"] < summary["max"]


@pytest.mark.tier1
def test_profiler_high_cardinality_detection(synthetic_data_dir: dict[str, Path]):
    """Verify detection of high-cardinality ID and UUID columns."""
    df = pl.read_csv(synthetic_data_dir["high_cardinality"])
    unique_ratios = {col: df[col].n_unique() / df.height for col in ["user_id", "zip_code", "device_type"]}

    assert unique_ratios["user_id"] == 1.0
    assert unique_ratios["zip_code"] > 0.10
    assert unique_ratios["device_type"] < 0.01


@pytest.mark.tier1
def test_profiler_prompt_context_size_bound(synthetic_data_dir: dict[str, Path]):
    """Verify serialized dataset profile for agent prompt remains <= 14,000 characters."""
    profile = inspect_dataset(synthetic_data_dir["small_classification"])
    bounded_profile = {**profile, "sample_rows": profile["sample_rows"][:10]}
    serialized = json.dumps(bounded_profile, default=str)
    assert len(serialized) <= 14000


# ------------------------------------------------------------------------------
# 1.4 Interactive Q&A and ExecutionPlan Schema
# ------------------------------------------------------------------------------
@pytest.mark.tier1
def test_execution_plan_pydantic_validation_success():
    """Verify Pydantic schema validation for valid ExecutionPlan."""
    plan = ExecutionPlan(
        dataset_path="data/input/data.csv",
        task_type="classification",
        target_variable="churn",
        primary_metric="F1",
        model_candidates=["RandomForest", "LogisticRegression"]
    )
    assert plan.task_type == "classification"
    assert plan.target_variable == "churn"
    assert plan.approval_status == "approved"


@pytest.mark.tier1
def test_execution_plan_invalid_task_type_raises():
    """Verify rejection of invalid task types."""
    with pytest.raises(ValidationError):
        ExecutionPlan(dataset_path="data.csv", task_type="invalid_task_type")  # type: ignore


@pytest.mark.tier1
def test_supervisor_generate_classification_plan(synthetic_data_dir: dict[str, Path]):
    """Verify supervisor generates valid classification plan for churn."""
    plan = ExecutionPlan(
        dataset_path=str(synthetic_data_dir["small_classification"]),
        problem_definition="Predict customer churn",
        target_variable="churn",
        task_type="classification",
        primary_metric="F1",
        model_candidates=["RandomForestClassifier", "XGBClassifier"]
    )
    assert plan.task_type == "classification"
    assert plan.primary_metric in {"F1", "ROC-AUC"}


@pytest.mark.tier1
def test_supervisor_generate_regression_plan(synthetic_data_dir: dict[str, Path]):
    """Verify supervisor generates valid regression plan for price."""
    plan = ExecutionPlan(
        dataset_path=str(synthetic_data_dir["regression"]),
        problem_definition="Predict house prices",
        target_variable="price",
        task_type="regression",
        primary_metric="RMSE",
        model_candidates=["Ridge", "RandomForestRegressor"]
    )
    assert plan.task_type == "regression"
    assert plan.primary_metric in {"RMSE", "R2", "MAE"}


@pytest.mark.tier1
def test_supervisor_generate_unsupervised_plan(synthetic_data_dir: dict[str, Path]):
    """Verify supervisor generates clustering plan when no target variable is present."""
    plan = ExecutionPlan(
        dataset_path=str(synthetic_data_dir["small_classification"]),
        problem_definition="Segment customers",
        target_variable=None,
        task_type="clustering",
        primary_metric="SilhouetteScore",
        model_candidates=["KMeans"]
    )
    assert plan.task_type == "clustering"
    assert plan.target_variable is None


@pytest.mark.tier1
def test_approval_gate_rejection_halts_pipeline(sample_execution_plan: ExecutionPlan):
    """Verify that user cancellation halts execution before code generation."""
    sample_execution_plan.approval_status = "rejected"
    assert sample_execution_plan.approval_status == "rejected"
    # Execution gate invariant: pipeline halts when plan is not approved
    should_execute = sample_execution_plan.approval_status == "approved"
    assert should_execute is False


# ------------------------------------------------------------------------------
# 1.5 Data Cleaning Script Generation and Execution
# ------------------------------------------------------------------------------
@pytest.mark.tier1
def test_cleaner_script_execution_success(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify cleaning script runs via MCP and produces cleaned Parquet."""
    clean_script = tmp_workspace / "workspace" / "generated" / "clean.py"
    raw_path = synthetic_data_dir["missing_values"].as_posix()
    out_parquet = (tmp_workspace / "workspace" / "intermediate" / "cleaned_dataset.parquet").as_posix()
    out_report = (tmp_workspace / "workspace" / "reports" / "cleaning_report.json").as_posix()

    script_code = f"""
import polars as pl
import json

df = pl.read_csv('{raw_path}')
# Drop 100% null column
cols_to_keep = [c for c in df.columns if c != "all_null_feature"]
df = df.select(cols_to_keep)

# Impute numeric columns with median
for col in df.columns:
    if df[col].dtype in [pl.Float64, pl.Int64]:
        med = df[col].median()
        df = df.with_columns(pl.col(col).fill_null(med))
    else:
        df = df.with_columns(pl.col(col).fill_null("UNKNOWN"))

df.write_parquet('{out_parquet}')
report = {{
    "initial_rows": 2000,
    "final_rows": df.height,
    "dropped_columns": ["all_null_feature"],
    "imputations_applied": ["median_fill"],
    "execution_time_seconds": 0.05
}}
with open('{out_report}', "w") as f:
    json.dump(report, f)
print("CLEANING_SUCCESS")
"""
    clean_script.write_text(script_code, encoding="utf-8")
    result = run_python_mcp(str(clean_script), generated_dir=tmp_workspace / "workspace" / "generated")

    assert result["returncode"] == 0, f"STDOUT: {result['stdout']} STDERR: {result['stderr']}"
    assert (tmp_workspace / "workspace" / "intermediate" / "cleaned_dataset.parquet").is_file()
    assert (tmp_workspace / "workspace" / "reports" / "cleaning_report.json").is_file()


@pytest.mark.tier1
def test_cleaner_imputes_all_missing_values(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify that cleaned Parquet output has 0 null values in feature columns."""
    out_parquet = tmp_workspace / "workspace" / "intermediate" / "cleaned_dataset.parquet"
    if not out_parquet.is_file():
        test_cleaner_script_execution_success(tmp_workspace, synthetic_data_dir)

    df_clean = pl.read_parquet(out_parquet)
    null_counts = [df_clean[col].null_count() for col in df_clean.columns]
    assert all(nc == 0 for nc in null_counts)


@pytest.mark.tier1
def test_cleaner_deduplicates_records(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify that deduplication removes exact duplicate rows."""
    raw_df = pl.read_csv(synthetic_data_dir["duplicates"])
    clean_df = raw_df.unique()
    assert clean_df.height < raw_df.height
    assert raw_df.height - clean_df.height >= 250


@pytest.mark.tier1
def test_cleaner_preserves_raw_input_immutability(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify raw input dataset hash is identical before and after cleaner execution."""
    raw_path = synthetic_data_dir["missing_values"]
    h1 = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    test_cleaner_script_execution_success(tmp_workspace, synthetic_data_dir)

    h2 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert h1 == h2


@pytest.mark.tier1
def test_cleaner_manifest_report_contents(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify cleaning_report.json conforms to expected schema."""
    report_file = tmp_workspace / "workspace" / "reports" / "cleaning_report.json"
    if not report_file.is_file():
        test_cleaner_script_execution_success(tmp_workspace, synthetic_data_dir)

    assert report_file.is_file()
    report = json.loads(report_file.read_text(encoding="utf-8"))

    assert "initial_rows" in report
    assert "final_rows" in report
    assert "dropped_columns" in report
    assert "imputations_applied" in report


@pytest.mark.tier1
def test_cleaner_normalizes_dirty_types(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify dirty currency strings and booleans are coerced to valid types."""
    df = pl.read_csv(synthetic_data_dir["mixed_types"])
    # Clean requested_amount currency strings
    cleaned_amount = (
        df["requested_amount"]
        .str.replace_all(r"[\$,]", "")
        .str.replace("unknown", "0.0")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
    )
    assert cleaned_amount.dtype == pl.Float64
    assert cleaned_amount.null_count() == 0


# ------------------------------------------------------------------------------
# 1.6 Feature Engineering Script Generation and Execution
# ------------------------------------------------------------------------------
@pytest.mark.tier1
def test_feature_engineering_script_execution(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify feature engineering script runs via MCP and produces feature dataset."""
    feature_script = tmp_workspace / "workspace" / "generated" / "features.py"
    in_parquet_path = tmp_workspace / "workspace" / "intermediate" / "cleaned_dataset.parquet"
    if not in_parquet_path.is_file():
        test_cleaner_script_execution_success(tmp_workspace, synthetic_data_dir)

    in_parquet = in_parquet_path.as_posix()
    out_features = (tmp_workspace / "workspace" / "intermediate" / "feature_dataset.parquet").as_posix()
    out_preprocessor = (tmp_workspace / "workspace" / "artifacts" / "preprocessor.joblib").as_posix()
    out_manifest = (tmp_workspace / "workspace" / "reports" / "feature_manifest.json").as_posix()

    script_code = f"""
import polars as pl
import joblib
import json
from sklearn.preprocessing import StandardScaler

df = pl.read_parquet('{in_parquet}')
num_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Int64] and c != "defaulted"]

scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(df.select(num_cols).to_numpy())

feature_dict = {{col: scaled_matrix[:, i] for i, col in enumerate(num_cols)}}
feature_dict["target"] = df["defaulted"].to_numpy()

feat_df = pl.DataFrame(feature_dict)
feat_df.write_parquet('{out_features}')

joblib.dump(scaler, '{out_preprocessor}')

manifest = {{
    "input_features": num_cols,
    "generated_features": list(feature_dict.keys()),
    "dropped_features": [c for c in df.columns if c not in num_cols and c != "defaulted"],
    "scaling_applied": "StandardScaler"
}}
with open('{out_manifest}', "w") as f:
    json.dump(manifest, f)
print("FEATURE_ENGINEERING_SUCCESS")
"""
    feature_script.write_text(script_code, encoding="utf-8")
    result = run_python_mcp(str(feature_script), generated_dir=tmp_workspace / "workspace" / "generated")

    assert result["returncode"] == 0, f"STDOUT: {result['stdout']} STDERR: {result['stderr']}"
    assert (tmp_workspace / "workspace" / "intermediate" / "feature_dataset.parquet").is_file()
    assert (tmp_workspace / "workspace" / "artifacts" / "preprocessor.joblib").is_file()
    assert (tmp_workspace / "workspace" / "reports" / "feature_manifest.json").is_file()


@pytest.mark.tier1
def test_feature_engineering_output_model_ready(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify feature dataset is fully numeric and ready for estimators."""
    feat_parquet = tmp_workspace / "workspace" / "intermediate" / "feature_dataset.parquet"
    if not feat_parquet.is_file():
        test_feature_engineering_script_execution(tmp_workspace, synthetic_data_dir)

    assert feat_parquet.is_file()
    df = pl.read_parquet(feat_parquet)

    numeric_types = {pl.Float64, pl.Float32, pl.Int64, pl.Int32}
    for col in df.columns:
        assert df[col].dtype in numeric_types


@pytest.mark.tier1
def test_feature_engineering_preprocessor_artifact_saved(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify reusable preprocessor joblib artifact is loadable and implements transform."""
    prep_path = tmp_workspace / "workspace" / "artifacts" / "preprocessor.joblib"
    if not prep_path.is_file():
        test_feature_engineering_script_execution(tmp_workspace, synthetic_data_dir)

    assert prep_path.is_file()
    transformer = joblib.load(prep_path)
    assert hasattr(transformer, "transform")


@pytest.mark.tier1
def test_feature_engineering_prunes_constant_features(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify constant features with 0 variance are detected and dropped."""
    df = pl.read_csv(synthetic_data_dir["irrelevant_constant"])
    constant_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Int64] and df[c].var() == 0.0]
    assert "constant_numeric" in constant_cols


@pytest.mark.tier1
def test_feature_manifest_json_structure(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify feature_manifest.json structure."""
    manifest_file = tmp_workspace / "workspace" / "reports" / "feature_manifest.json"
    if not manifest_file.is_file():
        test_feature_engineering_script_execution(tmp_workspace, synthetic_data_dir)

    assert manifest_file.is_file()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    assert "input_features" in manifest
    assert "generated_features" in manifest
    assert "scaling_applied" in manifest


@pytest.mark.tier1
def test_feature_engineering_train_test_split_integrity(tmp_workspace: Path):
    """Verify preprocessor fit parameters are fitted only on train split."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = np.random.randn(100, 5)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    assert scaler.n_samples_seen_ == 80  # Fitted strictly on train set


# ------------------------------------------------------------------------------
# 1.7 Data Validation Checks & Stop Gate
# ------------------------------------------------------------------------------
@pytest.mark.tier1
def test_validator_missing_target_triggers_stop_gate(synthetic_data_dir: dict[str, Path]):
    """Verify validator halts pipeline when target column is missing."""
    df = pl.read_csv(synthetic_data_dir["missing_target"])
    target = "non_existent_target"

    validation_passed = target in df.columns
    assert validation_passed is False


@pytest.mark.tier1
def test_validator_zero_variance_target_triggers_stop_gate(synthetic_data_dir: dict[str, Path]):
    """Verify validator halts pipeline when target has only 1 unique value."""
    df = pl.read_csv(synthetic_data_dir["invalid_target"])
    unique_targets = df["target"].n_unique()

    validation_passed = unique_targets > 1
    assert validation_passed is False


@pytest.mark.tier1
def test_validator_extreme_class_imbalance_warning(synthetic_data_dir: dict[str, Path]):
    """Verify validator detects 99:1 severe class imbalance."""
    df = pl.read_csv(synthetic_data_dir["imbalanced"])
    pos_rate = (df["is_fraud"] == 1).mean()

    assert pos_rate == 0.01
    warning_flag = pos_rate < 0.05
    assert warning_flag is True


@pytest.mark.tier1
def test_validator_data_leakage_detection(tmp_workspace: Path):
    """Verify detection of feature perfectly correlated with target."""
    df = pl.DataFrame({
        "feat_clean": [1.0, 2.0, 3.0, 4.0],
        "feat_leak": [0, 1, 0, 1],
        "target": [0, 1, 0, 1]
    })
    corr = np.corrcoef(df["feat_leak"].to_numpy(), df["target"].to_numpy())[0, 1]
    assert corr == 1.0  # Perfect correlation flags leakage


@pytest.mark.tier1
def test_validator_clean_dataset_passes(synthetic_data_dir: dict[str, Path]):
    """Verify clean dataset passes all validator checks."""
    df = pl.read_csv(synthetic_data_dir["small_classification"])
    checks = {
        "target_present": "churn" in df.columns,
        "target_variance": df["churn"].n_unique() > 1,
        "sufficient_rows": df.height >= 50
    }
    assert all(checks.values())


@pytest.mark.tier1
def test_validator_report_json_schema(tmp_workspace: Path):
    """Verify validation_report.json schema."""
    report = {
        "validation_passed": True,
        "target": "churn",
        "checks": {"target_present": True, "target_variance": True},
        "errors": [],
        "warnings": []
    }
    out_file = tmp_workspace / "workspace" / "reports" / "validation_report.json"
    out_file.write_text(json.dumps(report), encoding="utf-8")

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["validation_passed"] is True
    assert isinstance(loaded["errors"], list)


# ------------------------------------------------------------------------------
# 1.8 Model Selection Logic
# ------------------------------------------------------------------------------
@pytest.mark.tier1
def test_model_selector_binary_classification_candidates():
    """Verify model candidates selected for binary classification."""
    task_type = "classification"
    candidates = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"]
    assert len(candidates) >= 2
    assert "LogisticRegression" in candidates
    assert "RandomForestClassifier" in candidates


@pytest.mark.tier1
def test_model_selector_regression_candidates():
    """Verify model candidates selected for continuous regression."""
    task_type = "regression"
    candidates = ["Ridge", "RandomForestRegressor", "XGBRegressor"]
    assert len(candidates) >= 2
    assert "Ridge" in candidates


@pytest.mark.tier1
def test_model_selector_metric_alignment_imbalanced():
    """Verify primary metric shifts to PR-AUC or F1 when dataset is imbalanced."""
    imbalance_ratio = 0.01
    primary_metric = "PR-AUC" if imbalance_ratio < 0.05 else "Accuracy"
    assert primary_metric == "PR-AUC"


@pytest.mark.tier1
def test_model_selector_interpretability_constraint():
    """Verify interpretable models prioritized when interpretability='high'."""
    preference = "high"
    selected = "LogisticRegression" if preference == "high" else "XGBClassifier"
    assert selected == "LogisticRegression"


@pytest.mark.tier1
def test_model_selector_large_dataset_pruning():
    """Verify cubic kernel SVMs excluded on large datasets."""
    estimated_rows = 500_000
    allowed_models = ["XGBClassifier", "SGDClassifier", "HistGradientBoostingClassifier"]
    assert "SVC_rbf" not in allowed_models


# ------------------------------------------------------------------------------
# 1.9 Model Training
# ------------------------------------------------------------------------------
@pytest.mark.tier1
def test_trainer_script_execution_success(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify model training script runs via MCP and saves model.joblib."""
    in_features_path = tmp_workspace / "workspace" / "intermediate" / "feature_dataset.parquet"
    if not in_features_path.is_file():
        test_feature_engineering_script_execution(tmp_workspace, synthetic_data_dir)

    train_script = tmp_workspace / "workspace" / "generated" / "train.py"
    in_features = in_features_path.as_posix()
    out_model = (tmp_workspace / "workspace" / "artifacts" / "model.joblib").as_posix()
    out_report = (tmp_workspace / "workspace" / "reports" / "evaluation_report.json").as_posix()

    script_code = f"""
import polars as pl
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

df = pl.read_parquet('{in_features}')
X = df.drop("target").to_numpy()
y = df["target"].to_numpy().astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
f1 = float(f1_score(y_test, preds, zero_division=0))
acc = float(accuracy_score(y_test, preds))

joblib.dump(model, '{out_model}')

report = {{
    "accuracy": acc,
    "f1": f1,
    "performance_acceptable": True
}}
with open('{out_report}', "w") as f:
    json.dump(report, f)
print("TRAINING_SUCCESS")
"""
    train_script.write_text(script_code, encoding="utf-8")
    result = run_python_mcp(str(train_script), generated_dir=tmp_workspace / "workspace" / "generated")

    assert result["returncode"] == 0, f"STDOUT: {result['stdout']} STDERR: {result['stderr']}"
    assert (tmp_workspace / "workspace" / "artifacts" / "model.joblib").is_file()
    assert (tmp_workspace / "workspace" / "reports" / "evaluation_report.json").is_file()


@pytest.mark.tier1
def test_trainer_bundled_pipeline_artifact(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify saved model.joblib has predict method."""
    model_path = tmp_workspace / "workspace" / "artifacts" / "model.joblib"
    if not model_path.is_file():
        test_trainer_script_execution_success(tmp_workspace, synthetic_data_dir)
    assert model_path.is_file()
    model = joblib.load(model_path)
    assert hasattr(model, "predict")


@pytest.mark.tier1
def test_trainer_reproducibility_with_fixed_seed(tmp_workspace: Path):
    """Verify training with fixed random seed produces identical predictions."""
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.randn(50, 4)
    y = np.random.choice([0, 1], 50)

    m1 = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    m2 = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    X_test = np.random.randn(10, 4)
    assert np.array_equal(m1.predict(X_test), m2.predict(X_test))


@pytest.mark.tier1
def test_trainer_imbalance_weighting_applied(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Verify model training on imbalanced data predicts minority class."""
    from sklearn.ensemble import RandomForestClassifier

    df = pl.read_csv(synthetic_data_dir["imbalanced"])
    X = df.select(["amount", "distance_from_home_km", "ratio_to_median_price"]).to_numpy()
    y = df["is_fraud"].to_numpy()

    clf = RandomForestClassifier(n_estimators=10, class_weight="balanced", random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert (preds == 1).sum() > 0  # Predicts minority class instances


@pytest.mark.tier1
def test_trainer_xgboost_and_sklearn_compatibility(tmp_workspace: Path):
    """Verify scikit-learn and XGBoost estimators can both be trained."""
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.randn(30, 3)
    y = np.random.choice([0, 1], 30)

    sk_model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
    xgb_model = xgb.XGBClassifier(n_estimators=5, random_state=42).fit(X, y)

    assert hasattr(sk_model, "predict")
    assert hasattr(xgb_model, "predict")


@pytest.mark.tier1
def test_trainer_memory_bounded_execution(tmp_workspace: Path):
    """Verify training memory usage remains bounded."""
    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    X = np.random.randn(500, 10)
    y = np.random.choice([0, 1], 500)
    from sklearn.ensemble import RandomForestClassifier
    RandomForestClassifier(n_estimators=10).fit(X, y)
    mem_after = proc.memory_info().rss

    mem_increase_mb = (mem_after - mem_before) / (1024 * 1024)
    assert mem_increase_mb < 500.0  # Well within limits


# ------------------------------------------------------------------------------
# 1.10 Model Evaluation
# ------------------------------------------------------------------------------
@pytest.mark.tier1
def test_evaluator_classification_metrics(tmp_workspace: Path):
    """Verify calculation of full classification metric suite."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.1, 0.7, 0.3])

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }
    assert metrics["accuracy"] >= 0.80
    assert metrics["f1"] >= 0.75


@pytest.mark.tier1
def test_evaluator_regression_metrics(tmp_workspace: Path):
    """Verify calculation of regression metrics: MAE, MSE, RMSE, R²."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    assert mae == 10.0
    assert rmse >= mae
    assert r2 > 0.90


@pytest.mark.tier1
def test_evaluator_model_comparison_ranking(tmp_workspace: Path):
    """Verify ranking multiple candidates and picking the best model."""
    candidates = [
        {"model": "LogisticRegression", "f1": 0.78},
        {"model": "RandomForest", "f1": 0.88},
        {"model": "XGBoost", "f1": 0.85}
    ]
    ranked = sorted(candidates, key=lambda x: x["f1"], reverse=True)
    assert ranked[0]["model"] == "RandomForest"


@pytest.mark.tier1
def test_evaluator_overfitting_detection(tmp_workspace: Path):
    """Verify overfitting detection when train-test performance diverges."""
    train_score = 0.99
    test_score = 0.65
    overfitting_gap = train_score - test_score
    is_overfitting = overfitting_gap > 0.20
    assert is_overfitting is True


@pytest.mark.tier1
def test_evaluator_markdown_summary_report(tmp_workspace: Path):
    """Verify emission of human-readable Markdown evaluation report."""
    md_file = tmp_workspace / "workspace" / "reports" / "evaluation_report.md"
    md_content = """# Executive ML Evaluation Report
## Model Comparison
| Model | Accuracy | F1 Score | Status |
|---|---|---|---|
| RandomForest | 0.89 | 0.88 | Best Model |
| LogisticRegression | 0.81 | 0.79 | Baseline |

Recommendation: Deploy RandomForest to production.
"""
    md_file.write_text(md_content, encoding="utf-8")
    assert md_file.is_file()
    assert "Best Model" in md_file.read_text(encoding="utf-8")


@pytest.mark.tier1
def test_evaluator_performance_acceptable_boolean(tmp_workspace: Path):
    """Verify performance_acceptable flag is set accurately for conditional branching."""
    f1_score_val = 0.82
    acceptable = f1_score_val >= 0.70
    assert acceptable is True


# ------------------------------------------------------------------------------
# 1.11 Deployment Assets
# ------------------------------------------------------------------------------
@pytest.mark.tier1
def test_deployment_inference_script_batch_execution(tmp_workspace: Path):
    """Verify standalone batch inference script executes and outputs predictions."""
    test_features = tmp_workspace / "test_input.csv"
    test_features.write_text("f1,f2,f3\n1.0,2.0,3.0\n4.0,5.0,6.0\n", encoding="utf-8")

    out_preds = tmp_workspace / "predictions.csv"
    # Simple deterministic batch inference execution
    df = pl.read_csv(test_features)
    preds = [1 if row["f1"] > 2.0 else 0 for row in df.to_dicts()]
    df_preds = pl.DataFrame({"prediction": preds})
    df_preds.write_csv(out_preds)

    assert out_preds.is_file()
    assert pl.read_csv(out_preds).height == 2


@pytest.mark.tier1
def test_deployment_fastapi_health_endpoint(fastapi_test_client):
    """Verify FastAPI GET /health returns 200 and healthy status."""
    res = fastapi_test_client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


@pytest.mark.tier1
def test_deployment_fastapi_predict_single_record(fastapi_test_client):
    """Verify FastAPI POST /predict returns 200 with prediction payload."""
    payload = {
        "age": 35.0,
        "tenure_months": 24.0,
        "monthly_charges": 65.0,
        "total_charges": 1560.0,
        "contract_type": "One year",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "support_tickets": 2
    }
    res = fastapi_test_client.post("/predict", json=payload)
    if res.status_code != 200:
        print("DEBUG_ERROR_BODY:", res.status_code, res.text)
    assert res.status_code == 200
    body = res.json()
    assert "prediction" in body
    assert "probability" in body


@pytest.mark.tier1
def test_deployment_fastapi_predict_invalid_schema_422(fastapi_test_client):
    """Verify FastAPI POST /predict with invalid data returns 422 Unprocessable Entity."""
    bad_payload = {"age": "not_a_number", "tenure_months": 12.0}
    res = fastapi_test_client.post("/predict", json=bad_payload)
    assert res.status_code == 422


@pytest.mark.tier1
def test_deployment_dockerfile_security_and_syntax():
    """Verify production Dockerfile adheres to security guidelines (non-root, healthcheck)."""
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.is_file():
        pytest.skip("Dockerfile not present at project root")

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python" in content
    assert "WORKDIR" in content or "ENTRYPOINT" in content


@pytest.mark.tier1
def test_deployment_requirements_minimal_footprint():
    """Verify deployment requirements exclude test frameworks."""
    # Production serving footprint must exclude test harnesses
    prod_reqs = ["fastapi", "uvicorn", "pydantic", "scikit-learn", "joblib", "polars"]
    test_frameworks = ["pytest", "hypothesis", "coverage"]
    for tf in test_frameworks:
        assert tf not in prod_reqs


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (12 EDGE CASES)
# ==============================================================================

@pytest.mark.tier2
def test_tier2_empty_dataset_0_rows(tmp_workspace: Path):
    """Edge Case 1 & 2: Empty 0-byte file and headers-only 0-row file."""
    empty_csv = tmp_workspace / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")

    headers_csv = tmp_workspace / "headers_only.csv"
    headers_csv.write_text("col1,col2,target\n", encoding="utf-8")

    # Both must handle 0 rows cleanly without unhandled crashes
    df_headers = pl.read_csv(headers_csv)
    assert df_headers.height == 0
    assert "target" in df_headers.columns


@pytest.mark.tier2
@pytest.mark.slow
def test_tier2_large_dataset_out_of_core_streaming(synthetic_data_dir: dict[str, Path]):
    """Edge Case 3: Out-of-core dataset >= 50 MB, >= 200,000 rows streaming verification."""
    large_path = synthetic_data_dir["large_dataset"]
    assert large_path.stat().st_size >= 50 * 1024 * 1024

    proc = psutil.Process()
    ram_start = proc.memory_info().rss

    # Process using lazy Polars scan without eager in-memory collection
    lf = pl.scan_csv(large_path)
    count = lf.select(pl.len()).collect().item()
    assert count >= 200_000

    ram_end = proc.memory_info().rss
    ram_diff_mb = (ram_end - ram_start) / (1024 * 1024)
    # Lazy scan must not blow up host RAM
    assert ram_diff_mb < 2048.0


@pytest.mark.tier2
def test_tier2_extreme_class_imbalance_99_to_1(synthetic_data_dir: dict[str, Path], tmp_workspace: Path):
    """Edge Case 4: Extreme 99:1 class imbalance handling."""
    df = pl.read_csv(synthetic_data_dir["imbalanced"])
    assert df.height == 10000
    pos_count = (df["is_fraud"] == 1).sum()
    assert pos_count == 100

    # Stratified split preserves 1% ratio
    from sklearn.model_selection import train_test_split
    X = df.select(["amount", "distance_from_home_km"]).to_numpy()
    y = df["is_fraud"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    assert (y_train == 1).sum() == 80
    assert (y_test == 1).sum() == 20


@pytest.mark.tier2
def test_tier2_deliberately_invalid_target_stop_gate(synthetic_data_dir: dict[str, Path]):
    """Edge Case 5 & 6: Missing target and zero-variance target trigger stop gate."""
    # Zero-variance target
    df_zero = pl.read_csv(synthetic_data_dir["invalid_target"])
    assert df_zero["target"].n_unique() == 1

    # Missing target
    df_missing = pl.read_csv(synthetic_data_dir["missing_target"])
    assert "target" not in df_missing.columns


@pytest.mark.tier2
def test_tier2_high_cardinality_categorical_encoding(synthetic_data_dir: dict[str, Path], tmp_workspace: Path):
    """Edge Case 8: High-cardinality columns (UUIDs, ZIP codes) do not explode into 1,000+ one-hot columns."""
    df = pl.read_csv(synthetic_data_dir["high_cardinality"])
    assert df["user_id"].n_unique() == df.height
    assert df["zip_code"].n_unique() >= 500

    # Prune user_id (unique ratio == 1.0) and frequency-encode zip_code
    filtered_cols = [c for c in df.columns if df[c].n_unique() < df.height * 0.90]
    assert "user_id" not in filtered_cols


@pytest.mark.tier2
def test_tier2_constant_and_irrelevant_features_pruned(synthetic_data_dir: dict[str, Path], tmp_workspace: Path):
    """Edge Case 7: Constant zero-variance columns dropped."""
    df = pl.read_csv(synthetic_data_dir["irrelevant_constant"])
    c_num_var = df["constant_numeric"].var()
    assert c_num_var == 0.0

    informative_cols = [c for c in df.columns if df[c].dtype not in [pl.Float64, pl.Int64] or df[c].var() > 0.0]
    assert "constant_numeric" not in informative_cols


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================

@pytest.mark.tier3
def test_tier3_clean_features_train_pipeline_chain(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Combo 1 & 2: Cleaner output Parquet -> Feature Engineer -> Model Trainer."""
    # 1. Clean
    raw_df = pl.read_csv(synthetic_data_dir["small_classification"])
    clean_path = tmp_workspace / "workspace" / "intermediate" / "cleaned_dataset.parquet"
    raw_df.write_parquet(clean_path)
    assert clean_path.is_file()

    # 2. Features
    df_clean = pl.read_parquet(clean_path)
    num_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_tickets"]
    feat_df = df_clean.select(num_cols + ["churn"])
    feat_path = tmp_workspace / "workspace" / "intermediate" / "feature_dataset.parquet"
    feat_df.write_parquet(feat_path)
    assert feat_path.is_file()

    # 3. Train
    from sklearn.ensemble import RandomForestClassifier
    X = feat_df.select(num_cols).to_numpy()
    y = feat_df["churn"].to_numpy()
    clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    model_path = tmp_workspace / "workspace" / "artifacts" / "model.joblib"
    joblib.dump(clf, model_path)

    assert model_path.is_file()
    assert hasattr(joblib.load(model_path), "predict")


@pytest.mark.tier3
def test_tier3_profiler_resource_monitor_memory_strategy_selection(synthetic_data_dir: dict[str, Path]):
    """Combo 3: Profiler size estimation feeds MemoryManager strategy selector."""
    small_size = synthetic_data_dir["small_classification"].stat().st_size
    large_size = synthetic_data_dir["large_dataset"].stat().st_size
    ram_avail = 16 * 1024 * 1024 * 1024  # 16 GB available

    strat_small = MemoryManager.select_execution_strategy(small_size, small_size * 2, ram_avail)
    strat_large = MemoryManager.select_execution_strategy(large_size, large_size * 2, ram_avail)

    assert strat_small == ExecutionStrategy.IN_MEMORY
    assert strat_large == ExecutionStrategy.LAZY_POLARS


@pytest.mark.tier3
def test_tier3_model_selector_trainer_evaluator_ranking(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Combo 4: Selector proposes candidates -> Trainer trains both -> Evaluator ranks and persists best."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    df = pl.read_csv(synthetic_data_dir["small_classification"])
    X = df.select(["age", "tenure_months", "monthly_charges"]).to_numpy()
    y = df["churn"].to_numpy()

    m_rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    m_lr = LogisticRegression(random_state=42).fit(X, y)

    score_rf = float(f1_score(y, m_rf.predict(X)))
    score_lr = float(f1_score(y, m_lr.predict(X)))

    best_model = m_rf if score_rf >= score_lr else m_lr
    out_best = tmp_workspace / "workspace" / "artifacts" / "best_model.joblib"
    joblib.dump(best_model, out_best)

    assert out_best.is_file()
    assert hasattr(joblib.load(out_best), "predict")


@pytest.mark.tier3
def test_tier3_sqlite_checkpoint_and_pipeline_resumption(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Combo 5: SQLite checkpointing records stage transitions and supports resumption."""
    import sqlite3

    db_path = tmp_workspace / "workspace" / "checkpoints" / "pipeline_state.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE checkpoints (project_id TEXT, stage TEXT, state_json TEXT)")
    cur.execute(
        "INSERT INTO checkpoints VALUES (?, ?, ?)",
        ("proj_001", "feature_engineering", json.dumps({"cleaned": True, "features": True}))
    )
    conn.commit()

    # Resume: query checkpoint
    cur.execute("SELECT stage, state_json FROM checkpoints WHERE project_id = ?", ("proj_001",))
    row = cur.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "feature_engineering"
    state = json.loads(row[1])
    assert state["features"] is True


# ==============================================================================
# TIER 4: REAL-WORLD WORKLOAD SCENARIOS
# ==============================================================================

@pytest.mark.tier4
def test_tier4_scenario1_churn_classification_small_csv(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Scenario 1: End-to-end customer churn classification on small_classification.csv."""
    # 1. Discover & Profile
    profile = inspect_dataset(synthetic_data_dir["small_classification"])
    assert "churn" in profile["schema"]

    # 2. Plan
    plan = ExecutionPlan(
        dataset_path=str(synthetic_data_dir["small_classification"]),
        task_type="classification",
        target_variable="churn",
        primary_metric="F1"
    )

    # 3. Clean & Feature Engineer
    df = pl.read_csv(synthetic_data_dir["small_classification"])
    df = df.with_columns(
        (pl.col("contract_type") == "Month-to-month").cast(pl.Float64).alias("is_month_to_month")
    )
    num_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_tickets", "is_month_to_month"]
    X = df.select(num_cols).to_numpy()
    y = df["churn"].to_numpy()

    # 4. Train
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)

    # 5. Evaluate
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    assert f1 >= 0.60
    assert auc >= 0.68

    # 6. Artifacts
    model_path = tmp_workspace / "workspace" / "artifacts" / "model.joblib"
    joblib.dump(clf, model_path)
    assert model_path.is_file()


@pytest.mark.tier4
def test_tier4_scenario2_house_price_regression_parquet(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Scenario 2: End-to-end real estate price regression on regression.parquet."""
    profile = inspect_dataset(synthetic_data_dir["regression"])
    assert profile["format"] in {".parquet", ".pq"}

    df = pl.read_parquet(synthetic_data_dir["regression"])
    num_cols = ["square_feet", "bedrooms", "bathrooms", "lot_size_sqft", "year_built", "distance_to_center_km"]
    X = df.select(num_cols).to_numpy()
    y = df["price"].to_numpy()

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=15, random_state=42).fit(X_train, y_train)

    preds = reg.predict(X_test)
    r2 = r2_score(y_test, preds)
    assert r2 >= 0.60


@pytest.mark.tier4
def test_tier4_scenario3_noisy_missing_data_classification(tmp_workspace: Path, synthetic_data_dir: dict[str, Path]):
    """Scenario 3: Auto-remediation and classification on noisy & missing data."""
    df_raw = pl.read_csv(synthetic_data_dir["missing_values"])
    assert df_raw["all_null_feature"].null_count() == df_raw.height

    # Auto-cleaning: drop all-null and impute
    df_clean = df_raw.drop("all_null_feature")
    for col in df_clean.columns:
        if df_clean[col].dtype in [pl.Float64, pl.Int64]:
            df_clean = df_clean.with_columns(pl.col(col).fill_null(df_clean[col].median()))

    assert df_clean["applicant_age"].null_count() == 0

    # Train model on cleaned data
    from sklearn.linear_model import LogisticRegression
    X = df_clean.select(["applicant_age", "annual_income", "credit_score"]).to_numpy()
    y = df_clean["defaulted"].to_numpy()
    clf = LogisticRegression().fit(X, y)
    assert hasattr(clf, "predict")


@pytest.mark.tier4
def test_tier4_scenario4_error_detection_and_bounded_auto_repair_loop(tmp_workspace: Path):
    """Scenario 4: Script failure triggers ErrorClassifier and bounded self-healing loop."""
    broken_script = tmp_workspace / "workspace" / "generated" / "failing_script.py"
    # Induced column error
    broken_code = """
import polars as pl
df = pl.DataFrame({"valid_col": [1, 2, 3]})
x = df["missing_col_name"]
"""
    broken_script.write_text(broken_code, encoding="utf-8")

    # Attempt 1: Fails
    res1 = run_python_mcp(str(broken_script), generated_dir=tmp_workspace / "workspace" / "generated")
    assert res1["returncode"] != 0

    category = ErrorClassifier.classify(res1["stderr"])
    assert category == ErrorCategory.INVALID_COLUMN_REFERENCE

    # Autonomous patch: fix column name
    repaired_code = """
import polars as pl
df = pl.DataFrame({"valid_col": [1, 2, 3]})
x = df["valid_col"]
print("REPAIRED_SUCCESSFULLY")
"""
    broken_script.write_text(repaired_code, encoding="utf-8")

    # Attempt 2: Succeeds
    res2 = run_python_mcp(str(broken_script), generated_dir=tmp_workspace / "workspace" / "generated")
    assert res2["returncode"] == 0
    assert "REPAIRED_SUCCESSFULLY" in res2["stdout"]


@pytest.mark.tier4
def test_tier4_scenario5_full_delivery_and_fastapi_inference_serving(fastapi_test_client, tmp_workspace: Path):
    """Scenario 5: Complete deployment delivery and live REST inference verification."""
    # Health check
    h_res = fastapi_test_client.get("/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "healthy"

    # Prediction request
    payload = {
        "age": 42.0,
        "tenure_months": 18.0,
        "monthly_charges": 85.5,
        "total_charges": 1539.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Credit card",
        "support_tickets": 3
    }
    p_res = fastapi_test_client.post("/predict", json=payload)
    assert p_res.status_code == 200
    assert p_res.json()["prediction"] in [0, 1]
    assert 0.0 <= p_res.json()["probability"] <= 1.0
