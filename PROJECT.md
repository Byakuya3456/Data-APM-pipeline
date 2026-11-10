# Project: Autonomous MCP-Driven Data Intelligence & ML Engineering Platform

## Architecture
The platform is a local-first, autonomous machine learning engineering system with a worker-supervisor architecture orchestrated via LangGraph, communicating through the Model Context Protocol (MCP), and executing heavy data workloads through Polars and DuckDB out-of-core engines.

```
USER
 │
 ▼
CLI INTERFACE (main.py — 6 Interactive Options)
 │
 ▼
SUPERVISOR AGENT (agents/supervisor.py)
 ├── Dataset Discovery & Profiling (engine/data_profiler.py)
 ├── Multi-turn Q&A & Requirement Discovery
 ├── Structured Execution Plan (Pydantic ExecutionPlan)
 └── Human Approval Gate [Approve / Modify / Cancel]
 │
 ▼
LANGGRAPH ORCHESTRATION (orchestration/graph.py + SQLite Checkpointing)
 ├── DATA CLEANING WORKER (agents/cleaner_agent.py)
 ├── DATA VALIDATION WORKER (agents/validator_agent.py) [Gate: Stop on Failure]
 ├── FEATURE ENGINEERING WORKER (agents/feature_agent.py)
 ├── MODEL SELECTION WORKER (agents/model_selector_agent.py)
 ├── MODEL TRAINING WORKER (agents/trainer_agent.py)
 ├── MODEL EVALUATION WORKER (agents/evaluator_agent.py) [Branch: Acceptable? Yes/No]
 └── DEPLOYMENT WORKER (agents/deployment_agent.py)
 │
 ▼
MCP EXECUTION LAYER (mcp/client.py & mcp/executor.py)
 ├── Filesystem MCP Server (mcp/filesystem_server.py / npx @modelcontextprotocol/server-filesystem)
 ├── Command Execution MCP Server (mcp/command_server.py — MCPServer with 4 tools)
 └── Autonomous Repair Loop (engine/error_classifier.py — 10 categories, max 2-3 retries)
 │
 ▼
HIGH-VOLUME DATA ENGINE (engine/)
 ├── Polars LazyFrame Engine (streaming, sink_parquet, predicate pushdown)
 ├── DuckDB Query Engine (out-of-core memory_limit, temp_directory, SUMMARIZE)
 ├── Resource Monitor (RAM, CPU, NVIDIA RTX 3050 Laptop GPU 6GB telemetry)
 ├── Memory Manager (Execution Strategy Selector: IN_MEMORY, LAZY, DUCKDB, CHUNKED)
 └── Artifact Manager (intermediate Parquet, SHA256 manifests, resumption state)
 │
 ▼
DELIVERY ARTIFACTS (workspace/artifacts/ & deployment/)
 ├── Cleaned & Feature Datasets (Parquet)
 ├── Bundled Preprocessor & Model Artifact (Joblib)
 ├── Production FastAPI Server & Dockerfile
 ├── Standalone inference.py
 └── Executive Markdown Summary & Evaluation Report
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Dependency & Project Configuration | `pyproject.toml` with `uv`, adding `pydantic-settings`, `pytest`, `fastapi`, `uvicorn` | M1 | survey |
| 2 | Application Settings Management | `config/settings.py` via `pydantic-settings`, dynamic root discovery, `.env` support | M1 | survey |
| 3 | Structured Dual Logging | `config/logging_config.py` with colorized console output and rotating file logs | M1 | survey |
| 4 | MCP Filesystem Server | `mcp/filesystem_server.py` pure-Python implementation + config for official npx server | M1 | survey |
| 5 | MCP Command Execution Server | `mcp/command_server.py` using `MCPServer`, 4 tools, subprocess shell=False, AST check, timeouts, process tracking | M1 | survey |
| 6 | MCP Client & Stdio Transport | `mcp/client.py` and `mcp/executor.py` managing stdio sessions, discovery, reconnection, structured outputs | M1 | survey |
| 7 | Foundation Unit Tests | `tests/test_mcp.py` verifying MCP stdio protocol, tool discovery, execution security, and limits | M1 | survey |
| 8 | System Resource Telemetry | `engine/resource_monitor.py` querying RAM, CPU, NVIDIA RTX 3050 GPU VRAM, disk headroom | M2 | survey |
| 9 | Multi-Format Dataset Discovery | `engine/data_profiler.py` discovering CSV, TSV, Parquet, JSON, JSONL without full in-memory scans | M2 | survey |
| 10 | Polars Lazy Profiling | `engine/data_profiler.py` & `engine/polars_engine.py` bounded preview (<=100 rows), null counts, types, memory estimate | M2 | survey |
| 11 | DuckDB Out-of-Core Querying | `engine/duckdb_engine.py` out-of-core SQL queries, temp disk spilling, SUMMARIZE statistics | M2 | survey |
| 12 | Size-Aware Execution Selector | `engine/memory_manager.py` choosing in-memory vs lazy vs DuckDB vs chunked processing strategy | M2 | survey |
| 13 | Artifact & Storage Management | `engine/artifact_manager.py` managing intermediate Parquet artifacts, SHA256 manifests, resumption markers | M2 | survey |
| 14 | Profiler & Engine Tests | `tests/test_profiler.py` and `tests/test_resource_limits.py` | M2 | survey |
| 15 | Supervisor Interactive Q&A | `agents/supervisor.py` multi-turn interactive requirement discovery, target detection, fact/assumption split | M3 | survey |
| 16 | Structured Execution Plan | `agents/supervisor.py` Pydantic `ExecutionPlan` generation and human approval gate [Approve/Modify/Cancel] | M3 | survey |
| 17 | LangGraph Pipeline State Schema | `orchestration/state.py` Pydantic state schema tracking stages, artifacts, metrics, and error logs | M3 | survey |
| 18 | LangGraph Conditional Routing | `orchestration/routing.py` routing task branches (classification/regression/clustering/analysis) and gates | M3 | survey |
| 19 | Persistent SQLite Checkpointing | `orchestration/checkpoint.py` SQLite-backed checkpointer storing graph checkpoints for resumption | M3 | survey |
| 20 | Compiled Pipeline StateGraph | `orchestration/graph.py` compiling nodes, conditional edges, approval interrupts, and checkpointing | M3 | survey |
| 21 | Code Generation Templates | `templates/` cleaning, feature engineering, training, inference, FastAPI, and Dockerfile templates | M4 | survey |
| 22 | Data Cleaner Worker Agent | `agents/cleaner_agent.py` generating `cleaned_dataset.parquet`, `cleaning_report.json`, `cleaning_script.py` | M4 | survey |
| 23 | Feature Engineering Worker Agent | `agents/feature_agent.py` generating `feature_dataset.parquet`, `feature_manifest.json`, `feature_engineering.py` | M4 | survey |
| 24 | Data Validation Worker Agent | `agents/validator_agent.py` validating target availability, leakage, split feasibility, imbalance; gate stop | M4 | survey |
| 25 | Model Selection Worker Agent | `agents/model_selector_agent.py` selecting justified algorithm candidates based on dataset and resource limits | M4 | survey |
| 26 | Model Training Worker Agent | `agents/trainer_agent.py` reproducible training, CV, hyperparameter tuning, bundled preprocessor + model | M4 | survey |
| 27 | Model Evaluation Worker Agent | `agents/evaluator_agent.py` comprehensive metrics, overfitting/leakage detection, best model recommendation | M4 | survey |
| 28 | Deployment Worker Agent | `agents/deployment_agent.py` generating `inference.py`, FastAPI `serve_api.py`, `requirements.txt`, `Dockerfile` | M4 | survey |
| 29 | Autonomous Error Classifier & Repair | `engine/error_classifier.py` classifying 10 error categories, memory-aware remediation, max 2-3 retries | M5 | survey |
| 30 | Production FastAPI & Docker Delivery | `deployment/` FastAPI server with health checks, Pydantic schemas, and containerization assets | M5 | survey |
| 31 | Executive Reporting & Summaries | `engine/artifact_manager.py` / `agents/evaluator_agent.py` generating executive markdown project reports | M5 | survey |
| 32 | Interactive 6-Option CLI | `main.py` interactive terminal menu (New, Resume, Inspect, Reports, Resources, Exit) | M5 | survey |
| 33 | Synthetic Test Dataset Generator | `tests/generate_test_datasets.py` programmatically generating all 10 synthetic test scenarios | E2E | survey |
| 34 | E2E Requirement-Driven Test Suite | `tests/test_pipeline.py`, `TEST_INFRA.md`, `TEST_READY.md` covering Tiers 1-4 tests | E2E | survey |
| 35 | Final E2E Verification & Adversarial Hardening | 100% pass of E2E test suite + Tier 5 Challenger coverage hardening | M6 | survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Requirement-driven test suite (Tiers 1-4, 10 synthetic datasets, `TEST_INFRA.md`, `TEST_READY.md`) | none | IN_PROGRESS |
| M1 | Foundation & MCP Infrastructure | `pyproject.toml`, `config/settings.py`, `config/logging_config.py`, `mcp/filesystem_server.py`, `mcp/command_server.py`, `mcp/client.py`, `mcp/executor.py`, `tests/test_mcp.py` | none | DONE |
| M2 | Data Intelligence & Engine | `engine/resource_monitor.py`, `engine/data_profiler.py`, `engine/polars_engine.py`, `engine/duckdb_engine.py`, `engine/memory_manager.py`, `engine/artifact_manager.py`, `tests/test_profiler.py`, `tests/test_resource_limits.py` | M1 | IN_PROGRESS |
| M3 | Supervisor & LangGraph Orchestration | `agents/supervisor.py`, `orchestration/state.py`, `orchestration/routing.py`, `orchestration/checkpoint.py`, `orchestration/graph.py` | M1, M2 | PLANNED |
| M4 | 7 Specialized Workers & Templates | `templates/`, `agents/cleaner_agent.py`, `agents/feature_agent.py`, `agents/validator_agent.py`, `agents/model_selector_agent.py`, `agents/trainer_agent.py`, `agents/evaluator_agent.py`, `agents/deployment_agent.py` | M1, M2, M3 | PLANNED |
| M5 | Autonomous Repair, Delivery & CLI | `engine/error_classifier.py`, `deployment/`, `main.py` with 6 interactive menu options, `tests/test_pipeline.py` | M1, M2, M3, M4 | PLANNED |
| M6 | Final E2E Acceptance & Adversarial Hardening | Phase 1: 100% pass of E2E test suite; Phase 2: Tier 5 Challenger adversarial hardening | M1-M5, E2E | PLANNED |

## Interface Contracts

### `config.settings` ↔ All Subsystems
- `Settings` class (inherits `pydantic_settings.BaseSettings`):
  - `project_root: Path = Path.cwd()`
  - `data_input_dir: Path = Path("data/input")`
  - `workspace_dir: Path = Path("workspace")`
  - `temp_dir: Path = Path("temp")`
  - `openai_api_key: Optional[str] = None`
  - `openai_base_url: str = "https://api.openai.com/v1"`
  - `openai_model: str = "gpt-4o"`
  - `duckdb_memory_limit: str = "8GB"`
  - `duckdb_threads: int = 4`
  - `max_repair_attempts: int = 3`
  - Method `ensure_directories() -> None`

### `mcp.command_server` ↔ `mcp.client` / `mcp.executor`
- Tools registered on `MCPServer("command-server")`:
  - `run_python_script(script_path: str, timeout_seconds: int = 300, env_vars: Optional[dict] = None) -> dict`
    Returns: `{"execution_id": str, "status": "running"|"completed"|"failed"|"timeout", "exit_code": int, "stdout": str, "stderr": str, "duration_seconds": float, "peak_memory_mb": float}`
  - `get_execution_status(execution_id: str) -> dict`
  - `read_execution_logs(execution_id: str, tail_lines: int = 100) -> dict`
  - `inspect_process_resources(execution_id: str) -> dict`

### `engine.resource_monitor` ↔ `engine.memory_manager`
- `ResourceSnapshot`:
  - `ram_total_gb: float`, `ram_available_gb: float`, `ram_percent: float`
  - `cpu_percent: float`, `cpu_count: int`
  - `gpu_available: bool`, `gpu_name: Optional[str]`, `gpu_total_vram_mb: float`, `gpu_free_vram_mb: float`
  - `disk_free_gb: float`
- `MemoryManager.select_execution_strategy(dataset_path: Path, estimated_uncompressed_bytes: int, operation_type: str) -> ExecutionStrategy`
  Enum: `IN_MEMORY`, `LAZY_POLARS`, `DUCKDB_OUT_OF_CORE`, `CHUNKED_BATCH`, `INCREMENTAL_TRAINING`

### `agents.supervisor` ↔ `orchestration.graph`
- `ExecutionPlan` (Pydantic model):
  - `project_id: str`, `dataset_path: str`, `problem_definition: str`
  - `target_variable: Optional[str]`, `task_type: Literal["classification", "regression", "clustering", "analysis"]`
  - `cleaning_strategy: dict`, `feature_strategy: dict`, `validation_strategy: dict`
  - `model_candidates: List[str]`, `primary_metric: str`, `interpretability_vs_performance: str`
  - `resource_budget: dict`, `deployment_target: str`, `approval_status: Literal["approved", "modified", "rejected"]`

### `orchestration.state`
- `PipelineGraphState` (TypedDict / Pydantic):
  - `project_id: str`, `dataset_path: str`, `plan: ExecutionPlan`
  - `cleaned_dataset_path: Optional[str]`, `feature_dataset_path: Optional[str]`
  - `validation_report: Optional[dict]`, `validation_passed: bool`
  - `selected_models: List[str]`, `trained_model_artifacts: List[dict]`, `best_model_artifact: Optional[dict]`
  - `evaluation_report: Optional[dict]`, `performance_acceptable: bool`
  - `deployment_artifacts: Optional[dict]`
  - `error_history: List[dict]`, `stage_retry_counts: Dict[str, int]`

## Code Layout
```
autonomous-ml-pipeline/
├── main.py
├── pyproject.toml
├── uv.lock
├── .env
├── .env.example
├── README.md
├── PROJECT.md
├── TEST_INFRA.md
├── TEST_READY.md
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── logging_config.py
│   ├── mcp_config.json
│   └── resource_limits.yaml
├── mcp/
│   ├── __init__.py
│   ├── client.py
│   ├── filesystem_server.py
│   ├── command_server.py
│   └── executor.py
├── engine/
│   ├── __init__.py
│   ├── data_profiler.py
│   ├── polars_engine.py
│   ├── duckdb_engine.py
│   ├── resource_monitor.py
│   ├── memory_manager.py
│   ├── artifact_manager.py
│   └── error_classifier.py
├── agents/
│   ├── __init__.py
│   ├── supervisor.py
│   ├── profiler_agent.py
│   ├── cleaner_agent.py
│   ├── feature_agent.py
│   ├── validator_agent.py
│   ├── model_selector_agent.py
│   ├── trainer_agent.py
│   ├── evaluator_agent.py
│   └── deployment_agent.py
├── orchestration/
│   ├── __init__.py
│   ├── state.py
│   ├── routing.py
│   ├── checkpoint.py
│   └── graph.py
├── templates/
│   ├── cleaning_template.py
│   ├── feature_template.py
│   ├── training_template.py
│   ├── inference_template.py
│   ├── api_template.py
│   └── Dockerfile.template
├── deployment/
│   ├── serve_api.py
│   ├── Dockerfile
│   └── README.md
├── tests/
│   ├── generate_test_datasets.py
│   ├── test_mcp.py
│   ├── test_profiler.py
│   ├── test_resource_limits.py
│   └── test_pipeline.py
├── data/
│   ├── input/
│   └── samples/
├── workspace/
│   ├── generated/
│   ├── intermediate/
│   ├── artifacts/
│   ├── checkpoints/
│   ├── reports/
│   └── logs/
└── temp/
```
