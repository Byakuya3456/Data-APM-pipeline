# TEST_INFRA.md — Test Infrastructure & 4-Tier Opaque-Box Methodology

## 1. Testing Philosophy & Principles

The testing framework for the Autonomous Local ML Engineering Platform is designed around five foundational tenets:

1. **Opaque-Box Verification**:
   Tests interact strictly through external system surfaces: CLI commands, public module interfaces, filesystem artifacts, and HTTP endpoints. Assertions evaluate behavioral contracts, artifact schemas, and invariant preservation — never internal private functions, agent monologues, or brittle regex patterns.
2. **Local-First & Hardware-Respectful**:
   All tests execute on standard developer workstations (Windows 10/11, Python 3.11+, uv). System memory guardrails (24 GB RAM, NVIDIA RTX 3050 6 GB VRAM) are actively respected. Tests never cause host system out-of-memory crashes.
3. **Deterministic & Cost-Free CI Execution**:
   Testing must be 100% reproducible without requiring live external LLM API tokens. A deterministic local mock LLM provider intercepts code generation calls during testing, returning validated, syntactically correct code templates. Live LLM execution is an optional flag (`--live-llm`).
4. **Resilience & Fault Invariance**:
   The test suite actively injects errors, malformed data, and boundary conditions to verify that the platform fails gracefully, respects security jails, and autonomously repairs recoverable faults within bounded limits.
5. **Zero Test Pollution**:
   Every test executes in an isolated temporary workspace directory. Original input datasets are treated as read-only. Test artifacts are contained in temporary paths and cleaned up automatically.

---

## 2. 4-Tier Test Methodology

The test suite is structured into four progressive tiers of increasing scope:

```
┌────────────────────────────────────────────────────────┐
│  Tier 4: Real-World Workload Scenarios                 │
│  (5 End-to-End User Journeys: Churn, House Prices,     │
│   Dirty Data Auto-Remediation, Error Repair, Serving)  │
└───────────────────────────▲────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────┐
│  Tier 3: Cross-Feature Combinations                    │
│  (7 Pipeline Subsystems: Profiler->Memory->DuckDB,     │
│   Cleaner->Features->Validator, Trainer->Joblib, etc.) │
└───────────────────────────▲────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────┐
│  Tier 2: Boundary & Corner Cases                       │
│  (12 Edge Cases: Empty, 50MB+, 99:1 Imbalance,         │
│   Missing Target, High Cardinality, AST Security)      │
└───────────────────────────▲────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────┐
│  Tier 1: Feature Coverage (Atomic Isolation)           │
│  (19 Core Features × >=5 Tests = 95 Tests)             │
└────────────────────────────────────────────────────────┘
```

### Tier 1: Core Feature Coverage (Isolation)
Every core architectural feature is tested in isolation with dedicated test cases:

1. **MCP Filesystem Server (`mcp.filesystem_server`)**:
   - `test_fs_list_directory`: Lists directory contents, returning name, size, type.
   - `test_fs_read_file`: Reads valid file content within project root.
   - `test_fs_write_file`: Writes script to `workspace/generated/`, verifies content on disk.
   - `test_fs_path_traversal_jail`: Rejects attempts to escape root (`../../windows/system32`).
   - `test_fs_read_nonexistent_file`: Returns structured error without server crash.
2. **MCP Command Execution Server (`mcp.command_server`)**:
   - `test_cmd_run_python_script_success`: Executes valid script, returns exit code 0, stdout, telemetry.
   - `test_cmd_run_python_script_nonzero_exit`: Captures script exceptions and nonzero exit codes.
   - `test_cmd_timeout_enforcement`: Terminates runaway script exceeding timeout; sets `status='timeout'`.
   - `test_cmd_security_path_jail`: Blocks execution of scripts outside `workspace/generated/`.
   - `test_cmd_ast_sandbox_check`: AST pre-check rejects unauthorized imports or malicious OS operations.
3. **MCP Client & Stdio Protocol (`mcp.client` / `mcp.executor`)**:
   - `test_mcp_client_session_initialization`: Establishes stdio connection and initializes session.
   - `test_mcp_client_tool_discovery`: Discovers all 4 command server tools and filesystem tools.
   - `test_mcp_client_tool_call_roundtrip`: Dispatches tool call, verifies structured response schema.
   - `test_mcp_client_bounded_output`: Truncates large stdout/stderr buffers safely (<= 24,000 chars).
   - `test_mcp_client_resilience_and_reconnect`: Handles unexpected subprocess restarts cleanly.
4. **Multi-Format Dataset Discovery (`engine.data_profiler`)**:
   - `test_discovery_csv_tsv`: Detects CSV and TSV files, infers delimiters, extracts headers.
   - `test_discovery_parquet`: Reads Parquet schema metadata without loading full table rows.
   - `test_discovery_json_jsonl`: Identifies JSON array and line-delimited JSONL datasets.
   - `test_discovery_ignores_irrelevant_files`: Ignores `.txt`, `.tmp`, `.py`, images.
   - `test_discovery_empty_directory`: Gracefully handles empty input directory without crashing.
5. **Polars Lazy Profiling (`engine.data_profiler` / `engine.polars_engine`)**:
   - `test_profiling_bounded_sample_limit`: Generates preview containing exactly <= 100 rows.
   - `test_profiling_schema_types`: Categorizes column data types accurately.
   - `test_profiling_null_statistics`: Computes missing value counts and null percentages lazily.
   - `test_profiling_candidate_target_detection`: Flags heuristic target columns (`churn`, `target`, `price`).
   - `test_profiling_memory_estimation`: Calculates estimated uncompressed dataset byte size.
6. **DuckDB Out-of-Core Engine (`engine.duckdb_engine`)**:
   - `test_duckdb_connection_and_memory_limit`: Verifies memory limit configuration (`8GB`).
   - `test_duckdb_summarize_statistics`: Runs `SUMMARIZE` on CSV; verifies min/max/avg/quantiles.
   - `test_duckdb_parquet_query`: Executes SQL queries directly against Parquet files.
   - `test_duckdb_temp_disk_spilling`: Spills large aggregations to temporary disk directory.
   - `test_duckdb_sql_injection_defense`: Sanitizes queries against malicious table/column strings.
7. **System Resource Telemetry & Memory Manager (`engine.resource_monitor`, `engine.memory_manager`)**:
   - `test_resource_telemetry_ram_cpu`: Returns live RAM (total/available/percent) and CPU load.
   - `test_resource_telemetry_gpu_vram`: Queries NVIDIA GPU VRAM; falls back gracefully if absent.
   - `test_strategy_selection_in_memory`: Selects `IN_MEMORY` strategy for small datasets (< 10 MB).
   - `test_strategy_selection_lazy_polars`: Selects `LAZY_POLARS` for medium datasets (10 MB - 1 GB).
   - `test_strategy_selection_duckdb_or_chunked`: Selects `DUCKDB_OUT_OF_CORE` for large/complex workloads.
8. **Supervisor Interactive Q&A & Plan Generation (`agents.supervisor`)**:
   - `test_supervisor_target_clarification`: Asks clarifying questions when target is ambiguous.
   - `test_supervisor_task_classification`: Correctly categorizes classification vs regression vs clustering.
   - `test_supervisor_execution_plan_schema`: Produces Pydantic `ExecutionPlan` adhering to schema.
   - `test_supervisor_fact_vs_assumption_split`: Distinguishes confirmed user inputs from assumptions.
   - `test_supervisor_approval_gate`: Enforces user approval gate [Approve / Modify / Cancel].
9. **Data Cleaner Worker Agent (`agents.cleaner_agent`)**:
   - `test_cleaner_handles_missing_values`: Imputes or removes nulls according to cleaning plan.
   - `test_cleaner_deduplication`: Identifies and removes duplicate records.
   - `test_cleaner_type_casting`: Normalizes messy numbers, booleans, and date formats.
   - `test_cleaner_safe_vs_prompted_corrections`: Distinguishes automatic safe fixes from risky drops.
   - `test_cleaner_output_artifacts_exist`: Produces `cleaned_dataset.parquet` and `cleaning_report.json`.
10. **Feature Engineering Worker Agent (`agents.feature_agent`)**:
    - `test_feature_one_hot_or_target_encoding`: Encodes categorical features into numeric matrices.
    - `test_feature_datetime_decomposition`: Extracts year, month, day, hour, day-of-week components.
    - `test_feature_fit_on_train_only`: Enforces fit on train split only; applies transform to test.
    - `test_feature_high_correlation_filtering`: Identifies and eliminates collinear features (> 0.95).
    - `test_feature_manifest_schema`: Writes `feature_manifest.json` detailing all transformations.
11. **Data Validation Worker Agent (`agents.validator_agent`)**:
    - `test_validator_target_column_present`: Validates target existence and data type.
    - `test_validator_missing_target_gate_stop`: Triggers critical stop gate when target is absent.
    - `test_validator_zero_variance_target`: Halts execution if target has only one unique value.
    - `test_validator_train_test_split_feasibility`: Checks sample count sufficiency per class.
    - `test_validator_leakage_detection`: Flags features suspiciously correlated with target.
12. **Model Selection Worker Agent (`agents.model_selector_agent`)**:
    - `test_selector_classification_candidates`: Selects LogisticRegression, RandomForest, XGBoost.
    - `test_selector_regression_candidates`: Selects LinearRegression, RandomForestRegressor, XGBoost.
    - `test_selector_resource_awareness`: Adjusts algorithm candidates based on RAM/GPU availability.
    - `test_selector_interpretability_preference`: Prioritizes interpretable models when requested.
    - `test_selector_output_rationale`: Provides documented justification for model candidate set.
13. **Model Training Worker Agent (`agents.trainer_agent`)**:
    - `test_trainer_deterministic_seed`: Verifies seed=42 yields reproducible weights and scores.
    - `test_trainer_stratified_cv`: Executes cross-validation without data leakage across folds.
    - `test_trainer_handles_class_imbalance`: Configures `scale_pos_weight` or `class_weight='balanced'`.
    - `test_trainer_bundles_preprocessor_and_model`: Generates single loadable `.joblib` pipeline bundle.
    - `test_trainer_resource_telemetry_during_training`: Logs elapsed training time and memory peak.
14. **Model Evaluation Worker Agent (`agents.evaluator_agent`)**:
    - `test_evaluator_classification_metrics`: Computes Accuracy, F1, Precision, Recall, ROC-AUC, PR-AUC.
    - `test_evaluator_regression_metrics`: Computes MAE, MSE, RMSE, R²; asserts RMSE >= MAE.
    - `test_evaluator_overfitting_check`: Flags train/test performance divergence.
    - `test_evaluator_model_comparison_ranking`: Compares candidates and selects optimal model.
    - `test_evaluator_report_schema`: Generates `evaluation_report.json` and Markdown summary.
15. **Deployment Worker Agent (`agents.deployment_agent`)**:
    - `test_deployment_generates_inference_script`: Emits standalone executable `inference.py`.
    - `test_deployment_generates_fastapi_server`: Emits production `serve_api.py` with `/predict`.
    - `test_deployment_generates_dockerfile`: Emits containerized `Dockerfile` with uv/pip setup.
    - `test_deployment_inference_matches_test_prediction`: Standalone inference matches trainer output.
    - `test_deployment_fastapi_endpoint_response`: TestClient verifies 200 OK and valid JSON response.
16. **Autonomous Error Classifier & Repair Loop (`engine.error_classifier`, `mcp.executor`)**:
    - `test_classifier_syntax_error`: Categorizes syntax mistakes into `SYNTAX_ERROR`.
    - `test_classifier_missing_column_error`: Categorizes KeyError into `INVALID_COLUMN_REFERENCE`.
    - `test_classifier_memory_exhaustion`: Categorizes MemoryError into `MEMORY_EXHAUSTION`.
    - `test_repair_loop_bounded_retries`: Halts after max 3 failed repair attempts without runaway loops.
    - `test_repair_loop_memory_fallback`: Adapts execution strategy on memory exhaustion error.
17. **Persistent State Checkpointing & Resumption (`orchestration.checkpoint`, `orchestration.graph`)**:
    - `test_checkpoint_initial_state_save`: Writes initial graph state to SQLite checkpointer.
    - `test_checkpoint_stage_transitions`: Updates state accurately after each pipeline stage.
    - `test_checkpoint_resume_after_interrupt`: Reloads checkpoint and resumes at next pending stage.
    - `test_checkpoint_skip_completed_stages`: Skips previously finalized stages on resume.
    - `test_checkpoint_corruption_resilience`: Recovers cleanly from locked or missing SQLite files.
18. **Application Settings & Configuration (`config.settings`, `config.logging_config`)**:
    - `test_settings_dynamic_root_resolution`: Resolves project root dynamically on any host/drive.
    - `test_settings_directory_auto_creation`: Auto-creates required directory layout on startup.
    - `test_settings_env_override`: Overrides defaults via `.env` configuration.
    - `test_logging_dual_output`: Emits logs to both stdout console and rotating file `app.log`.
    - `test_logging_rotation`: Rotates file log when size threshold is exceeded.
19. **Interactive & Headless CLI Interface (`main.py`)**:
    - `test_cli_menu_display`: Displays 6 interactive menu options accurately.
    - `test_cli_headless_new_project`: Executes end-to-end pipeline headlessly via CLI flags.
    - `test_cli_inspect_dataset_option`: Executes Option 3 / `--inspect` to preview dataset.
    - `test_cli_resource_status_option`: Executes Option 5 / `--resources` to view live hardware stats.
    - `test_cli_resume_option`: Executes Option 2 / `--resume <project_id>` to continue existing work.

---

### Tier 2: Boundary & Corner Cases (12 Edge Cases)

1. **Empty Dataset (0 rows)**:
   - File: `empty_dataset.csv` (headers only).
   - Assertion: Validator halts pipeline with `ValidationError("Empty dataset: 0 rows found")`; exit code 0 or validation failure; no unhandled crash.
2. **Minimal Dataset (< 10 rows)**:
   - File: 3-row CSV dataset.
   - Assertion: Validator halts or warns about insufficient rows for train/test split.
3. **Out-of-Core Dataset (>= 50MB / >= 200,000 rows)**:
   - File: `large_dataset.csv`.
   - Assertion: MemoryManager chooses `LAZY_POLARS` or `DUCKDB_OUT_OF_CORE`; peak process RAM stays < 2.0 GB during profiling and cleaning; output Parquet generated.
4. **Extreme Class Imbalance (99:1 Ratio)**:
   - File: `imbalanced.csv` (10,000 rows, 1% positive class).
   - Assertion: Profiler detects imbalance; supervisor sets primary metric to `PR-AUC` / `F1`; trainer applies class weights; evaluator computes PR-AUC.
5. **Missing Target Column**:
   - File: `invalid_target.csv` with user request targeting missing column.
   - Assertion: Validation worker stops pipeline at Gate Stop; `validation_passed = False`; no model selection or training occurs.
6. **Zero-Variance Target (Constant Label)**:
   - File: Dataset where target column contains single unique value.
   - Assertion: Validator flags zero variance; stops pipeline with descriptive diagnostic.
7. **Irrelevant & Constant Features**:
   - File: `irrelevant_constant.csv` (constant columns, random Gaussian noise).
   - Assertion: Feature engineering drops constant features; logs dropped columns in manifest; training succeeds on remaining features.
8. **High-Cardinality Categorical Columns**:
   - File: `high_cardinality.csv` (UUIDs, ZIP codes, customer IDs).
   - Assertion: Feature engineer avoids memory-exploding one-hot encoding; uses target/frequency encoding or drops identifiers; process memory stays bounded.
9. **Messy & Mixed Data Types**:
   - File: `mixed_types.csv` (ISO dates, US dates, dirty booleans `'Y'/'N'`, dollar strings `"$1,200.50"`).
   - Assertion: Cleaner normalizes values into proper datatypes (`datetime`, `bool`, `float64`) without silent row drops.
10. **Duplicate Records & ID Collisions**:
    - File: `duplicates.csv` (exact duplicate rows and repeated IDs).
    - Assertion: Cleaner removes exact duplicate rows; records count in `cleaning_report.json`.
11. **Malicious Code Injection AST Rejection**:
    - Script: Attempted execution of unauthorized system commands (`os.system`, `subprocess` escaping sandbox).
    - Assertion: AST pre-execution check rejects script with `SecurityError`; execution aborted before process spawn.
12. **Subprocess Timeout Enforcement**:
    - Script: Script with infinite loop `while True: pass`.
    - Assertion: Command server terminates subprocess after `timeout_seconds`; returns `status='timeout'`.

---

### Tier 3: Cross-Feature Combinations (7 Integration Tests)

1. **Discovery → Profiling → Memory Strategy Selection**:
   - DataProfiler inspects dataset → extracts schema and row count → MemoryManager chooses correct `ExecutionStrategy` based on available RAM.
2. **Data Cleaner → Feature Engineering → Data Validator**:
   - Raw data cleaned to `cleaned_dataset.parquet` → Feature Engineer transforms to `feature_dataset.parquet` → Validator verifies schema integrity and zero leakage.
3. **Feature Engineering → Model Selector → Model Trainer**:
   - Transformed feature matrix feeds Model Selector → Candidate models chosen → Trainer fits all candidates using identical train/test splits.
4. **Model Trainer → Model Evaluator → Artifact Manager**:
   - Trained models evaluated against test set → Best model ranked by primary metric → Bundled preprocessor + model artifact saved to `workspace/artifacts/model.joblib`.
5. **LangGraph State Machine Checkpoint & Resume**:
   - Pipeline paused after Feature Engineering → Checkpointed to SQLite → New session invokes `--resume` → Execution resumes at Model Selection without re-running prior nodes.
6. **Code Generation Worker → MCP Execution → Disk Artifact Verification**:
   - Worker generates script → MCP Executor writes via Filesystem MCP → executes via Command MCP → outputs verified on disk with SHA256 checksums.
7. **Error Classifier → Worker Auto-Patch → Second Execution Success**:
   - Script with induced column typo fails with `KeyError` → Error Classifier categorizes as `INVALID_COLUMN_REFERENCE` → Worker generates patched script → Second execution succeeds within retry budget.

---

### Tier 4: Real-World Workload Scenarios (5 End-to-End Scenarios)

1. **Scenario 1: End-to-End Customer Churn Classification**:
   - Input: `data/test_fixtures/small_classification.csv` (1,000 rows, 10 columns, binary `churn`).
   - Workflow: Full autonomous cycle: Discovery → Profiling → ExecutionPlan → Cleaning → Features → Validation → Model Training (XGBoost / RandomForest) → Evaluation (F1 > 0.80) → Artifact Packaging → `inference.py` prediction verification.
2. **Scenario 2: End-to-End Real Estate Price Regression**:
   - Input: `data/test_fixtures/regression.parquet` (5,000 rows, continuous target `price`).
   - Workflow: Parquet metadata scan → Regression plan → Log transforms & scaling → Training (XGBoostRegressor / Ridge) → Evaluation (R² > 0.70, RMSE) → Artifact bundling.
3. **Scenario 3: Dirty & Missing Data Auto-Remediation**:
   - Input: `data/test_fixtures/mixed_types.csv` + `missing_values.csv`.
   - Workflow: Autonomous detection of dirty strings, dates, and missing values → Automated imputation and type conversion → Feature engineering → Viable model training.
4. **Scenario 4: Error Auto-Recovery Pipeline**:
   - Workflow: Pipeline encounters execution failure on attempt 1 → Error classified → Autonomous patch generated → Attempt 2 succeeds → Final artifacts verified.
5. **Scenario 5: Full Container & Serving Lifecycle**:
   - Workflow: Pipeline emits `deployment/serve_api.py`, `deployment/Dockerfile`, and `model.joblib` → Test launches FastAPI via `TestClient` → GET `/health` returns 200 `healthy` → POST `/predict` returns 200 with valid prediction payload.

---

## 3. Test Runner Harness Architecture

### Directory Layout
```
tests/
├── conftest.py                   # Pytest fixtures, isolated workspace, mock LLM
├── generate_test_datasets.py     # Deterministic generator for all 10 synthetic datasets
├── test_mcp.py                   # MCP infrastructure unit tests (M1)
├── test_profiler.py              # Data profiler & engine tests (M2)
├── test_resource_limits.py       # Telemetry & memory manager tests (M2)
└── test_pipeline.py              # Requirement-driven E2E test suite (Tiers 1-4)
```

### Pytest Configuration (`pyproject.toml`)
```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
markers = [
    "tier1: Tier 1 Feature Coverage unit/isolation tests",
    "tier2: Tier 2 Boundary and Corner Case tests",
    "tier3: Tier 3 Cross-Feature Combination and integration tests",
    "tier4: Tier 4 Real-World Workload end-to-end scenario tests",
    "slow: Tests taking more than 5 seconds or using large datasets",
    "mcp: Tests validating Model Context Protocol stdio communication",
    "resource_heavy: Tests requiring high RAM or CPU allocations",
]
```

### CLI Command Options
- Run all tests: `uv run pytest`
- Run specific tier: `uv run pytest -m "tier1"` (or `tier2`, `tier3`, `tier4`)
- Run fast tests (skipping slow 50MB+ datasets): `uv run pytest -m "not slow"`
- Run full regression including slow tests: `uv run pytest --run-slow`
- Generate datasets before testing: `uv run python tests/generate_test_datasets.py --all --seed 42`

### Exit Code Semantics
| Exit Code | Meaning | Action Required |
|-----------|---------|-----------------|
| `0` | All tests passed successfully | System verified ready for release |
| `1` | One or more tests failed | Inspect failure trace; repair code |
| `2` | Test run interrupted by user (Ctrl+C) | Resume or re-run tests |
| `3` | Internal harness crash or unhandled error | Check harness environment and dependencies |
| `4` | Pytest CLI syntax/usage error | Verify flags and arguments |
| `5` | No tests collected | Check test file paths and marker filters |
| `10` | Test Fixtures Missing | Run `python tests/generate_test_datasets.py --all` |
| `11` | Incompatible environment / missing dependency | Verify Python >= 3.11 and run `uv sync` |

### Environment Isolation & Mocking Fixtures (`conftest.py`)
- `isolated_workspace(tmp_path)`: Creates an isolated directory structure (`workspace/generated`, `workspace/artifacts`, `workspace/reports`, `temp`) for each test run; sets `PROJECT_ROOT` and `WORKSPACE_DIR` environment variables so tests never contaminate user workspace.
- `test_fixtures_dir`: Returns `data/test_fixtures/`; automatically invokes `generate_test_datasets.py` if datasets are missing.
- `mock_llm_responses`: Intercepts `ChatOpenAI` calls, returning validated Python code templates and JSON plans deterministically based on stage keywords (`cleaning`, `feature_engineering`, `training`), enabling 100% offline, zero-cost test execution.

---

## 4. Feature Inventory Checklist for E2E Tests

| # | Feature Name | Module / Path | Tier | Test IDs | Success Criteria / Invariants |
|---|--------------|---------------|------|----------|--------------------------------|
| 1 | Dependency & Config | `pyproject.toml`, `config/` | Tier 1 | `test_settings_dynamic_root_resolution` | `project_root` resolves dynamically; `.env` overrides defaults |
| 2 | Application Settings | `config/settings.py` | Tier 1 | `test_settings_directory_auto_creation` | `ensure_directories()` creates required directory layout |
| 3 | Structured Dual Logging | `config/logging_config.py` | Tier 1 | `test_logging_dual_output`, `test_logging_rotation` | Logs to console + `workspace/logs/app.log`; rotates at limit |
| 4 | MCP Filesystem Server | `mcp/filesystem_server.py` | Tier 1, 2 | `test_fs_*`, `test_fs_path_traversal_jail` | Read/write/list work; escapes outside root rejected |
| 5 | MCP Command Server | `mcp/command_server.py` | Tier 1, 2 | `test_cmd_*`, `test_cmd_ast_sandbox_check` | 4 tools present; shell=False; AST blocks unauthorized ops |
| 6 | MCP Client & Stdio | `mcp/client.py`, `executor.py`| Tier 1, 3 | `test_mcp_client_*`, `Combo 3.6` | Tool discovery succeeds; bounded stdout <= 24k chars |
| 7 | System Telemetry | `engine/resource_monitor.py` | Tier 1 | `test_resource_telemetry_ram_cpu`, `_gpu` | RAM, CPU, GPU VRAM reported; CPU fallback on non-NVIDIA |
| 8 | Multi-Format Discovery | `engine/data_profiler.py` | Tier 1 | `test_discovery_csv_tsv`, `_parquet`, `_json`| CSV, TSV, Parquet, JSON, JSONL identified without full scan |
| 9 | Polars Lazy Profiler | `engine/polars_engine.py` | Tier 1, 2 | `test_profiling_*`, `Case 2.1` | Preview <= 100 rows; schema & null statistics extracted |
| 10| DuckDB Query Engine | `engine/duckdb_engine.py` | Tier 1, 2 | `test_duckdb_*`, `Case 2.3` | Out-of-core queries; memory_limit=8GB; temp disk spilling |
| 11| Execution Selector | `engine/memory_manager.py` | Tier 1, 3 | `test_strategy_selection_*`, `Combo 3.1` | IN_MEMORY, LAZY_POLARS, DUCKDB chosen appropriately |
| 12| Artifact Manager | `engine/artifact_manager.py` | Tier 1, 3 | `test_cleaner_output_artifacts_exist`, `Combo 3.4`| Parquet artifacts written with SHA256 verification |
| 13| Supervisor Q&A | `agents/supervisor.py` | Tier 1 | `test_supervisor_target_clarification`, `_plan` | Interactive Q&A clarifies target; Pydantic `ExecutionPlan` |
| 14| Human Approval Gate | `agents/supervisor.py` | Tier 1 | `test_supervisor_approval_gate` | Approval required [Approve/Modify/Cancel]; stops on Cancel |
| 15| LangGraph Pipeline State| `orchestration/state.py` | Tier 1, 3 | `test_checkpoint_*`, `Combo 3.5` | `PipelineGraphState` tracks artifacts, metrics, retry counts |
| 16| SQLite Checkpointing | `orchestration/checkpoint.py`| Tier 1, 3 | `test_checkpoint_resume_after_interrupt` | Resumes from interrupted stage; skips completed nodes |
| 17| Data Cleaner Worker | `agents/cleaner_agent.py` | Tier 1, 2 | `test_cleaner_*`, `Case 2.9`, `Case 2.10` | Outputs `cleaned_dataset.parquet` and `cleaning_report.json` |
| 18| Feature Worker | `agents/feature_agent.py` | Tier 1, 2 | `test_feature_*`, `Case 2.7`, `Case 2.8` | Fits on train only; drops constant features; manifests written |
| 19| Data Validator Worker | `agents/validator_agent.py` | Tier 1, 2 | `test_validator_*`, `Case 2.5`, `Case 2.6` | Critical stop gate stops on missing/zero-variance target |
| 20| Model Selector Worker | `agents/model_selector_agent.py`| Tier 1 | `test_selector_classification_*`, `_regression_*`| Justified candidate set selected; no blind all-model training|
| 21| Model Trainer Worker | `agents/trainer_agent.py` | Tier 1, 3 | `test_trainer_deterministic_seed`, `Combo 3.4`| Seed 42 reproducible; bundles preprocessor + model joblib |
| 22| Model Evaluator Worker | `agents/evaluator_agent.py` | Tier 1, 4 | `test_evaluator_*`, `Scenario 4.1`, `Scenario 4.2`| Computes F1/ROC-AUC/PR-AUC/RMSE; detects overfitting |
| 23| Deployment Worker | `agents/deployment_agent.py` | Tier 1, 4 | `test_deployment_*`, `Scenario 4.5` | Emits `inference.py`, `serve_api.py`, `Dockerfile` |
| 24| Error Repair Loop | `engine/error_classifier.py`| Tier 1, 4 | `test_repair_loop_*`, `Scenario 4.4` | 10 error categories; max 3 retries; adapts on memory fault |
| 25| Production FastAPI | `deployment/serve_api.py` | Tier 1, 4 | `test_deployment_fastapi_endpoint_response` | `/health` returns 200; `/predict` returns valid schema |
| 26| 6-Option CLI Interface | `main.py` | Tier 1 | `test_cli_menu_display`, `test_cli_headless_*`| Supports interactive menu & non-interactive headless flags |
| 27| Synthetic Datasets | `tests/generate_test_datasets.py`| Tiers 1-4| `test_fixtures_dir` fixture | Generates all 10 synthetic datasets with seed 42 |
| 28| E2E Test Suite | `tests/test_pipeline.py` | Tiers 1-4| Full test execution | 100% pass across all 4 tiers |
