# Autonomous MCP-Driven ML & Data Pipeline

Local-first CLI that profiles a dataset, clarifies the objective, asks LLM workers to
generate Python scripts, and executes them through MCP. The LLM sees schema, a small
sample preview, requirements, and bounded errors—not the full dataset.

## Important implementation notes

- Supported initial inputs: CSV, TSV, Parquet.
- Polars lazy scans are used for schema/sample inspection. Worker prompts require lazy
  Polars / DuckDB and bounded materialization for processing.
- The generated workers are code generators, not data-processing agents.
- Filesystem MCP is the official `@modelcontextprotocol/server-filesystem`.
- `command_mcp_server.py` is a custom restricted MCP server. It runs only `.py` files
  inside `workspace/generated`, uses `shell=False`, and enforces a timeout.
- The repair loop is bounded (`MAX_REPAIR_ATTEMPTS`). It is not an unlimited autonomous
  retry mechanism.
- Generated code is still executable code. Review it before running on valuable data.
  For stronger isolation, run the command server in a dedicated low-privilege Windows
  account or container with read-only input mounts and a writable workspace only.

## Setup (Windows PowerShell)

1. Install Python 3.11+, Node.js (for `npx`), and `uv`.
2. Extract this project to `D:\project-data`.
3. Open PowerShell in `D:\project-data`:

```powershell
uv sync
Copy-Item .env.example .env
notepad .env
```

Set `AI_API_KEY`, `AI_BASE_URL`, and `AI_MODEL`. The endpoint must be OpenAI-compatible.
For LM Studio/Ollama-compatible local gateways, set the matching base URL and model.

4. Verify the filesystem MCP server can be launched:

```powershell
npx -y @modelcontextprotocol/server-filesystem D:\project-data
```

Stop it with Ctrl+C after the check. `npx` will fetch the server package if not cached.

5. Add a dataset under the project root, for example `D:\project-data\data\sales.parquet`.
6. Run:

```powershell
uv run python main.py --dataset D:\project-data\data\sales.parquet --request "Predict customer churn and prepare a Docker-deployable inference service"
```

The CLI shows schema and a bounded preview, then asks for target, task type, metric,
deployment target, constraints, and explicit approval.

## Output layout

```text
workspace/
  generated/       # generated Python scripts
  intermediate/    # cleaned/feature Parquet intermediates
  artifacts/       # model, metrics, metadata
  reports/         # run_summary.json and worker manifests
```

## Current MVP boundaries / next hardening steps

1. Add a dedicated data-profiler worker that emits aggregate-only statistics.
2. Add schema validation between stages and explicit artifact contracts.
3. Add dataset-size-aware model selection, streaming training where supported, and
   chunked training for estimators that support `partial_fit`.
4. Add process memory monitoring and OS-level resource limits. Python-level timeout
   alone is not an OOM guarantee.
5. Replace the simple linear LangGraph route with conditional routing based on task type
   (analysis-only, classification, regression, clustering, deep learning).
6. Add human approval gates before generated code execution and before deployment.
7. Add tests for data leakage, target leakage, class imbalance, drift, and reproducibility.
8. For large-scale inference, stream predictions to Parquet incrementally instead of
   retaining all prediction batches in a Python list.

## Security

Do not expose the command MCP server to a network. Do not run this project elevated.
Do not place secrets in generated scripts or prompts. The initial filesystem server has
access to the configured project root; keep unrelated personal files outside it.
