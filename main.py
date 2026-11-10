"""CLI supervisor and LangGraph routing for the local ML pipeline."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import TypedDict, Any

import polars as pl
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from agent_workers import DataCleanerWorker, FeatureEngineerWorker, MLTrainerWorker
from mcp_executor import MCPExecutor

load_dotenv()


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
        lf = pl.scan_csv(path, separator="\t" if suffix == ".tsv" else ",",
                         infer_schema_length=1000, ignore_errors=True)
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
    # Keep sample bounded in the prompt; user sees it locally in terminal.
    return profile


def ask_requirements(profile: dict, request: str) -> dict:
    print("\n=== Dataset preview (first <=100 rows) ===")
    print("File:", profile["file"], "| Size:", f'{profile["size_bytes"]:,}', "bytes")
    print("Schema:", json.dumps(profile["schema"], indent=2))
    print("Sample:")
    print(json.dumps(profile["sample_rows"][:10], indent=2, default=str)[:8000])
    print("\n=== Clarify the objective ===")
    print("Initial request:", request)
    target = input("Target column (or 'none' for analysis/clustering): ").strip()
    task = input("Task [classification/regression/clustering/analysis]: ").strip().lower()
    metric = input("Primary metric (e.g. F1, ROC-AUC, RMSE; blank for recommendation): ").strip()
    deployment = input("Deployment target [local/docker/fastapi/other]: ").strip().lower() or "local"
    constraints = input("Extra constraints (latency, interpretability, class imbalance, etc.): ").strip()
    confirmed = input("Proceed with this plan? [y/N]: ").strip().lower() == "y"
    if not confirmed:
        raise SystemExit("Plan not approved. Re-run with updated requirements.")
    return {
        "objective": request, "target": target or None, "task": task or "analysis",
        "metric": metric or "recommend", "deployment": deployment,
        "constraints": constraints,
    }


def build_graph():
    cleaner, feature, trainer = DataCleanerWorker(), FeatureEngineerWorker(), MLTrainerWorker()

    async def clean_node(state: PipelineState):
        ctx = _task_context(state, "cleaning")
        return {"cleaner_code": await cleaner.generate_code(ctx), "stage": "cleaning"}

    async def feature_node(state: PipelineState):
        ctx = _task_context(state, "feature engineering")
        return {"feature_code": await feature.generate_code(ctx), "stage": "feature_engineering"}

    async def train_node(state: PipelineState):
        ctx = _task_context(state, "model training and evaluation")
        return {"trainer_code": await trainer.generate_code(ctx), "stage": "training"}

    graph = StateGraph(PipelineState)
    graph.add_node("clean", clean_node)
    graph.add_node("features", feature_node)
    graph.add_node("train", train_node)
    graph.set_entry_point("clean")
    graph.add_edge("clean", "features")
    graph.add_edge("features", "train")
    graph.add_edge("train", END)
    return graph.compile(), (cleaner, feature, trainer)


def _task_context(state: PipelineState, stage: str) -> str:
    return (
        f"Stage: {stage}\nDataset path: {state['dataset']}\n"
        f"Dataset schema and bounded preview: {json.dumps(state['profile'], default=str)[:14000]}\n"
        f"User requirements: {json.dumps(state['requirements'], default=str)}\n"
        "Use the workspace paths workspace/intermediate and workspace/artifacts. "
        "Input files are read-only. Each stage must write a manifest JSON describing outputs."
    )


async def run_pipeline(args):
    root = Path(os.getenv("PROJECT_ROOT", args.project_root)).resolve()
    dataset = Path(args.dataset).resolve()
    if not dataset.is_file() or not dataset.is_relative_to(root):
        raise ValueError(f"Dataset must exist inside PROJECT_ROOT ({root})")
    for folder in ("workspace/generated", "workspace/intermediate", "workspace/artifacts", "workspace/reports"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    profile = inspect_dataset(dataset)
    requirements = ask_requirements(profile, args.request)
    state: PipelineState = {
        "dataset": str(dataset), "user_request": args.request,
        "profile": {**profile, "sample_rows": profile["sample_rows"][:10]},
        "requirements": requirements,
    }
    app, workers = build_graph()
    generated = await app.ainvoke(state)

    executor = MCPExecutor(root / "mcp_server_config.json", root)
    stages = [
        ("clean.py", generated["cleaner_code"], workers[0], "cleaning"),
        ("features.py", generated["feature_code"], workers[1], "feature engineering"),
        ("train.py", generated["trainer_code"], workers[2], "training"),
    ]
    outcomes = []
    for filename, code, worker, context in stages:
        print(f"\n--- Executing {context} ---")
        outcome = await executor.execute_with_repair(
            relative_path=f"workspace/generated/{filename}",
            initial_code=code,
            worker=worker,
            task_context=_task_context({**state, **generated}, context),
            max_attempts=int(os.getenv("MAX_REPAIR_ATTEMPTS", "2")),
            timeout_seconds=int(os.getenv("MAX_SCRIPT_SECONDS", "3600")),
        )
        outcomes.append({"stage": context, **outcome})
        print(json.dumps(outcome, indent=2, default=str)[:6000])
        if not outcome["success"]:
            print(f"Stopping: {context} failed after bounded repair attempts.")
            break

    report = root / "workspace/reports/run_summary.json"
    report.write_text(json.dumps({
        "requirements": requirements,
        "profile": {k: v for k, v in profile.items() if k != "sample_rows"},
        "outcomes": outcomes,
    }, indent=2, default=str), encoding="utf-8")
    print("\nRun summary:", report)


def main():
    parser = argparse.ArgumentParser(description="Autonomous MCP-driven ML pipeline")
    parser.add_argument("--project-root", default=os.getenv("PROJECT_ROOT", r"D:\project-data"))
    parser.add_argument("--dataset", required=True, help="Dataset path inside PROJECT_ROOT")
    parser.add_argument("--request", required=True, help="Describe the desired model/outcome")
    args = parser.parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
