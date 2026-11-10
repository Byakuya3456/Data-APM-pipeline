"""LLM workers generate code; they never ingest full datasets."""
from __future__ import annotations

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


COMMON = """
You are a senior Python data/ML engineer. Generate executable Python 3.11 code only.
The code runs locally. Never print, upload, or send raw dataset rows to an API.
Use Polars scan_csv/scan_parquet and lazy expressions for large data. Prefer Parquet
intermediates. Use DuckDB SQL for out-of-core joins/aggregations when appropriate.
Do not call collect() on an unbounded LazyFrame. Collect only bounded samples or
aggregations. Avoid pandas for full datasets. Write artifacts under the supplied
workspace/artifacts directory. Include logging, deterministic seeds, schema checks,
and clear exceptions. Never use shell commands or modify files outside the workspace.
"""

class CodeWorker:
    def __init__(self, role: str):
        self.role = role
        self.llm = ChatOpenAI(
            model=os.getenv("AI_MODEL", "replace-with-your-model"),
            api_key=os.getenv("AI_API_KEY", "missing"),
            base_url=os.getenv("AI_BASE_URL") or None,
            temperature=float(os.getenv("AI_TEMPERATURE", "0")),
        )

    async def generate_code(self, context: str) -> str:
        prompt = f"{COMMON}\nROLE: {self.role}\nReturn only Python source code.\n\n{context}"
        response = await self.llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Generate the implementation script now.")
        ])
        return _strip_fences(response.content)

    async def repair_code(self, code: str, error: str, task_context: str) -> str:
        prompt = (
            f"{COMMON}\nROLE: {self.role}\n"
            "Repair the script with the smallest safe change. Return the complete Python script only.\n"
            f"TASK:\n{task_context}\n\nCURRENT SCRIPT:\n{code[:24000]}\n\n"
            f"BOUNDED EXECUTION ERROR:\n{error[:12000]}"
        )
        response = await self.llm.ainvoke([SystemMessage(content=prompt)])
        return _strip_fences(response.content)


def _strip_fences(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines)
    return value.strip()


class DataCleanerWorker(CodeWorker):
    def __init__(self):
        super().__init__("Data Cleaner: profile missingness, duplicates, invalid values, types, and produce a reproducible cleaning script.")


class FeatureEngineerWorker(CodeWorker):
    def __init__(self):
        super().__init__("Feature Engineer: build leakage-aware, task-appropriate features and train/validation/test splits.")


class MLTrainerWorker(CodeWorker):
    def __init__(self):
        super().__init__("ML Trainer: train and evaluate scikit-learn or XGBoost models with bounded memory and save model/metrics.")
