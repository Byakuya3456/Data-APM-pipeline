"""High-level domain-specific execution engine wrapping MCP client with typed models and self-correction loop."""
from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from mcp.client import MCPClient, ToolResult

logger = logging.getLogger("mlpipe.mcp.executor")


class ScriptExecutionResult(BaseModel):
    """Structured response model for Python script execution via MCP command server."""
    execution_id: str
    status: Literal["completed", "failed", "timeout", "running"]
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    peak_memory_mb: float = 0.0

    @property
    def is_success(self) -> bool:
        """True if the script finished cleanly with exit code 0."""
        return self.status == "completed" and self.exit_code == 0


class ProcessResources(BaseModel):
    """Process resource telemetry snapshot."""
    execution_id: str
    pid: Optional[int] = None
    status: str = "unknown"
    memory_rss_mb: float = 0.0
    memory_vms_mb: float = 0.0
    cpu_percent: float = 0.0
    duration_seconds: float = 0.0

    @property
    def is_running(self) -> bool:
        return self.status == "running"


class MCPExecutor:
    """High-level domain operations coordinator backed by MCP filesystem and command execution servers."""

    def __init__(
        self,
        client: Optional[MCPClient] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.client = client or MCPClient(settings=self.settings)

    async def __aenter__(self) -> MCPExecutor:
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.client.__aexit__(exc_type, exc_val, exc_tb)

    # ------------------------------------------------------------------
    # Filesystem Domain Operations
    # ------------------------------------------------------------------

    async def write_script(
        self,
        relative_path: str | Path,
        code: str,
        validate_syntax: bool = True,
    ) -> Path:
        """Write generated Python code via the filesystem MCP server.

        Validates Python syntax via AST pre-check and guards against path traversal.
        """
        rel_str = str(relative_path).replace("\\", "/")
        if ".." in Path(rel_str).parts or Path(rel_str).is_absolute():
            raise ValueError(f"Path traversal rejected: '{relative_path}'")

        if not rel_str.endswith(".py"):
            raise ValueError(f"Only .py scripts may be written: '{relative_path}'")

        if validate_syntax:
            try:
                ast.parse(code, filename=rel_str)
            except SyntaxError as e:
                raise ValueError(
                    f"AST SyntaxError in generated code for '{rel_str}' at line {e.lineno}, col {e.offset}: {e.msg}"
                )

        # Write via MCP filesystem tool
        result = await self.client.call_tool(
            "write_file",
            {"path": rel_str, "content": code},
            server_name="filesystem",
        )
        if result.is_error:
            raise RuntimeError(f"Filesystem server write failed: {result.error_message}")

        return (self.settings.project_root / rel_str).resolve()

    async def read_file(self, relative_path: str | Path, max_bytes: int = 1048576) -> str:
        """Read file contents via filesystem MCP server."""
        rel_str = str(relative_path).replace("\\", "/")
        result = await self.client.call_tool(
            "read_file",
            {"path": rel_str, "max_bytes": max_bytes},
            server_name="filesystem",
        )
        if result.is_error:
            raise RuntimeError(f"Filesystem server read failed: {result.error_message}")

        return result.data.get("content", result.text)

    async def list_files(self, relative_path: str = ".") -> List[Dict[str, Any]]:
        """List directory contents via filesystem MCP server."""
        result = await self.client.call_tool(
            "list_directory",
            {"path": relative_path},
            server_name="filesystem",
        )
        if result.is_error:
            raise RuntimeError(f"Filesystem server list failed: {result.error_message}")

        return result.data.get("entries", [])

    async def get_file_info(self, relative_path: str | Path) -> Dict[str, Any]:
        """Inspect file metadata via filesystem MCP server."""
        rel_str = str(relative_path).replace("\\", "/")
        result = await self.client.call_tool(
            "get_file_info",
            {"path": rel_str},
            server_name="filesystem",
        )
        if result.is_error:
            raise RuntimeError(f"Filesystem server get_file_info failed: {result.error_message}")

        return result.data

    # ------------------------------------------------------------------
    # Command Execution Domain Operations
    # ------------------------------------------------------------------

    async def run_script(
        self,
        script_path: str | Path,
        timeout_seconds: int = 300,
        env_vars: Optional[Dict[str, str]] = None,
        background: bool = False,
    ) -> ScriptExecutionResult:
        """Execute a Python script via MCP command server and return structured execution result."""
        p = Path(script_path)
        if p.is_absolute():
            try:
                rel_str = str(p.relative_to(self.settings.project_root)).replace("\\", "/")
            except ValueError:
                rel_str = str(p).replace("\\", "/")
        else:
            rel_str = str(p).replace("\\", "/")

        result = await self.client.call_tool(
            "run_python_script",
            {
                "script_path": rel_str,
                "timeout_seconds": timeout_seconds,
                "env_vars": env_vars,
                "background": background,
            },
            server_name="command",
        )

        if result.is_error:
            return ScriptExecutionResult(
                execution_id=result.data.get("execution_id", "unknown"),
                status="failed",
                exit_code=-1,
                stderr=result.error_message or "Execution failed",
                duration_seconds=result.duration_seconds,
            )

        data = result.data
        return ScriptExecutionResult(
            execution_id=data.get("execution_id", "unknown"),
            status=data.get("status", "failed"),
            exit_code=data.get("exit_code"),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            duration_seconds=data.get("duration_seconds", result.duration_seconds),
            peak_memory_mb=data.get("peak_memory_mb", 0.0),
        )

    async def get_execution_status(self, execution_id: str) -> ScriptExecutionResult:
        """Query execution status for an active or completed execution."""
        result = await self.client.call_tool(
            "get_execution_status",
            {"execution_id": execution_id},
            server_name="command",
        )
        if result.is_error:
            raise RuntimeError(f"Failed to get execution status: {result.error_message}")

        data = result.data
        return ScriptExecutionResult(
            execution_id=data.get("execution_id", execution_id),
            status=data.get("status", "failed"),
            exit_code=data.get("exit_code"),
            stdout="",
            stderr="",
            duration_seconds=data.get("duration_seconds", 0.0),
            peak_memory_mb=data.get("peak_memory_mb", 0.0),
        )

    async def read_execution_logs(self, execution_id: str, tail_lines: int = 100) -> Dict[str, Any]:
        """Read stdout and stderr log tails for a script execution."""
        result = await self.client.call_tool(
            "read_execution_logs",
            {"execution_id": execution_id, "tail_lines": tail_lines},
            server_name="command",
        )
        if result.is_error:
            raise RuntimeError(f"Failed to read execution logs: {result.error_message}")

        return result.data

    async def inspect_resources(self, execution_id: str) -> ProcessResources:
        """Inspect memory and CPU utilization of an execution."""
        result = await self.client.call_tool(
            "inspect_process_resources",
            {"execution_id": execution_id},
            server_name="command",
        )
        if result.is_error:
            raise RuntimeError(f"Failed to inspect process resources: {result.error_message}")

        data = result.data
        return ProcessResources(
            execution_id=data.get("execution_id", execution_id),
            pid=data.get("pid"),
            status=data.get("status", "unknown"),
            memory_rss_mb=data.get("memory_rss_mb", 0.0),
            memory_vms_mb=data.get("memory_vms_mb", 0.0),
            cpu_percent=data.get("cpu_percent", 0.0),
            duration_seconds=data.get("duration_seconds", 0.0),
        )

    # ------------------------------------------------------------------
    # Autonomous Execution and Repair Loop
    # ------------------------------------------------------------------

    async def execute_with_repair(
        self,
        *,
        relative_path: str,
        initial_code: str,
        worker: Optional[Any] = None,
        task_context: str = "",
        max_attempts: int = 3,
        timeout_seconds: int = 300,
    ) -> Dict[str, Any]:
        """Execute a generated script with automatic bounded repair attempts on failure.

        Args:
            relative_path: Destination path under workspace/generated/.
            initial_code: Python code string.
            worker: Optional worker agent exposing `repair_code(code, error, task_context)`.
            task_context: Context string informing repair generation.
            max_attempts: Maximum retry count (default 3).
            timeout_seconds: Execution timeout per run.

        Returns:
            Dictionary with 'success' (bool), 'result' (ScriptExecutionResult), 'history' (list), and 'attempts' (int).
        """
        code = initial_code
        history: List[Dict[str, Any]] = []

        for attempt in range(max_attempts):
            logger.info(f"Executing '{relative_path}' (attempt {attempt + 1}/{max_attempts})...")
            # 1. Write script via filesystem MCP
            script_path = await self.write_script(relative_path, code, validate_syntax=True)

            # 2. Run script via command MCP
            res = await self.run_script(script_path, timeout_seconds=timeout_seconds)
            history.append({"attempt": attempt + 1, "result": res.model_dump()})

            if res.is_success:
                logger.info(f"Execution of '{relative_path}' succeeded on attempt {attempt + 1}")
                return {"success": True, "result": res, "history": history, "attempts": attempt + 1}

            logger.warning(
                f"Execution failed on attempt {attempt + 1} (status={res.status}, exit_code={res.exit_code})"
            )

            if attempt + 1 >= max_attempts or not worker or not hasattr(worker, "repair_code"):
                break

            # 3. Formulate bounded error context (never raw datasets)
            error_payload = json.dumps(
                {
                    "exit_code": res.exit_code,
                    "status": res.status,
                    "stdout_tail": res.stdout[-4000:] if res.stdout else "",
                    "stderr_tail": res.stderr[-8000:] if res.stderr else "",
                },
                ensure_ascii=False,
            )

            # 4. Invoke worker repair callback
            logger.info(f"Requesting code repair from worker for attempt {attempt + 2}...")
            code = await worker.repair_code(
                code=code,
                error=error_payload,
                task_context=task_context,
            )

        return {"success": False, "result": history[-1]["result"], "history": history, "attempts": len(history)}
