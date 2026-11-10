"""Restricted Python Command Execution Model Context Protocol (MCP) Server.

Implements official MCP 2.3.0 server providing secure Python script execution,
AST pre-validation, environment scrubbing, process tree tracking, timeouts, and resource monitoring.
"""
from __future__ import annotations

import ast
import collections
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from mcp import types
from mcp.server.mcpserver import MCPServer

server = MCPServer(name="command-server")

ALLOWED_SCRIPT_SUBDIRS = ["workspace/generated", "workspace/scripts", "workspace", "temp"]


def get_project_root() -> Path:
    """Dynamically discover project root directory."""
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        candidate = Path(str(env_root).strip('"\'')).expanduser()
        try:
            resolved = candidate.resolve()
            if resolved.exists() and resolved.is_dir():
                return resolved
        except Exception:
            pass

    current = Path(__file__).resolve().parent
    for candidate in [current] + list(current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists() or (candidate / "PROJECT.md").exists():
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in [cwd] + list(cwd.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists() or (candidate / "PROJECT.md").exists():
            return candidate

    return cwd


class ExecutionRecord:
    """Tracks state, timing, and resource utilization for an execution."""

    def __init__(self, execution_id: str, script_path: str, root_dir: Path):
        self.execution_id = execution_id
        self.script_path = script_path
        self.status = "running"  # "running" | "completed" | "failed" | "timeout"
        self.exit_code: Optional[int] = None
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.duration_seconds = 0.0
        self.peak_memory_mb = 0.0
        self.pid: Optional[int] = None

        log_dir = root_dir / "workspace" / "logs" / "executions"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_log = log_dir / f"{execution_id}.stdout.log"
        self.stderr_log = log_dir / f"{execution_id}.stderr.log"


_executions: Dict[str, ExecutionRecord] = {}
_executions_lock = threading.Lock()


def validate_and_parse_script(script_path: str, root: Path) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    """Validate script containment, extension, existence, allowed dir, and AST syntax."""
    raw = Path(script_path)
    resolved = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()

    try:
        resolved.relative_to(root)
    except ValueError:
        return None, None, f"Access denied: script '{script_path}' is outside project root"

    if resolved.suffix.lower() != ".py":
        return None, None, f"Script must be a .py file: '{script_path}'"

    if not resolved.is_file():
        return None, None, f"Script not found: '{script_path}'"

    # Verify allowed directory containment
    allowed_roots = [(root / sub).resolve() for sub in ALLOWED_SCRIPT_SUBDIRS]
    if not any(resolved == ar or resolved.is_relative_to(ar) for ar in allowed_roots):
        return None, None, f"Script must reside inside one of {ALLOWED_SCRIPT_SUBDIRS}"

    code = resolved.read_text(encoding="utf-8")
    try:
        ast.parse(code, filename=str(resolved))
    except SyntaxError as e:
        return None, None, f"AST SyntaxError in '{script_path}' at line {e.lineno}, col {e.offset}: {e.msg}\n{e.text}"

    return resolved, code, None


def sanitize_environment(root: Path, custom_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Sanitize environment variables removing sensitive tokens and secrets."""
    safe_env: Dict[str, str] = {}
    SENSITIVE_PATTERNS = ("KEY", "SECRET", "TOKEN", "PASSWD", "PASSWORD", "AUTH", "CREDENTIAL", "PRIVATE")

    for k, v in os.environ.items():
        if any(pat in k.upper() for pat in SENSITIVE_PATTERNS):
            continue
        safe_env[k] = v

    safe_env["PYTHONUNBUFFERED"] = "1"
    existing_pp = safe_env.get("PYTHONPATH", "")
    safe_env["PYTHONPATH"] = str(root) + (os.pathsep + existing_pp if existing_pp else "")

    if custom_env:
        for k, v in custom_env.items():
            safe_env[k] = str(v)

    return safe_env


def terminate_process_tree(pid: int, timeout: float = 3.0) -> None:
    """Terminate a process and all of its recursive child processes on Windows."""
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return

    try:
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []

    all_procs = children + [parent]
    for p in all_procs:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    gone, alive = psutil.wait_procs(all_procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


@server.tool()
def run_python_script(
    script_path: str,
    timeout_seconds: int = 300,
    env_vars: Optional[Dict[str, str]] = None,
    background: bool = False,
):
    """Execute a Python script in a restricted subprocess with AST pre-check and resource monitoring.

    Args:
        script_path: Path to .py script relative to project root.
        timeout_seconds: Maximum execution time before termination (default 300s).
        env_vars: Optional dictionary of additional environment variables.
        background: If True, returns immediately while execution runs asynchronously.
    """
    root = get_project_root()
    resolved, _, err = validate_and_parse_script(script_path, root)
    if err:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=err)],
        )

    assert resolved is not None
    execution_id = str(uuid.uuid4())
    record = ExecutionRecord(execution_id, script_path, root)

    with _executions_lock:
        _executions[execution_id] = record

    safe_env = sanitize_environment(root, env_vars)

    out_f = open(record.stdout_log, "w", encoding="utf-8")
    err_f = open(record.stderr_log, "w", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            [sys.executable, str(resolved)],
            cwd=str(root),
            stdout=out_f,
            stderr=err_f,
            env=safe_env,
            shell=False,
        )
        record.pid = proc.pid
    except Exception as exc:
        out_f.close()
        err_f.close()
        record.status = "failed"
        record.exit_code = -1
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=f"Failed to spawn process for '{script_path}': {exc}")],
        )

    # Initial memory sample
    try:
        p = psutil.Process(proc.pid)
        init_mem = p.memory_info().rss / (1024 * 1024)
        record.peak_memory_mb = max(record.peak_memory_mb, init_mem)
    except Exception:
        record.peak_memory_mb = max(record.peak_memory_mb, 1.0)

    def monitor_process() -> None:
        timed_out = False
        while proc.poll() is None:
            elapsed = time.time() - record.start_time
            if elapsed > timeout_seconds:
                timed_out = True
                terminate_process_tree(proc.pid)
                break

            try:
                p = psutil.Process(proc.pid)
                mem = p.memory_info().rss
                for c in p.children(recursive=True):
                    try:
                        mem += c.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                mem_mb = mem / (1024 * 1024)
                if mem_mb > record.peak_memory_mb:
                    record.peak_memory_mb = mem_mb
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            time.sleep(0.05)

        out_f.close()
        err_f.close()
        record.end_time = time.time()
        record.duration_seconds = record.end_time - record.start_time

        if timed_out:
            record.status = "timeout"
            record.exit_code = -1
        else:
            record.exit_code = proc.returncode
            record.status = "completed" if record.exit_code == 0 else "failed"

    if background:
        thread = threading.Thread(target=monitor_process, daemon=True)
        thread.start()
        return {
            "execution_id": execution_id,
            "status": "running",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
            "peak_memory_mb": round(record.peak_memory_mb, 2),
        }

    # Synchronous execution
    monitor_process()

    stdout_content = ""
    stderr_content = ""
    if record.stdout_log.exists():
        try:
            stdout_content = record.stdout_log.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if record.stderr_log.exists():
        try:
            stderr_content = record.stderr_log.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    bounded_stdout = stdout_content[-24000:] if len(stdout_content) > 24000 else stdout_content
    bounded_stderr = stderr_content[-24000:] if len(stderr_content) > 24000 else stderr_content

    return {
        "execution_id": record.execution_id,
        "status": record.status,
        "exit_code": record.exit_code,
        "stdout": bounded_stdout,
        "stderr": bounded_stderr,
        "duration_seconds": round(record.duration_seconds, 3),
        "peak_memory_mb": round(record.peak_memory_mb, 2),
    }


@server.tool()
def get_execution_status(execution_id: str):
    """Retrieve execution status for a given execution ID.

    Args:
        execution_id: UUID string returned by run_python_script.
    """
    with _executions_lock:
        record = _executions.get(execution_id)

    if not record:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=f"Unknown execution ID: '{execution_id}'")],
        )

    duration = (time.time() - record.start_time) if record.end_time is None else record.duration_seconds
    return {
        "execution_id": execution_id,
        "status": record.status,
        "exit_code": record.exit_code,
        "start_time": record.start_time,
        "end_time": record.end_time,
        "duration_seconds": round(duration, 3),
        "script_path": record.script_path,
    }


@server.tool()
def read_execution_logs(execution_id: str, tail_lines: int = 100):
    """Read stdout and stderr log tails from disk for an execution.

    Args:
        execution_id: UUID string returned by run_python_script.
        tail_lines: Number of trailing log lines to retrieve (default 100).
    """
    with _executions_lock:
        record = _executions.get(execution_id)

    if not record:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=f"Unknown execution ID: '{execution_id}'")],
        )

    stdout_lines: List[str] = []
    if record.stdout_log.exists():
        try:
            with open(record.stdout_log, "r", encoding="utf-8", errors="replace") as f:
                stdout_lines = list(collections.deque(f, maxlen=tail_lines))
        except Exception:
            pass

    stderr_lines: List[str] = []
    if record.stderr_log.exists():
        try:
            with open(record.stderr_log, "r", encoding="utf-8", errors="replace") as f:
                stderr_lines = list(collections.deque(f, maxlen=tail_lines))
        except Exception:
            pass

    return {
        "execution_id": execution_id,
        "status": record.status,
        "tail_lines": tail_lines,
        "stdout": "".join(stdout_lines),
        "stderr": "".join(stderr_lines),
        "stdout_tail": [line.rstrip("\r\n") for line in stdout_lines],
        "stderr_tail": [line.rstrip("\r\n") for line in stderr_lines],
        "total_stdout_lines": len(stdout_lines),
        "total_stderr_lines": len(stderr_lines),
    }


@server.tool()
def inspect_process_resources(execution_id: str):
    """Inspect CPU and memory resource consumption of an active or completed process.

    Args:
        execution_id: UUID string returned by run_python_script.
    """
    with _executions_lock:
        record = _executions.get(execution_id)

    if not record:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=f"Unknown execution ID: '{execution_id}'")],
        )

    rss_mb = record.peak_memory_mb
    vms_mb = 0.0
    cpu_pct = 0.0
    is_alive = False

    if record.pid is not None and record.status == "running":
        try:
            p = psutil.Process(record.pid)
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                is_alive = True
                mem_info = p.memory_info()
                rss = mem_info.rss
                vms = mem_info.vms
                for c in p.children(recursive=True):
                    try:
                        c_mem = c.memory_info()
                        rss += c_mem.rss
                        vms += c_mem.vms
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                rss_mb = rss / (1024 * 1024)
                vms_mb = vms / (1024 * 1024)
                cpu_pct = p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            is_alive = False

    duration = (time.time() - record.start_time) if record.end_time is None else record.duration_seconds
    return {
        "execution_id": execution_id,
        "pid": record.pid,
        "status": record.status if is_alive else ("completed" if record.status == "running" else record.status),
        "memory_rss_mb": round(rss_mb, 2),
        "memory_vms_mb": round(vms_mb, 2),
        "cpu_percent": round(cpu_pct, 2),
        "duration_seconds": round(duration, 3),
        "is_alive": is_alive,
    }


def main() -> None:
    """Entry point for stdio MCP command server."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
