"""Restricted MCP command server.

Only executes Python scripts located under PROJECT_ROOT/workspace/generated.
No shell=True, no arbitrary executable selection, bounded timeout and output.
Run this server only on a machine and directory you control.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

ROOT = Path(os.getenv("PROJECT_ROOT", r"D:\project-data")).resolve()
GENERATED = (ROOT / "workspace" / "generated").resolve()
MAX_SECONDS = int(os.getenv("MAX_SCRIPT_SECONDS", "3600"))
MAX_OUTPUT = 24000

mcp = FastMCP("restricted-python-executor")


def _validate_script(script_path: str) -> Path:
    candidate = Path(script_path).resolve()
    if not candidate.is_relative_to(GENERATED):
        raise ValueError(f"Script must be inside {GENERATED}")
    if candidate.suffix.lower() != ".py" or not candidate.is_file():
        raise ValueError("A real .py script is required")
    return candidate


@mcp.tool()
def run_python(script_path: str, timeout_seconds: int = MAX_SECONDS) -> dict:
    """Run one Python script from workspace/generated and return bounded output."""
    script = _validate_script(script_path)
    timeout = max(1, min(int(timeout_seconds), MAX_SECONDS))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
