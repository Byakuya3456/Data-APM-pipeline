"""Pure-Python Model Context Protocol (MCP) Filesystem Server.

Implements official MCP 2.3.0 server providing secure file operations restricted
to the designated project root directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import types
from mcp.server.mcpserver import MCPServer

# Initialize MCP server
server = MCPServer(name="filesystem-server")


def get_project_root() -> Path:
    """Dynamically discover project root directory.

    Order of precedence:
    1. PROJECT_ROOT environment variable (if exists and is a directory).
    2. Parent traversal searching for pyproject.toml, .git, or PROJECT.md.
    3. Fallback to current working directory.
    """
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


def validate_path(rel_or_abs_path: str, must_exist: bool = False) -> tuple[Path, Optional[str]]:
    """Validate that path resides strictly inside project root and optionally exists."""
    root = get_project_root()
    raw = Path(rel_or_abs_path)
    resolved = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved, f"Access denied: path '{rel_or_abs_path}' escapes project root '{root}'"

    if must_exist and not resolved.exists():
        return resolved, f"Path does not exist: '{rel_or_abs_path}'"

    return resolved, None


@server.tool()
def read_file(path: str, max_bytes: int = 1048576):
    """Read file content with strict project root containment and byte bounding.

    Args:
        path: Path relative to project root or absolute path within project root.
        max_bytes: Maximum bytes to read before truncating (default 1 MB).
    """
    resolved, err = validate_path(path, must_exist=True)
    if err:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=err)],
        )

    if not resolved.is_file():
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=f"Path is not a file: '{path}'")],
        )

    root = get_project_root()
    try:
        file_stat = resolved.stat()
        with open(resolved, "rb") as f:
            raw_bytes = f.read(max_bytes + 1)
        truncated = len(raw_bytes) > max_bytes
        content_bytes = raw_bytes[:max_bytes] if truncated else raw_bytes
        content_str = content_bytes.decode("utf-8", errors="replace")
        rel_path = str(resolved.relative_to(root)).replace("\\", "/")
        return {
            "path": rel_path,
            "content": content_str,
            "size_bytes": file_stat.st_size,
            "truncated": truncated,
        }
    except Exception as exc:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=f"Failed to read '{path}': {exc}")],
        )


@server.tool()
def write_file(path: str, content: str):
    """Write text content to a file within project root, creating parent directories.

    Args:
        path: Path relative to project root or absolute path within project root.
        content: Text content to write encoded as UTF-8.
    """
    resolved, err = validate_path(path, must_exist=False)
    if err:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=err)],
        )

    root = get_project_root()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        rel_path = str(resolved.relative_to(root)).replace("\\", "/")
        return {
            "path": rel_path,
            "bytes_written": len(content.encode("utf-8")),
            "status": "success",
        }
    except Exception as exc:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=f"Failed to write '{path}': {exc}")],
        )


@server.tool()
def list_directory(path: str = "."):
    """List directory contents within project root.

    Args:
        path: Directory path relative to project root (defaults to root).
    """
    resolved, err = validate_path(path, must_exist=True)
    if err:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=err)],
        )

    if not resolved.is_dir():
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=f"Path is not a directory: '{path}'")],
        )

    root = get_project_root()
    try:
        entries: List[Dict[str, Any]] = []
        for entry in os.scandir(resolved):
            if entry.name in {".git", ".venv", "__pycache__"}:
                continue
            try:
                stat = entry.stat()
                entry_path = Path(entry.path).resolve()
                rel_path = str(entry_path.relative_to(root)).replace("\\", "/")
                entries.append({
                    "name": entry.name,
                    "path": rel_path,
                    "is_dir": entry.is_dir(),
                    "is_file": entry.is_file(),
                    "size_bytes": stat.st_size if entry.is_file() else None,
                    "modified_time": stat.st_mtime,
                })
            except (OSError, PermissionError):
                continue

        rel_dir = str(resolved.relative_to(root)).replace("\\", "/") if resolved != root else "."
        return {
            "path": rel_dir,
            "entries": entries,
            "total_count": len(entries),
        }
    except Exception as exc:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=f"Failed to list directory '{path}': {exc}")],
        )


@server.tool()
def get_file_info(path: str):
    """Inspect file or directory metadata within project root.

    Args:
        path: Path relative to project root or absolute path within project root.
    """
    resolved, err = validate_path(path, must_exist=False)
    if err:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=err)],
        )

    root = get_project_root()
    try:
        rel_path = str(resolved.relative_to(root)).replace("\\", "/")
    except ValueError:
        rel_path = str(resolved).replace("\\", "/")

    if not resolved.exists():
        return {
            "path": rel_path,
            "exists": False,
        }

    try:
        stat = resolved.stat()
        return {
            "path": rel_path,
            "exists": True,
            "is_file": resolved.is_file(),
            "is_dir": resolved.is_dir(),
            "size_bytes": stat.st_size if resolved.is_file() else None,
            "created_time": stat.st_ctime,
            "modified_time": stat.st_mtime,
            "extension": resolved.suffix,
        }
    except Exception as exc:
        return types.CallToolResult(
            is_error=True,
            content=[types.TextContent(type="text", text=f"Failed to get info for '{path}': {exc}")],
        )


def main() -> None:
    """Entry point for stdio MCP server execution."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
