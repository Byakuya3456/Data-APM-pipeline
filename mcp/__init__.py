"""Model Context Protocol (MCP) subsystem package.

Integrates the official MCP SDK 2.3.0 with autonomous platform MCP components:
- filesystem_server
- command_server
- client (MCPClient, ToolResult, ServerConfig, ServerSession)
- executor (MCPExecutor, ScriptExecutionResult, ProcessResources)
"""
from __future__ import annotations

import sys
import importlib
import importlib.util
from pathlib import Path

# Locate site-packages mcp
_local_dir = Path(__file__).resolve().parent
_sp_mcp = None
for _p in sys.path:
    try:
        _cand = Path(_p) / "mcp"
        if _cand.is_dir() and (_cand / "__init__.py").exists():
            if _cand.resolve() != _local_dir:
                _sp_mcp = _cand.resolve()
                break
    except Exception:
        continue

# Ensure site-packages mcp is first in __path__, followed by local dir
if _sp_mcp:
    __path__ = [str(_sp_mcp), str(_local_dir)]
    _init_file = _sp_mcp / "__init__.py"
    with open(_init_file, "rb") as _f:
        _code = compile(_f.read(), str(_init_file), "exec")
        exec(_code, globals())
else:
    __path__ = [str(_local_dir)]

# Load local client implementation and attach to mcp.client and mcp
_client_file = _local_dir / "client.py"
if _client_file.exists():
    _spec = importlib.util.spec_from_file_location("mcp._client_impl", _client_file)
    if _spec and _spec.loader:
        _client_mod = importlib.util.module_from_spec(_spec)
        sys.modules["mcp._client_impl"] = _client_mod
        _spec.loader.exec_module(_client_mod)

        # Attach custom classes to mcp.client
        if "mcp.client" in sys.modules:
            _client_pkg = sys.modules["mcp.client"]
            _client_pkg.MCPClient = _client_mod.MCPClient  # type: ignore[attr-defined]
            _client_pkg.ToolResult = _client_mod.ToolResult  # type: ignore[attr-defined]
            _client_pkg.ServerConfig = _client_mod.ServerConfig  # type: ignore[attr-defined]
            _client_pkg.ServerSession = _client_mod.ServerSession  # type: ignore[attr-defined]

        # Also expose on mcp root
        MCPClient = _client_mod.MCPClient
        ToolResult = _client_mod.ToolResult
        ServerConfig = _client_mod.ServerConfig
        ServerSession = _client_mod.ServerSession

# Import and attach executor
from mcp.executor import MCPExecutor, ProcessResources, ScriptExecutionResult  # noqa: E402

__all__ = [
    "MCPClient",
    "ToolResult",
    "ServerConfig",
    "ServerSession",
    "MCPExecutor",
    "ScriptExecutionResult",
    "ProcessResources",
]
