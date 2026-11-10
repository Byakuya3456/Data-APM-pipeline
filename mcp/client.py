"""Model Context Protocol (MCP) Client managing multi-server stdio sessions, reconnection, and tool execution."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings

logger = logging.getLogger("mlpipe.mcp.client")


class ToolResult(BaseModel):
    """Structured response object returned by MCP tool executions."""
    tool_name: str
    server_name: str
    is_error: bool = False
    content: List[Dict[str, Any]] = Field(default_factory=list)
    text: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    error_message: Optional[str] = None


@dataclass
class ServerConfig:
    """Configuration for an individual MCP stdio server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[Path] = None


class ServerSession:
    """Manages an active stdio connection and ClientSession to an MCP server in an isolated task."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, types.Tool] = {}
        self.is_connected: bool = False
        self.reconnect_count: int = 0
        self._ready_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._error: Optional[Exception] = None

    async def connect(self, default_timeout: float = 300.0) -> None:
        """Establish stdio transport and initialize MCP ClientSession in an isolated task."""
        if self.is_connected and self.session:
            return

        await self.disconnect()

        self._ready_event.clear()
        self._stop_event.clear()
        self._error = None

        self._task = asyncio.create_task(self._run(default_timeout=default_timeout))
        await self._ready_event.wait()

        if self._error:
            exc = self._error
            await self.disconnect()
            raise exc

    async def _run(self, default_timeout: float = 300.0) -> None:
        stack = AsyncExitStack()
        try:
            # Determine repository root to guarantee module resolution
            repo_root = str(Path(__file__).resolve().parent.parent)
            env = {**os.environ, **self.config.env}
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = repo_root + (os.pathsep + existing_pp if existing_pp else "")

            # Ensure command path is resolved
            cmd = self.config.command
            if cmd in ("uv", "python"):
                resolved_cmd = shutil.which(cmd) or cmd
            else:
                resolved_cmd = cmd

            params = StdioServerParameters(
                command=resolved_cmd,
                args=self.config.args,
                env=env,
                cwd=self.config.cwd,
            )
            read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream, read_timeout_seconds=default_timeout)
            )
            await session.initialize()

            # Discover registered tools
            tools_result = await session.list_tools()
            self.tools = {tool.name: tool for tool in tools_result.tools}
            self.session = session
            self.is_connected = True
            logger.info(
                f"Connected to MCP server '{self.config.name}' via stdio. "
                f"Discovered {len(self.tools)} tools: {list(self.tools.keys())}"
            )
            self._ready_event.set()

            # Wait until cancellation or stop request
            await self._stop_event.wait()
        except Exception as exc:
            self._error = exc
            self.is_connected = False
            self.session = None
            logger.error(f"Error in MCP server session '{self.config.name}': {exc}")
            self._ready_event.set()
        finally:
            self.is_connected = False
            self.session = None
            try:
                await stack.aclose()
            except Exception as e:
                logger.debug(f"Stack aclose error for '{self.config.name}': {e}")

    async def disconnect(self) -> None:
        """Cleanly tear down stdio transport and child process."""
        self._stop_event.set()
        if self._task:
            try:
                await self._task
            except Exception as exc:
                logger.debug(f"Error during task join of server '{self.config.name}': {exc}")
            finally:
                self._task = None
        self.is_connected = False
        self.session = None


class MCPClient:
    """Multi-server MCP client managing stdio sessions, tool routing, and auto-reconnection."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        settings: Optional[Settings] = None,
        auto_reconnect: bool = True,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        default_timeout: float = 300.0,
    ):
        self.settings = settings or get_settings()
        self.config_path = config_path or (self.settings.project_root / "config" / "mcp_config.json")
        self.auto_reconnect = auto_reconnect
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_timeout = default_timeout

        self._server_configs: Dict[str, ServerConfig] = {}
        self._servers: Dict[str, ServerSession] = {}
        self._tool_to_server_map: Dict[str, str] = {}
        self._load_server_configs()

    def _load_server_configs(self) -> None:
        """Load server definitions from config JSON or apply robust defaults."""
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                for name, spec in data.get("mcpServers", {}).items():
                    cmd = spec.get("command", sys.executable)
                    args = spec.get("args", [])
                    raw_env = spec.get("env", {})
                    server_env: Dict[str, str] = {}
                    for k, v in raw_env.items():
                        if k == "PROJECT_ROOT" and (v == "." or not Path(v).is_absolute()):
                            server_env[k] = str(self.settings.project_root)
                        else:
                            server_env[k] = str(v)

                    self._server_configs[name] = ServerConfig(
                        name=name,
                        command=cmd,
                        args=args,
                        env=server_env,
                        cwd=self.settings.project_root,
                    )
            except Exception as exc:
                logger.warning(f"Error loading {self.config_path}, falling back to defaults: {exc}")

        # Fallback default configurations if none loaded
        if "filesystem" not in self._server_configs:
            self._server_configs["filesystem"] = ServerConfig(
                name="filesystem",
                command=sys.executable,
                args=["-m", "mcp.filesystem_server"],
                env={"PROJECT_ROOT": str(self.settings.project_root)},
                cwd=self.settings.project_root,
            )
        if "command" not in self._server_configs:
            self._server_configs["command"] = ServerConfig(
                name="command",
                command=sys.executable,
                args=["-m", "mcp.command_server"],
                env={
                    "PROJECT_ROOT": str(self.settings.project_root),
                    "MAX_SCRIPT_SECONDS": str(self.settings.max_script_seconds),
                },
                cwd=self.settings.project_root,
            )

        for name, cfg in self._server_configs.items():
            self._servers[name] = ServerSession(cfg)

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self, server_names: Optional[List[str]] = None) -> None:
        """Connect to specified servers or all configured servers."""
        targets = server_names or list(self._servers.keys())
        for name in targets:
            if name in self._servers:
                await self._servers[name].connect(default_timeout=self.default_timeout)

        self._update_tool_index()

    def _update_tool_index(self) -> None:
        """Update global tool routing map from all active servers."""
        for server_name, server in self._servers.items():
            if server.is_connected:
                for tool_name in server.tools:
                    self._tool_to_server_map[tool_name] = server_name

    async def close(self) -> None:
        """Disconnect all active server sessions cleanly."""
        for server in self._servers.values():
            await server.disconnect()
        self._tool_to_server_map.clear()

    async def reconnect_server(self, server_name: str) -> None:
        """Reconnect a specific server session."""
        if server_name not in self._servers:
            raise ValueError(f"Unknown server '{server_name}'")
        server = self._servers[server_name]
        server.reconnect_count += 1
        logger.warning(f"Reconnecting to MCP server '{server_name}' (attempt {server.reconnect_count})...")
        await server.disconnect()
        await server.connect(default_timeout=self.default_timeout)
        self._update_tool_index()

    def is_connected(self, server_name: str) -> bool:
        """Check if a specific server is currently connected."""
        server = self._servers.get(server_name)
        return bool(server and server.is_connected and server.session)

    async def list_tools(self, server_name: Optional[str] = None) -> List[types.Tool]:
        """List available tools, optionally filtered by server name."""
        if server_name:
            server = self._servers.get(server_name)
            if not server:
                return []
            if not server.is_connected:
                await self.connect([server_name])
            return list(self._servers[server_name].tools.values())

        # All servers
        tools: List[types.Tool] = []
        for name, server in self._servers.items():
            if not server.is_connected:
                try:
                    await server.connect(default_timeout=self.default_timeout)
                except Exception as exc:
                    logger.warning(f"Could not connect to '{server.config.name}' to list tools: {exc}")
            tools.extend(server.tools.values())
        self._update_tool_index()
        return tools

    def find_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Find the server name that provides the requested tool."""
        server_name = self._tool_to_server_map.get(tool_name)
        if server_name:
            return server_name

        # Check in active servers tool dictionaries
        for name, server in self._servers.items():
            if tool_name in server.tools:
                self._tool_to_server_map[tool_name] = name
                return name

        # Well-known mappings as fallback
        if tool_name in {"read_file", "write_file", "list_directory", "get_file_info"}:
            return "filesystem"
        if tool_name in {"run_python_script", "get_execution_status", "read_execution_logs", "inspect_process_resources"}:
            return "command"

        return None

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        server_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ToolResult:
        """Invoke an MCP tool with auto-routing, reconnection resilience, and structured return."""
        args = arguments or {}
        effective_timeout = timeout_seconds or self.default_timeout

        target_server = server_name or self.find_server_for_tool(tool_name)
        if not target_server:
            # Attempt tool discovery once to update mapping
            try:
                await self.list_tools()
            except Exception:
                pass
            target_server = server_name or self.find_server_for_tool(tool_name)

        if not target_server or target_server not in self._servers:
            return ToolResult(
                tool_name=tool_name,
                server_name=target_server or "unknown",
                is_error=True,
                error_message=f"No connected MCP server provides tool '{tool_name}'",
            )

        server = self._servers[target_server]
        attempts = 0
        last_error: Optional[Exception] = None

        max_allowed_attempts = self.max_retries if self.auto_reconnect else 1

        while attempts < max_allowed_attempts:
            attempts += 1
            t_start = time.time()
            try:
                if not server.is_connected or not server.session:
                    await server.connect(default_timeout=self.default_timeout)
                    self._update_tool_index()

                assert server.session is not None
                res = await server.session.call_tool(
                    name=tool_name,
                    arguments=args,
                    read_timeout_seconds=effective_timeout,
                )
                duration = time.time() - t_start
                return self._parse_tool_result(res, tool_name, target_server, duration)

            except (MCPError, ConnectionError, anyio.ClosedResourceError, anyio.BrokenResourceError, OSError) as exc:
                duration = time.time() - t_start
                last_error = exc
                logger.warning(
                    f"MCP tool call '{tool_name}' on '{target_server}' failed (attempt {attempts}/{max_allowed_attempts}): {exc}"
                )
                if self.auto_reconnect and attempts < max_allowed_attempts:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempts - 1)))
                    try:
                        await self.reconnect_server(target_server)
                    except Exception as reconn_err:
                        logger.error(f"Reconnection to '{target_server}' failed: {reconn_err}")
                else:
                    break

        return ToolResult(
            tool_name=tool_name,
            server_name=target_server,
            is_error=True,
            error_message=f"Tool call '{tool_name}' failed after {attempts} attempts: {last_error}",
        )

    def _parse_tool_result(
        self,
        result: Any,
        tool_name: str,
        server_name: str,
        duration: float,
    ) -> ToolResult:
        """Parse official CallToolResult into a structured, typed ToolResult."""
        content_items: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        parsed_data: Dict[str, Any] = {}
        is_error = getattr(result, "is_error", False)

        structured = getattr(result, "structured_content", None)
        if isinstance(structured, dict):
            parsed_data = structured

        raw_content = getattr(result, "content", None) or []
        for item in raw_content:
            if hasattr(item, "model_dump"):
                content_items.append(item.model_dump())
            elif isinstance(item, dict):
                content_items.append(item)

            if getattr(item, "type", None) == "text" or isinstance(item, types.TextContent):
                text_parts.append(getattr(item, "text", ""))

        combined_text = "\n".join(text_parts).strip()

        if not parsed_data and combined_text:
            try:
                parsed_json = json.loads(combined_text)
                if isinstance(parsed_json, dict):
                    parsed_data = parsed_json
            except (json.JSONDecodeError, ValueError):
                pass

        error_msg = None
        if is_error:
            error_msg = combined_text or "MCP tool returned error status"

        return ToolResult(
            tool_name=tool_name,
            server_name=server_name,
            is_error=is_error,
            content=content_items,
            text=combined_text,
            data=parsed_data,
            duration_seconds=round(duration, 3),
            error_message=error_msg,
        )
