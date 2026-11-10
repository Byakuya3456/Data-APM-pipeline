"""MCP client and bounded self-correction loop."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPExecutor:
    def __init__(self, config_path: Path, project_root: Path):
        self.config_path = config_path
        self.project_root = project_root.resolve()
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

    async def _call(self, server_name: str, tool_name: str, arguments: dict[str, Any]):
        spec = self.config["mcpServers"][server_name]
        params = StdioServerParameters(
            command=spec["command"],
            args=spec.get("args", []),
            env={**os.environ, **spec.get("env", {})},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                # MCP SDK returns content blocks; preserve text and structured data.
                text_parts = [
                    item.text for item in (result.content or [])
                    if getattr(item, "type", None) == "text"
                ]
                structured = getattr(result, "structuredContent", None)
                return structured if structured is not None else "\n".join(text_parts)

    async def write_script(self, relative_path: str, code: str) -> str:
        """Write generated code using the filesystem MCP server."""
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Path traversal rejected")
        if rel.suffix != ".py":
            raise ValueError("Only .py scripts may be written")
        target = (self.project_root / rel).resolve()
        if not target.is_relative_to(self.project_root):
            raise ValueError("Path escapes project root")
        # The official filesystem server exposes write_file.
        await self._call("filesystem", "write_file", {
            "path": str(target),
            "content": code
        })
        return str(target)

    async def run_script(self, absolute_path: str, timeout_seconds: int = 3600):
        return await self._call("command", "run_python", {
            "script_path": absolute_path,
            "timeout_seconds": timeout_seconds
        })

    async def execute_with_repair(
        self, *, relative_path: str, initial_code: str, worker,
        task_context: str, max_attempts: int = 2, timeout_seconds: int = 3600
    ) -> dict:
        code = initial_code
        history = []
        for attempt in range(max_attempts + 1):
            script_path = await self.write_script(relative_path, code)
            result = await self.run_script(script_path, timeout_seconds)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {"returncode": -1, "stdout": result, "stderr": ""}
            history.append({"attempt": attempt + 1, "result": result})
            if int(result.get("returncode", -1)) == 0:
                return {"success": True, "result": result, "history": history}
            if attempt >= max_attempts:
                break
            # Only bounded stderr/stdout are sent to the model, never dataset rows.
            code = await worker.repair_code(
                code=code,
                error=json.dumps(result, ensure_ascii=False)[:12000],
                task_context=task_context,
            )
        return {"success": False, "result": history[-1]["result"], "history": history}
