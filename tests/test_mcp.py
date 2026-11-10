"""Comprehensive test suite for Settings, Logging, MCP Servers, Client, and Executor."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from config.logging_config import ColoredConsoleFormatter, get_logger, setup_logging
from config.settings import Settings, discover_project_root, get_settings
from mcp.client import MCPClient, ToolResult
from mcp.executor import MCPExecutor, ProcessResources, ScriptExecutionResult


# ----------------------------------------------------------------------
# Windows Asynchronous Test Fixture
# ----------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def setup_windows_event_loop():
    """Guarantee WindowsProactorEventLoopPolicy is active for subprocess pipes on Windows."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def run_async(coro):
    """Synchronous helper to run coroutines cleanly under a fresh Proactor loop on Windows."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.run(coro)


# ----------------------------------------------------------------------
# Part 1: Settings Tests
# ----------------------------------------------------------------------

class TestSettings:
    """Validate Pydantic BaseSettings, dynamic root discovery, and directory management."""

    def test_default_settings_instantiation(self):
        settings = Settings()
        assert settings.project_root.exists()
        assert settings.project_root.is_absolute()
        assert settings.duckdb_memory_limit == "8GB"
        assert settings.duckdb_threads == 4
        assert settings.max_repair_attempts == 3
        assert settings.max_script_seconds == 3600
        assert settings.max_data_sample_rows == 100

    def test_dynamic_root_discovery(self):
        root = discover_project_root()
        assert root.is_absolute()
        assert (root / "pyproject.toml").exists() or (root / "PROJECT.md").exists()

    def test_ensure_directories(self, tmp_path):
        custom_settings = Settings(
            project_root=tmp_path,
            data_input_dir=Path("data/input"),
            workspace_dir=Path("workspace"),
            temp_dir=Path("temp"),
        )
        custom_settings.ensure_directories()
        assert (tmp_path / "data" / "input").is_dir()
        assert (tmp_path / "data" / "samples").is_dir()
        assert (tmp_path / "workspace" / "generated").is_dir()
        assert (tmp_path / "workspace" / "artifacts").is_dir()
        assert (tmp_path / "workspace" / "logs").is_dir()
        assert (tmp_path / "temp").is_dir()

    def test_environment_variable_overrides(self, monkeypatch):
        monkeypatch.setenv("DUCKDB_MEMORY_LIMIT", "16GB")
        monkeypatch.setenv("AI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("MAX_REPAIR_ATTEMPTS", "5")
        settings = Settings()
        assert settings.duckdb_memory_limit == "16GB"
        assert settings.openai_model == "gpt-4o-mini"
        assert settings.max_repair_attempts == 5


# ----------------------------------------------------------------------
# Part 2: Logging Tests
# ----------------------------------------------------------------------

class TestLogging:
    """Validate dual logging configuration, ANSI color formatting, and file rotation."""

    def test_setup_logging_and_file_creation(self, tmp_path):
        log_file = tmp_path / "logs" / "test_app.log"
        logger = setup_logging(log_level="DEBUG", log_file=log_file)
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")

        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "Test info message" in content
        assert "Test warning message" in content
        assert "Test error message" in content
        # Ensure file output is stripped of ANSI color codes
        assert "\033[" not in content

    def test_colored_console_formatter(self):
        import logging
        formatter = ColoredConsoleFormatter(fmt="%(levelname)s: %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=10,
            msg="Sample error",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "\033[31m" in formatted  # Red ANSI code
        assert "\033[0m" in formatted   # Reset ANSI code
        # Verify original record levelname is restored
        assert record.levelname == "ERROR"


# ----------------------------------------------------------------------
# Part 3: Filesystem Server Unit Tests
# ----------------------------------------------------------------------

class TestFilesystemServerUnit:
    """Validate filesystem server tools in-process via MCPServer.call_tool()."""

    @pytest.fixture
    def fs_server(self, monkeypatch, tmp_path):
        from mcp.filesystem_server import server
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        return server

    def test_write_and_read_file(self, fs_server, tmp_path):
        async def run():
            # Write file
            w_res = await fs_server.call_tool("write_file", {"path": "test.txt", "content": "Hello World!"})
            assert not w_res.is_error
            assert (tmp_path / "test.txt").read_text(encoding="utf-8") == "Hello World!"

            # Read file
            r_res = await fs_server.call_tool("read_file", {"path": "test.txt"})
            assert not r_res.is_error
            data = json.loads(r_res.content[0].text)
            assert data["content"] == "Hello World!"
            assert data["truncated"] is False

        run_async(run())

    def test_list_directory_and_file_info(self, fs_server, tmp_path):
        async def run():
            (tmp_path / "sub").mkdir()
            (tmp_path / "sub" / "file1.txt").write_text("abc", encoding="utf-8")

            list_res = await fs_server.call_tool("list_directory", {"path": "sub"})
            assert not list_res.is_error
            data = json.loads(list_res.content[0].text)
            assert data["total_count"] == 1
            assert data["entries"][0]["name"] == "file1.txt"

            info_res = await fs_server.call_tool("get_file_info", {"path": "sub/file1.txt"})
            assert not info_res.is_error
            info = json.loads(info_res.content[0].text)
            assert info["exists"] is True
            assert info["size_bytes"] == 3

        run_async(run())

    def test_security_path_traversal_rejected(self, fs_server):
        async def run():
            # Attempt directory traversal with ..
            res = await fs_server.call_tool("read_file", {"path": "../../secret.txt"})
            assert res.is_error
            assert "escapes project root" in res.content[0].text

        run_async(run())


# ----------------------------------------------------------------------
# Part 4: Command Server Unit Tests & Security
# ----------------------------------------------------------------------

class TestCommandServerUnit:
    """Validate command server AST checks, timeouts, security, and telemetry in-process."""

    @pytest.fixture
    def cmd_server(self, monkeypatch, tmp_path):
        from mcp.command_server import server
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        (tmp_path / "workspace" / "generated").mkdir(parents=True, exist_ok=True)
        (tmp_path / "workspace" / "logs" / "executions").mkdir(parents=True, exist_ok=True)
        return server

    def test_run_python_script_success(self, cmd_server, tmp_path):
        async def run():
            script = tmp_path / "workspace" / "generated" / "ok.py"
            script.write_text("import json\nprint(json.dumps({'status': 'ok', 'count': 42}))\n", encoding="utf-8")

            res = await cmd_server.call_tool("run_python_script", {"script_path": "workspace/generated/ok.py"})
            assert not res.is_error
            data = json.loads(res.content[0].text)
            assert data["status"] == "completed"
            assert data["exit_code"] == 0
            assert "42" in data["stdout"]
            assert data["duration_seconds"] >= 0
            assert data["peak_memory_mb"] > 0

        run_async(run())

    def test_ast_syntax_error_precheck(self, cmd_server, tmp_path):
        async def run():
            script = tmp_path / "workspace" / "generated" / "bad_syntax.py"
            script.write_text("def broken_func(\n    print('fail')\n", encoding="utf-8")

            res = await cmd_server.call_tool("run_python_script", {"script_path": "workspace/generated/bad_syntax.py"})
            assert res.is_error
            assert "AST SyntaxError" in res.content[0].text

        run_async(run())

    def test_timeout_enforcement(self, cmd_server, tmp_path):
        async def run():
            script = tmp_path / "workspace" / "generated" / "sleepy.py"
            script.write_text("import time\ntime.sleep(10)\n", encoding="utf-8")

            t0 = time.time()
            res = await cmd_server.call_tool(
                "run_python_script",
                {"script_path": "workspace/generated/sleepy.py", "timeout_seconds": 1},
            )
            elapsed = time.time() - t0
            assert not res.is_error
            data = json.loads(res.content[0].text)
            assert data["status"] == "timeout"
            assert data["exit_code"] == -1
            assert elapsed < 4.0

        run_async(run())

    def test_get_execution_status_and_logs(self, cmd_server, tmp_path):
        async def run():
            script = tmp_path / "workspace" / "generated" / "log_test.py"
            script.write_text("print('Line 1')\nprint('Line 2')\n", encoding="utf-8")

            res = await cmd_server.call_tool("run_python_script", {"script_path": "workspace/generated/log_test.py"})
            data = json.loads(res.content[0].text)
            exec_id = data["execution_id"]

            status_res = await cmd_server.call_tool("get_execution_status", {"execution_id": exec_id})
            assert not status_res.is_error
            st = json.loads(status_res.content[0].text)
            assert st["status"] == "completed"

            logs_res = await cmd_server.call_tool("read_execution_logs", {"execution_id": exec_id, "tail_lines": 10})
            assert not logs_res.is_error
            lg = json.loads(logs_res.content[0].text)
            assert "Line 1" in "\n".join(lg["stdout_tail"])

        run_async(run())

    def test_environment_variable_sanitization(self, cmd_server, tmp_path, monkeypatch):
        async def run():
            monkeypatch.setenv("SECRET_API_KEY", "super_secret_value")
            script = tmp_path / "workspace" / "generated" / "env_check.py"
            script.write_text("import os\nprint('SECRET_KEY_FOUND:' + str('SECRET_API_KEY' in os.environ))\n", encoding="utf-8")

            res = await cmd_server.call_tool("run_python_script", {"script_path": "workspace/generated/env_check.py"})
            data = json.loads(res.content[0].text)
            assert "SECRET_KEY_FOUND:False" in data["stdout"]

        run_async(run())


# ----------------------------------------------------------------------
# Part 5: End-to-End MCP Client & Executor Integration Tests
# ----------------------------------------------------------------------

class TestMCPClientAndExecutorIntegration:
    """Validate MCPClient and MCPExecutor over live stdio transport on Windows."""

    @pytest.fixture
    def test_settings(self, tmp_path):
        s = Settings(
            project_root=tmp_path,
            data_input_dir=Path("data/input"),
            workspace_dir=Path("workspace"),
            temp_dir=Path("temp"),
        )
        s.ensure_directories()
        return s

    def test_client_connect_and_tool_discovery(self, test_settings):
        async def run():
            async with MCPClient(settings=test_settings) as client:
                tools = await client.list_tools()
                tool_names = [t.name for t in tools]
                # Filesystem tools
                assert "read_file" in tool_names
                assert "write_file" in tool_names
                assert "list_directory" in tool_names
                # Command execution tools
                assert "run_python_script" in tool_names
                assert "get_execution_status" in tool_names
                assert "inspect_process_resources" in tool_names

        run_async(run())

    def test_executor_write_and_run_script_e2e(self, test_settings):
        async def run():
            async with MCPExecutor(settings=test_settings) as executor:
                code = "print('E2E MCP execution success!')\n"
                path = await executor.write_script("workspace/generated/e2e.py", code)
                assert path.exists()

                exec_res = await executor.run_script(path)
                assert isinstance(exec_res, ScriptExecutionResult)
                assert exec_res.is_success
                assert "E2E MCP execution success!" in exec_res.stdout
                assert exec_res.peak_memory_mb > 0

        run_async(run())

    def test_executor_execute_with_repair_loop(self, test_settings):
        async def run():
            class MockRepairWorker:
                def __init__(self):
                    self.call_count = 0

                async def repair_code(self, code: str, error: str, task_context: str) -> str:
                    self.call_count += 1
                    return "print('Repair succeeded!')\n"

            worker = MockRepairWorker()
            broken_code = "raise RuntimeError('Initial execution failed!')\n"

            async with MCPExecutor(settings=test_settings) as executor:
                result = await executor.execute_with_repair(
                    relative_path="workspace/generated/repaired.py",
                    initial_code=broken_code,
                    worker=worker,
                    max_attempts=3,
                )
                assert result["success"] is True
                assert result["attempts"] == 2
                assert worker.call_count == 1
                assert "Repair succeeded!" in result["result"].stdout

        run_async(run())


# ----------------------------------------------------------------------
# Part 6: Client Resilience & Reconnection Tests
# ----------------------------------------------------------------------

class TestClientResilience:
    """Validate client resilience, reconnection upon server drop, and error handling."""

    def test_unknown_tool_returns_error_result(self, tmp_path):
        async def run():
            s = Settings(project_root=tmp_path)
            s.ensure_directories()
            async with MCPClient(settings=s) as client:
                res = await client.call_tool("non_existent_tool_xyz", {})
                assert isinstance(res, ToolResult)
                assert res.is_error is True
                assert "No connected MCP server provides tool" in res.error_message

        run_async(run())

    def test_auto_reconnect_after_server_disconnect(self, tmp_path):
        async def run():
            s = Settings(project_root=tmp_path)
            s.ensure_directories()
            async with MCPClient(settings=s) as client:
                await client.connect(["filesystem"])
                assert client.is_connected("filesystem")

                # Deliberately disconnect the session
                await client._servers["filesystem"].disconnect()
                assert not client.is_connected("filesystem")

                # Next tool call should trigger transparent auto-reconnection
                res = await client.call_tool("write_file", {"path": "reconn.txt", "content": "alive"})
                assert not res.is_error
                assert client.is_connected("filesystem")
                assert (tmp_path / "reconn.txt").read_text(encoding="utf-8") == "alive"

        run_async(run())
