"""DuckDB SQL execution engine with out-of-core memory management and disk spilling."""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import polars as pl

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger("engine.duckdb_engine")


class DuckDBEngine:
    """Out-of-core SQL query engine powered by DuckDB.

    Features:
    - Bounded memory limits (default '8GB') and disk spilling to dedicated temp_directory.
    - Zero-copy conversion to Polars DataFrame via relation .pl().
    - In-place streaming execution on Parquet, CSV, TSV, and JSON/JSONL without Python copying.
    - Out-of-core SUMMARIZE statistical profiling.
    - Windows file locking mitigation via context management and handle disposal.
    """

    def __init__(
        self,
        memory_limit: Optional[str] = None,
        temp_directory: Optional[Union[Path, str]] = None,
        threads: Optional[int] = None,
        database: str = ":memory:",
    ) -> None:
        settings = get_settings()
        self.memory_limit: str = memory_limit or settings.duckdb_memory_limit
        self.threads: int = threads or settings.duckdb_threads

        if temp_directory is not None:
            self.temp_directory: Path = Path(temp_directory).resolve()
        else:
            self.temp_directory = settings.temp_dir / "duckdb_spill"

        self.database: str = database
        self._con: Optional[duckdb.DuckDBPyConnection] = None
        self._init_connection()

    def _init_connection(self) -> None:
        """Initialize DuckDB connection and enforce out-of-core settings."""
        self.temp_directory.mkdir(parents=True, exist_ok=True)
        posix_temp = self.temp_directory.as_posix().replace("'", "''")

        self._con = duckdb.connect(database=self.database)
        self._con.execute(f"SET memory_limit = '{self.memory_limit}';")
        self._con.execute(f"SET temp_directory = '{posix_temp}';")
        self._con.execute(f"SET threads = {self.threads};")
        self._con.execute("SET preserve_insertion_order = false;")
        logger.debug(
            "DuckDB initialized (memory_limit=%s, threads=%d, temp_dir=%s)",
            self.memory_limit,
            self.threads,
            self.temp_directory,
        )

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Return the active connection or re-initialize if closed."""
        if self._con is None:
            self._init_connection()
        return self._con

    def query(
        self,
        sql: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ) -> duckdb.DuckDBPyConnection:
        """Execute SQL query returning the connection for fetchall/fetchone operations."""
        con = self.connect()
        if params:
            return con.execute(sql, params)
        return con.execute(sql)

    def query_relation(
        self,
        sql: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ) -> duckdb.DuckDBPyRelation:
        """Execute SQL query returning DuckDBPyRelation supporting zero-copy .pl()."""
        con = self.connect()
        if params:
            return con.query(sql, params=params)
        return con.query(sql)

    def query_df(
        self,
        sql: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ) -> pl.DataFrame:
        """Execute SQL query and return zero-copy Polars DataFrame."""
        rel = self.query_relation(sql, params=params)
        return rel.pl()

    def query_to_parquet(
        self,
        sql: str,
        output_path: Union[Path, str],
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
        compression: str = "zstd",
    ) -> Path:
        """Stream SQL query results directly into Parquet file on disk out-of-core."""
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        posix_target = target.as_posix().replace("'", "''")

        copy_sql = f"COPY ({sql}) TO '{posix_target}' (FORMAT PARQUET, COMPRESSION '{compression.upper()}');"
        con = self.connect()
        if params:
            con.execute(copy_sql, params)
        else:
            con.execute(copy_sql)
        logger.info("Streamed query result directly to parquet: %s", target)
        return target

    @staticmethod
    def get_read_function(file_path: Union[Path, str]) -> str:
        """Map file extension to optimal DuckDB reader table function."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            return "read_parquet"
        elif suffix in {".csv"}:
            return "read_csv_auto"
        elif suffix in {".tsv"}:
            return "read_csv_auto"
        elif suffix in {".json", ".jsonl", ".ndjson"}:
            return "read_json_auto"
        else:
            raise ValueError(f"Unsupported dataset format for DuckDB direct read: {suffix}")

    def summarize_file(self, file_path: Union[Path, str]) -> pl.DataFrame:
        """Execute single-pass out-of-core SUMMARIZE profiling on a file."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        read_fn = self.get_read_function(path)
        posix_path = path.as_posix().replace("'", "''")

        if path.suffix.lower() == ".tsv":
            sql = f"SUMMARIZE SELECT * FROM {read_fn}('{posix_path}', delim='\\t');"
        else:
            sql = f"SUMMARIZE SELECT * FROM {read_fn}('{posix_path}');"

        return self.query_df(sql)

    def register_view(self, view_name: str, file_path: Union[Path, str]) -> None:
        """Create temporary SQL view directly over file without memory copy."""
        path = Path(file_path).resolve()
        read_fn = self.get_read_function(path)
        posix_path = path.as_posix().replace("'", "''")
        if path.suffix.lower() == ".tsv":
            sql = f"CREATE OR REPLACE TEMPORARY VIEW {view_name} AS SELECT * FROM {read_fn}('{posix_path}', delim='\\t');"
        else:
            sql = f"CREATE OR REPLACE TEMPORARY VIEW {view_name} AS SELECT * FROM {read_fn}('{posix_path}');"
        self.connect().execute(sql)

    def unregister_view(self, view_name: str) -> None:
        """Drop temporary view."""
        self.connect().execute(f"DROP VIEW IF EXISTS {view_name};")

    def close(self) -> None:
        """Explicitly close connection and release Windows file handles."""
        if self._con is not None:
            try:
                self._con.close()
            except Exception as e:
                logger.warning("Error closing DuckDB connection: %s", e)
            finally:
                self._con = None
                gc.collect()

    def __enter__(self) -> DuckDBEngine:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
