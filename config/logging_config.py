"""Structured dual logging with ANSI colored console and rotating file output."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import get_settings


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter applying ANSI color sequences according to log levels."""

    COLOR_MAP = {
        logging.DEBUG: "\033[36m",       # Cyan
        logging.INFO: "\033[32m",        # Green
        logging.WARNING: "\033[33m",     # Yellow
        logging.ERROR: "\033[31m",       # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLOR_MAP.get(record.levelno, "")
        reset = self.RESET if color else ""
        orig_levelname = record.levelname
        # Non-destructively pad and colorize levelname
        record.levelname = f"{color}{orig_levelname:<8}{reset}"
        try:
            formatted = super().format(record)
        finally:
            # Restore plain levelname so file handlers process clean text
            record.levelname = orig_levelname
        return formatted


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure structured dual logging for the platform.

    Args:
        log_level: Desired log level string (e.g. 'DEBUG', 'INFO', 'WARNING'). Defaults to Settings.log_level.
        log_file: Optional path for rotating file logs. Defaults to workspace/logs/app.log.

    Returns:
        Configured root 'mlpipe' logger instance.
    """
    settings = get_settings()
    effective_level_str = log_level or settings.log_level
    numeric_level = getattr(logging, effective_level_str.upper(), logging.INFO)

    logger = logging.getLogger("mlpipe")
    logger.setLevel(numeric_level)
    logger.handlers.clear()

    # 1. Colored Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = ColoredConsoleFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 2. Rotating File Handler (workspace/logs/app.log)
    target_log_file = log_file or (settings.workspace_logs_dir / "app.log")
    target_log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=target_log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,               # Keep up to 5 rotations (50 MB total)
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # Capture debug logs to file
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str = "mlpipe") -> logging.Logger:
    """Retrieve a namespaced child logger under the 'mlpipe' root hierarchy."""
    if not logging.getLogger("mlpipe").handlers:
        setup_logging()

    if name == "mlpipe" or name.startswith("mlpipe."):
        return logging.getLogger(name)
    return logging.getLogger(f"mlpipe.{name}")
