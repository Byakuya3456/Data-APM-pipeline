"""Real-time system telemetry and resource monitoring for local hardware.

Collects RAM, swap, CPU, NVIDIA GPU (RTX 3050 6GB VRAM), and storage metrics.
Provides memory pressure detection and the ProcessTracker context manager.
Zero-exception design ensures continuous stability across all host configurations.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil
from pydantic import BaseModel, Field

logger = logging.getLogger("autonomous_ml.resource_monitor")

# Module-level warm-up for psutil CPU measurement
try:
    psutil.cpu_percent(interval=None)
except Exception:
    pass


class MemoryPressureLevel(str, Enum):
    """Classification of host memory pressure severity."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class MemoryPressureStatus(BaseModel):
    """Detailed memory pressure diagnostic report."""
    level: MemoryPressureLevel
    is_under_pressure: bool
    ram_percent: float
    ram_available_gb: float
    swap_percent: float
    swap_free_gb: float
    remediation_advice: Optional[str] = None


class SystemTelemetry(BaseModel):
    """Comprehensive snapshot of hardware resources and utilization."""
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # RAM & Swap
    ram_total_gb: float
    ram_available_gb: float
    ram_used_gb: float
    ram_percent: float
    swap_total_gb: float = 0.0
    swap_free_gb: float = 0.0
    swap_percent: float = 0.0

    # CPU
    cpu_cores_physical: int
    cpu_threads_logical: int
    cpu_percent: float

    # Storage
    disk_total_gb: float = 0.0
    disk_free_gb: float
    disk_used_gb: float = 0.0
    disk_percent: float = 0.0

    # GPU (NVIDIA RTX 3050 or fallback)
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_vram_total_mb: Optional[float] = None
    gpu_vram_free_mb: Optional[float] = None
    gpu_vram_used_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None

    # Bidirectional contract properties matching PROJECT.md ResourceSnapshot
    @property
    def cpu_count(self) -> int:
        return self.cpu_threads_logical

    @property
    def gpu_total_vram_mb(self) -> Optional[float]:
        return self.gpu_vram_total_mb

    @property
    def gpu_free_vram_mb(self) -> Optional[float]:
        return self.gpu_vram_free_mb


# Alias for contract compatibility with PROJECT.md
ResourceSnapshot = SystemTelemetry


def _find_nvidia_smi() -> Optional[str]:
    """Locate nvidia-smi executable on Windows or POSIX."""
    smi = shutil.which("nvidia-smi")
    if smi:
        return smi

    # Standard Windows install locations
    windows_candidates = [
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    for candidate in windows_candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def get_gpu_telemetry() -> Dict[str, Any]:
    """Extract GPU telemetry via nvidia-smi with zero-exception guarantee."""
    fallback_result: Dict[str, Any] = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_total_mb": None,
        "gpu_vram_free_mb": None,
        "gpu_vram_used_mb": None,
        "gpu_utilization_percent": None,
    }

    smi_path = _find_nvidia_smi()
    if not smi_path:
        logger.debug("nvidia-smi not detected on system PATH or standard directories.")
        return fallback_result

    cmd = [
        smi_path,
        "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3.0,
            shell=False,
        )
        if proc.returncode != 0:
            logger.debug("nvidia-smi returned non-zero code %d: %s", proc.returncode, proc.stderr.strip())
            return fallback_result

        lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
        if not lines:
            return fallback_result

        # Parse primary GPU
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) < 5:
            return fallback_result

        def _safe_float(val: str) -> Optional[float]:
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        return {
            "gpu_available": True,
            "gpu_name": parts[0],
            "gpu_vram_total_mb": _safe_float(parts[1]),
            "gpu_vram_free_mb": _safe_float(parts[2]),
            "gpu_vram_used_mb": _safe_float(parts[3]),
            "gpu_utilization_percent": _safe_float(parts[4]),
        }
    except Exception as exc:
        logger.debug("Safe catch of GPU telemetry query failure: %s", exc)
        return fallback_result


def get_system_telemetry(
    target_disk_path: Union[str, Path] = ".",
    cpu_sample_interval: Optional[float] = None,
) -> SystemTelemetry:
    """Acquire instantaneous snapshot of host hardware telemetry."""
    # RAM & Swap
    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # CPU
    cpu_pct = psutil.cpu_percent(interval=cpu_sample_interval)
    cpu_logical = psutil.cpu_count(logical=True) or 1
    cpu_physical = psutil.cpu_count(logical=False) or 1

    # Disk
    disk_path = str(Path(target_disk_path).resolve())
    try:
        disk = psutil.disk_usage(disk_path)
    except Exception:
        try:
            disk = psutil.disk_usage(".")
        except Exception:
            disk = None

    # GPU
    gpu_data = get_gpu_telemetry()

    return SystemTelemetry(
        ram_total_gb=round(vmem.total / (1024**3), 2),
        ram_available_gb=round(vmem.available / (1024**3), 2),
        ram_used_gb=round(vmem.used / (1024**3), 2),
        ram_percent=float(vmem.percent),
        swap_total_gb=round(swap.total / (1024**3), 2),
        swap_free_gb=round(swap.free / (1024**3), 2),
        swap_percent=float(swap.percent),
        cpu_cores_physical=cpu_physical,
        cpu_threads_logical=cpu_logical,
        cpu_percent=float(cpu_pct),
        disk_total_gb=round(disk.total / (1024**3), 2) if disk else 0.0,
        disk_free_gb=round(disk.free / (1024**3), 2) if disk else 0.0,
        disk_used_gb=round(disk.used / (1024**3), 2) if disk else 0.0,
        disk_percent=float(disk.percent) if disk else 0.0,
        **gpu_data,
    )


def check_memory_pressure(
    threshold_percent: float = 85.0,
    min_available_gb: float = 3.0,
) -> bool:
    """Return True if system is experiencing acute memory pressure."""
    try:
        vmem = psutil.virtual_memory()
        avail_gb = vmem.available / (1024**3)
        return bool(vmem.percent >= threshold_percent or avail_gb < min_available_gb)
    except Exception as exc:
        logger.warning("Error checking memory pressure: %s", exc)
        return False


def get_memory_pressure_status() -> MemoryPressureStatus:
    """Generate structured memory pressure diagnostic status and advice."""
    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    avail_gb = round(vmem.available / (1024**3), 2)
    ram_pct = float(vmem.percent)
    swap_pct = float(swap.percent)
    swap_free_gb = round(swap.free / (1024**3), 2)

    if ram_pct >= 90.0 or avail_gb < 2.0 or swap_pct >= 90.0:
        level = MemoryPressureLevel.CRITICAL
        under_pressure = True
        advice = (
            "CRITICAL: Host RAM exhaustion imminent (<2GB available or >90% utilized). "
            "Halt in-memory operations. Switch all queries to DuckDB out-of-core with disk spilling. "
            "Invoke gc.collect() immediately."
        )
    elif ram_pct >= 75.0 or avail_gb < 4.0:
        level = MemoryPressureLevel.WARNING
        under_pressure = True
        advice = (
            "WARNING: Elevated memory pressure detected. "
            "Use Polars LazyFrames with streaming sink_parquet or DuckDB. "
            "Reduce chunk size to 25,000 rows."
        )
    else:
        level = MemoryPressureLevel.NORMAL
        under_pressure = False
        advice = None

    return MemoryPressureStatus(
        level=level,
        is_under_pressure=under_pressure,
        ram_percent=ram_pct,
        ram_available_gb=avail_gb,
        swap_percent=swap_pct,
        swap_free_gb=swap_free_gb,
        remediation_advice=advice,
    )


class ProcessTracker:
    """Context manager measuring peak memory, CPU time, and duration of operations."""

    def __init__(self, name: Optional[str] = None, sample_interval_s: float = 0.02):
        self.name = name
        self.sample_interval_s = sample_interval_s
        self._proc = psutil.Process()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_seconds: float = 0.0

        self.initial_memory_bytes: int = 0
        self.final_memory_bytes: int = 0
        self.peak_memory_bytes: int = 0

        self._initial_cpu_times: Optional[Any] = None
        self._final_cpu_times: Optional[Any] = None

    def _sample_loop(self) -> None:
        while self._running:
            try:
                current_rss = self._proc.memory_info().rss
                for child in self._proc.children(recursive=True):
                    try:
                        current_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                with self._lock:
                    if current_rss > self.peak_memory_bytes:
                        self.peak_memory_bytes = current_rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(self.sample_interval_s)

    def start(self) -> ProcessTracker:
        self.start_time = time.perf_counter()
        try:
            self.initial_memory_bytes = self._proc.memory_info().rss
            self._initial_cpu_times = self._proc.cpu_times()
        except Exception:
            self.initial_memory_bytes = 0
        self.peak_memory_bytes = self.initial_memory_bytes

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

        self.end_time = time.perf_counter()
        self.duration_seconds = max(0.0, self.end_time - self.start_time)
        try:
            self.final_memory_bytes = self._proc.memory_info().rss
            self._final_cpu_times = self._proc.cpu_times()
        except Exception:
            self.final_memory_bytes = self.peak_memory_bytes

        with self._lock:
            if self.final_memory_bytes > self.peak_memory_bytes:
                self.peak_memory_bytes = self.final_memory_bytes

    def __enter__(self) -> ProcessTracker:
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    @property
    def peak_memory_mb(self) -> float:
        return round(self.peak_memory_bytes / (1024**2), 2)

    @property
    def initial_memory_mb(self) -> float:
        return round(self.initial_memory_bytes / (1024**2), 2)

    @property
    def final_memory_mb(self) -> float:
        return round(self.final_memory_bytes / (1024**2), 2)

    @property
    def memory_delta_mb(self) -> float:
        return round((self.final_memory_bytes - self.initial_memory_bytes) / (1024**2), 2)

    @property
    def cpu_time_seconds(self) -> float:
        if self._initial_cpu_times and self._final_cpu_times:
            user = self._final_cpu_times.user - self._initial_cpu_times.user
            system = self._final_cpu_times.system - self._initial_cpu_times.system
            return round(max(0.0, user + system), 3)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_seconds": round(self.duration_seconds, 4),
            "peak_memory_mb": self.peak_memory_mb,
            "initial_memory_mb": self.initial_memory_mb,
            "final_memory_mb": self.final_memory_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "cpu_time_seconds": self.cpu_time_seconds,
        }
