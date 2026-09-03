#!/usr/bin/env python3
"""
GPU & CUDA Detection Module for ARA Document Intelligence.
Detects NVIDIA hardware, driver version, driver-supported CUDA runtime capability,
and host CUDA Toolkit version.
"""

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SystemGPUInfo:
    has_gpu: bool
    gpu_name: Optional[str] = None
    driver_version: Optional[str] = None
    reported_cuda_version: Optional[str] = None
    toolkit_cuda_version: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def run_command(command: list[str]) -> str:
    """Execute a command and return stripped stdout."""
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def parse_version(version: str) -> tuple[int, ...]:
    """Convert version string like '13.3.1' -> (13, 3, 1)."""
    numbers = re.findall(r"\d+", version)
    return tuple(map(int, numbers))


def normalize_cuda_major_minor(version: str) -> str:
    """Normalize '13.3.1' -> '13.3'."""
    parts = re.findall(r"\d+", version)
    if len(parts) < 2:
        raise ValueError(f"Invalid CUDA version format: {version}")
    return f"{int(parts[0])}.{int(parts[1])}"


def detect_installed_cuda_toolkit() -> Optional[str]:
    """
    Detect locally installed CUDA Toolkit (nvcc / version.json).
    Informational: modern Paddle GPU wheels bundle dependencies.
    """
    if command_exists("nvcc"):
        try:
            output = run_command(["nvcc", "--version"])
            match = re.search(r"release\s+([0-9]+\.[0-9]+)", output)
            if match:
                return match.group(1)
        except Exception:
            pass

    possible_files = [
        "/usr/local/cuda/version.json",
        "/usr/local/cuda/version.txt",
    ]

    for path_str in possible_files:
        path = Path(path_str)
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
                if path_str.endswith(".json"):
                    data = json.loads(content)
                    version = data.get("cuda", {}).get("version")
                    if version:
                        return normalize_cuda_major_minor(version)
                else:
                    match = re.search(r"CUDA Version\s+([0-9]+\.[0-9]+)", content)
                    if match:
                        return match.group(1)
            except Exception:
                continue

    return None


def detect_nvidia_system() -> SystemGPUInfo:
    """
    Detect NVIDIA GPU hardware, driver version, and driver CUDA capability.
    """
    if not command_exists("nvidia-smi"):
        return SystemGPUInfo(
            has_gpu=False,
            error_message="nvidia-smi not found. No NVIDIA driver/GPU available.",
        )

    try:
        query_out = run_command([
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ])
    except Exception as e:
        return SystemGPUInfo(
            has_gpu=False,
            error_message=f"nvidia-smi execution failed: {e}",
        )

    if not query_out.strip():
        return SystemGPUInfo(
            has_gpu=False,
            error_message="nvidia-smi returned no GPUs.",
        )

    first_gpu = query_out.splitlines()[0]
    parts = [x.strip() for x in first_gpu.split(",")]
    if len(parts) < 2:
        return SystemGPUInfo(
            has_gpu=False,
            error_message=f"Could not parse nvidia-smi output: {first_gpu}",
        )

    gpu_name, driver_version = parts[0], parts[1]

    # Parse max CUDA capability from nvidia-smi banner (handles 'CUDA Version:' and 'CUDA UMD Version:')
    reported_cuda = None
    try:
        smi_banner = run_command(["nvidia-smi"])
        cuda_match = re.search(r"CUDA(?:\s+\w+)?\s+Version:\s*([0-9]+\.[0-9]+)", smi_banner, re.IGNORECASE)
        if cuda_match:
            reported_cuda = cuda_match.group(1)
    except Exception:
        pass

    toolkit_cuda = detect_installed_cuda_toolkit()

    return SystemGPUInfo(
        has_gpu=True,
        gpu_name=gpu_name,
        driver_version=driver_version,
        reported_cuda_version=reported_cuda,
        toolkit_cuda_version=toolkit_cuda,
    )


if __name__ == "__main__":
    info = detect_nvidia_system()
    print("=" * 60)
    print("NVIDIA SYSTEM HARDWARE & CUDA CAPABILITY")
    print("=" * 60)
    print(f"GPU Detected:            {info.has_gpu}")
    print(f"GPU Model:               {info.gpu_name or 'N/A'}")
    print(f"Driver Version:          {info.driver_version or 'N/A'}")
    print(f"CUDA Driver Capability:  {info.reported_cuda_version or 'N/A'}")
    print(f"CUDA Toolkit (Local):    {info.toolkit_cuda_version or 'Not detected'}")
    if info.error_message:
        print(f"Status Note:             {info.error_message}")
    print("=" * 60)
