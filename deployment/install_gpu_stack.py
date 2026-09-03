#!/usr/bin/env python3
"""
Capability-Based Unified GPU Stack Installer for ARA Document Intelligence.
Resolves and installs:
  1. paddlepaddle-gpu (using Paddle CUDA wheel matrix)
  2. PyTorch & torchvision (using PyTorch CUDA wheel repository)
Supports pip, uv pip, and dry-run preview.
Protects against PEP 668 externally-managed Python environments.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add deployment dir to path
DEPLOYMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOYMENT_DIR.parent
sys.path.insert(0, str(DEPLOYMENT_DIR))

from gpu_detector import (  # noqa: E402
    detect_nvidia_system,
    parse_version,
)
from install_paddle import (  # noqa: E402
    determine_installer,
    load_compatibility_matrix,
    resolve_paddle_build,
    resolve_target_python,
)


def resolve_pytorch_build(
    reported_cuda_version: str,
    matrix: dict,
    target_torch_version: str = None,
) -> dict:
    """Selects the highest compatible PyTorch CUDA wheel <= reported_cuda_version."""
    pytorch_config = matrix.get("pytorch_versions", {})
    cuda_builds = pytorch_config.get("cuda_builds", {})
    driver_cuda = parse_version(reported_cuda_version)

    compatible_builds = []
    for cuda_ver_str, config in cuda_builds.items():
        build_cuda = parse_version(cuda_ver_str)
        if build_cuda <= driver_cuda:
            compatible_builds.append((build_cuda, cuda_ver_str, config))

    if not compatible_builds:
        available = list(cuda_builds.keys())
        raise RuntimeError(
            f"Host driver supports up to CUDA {reported_cuda_version}, "
            f"but no configured PyTorch build is <= {reported_cuda_version}. "
            f"Configured builds: {available}"
        )

    compatible_builds.sort(reverse=True, key=lambda x: x[0])
    _, selected_version_str, selected_config = compatible_builds[0]

    return {
        "pytorch_version": target_torch_version or pytorch_config.get("default_version", "2.4.0"),
        "cuda_version": selected_version_str,
        **selected_config,
    }


def install_gpu_stack(
    paddle_build: dict,
    pytorch_build: dict,
    python_exec: str = None,
    dry_run: bool = False,
):
    executable = resolve_target_python(python_exec)
    installer_name, base_cmd = determine_installer(executable)

    print("\n" + "=" * 60)
    print("UNIFIED GPU STACK INSTALLATION PLAN")
    print("=" * 60)
    print(f"Target Python:      {executable}")
    print(f"Installer Tool:     {installer_name} ({' '.join(base_cmd)})")
    print(f"Paddle Version:     {paddle_build['paddle_version']} (CUDA {paddle_build['cuda_version']} - {paddle_build['tag']})")
    print(f"Paddle Index URL:   {paddle_build['index_url']}")
    print(f"PyTorch Version:    {pytorch_build['pytorch_version']} (CUDA {pytorch_build['cuda_version']} - {pytorch_build['tag']})")
    print(f"PyTorch Index URL:  {pytorch_build['index_url']}")
    print("=" * 60)

    if dry_run:
        print("[DRY-RUN] Skipping actual package installations.")
        return

    # Custom disk-backed TMPDIR
    env = os.environ.copy()
    tmp_dir = Path.home() / ".cache" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp_dir)

    # 1. Clean up conflicting CPU / old Paddle packages
    print("\nStep 1/3: Removing conflicting CPU / previous paddle packages...")
    if installer_name == "uv":
        uninstall_cmd = base_cmd + ["uninstall", "--python", executable, "paddlepaddle", "paddlepaddle-gpu"]
    else:
        uninstall_cmd = base_cmd + ["uninstall", "-y", "paddlepaddle", "paddlepaddle-gpu"]
    subprocess.run(uninstall_cmd, check=False, env=env)

    # 2. Install PaddlePaddle GPU
    print("\nStep 2/3: Installing PaddlePaddle GPU...")
    if installer_name == "uv":
        paddle_cmd = base_cmd + [
            "install",
            "--python",
            executable,
            "--upgrade",
            f"paddlepaddle-gpu=={paddle_build['paddle_version']}",
            "--extra-index-url",
            paddle_build["index_url"],
        ]
    else:
        paddle_cmd = base_cmd + [
            "install",
            "--upgrade",
            f"paddlepaddle-gpu=={paddle_build['paddle_version']}",
            "--extra-index-url",
            paddle_build["index_url"],
        ]

    print(f"Running: {' '.join(paddle_cmd)}")
    subprocess.run(paddle_cmd, check=True, env=env)

    # 3. Install PyTorch GPU
    print("\nStep 3/3: Installing PyTorch & torchvision with CUDA...")
    if installer_name == "uv":
        torch_cmd = base_cmd + [
            "install",
            "--python",
            executable,
            "--upgrade",
            "torch",
            "torchvision",
            "--extra-index-url",
            pytorch_build["index_url"],
        ]
    else:
        torch_cmd = base_cmd + [
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "--extra-index-url",
            pytorch_build["index_url"],
        ]

    print(f"Running: {' '.join(torch_cmd)}")
    subprocess.run(torch_cmd, check=True, env=env)

    print("\n[PASS] Unified GPU Stack installation completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Capability-based GPU Installer for PaddleOCR & LayoutXLM"
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=None,
        help="Path to target python executable (defaults to detected project virtual environment or active Python)",
    )
    parser.add_argument(
        "--paddle-version",
        type=str,
        default=None,
        help="Specific PaddlePaddle version",
    )
    parser.add_argument(
        "--torch-version",
        type=str,
        default=None,
        help="Specific PyTorch version",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate resolution without installing packages",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ARA GPU CAPABILITY RESOLVER")
    print("=" * 60)

    info = detect_nvidia_system()
    if not info.has_gpu or not info.reported_cuda_version:
        print(f"ERROR: Cannot proceed with GPU install: {info.error_message or 'No CUDA driver capability'}")
        sys.exit(1)

    print(f"Detected GPU:     {info.gpu_name}")
    print(f"Driver Version:   {info.driver_version}")
    print(f"Max CUDA Driver:  {info.reported_cuda_version}")

    matrix = load_compatibility_matrix()
    paddle_build = resolve_paddle_build(
        info.reported_cuda_version,
        matrix,
        target_paddle_version=args.paddle_version,
    )
    pytorch_build = resolve_pytorch_build(
        info.reported_cuda_version,
        matrix,
        target_torch_version=args.torch_version,
    )

    install_gpu_stack(
        paddle_build,
        pytorch_build,
        python_exec=args.python_bin,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
