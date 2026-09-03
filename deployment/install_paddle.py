#!/usr/bin/env python3
"""
Capability-Based PaddlePaddle GPU Installer.
Resolves the highest officially supported PaddlePaddle CUDA wheel compatible
with the installed NVIDIA driver, based on deployment/compatibility.json.
Automatically adapts to standard pip, uv pip, or bootstraps ensurepip if missing.
Guards against PEP 668 externally-managed system Python environments.
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


def load_compatibility_matrix() -> dict:
    matrix_path = DEPLOYMENT_DIR / "compatibility.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Compatibility matrix missing at {matrix_path}")
    with open(matrix_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_paddle_build(
    reported_cuda_version: str,
    matrix: dict,
    target_paddle_version: str = None,
) -> dict:
    """
    Selects the highest Paddle CUDA build <= reported_cuda_version.
    """
    version_key = target_paddle_version or matrix.get("default_paddle_version", "3.3.0")
    paddle_config = matrix.get("paddle_versions", {}).get(version_key)
    if not paddle_config:
        raise ValueError(f"No compatibility matrix found for PaddlePaddle {version_key}")

    cuda_builds = paddle_config.get("cuda_builds", {})
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
            f"but no configured Paddle build is <= {reported_cuda_version}. "
            f"Configured builds: {available}"
        )

    compatible_builds.sort(reverse=True, key=lambda x: x[0])
    _, selected_version_str, selected_config = compatible_builds[0]

    return {
        "paddle_version": version_key,
        "cuda_version": selected_version_str,
        **selected_config,
    }


def resolve_target_python(user_python_bin: str = None) -> str:
    """
    Resolves the appropriate target Python environment.
    Automatically detects and selects project virtual environments to prevent
    PEP 668 externally-managed-environment collisions on Linux distributions.
    """
    if user_python_bin:
        path = Path(user_python_bin).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Specified Python binary not found: {user_python_bin}")
        return str(path)

    # 1. Check if VIRTUAL_ENV is active
    if "VIRTUAL_ENV" in os.environ:
        venv_py = Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python"
        if venv_py.is_file():
            return str(venv_py)

    # 2. Check if currently executing inside a virtual environment
    if sys.prefix != sys.base_prefix:
        return str(sys.executable)

    # 3. Check for repository virtual environments (Windows and POSIX)
    candidate_paths = [
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / "backend" / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / "backend" / ".venv" / "bin" / "python",
        REPO_ROOT / "venv" / "Scripts" / "python.exe",
        REPO_ROOT / "venv" / "bin" / "python",
    ]
    for cand in candidate_paths:
        if cand.is_file():
            print(f"[INFO] Detected local virtual environment at {cand}. Using it to prevent PEP 668 errors.")
            return str(cand)

    # 4. Check if current sys.executable is externally managed
    ext_managed_marker = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "EXTERNALLY-MANAGED"
    if ext_managed_marker.exists() or Path("/usr/lib/python3.14/EXTERNALLY-MANAGED").exists():
        uv_path = shutil.which("uv")
        if uv_path:
            venv_target = REPO_ROOT / ".venv"
            print(f"[INFO] System Python is externally managed. Creating virtual environment with uv at {venv_target}...")
            subprocess.run([uv_path, "venv", str(venv_target), "--python", "3.11"], check=True)
            new_py = venv_target / "bin" / "python"
            if new_py.is_file():
                return str(new_py)

        raise RuntimeError(
            "System Python is externally managed (PEP 668). Please run within a virtualenv:\n"
            "  uv venv .venv --python 3.11\n"
            "  source .venv/bin/activate\n"
            "  ./deployment/install_paddle.py\n"
            "Or pass --python-bin /path/to/venv/bin/python"
        )

    return str(sys.executable)


def determine_installer(python_exec: str) -> tuple[str, list[str]]:
    """
    Determines whether to use uv pip or standard python -m pip.
    Prefers uv pip when available for speed and isolated environment target support.
    """
    uv_path = shutil.which("uv")
    if uv_path:
        return ("uv", [uv_path, "pip"])

    res = subprocess.run(
        [python_exec, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        return ("pip", [python_exec, "-m", "pip"])

    print("pip not found in target environment; attempting ensurepip...")
    try:
        subprocess.run([python_exec, "-m", "ensurepip", "--default-pip"], check=True)
        return ("pip", [python_exec, "-m", "pip"])
    except Exception as e:
        raise RuntimeError(
            f"No working pip or uv found for {python_exec}. Detail: {e}"
        )


def install_paddle_gpu(build: dict, python_exec: str = None, dry_run: bool = False):
    """
    Installs paddlepaddle-gpu for the resolved build.
    """
    executable = resolve_target_python(python_exec)
    installer_name, base_cmd = determine_installer(executable)

    print("\n" + "=" * 60)
    print("PADDLEPADDLE GPU INSTALLATION PLAN")
    print("=" * 60)
    print(f"Target Python:   {executable}")
    print(f"Installer Tool:  {installer_name} ({' '.join(base_cmd)})")
    print(f"Paddle Version:  {build['paddle_version']}")
    print(f"Selected CUDA:   {build['cuda_version']} ({build['tag']})")
    print(f"Index URL:       {build['index_url']}")
    print("=" * 60)

    if dry_run:
        print("[DRY-RUN] Skipping actual installation.")
        return

    env = os.environ.copy()
    tmp_dir = Path.home() / ".cache" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp_dir)

    print("\nStep 1/2: Removing existing / conflicting paddle packages...")
    if installer_name == "uv":
        uninstall_cmd = base_cmd + ["uninstall", "--python", executable, "paddlepaddle", "paddlepaddle-gpu"]
    else:
        uninstall_cmd = base_cmd + ["uninstall", "-y", "paddlepaddle", "paddlepaddle-gpu"]

    subprocess.run(uninstall_cmd, check=False, env=env)

    print("\nStep 2/2: Installing target paddlepaddle-gpu build...")
    if installer_name == "uv":
        install_cmd = base_cmd + [
            "install",
            "--python",
            executable,
            "--upgrade",
            f"paddlepaddle-gpu=={build['paddle_version']}",
            "--extra-index-url",
            build["index_url"],
        ]
    else:
        install_cmd = base_cmd + [
            "install",
            "--upgrade",
            f"paddlepaddle-gpu=={build['paddle_version']}",
            "--extra-index-url",
            build["index_url"],
        ]

    print(f"Running: {' '.join(install_cmd)}")
    subprocess.run(install_cmd, check=True, env=env)
    print("\n[PASS] PaddlePaddle GPU installation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Capability-based PaddlePaddle GPU installer"
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=None,
        help="Path to python executable (defaults to detected project virtual environment or active Python)",
    )
    parser.add_argument(
        "--paddle-version",
        type=str,
        default=None,
        help="Specific Paddle version (e.g. 3.3.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate resolution without installing packages",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PADDLEPADDLE GPU CAPABILITY RESOLVER")
    print("=" * 60)

    info = detect_nvidia_system()
    if not info.has_gpu or not info.reported_cuda_version:
        print(f"ERROR: Cannot proceed with GPU install: {info.error_message or 'No CUDA driver capability'}")
        sys.exit(1)

    print(f"Detected GPU:     {info.gpu_name}")
    print(f"Driver Version:   {info.driver_version}")
    print(f"Max CUDA Driver:  {info.reported_cuda_version}")
    if info.toolkit_cuda_version:
        print(f"Host Toolkit:     {info.toolkit_cuda_version} (informational)")

    matrix = load_compatibility_matrix()
    build = resolve_paddle_build(
        info.reported_cuda_version,
        matrix,
        target_paddle_version=args.paddle_version,
    )

    install_paddle_gpu(build, python_exec=args.python_bin, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
