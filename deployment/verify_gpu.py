#!/usr/bin/env python3
"""
Stage 1 GPU Verification: PaddlePaddle CUDA Backend & Matrix Operations.
Checks device detection, CUDA runtime compilation, memory allocation, and kernel execution.
"""

import sys


def verify_paddle_gpu_core() -> bool:
    print("=" * 60)
    print("STAGE 1: PADDLEPADDLE CUDA TENSOR VERIFICATION")
    print("=" * 60)

    try:
        import paddle
    except ImportError as e:
        print(f"FAILED: Could not import paddle: {e}")
        return False

    print(f"PaddlePaddle Version:      {paddle.__version__}")
    is_cuda_compiled = paddle.is_compiled_with_cuda()
    print(f"Compiled with CUDA:        {is_cuda_compiled}")

    if not is_cuda_compiled:
        print("ERROR: Installed PaddlePaddle binary is not compiled with CUDA support.")
        return False

    gpu_count = paddle.device.cuda.device_count()
    print(f"Visible GPU Count:         {gpu_count}")
    if gpu_count < 1:
        print("ERROR: PaddlePaddle reports 0 visible CUDA GPUs.")
        return False

    try:
        paddle.set_device("gpu:0")
        active_dev = paddle.get_device()
        print(f"Active Execution Device:   {active_dev}")

        # Run real GPU tensor math
        print("\nExecuting 4096 x 4096 tensor matrix multiplication on GPU...")
        x = paddle.randn([4096, 4096])
        y = paddle.matmul(x, x)
        paddle.device.cuda.synchronize()

        # Check tensor device
        place = str(y.place)
        print(f"Result Tensor Memory Place: {place}")
        if "CUDAPlace" not in place and "gpu" not in place.lower():
            print(f"ERROR: Tensor place {place} is not on CUDA device.")
            return False

        print("[PASS] CUDA kernel execution and synchronization succeeded.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"ERROR during GPU tensor verification: {e}")
        return False


if __name__ == "__main__":
    success = verify_paddle_gpu_core()
    sys.exit(0 if success else 1)
