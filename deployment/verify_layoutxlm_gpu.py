#!/usr/bin/env python3
"""
Stage 3 GPU Verification: PyTorch CUDA Backend & LayoutXLM / LayoutLMv3 Multimodal Inference.
Checks PyTorch CUDA availability, GPU device properties, CUDA tensor kernel execution,
and executes a multimodal forward pass with LayoutLMv3 on the target dGPU.
"""

import sys
import tempfile
import numpy as np
from pathlib import Path


def verify_pytorch_gpu_core() -> bool:
    print("=" * 60)
    print("STAGE 3A: PYTORCH CUDA TENSOR VERIFICATION")
    print("=" * 60)

    try:
        import torch
    except ImportError as e:
        print(f"FAILED: Could not import torch: {e}")
        return False

    print(f"PyTorch Version:           {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:            {cuda_available}")

    if not cuda_available:
        print("ERROR: PyTorch does not have CUDA enabled or no NVIDIA driver accessible.")
        return False

    gpu_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"Visible GPU Count:         {gpu_count}")
    print(f"Primary Device Name:       {device_name}")
    print(f"Compute Capability:        {capability[0]}.{capability[1]}")
    print(f"Total VRAM:                {total_mem_gb:.2f} GB")

    try:
        device = torch.device("cuda:0")
        print("\nExecuting 4096 x 4096 tensor matrix multiplication on GPU...")
        a = torch.randn(4096, 4096, device=device, dtype=torch.float32)
        b = torch.randn(4096, 4096, device=device, dtype=torch.float32)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        print(f"Result Tensor Device:      {c.device}")
        if c.device.type != "cuda":
            print(f"ERROR: Tensor device {c.device} is not CUDA.")
            return False

        print("[PASS] PyTorch CUDA kernel execution & memory allocation succeeded.")
        return True
    except Exception as e:
        print(f"ERROR during PyTorch GPU verification: {e}")
        return False


def verify_layoutxlm_gpu_inference() -> bool:
    print("\n" + "=" * 60)
    print("STAGE 3B: LAYOUTXLM / LAYOUTLMV3 GPU INFERENCE VERIFICATION")
    print("=" * 60)

    try:
        import torch
        from PIL import Image
        from transformers import (
            LayoutLMv3Config,
            LayoutLMv3ForSequenceClassification,
            LayoutLMv3Processor,
        )
    except ImportError as e:
        print(f"FAILED: Required library missing: {e}")
        return False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: Cannot run LayoutXLM GPU verification because CUDA is not available.")
        return False

    try:
        print("Instantiating LayoutLMv3 sequence classifier...")
        repo_root = Path(__file__).resolve().parent.parent
        model_dir = repo_root / "models" / "layoutxlm"
        has_weights = (
            (model_dir / "model.safetensors").exists()
            or (model_dir / "pytorch_model.bin").exists()
        )

        from transformers import AutoTokenizer, LayoutLMv3ImageProcessor

        if model_dir.exists() and has_weights:
            print(f"Loading local weights from: {model_dir}")
            model = LayoutLMv3ForSequenceClassification.from_pretrained(str(model_dir), num_labels=6)
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)
        elif model_dir.exists() and (model_dir / "config.json").exists():
            print(f"Loading local architecture configuration from: {model_dir}")
            config = LayoutLMv3Config.from_pretrained(str(model_dir), num_labels=6)
            model = LayoutLMv3ForSequenceClassification(config)
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)
        else:
            print("Using microsoft/layoutlmv3-base configuration...")
            config = LayoutLMv3Config(num_labels=6)
            model = LayoutLMv3ForSequenceClassification(config)
            tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
            image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)

        model.eval()
        model.to(device)
        print(f"Model transferred to device: {device}")

        # Synthetic multimodal inputs
        synthetic_img = Image.new("RGB", (224, 224), color=(240, 240, 240))
        words = ["GOVERNMENT", "MAHARASHTRA", "CASTE", "CERTIFICATE", "VALIDITY"]
        boxes = [
            [50, 50, 200, 80],
            [210, 50, 400, 80],
            [50, 100, 150, 130],
            [160, 100, 300, 130],
            [50, 150, 200, 180],
        ]

        print("Tokenizing and encoding multimodal inputs for GPU...")
        encoding = processor(
            synthetic_img,
            words,
            boxes=boxes,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        encoding = {k: v.to(device) for k, v in encoding.items()}

        print("Executing forward pass on GPU...")
        with torch.inference_mode():
            outputs = model(**encoding)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs_np = probs.cpu().numpy()[0]

        top_class_idx = int(np.argmax(probs_np))
        top_prob = float(probs_np[top_class_idx])

        print(f"Inference output logits shape: {logits.shape}")
        print(f"Top predicted class index:    {top_class_idx} (confidence: {top_prob:.4f})")
        print("[PASS] LayoutXLM / LayoutLMv3 GPU forward pass verified successfully.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"ERROR during LayoutXLM GPU forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    core_ok = verify_pytorch_gpu_core()
    if not core_ok:
        sys.exit(1)

    inference_ok = verify_layoutxlm_gpu_inference()
    sys.exit(0 if inference_ok else 1)


if __name__ == "__main__":
    main()
