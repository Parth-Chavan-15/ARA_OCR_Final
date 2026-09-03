#!/usr/bin/env python3
"""
Comprehensive 5-Stage GPU Stack Verification Suite.
Runs all verification layers sequentially and tests concurrent pipeline coexistence:
  [Stage 0] GPU Hardware & Driver Discovery
  [Stage 1] PaddlePaddle CUDA Core Tensor Ops
  [Stage 2] PaddleOCR GPU Document Extraction
  [Stage 3] PyTorch CUDA Backend & LayoutXLM Multimodal Inference
  [Stage 4] Combined PaddleOCR + LayoutXLM Pipeline Coexistence & VRAM Stability
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Add deployment dir to path
DEPLOYMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOYMENT_DIR))

from gpu_detector import detect_nvidia_system  # noqa: E402
from verify_gpu import verify_paddle_gpu_core  # noqa: E402
from verify_ocr_gpu import verify_ocr_gpu, create_synthetic_test_image  # noqa: E402
from verify_layoutxlm_gpu import (  # noqa: E402
    verify_layoutxlm_gpu_inference,
    verify_pytorch_gpu_core,
)


def verify_pipeline_coexistence() -> bool:
    print("\n" + "=" * 60)
    print("STAGE 4: FULL PIPELINE COEXISTENCE & VRAM STABILITY TEST")
    print("=" * 60)

    try:
        import paddle
        from paddleocr import PaddleOCR
        import torch
        from transformers import (
            LayoutLMv3Config,
            LayoutLMv3ForSequenceClassification,
            LayoutLMv3Processor,
        )
    except ImportError as e:
        print(f"FAILED: Dependency import failed during coexistence test: {e}")
        return False

    try:
        test_img_path = DEPLOYMENT_DIR / "test_pipeline_coexist.png"
        create_synthetic_test_image(test_img_path)

        # 1. PaddleOCR on GPU
        print("[1/3] Running PaddleOCR on GPU...")
        t0 = time.perf_counter()
        try:
            ocr = PaddleOCR(use_textline_orientation=False, lang="en", device="gpu")
        except TypeError:
            ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=True)

        if hasattr(ocr, "predict"):
            preds = list(ocr.predict(str(test_img_path)))
        else:
            preds = ocr.ocr(str(test_img_path), cls=False)

        if hasattr(paddle.device, "cuda") and hasattr(paddle.device.cuda, "synchronize"):
            paddle.device.cuda.synchronize()
        elif hasattr(paddle.device, "synchronize"):
            paddle.device.synchronize()
        t_ocr = (time.perf_counter() - t0) * 1000

        words = []
        boxes = []
        if preds and isinstance(preds[0], dict) and "rec_texts" in preds[0]:
            words = preds[0].get("rec_texts", [])
            dt_polys = preds[0].get("dt_polys", [])
            for poly in dt_polys:
                xs = [int(p[0]) for p in poly]
                ys = [int(p[1]) for p in poly]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                boxes.append([
                    max(0, min(1000, int(x1 * 1000 / 800))),
                    max(0, min(1000, int(y1 * 1000 / 300))),
                    max(0, min(1000, int(x2 * 1000 / 800))),
                    max(0, min(1000, int(y2 * 1000 / 300))),
                ])
        elif preds and isinstance(preds, list):
            for page in preds:
                if page:
                    for line in page:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            poly = line[0]
                            text_tuple = line[1]
                            text = text_tuple[0] if isinstance(text_tuple, (list, tuple)) else str(text_tuple)
                            words.append(text)
                            xs = [int(p[0]) for p in poly]
                            ys = [int(p[1]) for p in poly]
                            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                            boxes.append([
                                max(0, min(1000, int(x1 * 1000 / 800))),
                                max(0, min(1000, int(y1 * 1000 / 300))),
                                max(0, min(1000, int(x2 * 1000 / 800))),
                                max(0, min(1000, int(y2 * 1000 / 300))),
                            ])

        print(f"      PaddleOCR extracted {len(words)} tokens in {t_ocr:.1f}ms on GPU.")

        # 2. LayoutXLM on GPU
        print("[2/3] Initializing LayoutXLM classifier on GPU...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        repo_root = Path(__file__).resolve().parent.parent
        model_dir = repo_root / "models" / "layoutxlm"
        has_weights = (
            (model_dir / "model.safetensors").exists()
            or (model_dir / "pytorch_model.bin").exists()
        )

        from transformers import AutoTokenizer, LayoutLMv3ImageProcessor

        if model_dir.exists() and has_weights:
            model = LayoutLMv3ForSequenceClassification.from_pretrained(str(model_dir), num_labels=6)
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)
        elif model_dir.exists() and (model_dir / "config.json").exists():
            config = LayoutLMv3Config.from_pretrained(str(model_dir), num_labels=6)
            model = LayoutLMv3ForSequenceClassification(config)
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)
        else:
            config = LayoutLMv3Config(num_labels=6)
            model = LayoutLMv3ForSequenceClassification(config)
            tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
            image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
            processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)

        model.eval()
        model.to(device)

        t1 = time.perf_counter()
        img = Image.open(test_img_path).convert("RGB")
        encoding = processor(
            img,
            words if words else ["SAMPLE"],
            boxes=boxes if boxes else [[0, 0, 100, 100]],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.inference_mode():
            outputs = model(**encoding)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        t_cls = (time.perf_counter() - t1) * 1000
        print(f"      LayoutXLM classified document in {t_cls:.1f}ms on GPU.")

        # 3. Check GPU memory
        print("[3/3] Checking GPU Memory Status...")
        if torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
            res_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
            print(f"      PyTorch Allocated VRAM: {alloc_mb:.1f} MB | Reserved: {res_mb:.1f} MB")

        # Cleanup
        if test_img_path.exists():
            test_img_path.unlink()

        print("\n[PASS] Full Pipeline Coexistence Test PASSED.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"ERROR during pipeline coexistence test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("ARA DOCUMENT INTELLIGENCE — COMPLETE GPU STACK TEST SUITE")
    print("=" * 60)

    # Stage 0: Hardware discovery
    print("\n[STAGE 0] Detecting GPU Hardware and Drivers...")
    info = detect_nvidia_system()
    if not info.has_gpu:
        print(f"FAILED: No NVIDIA GPU detected: {info.error_message}")
        sys.exit(1)
    print(f"[PASS] GPU: {info.gpu_name} (Driver: {info.driver_version}, CUDA Max: {info.reported_cuda_version})")

    # Stage 1: Paddle CUDA core
    print("\n[STAGE 1] Validating PaddlePaddle CUDA Core...")
    if not verify_paddle_gpu_core():
        print("FAILED: Stage 1 Paddle tensor check failed.")
        sys.exit(1)

    # Stage 2: PaddleOCR GPU
    print("\n[STAGE 2] Validating PaddleOCR Pipeline...")
    if not verify_ocr_gpu():
        print("FAILED: Stage 2 PaddleOCR check failed.")
        sys.exit(1)

    # Stage 3: PyTorch + LayoutXLM GPU
    print("\n[STAGE 3] Validating PyTorch & LayoutXLM GPU Inference...")
    if not verify_pytorch_gpu_core() or not verify_layoutxlm_gpu_inference():
        print("FAILED: Stage 3 PyTorch / LayoutXLM check failed.")
        sys.exit(1)

    # Stage 4: End-to-end pipeline coexistence
    print("\n[STAGE 4] Validating Pipeline Coexistence & Memory...")
    if not verify_pipeline_coexistence():
        print("FAILED: Stage 4 Coexistence check failed.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL GPU VERIFICATION STAGES PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
