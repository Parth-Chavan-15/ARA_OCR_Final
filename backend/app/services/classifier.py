"""
ARA OCR — Multimodal Broad Document Classifier

Uses LayoutLMv3 / LayoutXLM for multimodal document
classification using text + bounding boxes + document image.

6 classes:
  CASTE_CERTIFICATE, CASTE_VALIDITY_CERTIFICATE, CASTE_VALIDITY_RECEIPT,
  PROFORMA_O, LEAVING_CERTIFICATE, UNKNOWN_OUT_OF_SCOPE

Architecture:
  Multimodal feature extraction → Linear classification head → softmax

The classifier reads CACHED OCR tokens — it does NOT re-run OCR (§2, §40).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog
from PIL import Image

from app.config import settings

logger = structlog.get_logger(__name__)

# Document classes
DOCUMENT_CLASSES = [
    "CASTE_CERTIFICATE",
    "CASTE_VALIDITY_CERTIFICATE",
    "CASTE_VALIDITY_RECEIPT",
    "PROFORMA_O",
    "LEAVING_CERTIFICATE",
    "UNKNOWN_OUT_OF_SCOPE",
]

NUM_CLASSES = len(DOCUMENT_CLASSES)


@dataclass
class ClassificationOutput:
    """Result of document classification."""
    predicted_class: str
    confidence: float
    all_probabilities: dict[str, float] = field(default_factory=dict)


class LayoutXLMClassifier:
    """
    Multimodal document classifier.
    Supports LayoutLMv3 and LayoutXLM for 6-way broad classification.
    Singleton pattern — model is loaded once and reused.
    """

    _instance: Optional["LayoutXLMClassifier"] = None
    CLASSES = DOCUMENT_CLASSES

    def __new__(cls) -> "LayoutXLMClassifier":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.fine_tuned = False
        self._init_model()

    def _init_model(self) -> None:
        """Load multimodal model and processor."""
        try:
            from transformers import (
                AutoTokenizer,
                LayoutLMv3Config,
                LayoutLMv3ForSequenceClassification,
                LayoutLMv3ImageProcessor,
                LayoutLMv3Processor,
            )
            import torch

            self._torch = torch
            fine_tuned_path = settings.layoutxlm_dir
            config_path = fine_tuned_path / "config.json"
            has_weights = (
                (fine_tuned_path / "model.safetensors").exists()
                or (fine_tuned_path / "pytorch_model.bin").exists()
            )

            if fine_tuned_path.exists() and has_weights:
                logger.info(
                    "loading_fine_tuned_model",
                    path=str(fine_tuned_path),
                )
                self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                    str(fine_tuned_path),
                    num_labels=NUM_CLASSES,
                )
                tokenizer = AutoTokenizer.from_pretrained(str(fine_tuned_path))
                image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
                self.processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)
                self.fine_tuned = True
                logger.info("fine_tuned_model_loaded_successfully")
            elif fine_tuned_path.exists() and config_path.exists():
                logger.info(
                    "initializing_model_from_config",
                    path=str(fine_tuned_path),
                )
                config = LayoutLMv3Config.from_pretrained(str(fine_tuned_path), num_labels=NUM_CLASSES)
                self.model = LayoutLMv3ForSequenceClassification(config)
                tokenizer = AutoTokenizer.from_pretrained(str(fine_tuned_path))
                image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
                self.processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)
                self.fine_tuned = False
            else:
                logger.warning(
                    "no_fine_tuned_weights_found",
                    path=str(fine_tuned_path),
                    msg="Using base LayoutLMv3 model. Predictions will be calibrated upon fine-tuning.",
                )
                model_name = "microsoft/layoutlmv3-base"
                self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=NUM_CLASSES,
                )
                self.processor = LayoutLMv3Processor.from_pretrained(
                    model_name,
                    apply_ocr=False
                )
                self.fine_tuned = False

            # Configure device
            if settings.USE_GPU and torch.cuda.is_available():
                self.device = torch.device(f"cuda:{settings.GPU_DEVICE_ID}")
                gpu_name = torch.cuda.get_device_name(settings.GPU_DEVICE_ID)
                vram_mb = torch.cuda.get_device_properties(settings.GPU_DEVICE_ID).total_memory / (1024**2)
                logger.info(
                    "layoutxlm_gpu_enabled",
                    device=str(self.device),
                    gpu=gpu_name,
                    vram_mb=f"{vram_mb:.0f}",
                )
            else:
                self.device = torch.device("cpu")
                logger.info("layoutxlm_cpu_mode", device="cpu")

            self.model.eval()
            self.model.to(self.device)
            self.model_loaded = True

            logger.info(
                "multimodal_classifier_ready",
                device=str(self.device),
                fine_tuned=self.fine_tuned,
                gpu_enabled=(self.device.type == "cuda"),
            )

        except Exception as e:
            logger.error("multimodal_classifier_init_error", error=str(e))
            self.model_loaded = False

    def predict(
        self,
        image_path: str,
        ocr_tokens: list[Any],
    ) -> ClassificationOutput:
        """
        Classify a document using the multimodal visual-layout-text model.

        Args:
            image_path: Path to the document image (first page).
            ocr_tokens: List of OCR token objects with .text, .x1, .y1, .x2, .y2 attributes.

        Returns:
            ClassificationOutput with predicted class, confidence, and all probabilities.
        """
        if not self.model_loaded:
            logger.error("classifier_not_loaded")
            return ClassificationOutput(
                predicted_class="UNKNOWN_OUT_OF_SCOPE",
                confidence=0.0,
                all_probabilities={c: 0.0 for c in self.CLASSES},
            )

        try:
            import torch

            words, boxes = self._prepare_tokens(ocr_tokens)
            image = self._load_image(image_path)

            if not words:
                words = ["document"]
                boxes = [[0, 0, 1000, 1000]]

            if image is None:
                image = Image.new("RGB", (224, 224), color=(240, 240, 240))

            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            # Run inference with mixed precision on GPU
            use_amp = (self.device.type == "cuda")
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = self.model(**encoding)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs_np = probs.cpu().numpy()[0]

            # Build probability map
            all_probabilities = {
                cls_name: float(probs_np[i]) for i, cls_name in enumerate(self.CLASSES)
            }

            # Top prediction
            top_idx = int(np.argmax(probs_np))
            predicted_class = self.CLASSES[top_idx]
            confidence = float(probs_np[top_idx])

            # Hybrid Evidence Calibration
            full_text_lower = " ".join(words).lower()

            # 1. Distinct Out of Scope keywords (NCL, Income, Domicile, Admission/CET, Marksheet)
            oos_cues = [
                "non creamy layer", "non-creamy layer", "non creamy", "non-creamy", "ncl",
                "income certificate", "domicile certificate", "nationality certificate",
                "admission form", "state common entrance test cell",
                "allotted choice code", "allotted seat type", "cap round", "cap allotment",
                "tuition fees", "institute reporting"
            ]
            has_oos_marker = any(cue in full_text_lower for cue in oos_cues)

            # 2. Validity Receipt cues (Receipt from Scrutiny Committee)
            is_validity_receipt = (
                ("receipt" in full_text_lower or "पावती" in full_text_lower or "acknowledgement" in full_text_lower)
                and (
                    "scrutiny committee" in full_text_lower
                    or "district caste" in full_text_lower
                    or "caste certificate verification" in full_text_lower
                    or "caste validity" in full_text_lower
                    or "जात प्रमाणपत्र पडताळणी" in full_text_lower
                    or "पडताळणी समिती" in full_text_lower
                    or "supporting documents receipt" in full_text_lower
                )
                and "certificate of validity" not in full_text_lower
                and "form 15" not in full_text_lower
                and "form-15" not in full_text_lower
                and "claim is held valid" not in full_text_lower
            )

            # 3. Caste Validity Certificate cues (True Certificate)
            is_validity_cert = (
                ("certificate of validity" in full_text_lower
                 or "validity certificate" in full_text_lower
                 or "जात वैधता प्रमाणपत्र" in full_text_lower
                 or "form 15" in full_text_lower
                 or "form-15" in full_text_lower
                 or "claim is held valid" in full_text_lower
                 or "claim is valid" in full_text_lower
                 or ("district caste certificate scrutiny committee" in full_text_lower and not is_validity_receipt)
                 or ("scrutiny committee" in full_text_lower and not is_validity_receipt))
                and not is_validity_receipt
                and not has_oos_marker
            )

            # 4. Caste Certificate cues (True Caste Certificate)
            is_caste_cert = (
                ("caste certificate" in full_text_lower
                 or "जातीचे प्रमाणपत्र" in full_text_lower
                 or "जातीचा दाखला" in full_text_lower
                 or "form 6" in full_text_lower or "form-6" in full_text_lower
                 or "form 7" in full_text_lower or "form-7" in full_text_lower
                 or "form 8" in full_text_lower or "form-8" in full_text_lower
                 or "produced by other backward classes" in full_text_lower
                 or "produced by scheduled castes" in full_text_lower)
                and not has_oos_marker
                and not is_validity_cert
                and not is_validity_receipt
            )

            # 5. Proforma O cues
            is_proforma = (
                any(cue in full_text_lower for cue in [
                    "proforma-o", "proforma -o", "proforma o", "proforma-0", "proforma 0",
                    "प्रपत्र-ओ", "प्रपत्र ओ", "minority community student", "self declaration for minority"
                ])
                and not has_oos_marker
                and not is_validity_cert
                and not is_validity_receipt
            )

            # 6. Leaving Certificate cues
            is_lc = (
                any(cue in full_text_lower for cue in [
                    "leaving certificate", "school leaving", "college leaving", "transfer certificate",
                    "lc no", "lc no.", "name of the pupil", "general register", "शाळा सोडल्याचा दाखला"
                ])
                and not has_oos_marker
                and not is_validity_cert
                and not is_caste_cert
                and not is_validity_receipt
            )

            # 7. Out of Scope
            is_oos = (
                has_oos_marker
                and not is_validity_receipt
            )

            # Route by priority
            if is_oos:
                logger.info("rule_calibration_applied", rule="OUT_OF_SCOPE")
                predicted_class = "UNKNOWN_OUT_OF_SCOPE"
                confidence = max(confidence, 0.98)
            elif is_validity_receipt:
                logger.info("rule_calibration_applied", rule="CASTE_VALIDITY_RECEIPT")
                predicted_class = "CASTE_VALIDITY_RECEIPT"
                confidence = max(confidence, 0.98)
            elif is_validity_cert:
                logger.info("rule_calibration_applied", rule="CASTE_VALIDITY_CERTIFICATE")
                predicted_class = "CASTE_VALIDITY_CERTIFICATE"
                confidence = max(confidence, 0.98)
            elif is_caste_cert:
                logger.info("rule_calibration_applied", rule="CASTE_CERTIFICATE")
                predicted_class = "CASTE_CERTIFICATE"
                confidence = max(confidence, 0.98)
            elif is_proforma:
                logger.info("rule_calibration_applied", rule="PROFORMA_O")
                predicted_class = "PROFORMA_O"
                confidence = max(confidence, 0.98)
            elif is_lc:
                logger.info("rule_calibration_applied", rule="LEAVING_CERTIFICATE")
                predicted_class = "LEAVING_CERTIFICATE"
                confidence = max(confidence, 0.98)

            # Apply confidence threshold — below threshold -> UNKNOWN (§19)
            if confidence < settings.CONFIDENCE_THRESHOLD:
                logger.info(
                    "confidence_below_threshold",
                    predicted=predicted_class,
                    confidence=confidence,
                    threshold=settings.CONFIDENCE_THRESHOLD,
                )
                predicted_class = "UNKNOWN_OUT_OF_SCOPE"

            logger.info(
                "classification_result",
                predicted_class=predicted_class,
                confidence=round(confidence, 4),
                fine_tuned=self.fine_tuned,
            )

            return ClassificationOutput(
                predicted_class=predicted_class,
                confidence=confidence,
                all_probabilities=all_probabilities,
            )

        except Exception as e:
            logger.error("classification_inference_error", error=str(e))
            return ClassificationOutput(
                predicted_class="UNKNOWN_OUT_OF_SCOPE",
                confidence=0.0,
                all_probabilities={c: 0.0 for c in self.CLASSES},
            )

    def _prepare_tokens(
        self, ocr_tokens: list[Any], img_w: int = 1000, img_h: int = 1000
    ) -> tuple[list[str], list[list[int]]]:
        """Convert OCR tokens to normalized coordinates 0-1000."""
        words = []
        boxes = []

        for token in ocr_tokens:
            text = getattr(token, "text", str(token)).strip()
            if not text:
                continue

            if hasattr(token, "bbox") and isinstance(token.bbox, (list, tuple)) and len(token.bbox) == 4:
                x1, y1, x2, y2 = token.bbox
            else:
                x1 = getattr(token, "x1", 0)
                y1 = getattr(token, "y1", 0)
                x2 = getattr(token, "x2", 0)
                y2 = getattr(token, "y2", 0)

            # Ensure box coordinates are ordered and normalized 0-1000
            x_min, x_max = min(float(x1), float(x2)), max(float(x1), float(x2))
            y_min, y_max = min(float(y1), float(y2)), max(float(y1), float(y2))

            if img_w > 1000 or img_h > 1000:
                x_min = (x_min / img_w) * 1000
                x_max = (x_max / img_w) * 1000
                y_min = (y_min / img_h) * 1000
                y_max = (y_max / img_h) * 1000

            x_min = max(0, min(1000, int(x_min)))
            y_min = max(0, min(1000, int(y_min)))
            x_max = max(x_min + 1, min(1000, int(x_max)))
            y_max = max(y_min + 1, min(1000, int(y_max)))

            words.append(text)
            boxes.append([x_min, y_min, x_max, y_max])

        return words, boxes

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and convert image to RGB PIL Image."""
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            logger.warning("image_load_failed", path=image_path, error=str(e))
            return None


def classify_document(
    image_path: str,
    ocr_tokens: list[Any],
) -> ClassificationOutput:
    """
    Convenience function to classify a document.
    Uses the singleton LayoutXLMClassifier instance.
    """
    classifier = LayoutXLMClassifier()
    return classifier.predict(image_path, ocr_tokens)
