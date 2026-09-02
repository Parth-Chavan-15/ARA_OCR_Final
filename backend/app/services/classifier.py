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
                LayoutLMv3ForSequenceClassification,
                LayoutLMv3Processor,
            )
            import torch

            self._torch = torch
            fine_tuned_path = settings.layoutxlm_dir
            config_path = fine_tuned_path / "config.json"

            if fine_tuned_path.exists() and config_path.exists():
                logger.info(
                    "loading_fine_tuned_model",
                    path=str(fine_tuned_path),
                )
                self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                    str(fine_tuned_path),
                    num_labels=NUM_CLASSES,
                )
                self.processor = LayoutLMv3Processor.from_pretrained(
                    str(fine_tuned_path),
                    apply_ocr=False
                )
                self.fine_tuned = True
                logger.info("fine_tuned_model_loaded_successfully")
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

            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model_loaded = True

            logger.info(
                "multimodal_classifier_ready",
                device=str(self.device),
                fine_tuned=self.fine_tuned,
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

            # Run inference
            with torch.no_grad():
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

            # Apply confidence threshold — below threshold -> UNKNOWN (§19)
            if confidence < settings.CLASSIFICATION_CONFIDENCE_THRESHOLD:
                logger.info(
                    "low_confidence_classification",
                    predicted=predicted_class,
                    confidence=confidence,
                    threshold=settings.CLASSIFICATION_CONFIDENCE_THRESHOLD,
                )
                predicted_class = "UNKNOWN_OUT_OF_SCOPE"

            logger.info(
                "classification_result",
                predicted_class=predicted_class,
                confidence=f"{confidence:.4f}",
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
        self, ocr_tokens: list[Any]
    ) -> tuple[list[str], list[list[int]]]:
        """Convert OCR tokens to normalized coordinates 0-1000."""
        words = []
        boxes = []

        for token in ocr_tokens:
            text = getattr(token, "text", str(token)).strip()
            if not text:
                continue

            x1 = getattr(token, "x1", 0)
            y1 = getattr(token, "y1", 0)
            x2 = getattr(token, "x2", 0)
            y2 = getattr(token, "y2", 0)

            norm_box = [
                max(0, min(1000, int(x1))),
                max(0, min(1000, int(y1))),
                max(0, min(1000, int(x2))),
                max(0, min(1000, int(y2))),
            ]

            words.append(text)
            boxes.append(norm_box)

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
