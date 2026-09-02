"""
ARA OCR — OCR Pydantic Schemas

Request/response models for OCR results and tokens.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class OcrTokenResponse(BaseModel):
    """Single OCR token with bounding box and confidence."""

    token_id: str
    page_number: int
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class OcrResultResponse(BaseModel):
    """Full OCR result for a document."""

    ocr_id: str
    document_id: str
    model_version: str
    language: Optional[str] = None
    raw_text: str
    overall_confidence: float
    created_at: datetime
    tokens: list[OcrTokenResponse] = []

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class OcrResultSummary(BaseModel):
    """Lightweight OCR summary without individual tokens."""

    ocr_id: str
    document_id: str
    model_version: str
    language: Optional[str] = None
    overall_confidence: float
    token_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}
