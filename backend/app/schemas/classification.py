"""
ARA OCR — Classification Pydantic Schemas

Request/response models for classification results, evidence,
and dashboard statistics.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Single piece of classification evidence."""

    text: str = Field(..., description="Matched text/phrase from the document")
    page: Optional[int] = Field(None, description="Page number where found")
    bbox: Optional[list[float]] = Field(
        None, description="Bounding box [x1, y1, x2, y2]"
    )
    relevance: Optional[str] = Field(
        None, description="Why this is relevant to the classification"
    )


class ClassificationResultResponse(BaseModel):
    """Full classification result for a document."""

    id: str
    document_id: str
    predicted_class: str
    confidence: float
    raw_confidence: Optional[float] = None
    rule_applied: Optional[str] = None
    confidence_label: Optional[str] = None
    evidence: dict[str, Any]
    extracted_fields: Optional[dict[str, Any]] = None
    model_version: str
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class ClassificationSummary(BaseModel):
    """Summary view of classification for the incoming-documents list."""

    document_id: str
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    raw_confidence: Optional[float] = None
    confidence_label: Optional[str] = None
    status: str


class ClassifyRequest(BaseModel):
    """Request body for the classify endpoint (currently empty — document_id comes from path)."""
    pass


class ClassifyResponse(BaseModel):
    """Full response from the classification pipeline."""

    document_id: str
    status: str
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    raw_confidence: Optional[float] = None
    rule_applied: Optional[str] = None
    confidence_label: Optional[str] = None
    language: Optional[str] = None
    ocr_confidence: Optional[float] = None
    evidence: Optional[dict[str, Any]] = None
    extracted_fields: Optional[dict[str, Any]] = None
    model_version: Optional[str] = None
    error: Optional[str] = None
    model_config = {"protected_namespaces": ()}

    pipeline_stages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Stage-by-stage processing log",
    )


class DashboardStats(BaseModel):
    """Aggregate statistics for the dashboard view."""

    total_candidates: int = 0
    total_documents: int = 0
    processed: int = 0
    classified: int = 0
    unknown: int = 0
    low_confidence: int = 0
    errors: int = 0
    pending: int = 0
