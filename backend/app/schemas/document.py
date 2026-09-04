"""
ARA OCR — Document Pydantic Schemas

Request/response models for the Document API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentBase(BaseModel):
    """Shared document fields."""

    filename: str = Field(..., description="Original filename")
    candidate_id: str = Field(..., description="ID of the owning candidate")


class DocumentCreate(DocumentBase):
    """Schema for registering a new document."""
    pass


class DocumentResponse(BaseModel):
    """Schema returned when reading a document."""

    document_id: str
    candidate_id: str
    filename: str
    file_path: str
    document_type: Optional[str] = None
    status: str
    uploaded_at: datetime
    confidence: Optional[float] = None
    raw_confidence: Optional[float] = None
    confidence_label: Optional[str] = None

    model_config = {"from_attributes": True}


class DocumentStatusResponse(BaseModel):
    """Current processing status of a document."""

    document_id: str
    status: str
    document_type: Optional[str] = None
    message: Optional[str] = None

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""

    document_id: str
    candidate_id: str
    filename: str
    status: str
    message: str
