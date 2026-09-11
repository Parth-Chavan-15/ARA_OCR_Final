"""
ARA OCR — Candidate Pydantic Schemas

Request/response models for the Candidate API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CandidateBase(BaseModel):
    """Shared fields for candidate creation and display."""

    enrollment_number: str = Field(
        ..., min_length=1, max_length=64, description="Unique enrollment number"
    )
    name: str = Field(
        ..., min_length=1, max_length=256, description="Candidate full name"
    )
    reserve_category: Optional[str] = Field(
        None,
        max_length=64,
        description="Reserve category (e.g. SC, OBC, VJNT, SBC, OPEN)",
    )
    institute_code: str = Field(
        default="06122",
        max_length=64,
        description="5-digit or alphanumeric Institute Code",
    )
    stream: str = Field(
        default="MHT-CET Engineering",
        max_length=128,
        description="Stream or course (e.g. NEET(UG) MBBS, MHT-CET Engineering)",
    )
    institute_name: Optional[str] = Field(
        None,
        max_length=256,
        description="Institute full name if available",
    )


class CandidateCreate(CandidateBase):
    """Schema for creating a new candidate."""
    pass


class CandidateResponse(CandidateBase):
    """Schema returned when reading a candidate."""

    candidate_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CandidateWithDocuments(CandidateResponse):
    """Candidate response including all linked documents."""

    documents: list["DocumentResponse"] = []


# Avoid circular import — resolve forward reference
from app.schemas.document import DocumentResponse  # noqa: E402

CandidateWithDocuments.model_rebuild()
