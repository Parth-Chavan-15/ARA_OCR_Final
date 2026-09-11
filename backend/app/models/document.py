"""
ARA OCR — Document ORM Model

Every document belongs to a candidate.
document_type is null until classification completes.
status tracks the processing state through the pipeline (§8, §22).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    candidate_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("candidates.candidate_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    document_type: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PENDING"
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    candidate: Mapped["Candidate"] = relationship(
        "Candidate", back_populates="documents"
    )
    ocr_results: Mapped[list["OcrResult"]] = relationship(
        "OcrResult", back_populates="document", lazy="selectin"
    )
    classification_results: Mapped[list["ClassificationResult"]] = relationship(
        "ClassificationResult", back_populates="document", lazy="selectin"
    )

    @property
    def confidence(self) -> float | None:
        if self.classification_results:
            return self.classification_results[-1].confidence
        return None

    @property
    def raw_confidence(self) -> float | None:
        if self.classification_results and isinstance(self.classification_results[-1].evidence, dict):
            return self.classification_results[-1].evidence.get("raw_confidence")
        return None

    @property
    def confidence_label(self) -> str | None:
        if self.classification_results and isinstance(self.classification_results[-1].evidence, dict):
            return self.classification_results[-1].evidence.get("confidence_label")
        return None

    @property
    def extracted_fields(self) -> dict | None:
        if self.classification_results and hasattr(self.classification_results[-1], "extracted_fields"):
            return self.classification_results[-1].extracted_fields
        return None

    def __repr__(self) -> str:
        return (
            f"<Document {self.document_id} "
            f"file={self.filename} "
            f"status={self.status}>"
        )
