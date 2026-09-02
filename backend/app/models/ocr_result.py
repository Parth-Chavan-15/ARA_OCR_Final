"""
ARA OCR — OCR Result ORM Model

Stores the overall OCR result for a document, including model version,
detected language, raw text, and overall confidence (§14, §22).
A document may have multiple OCR results if re-processed with a
different model version.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class OcrResult(Base):
    __tablename__ = "ocr_results"

    ocr_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_version: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    language: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    raw_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    overall_confidence: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="ocr_results"
    )
    tokens: Mapped[list["OcrToken"]] = relationship(
        "OcrToken",
        back_populates="ocr_result",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<OcrResult {self.ocr_id} "
            f"doc={self.document_id} "
            f"model={self.model_version} "
            f"lang={self.language}>"
        )
