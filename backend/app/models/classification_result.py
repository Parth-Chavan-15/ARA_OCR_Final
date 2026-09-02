"""
ARA OCR — Classification Result ORM Model

Stores the broad document classification result with confidence,
evidence, and model version (§21, §22).
Designed to be extensible for future template_type, extracted_fields,
validation_result, and cross_document_findings columns.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class ClassificationResult(Base):
    __tablename__ = "classification_results"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    predicted_class: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    model_version: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # ── Future columns (deferred §37) ──
    # template_type: Mapped[str | None]          (Form 6/7/8)
    # extracted_fields: Mapped[dict | None]      (field extraction)
    # validation_result: Mapped[dict | None]     (validation findings)
    # cross_document_findings: Mapped[dict | None]

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="classification_results"
    )

    def __repr__(self) -> str:
        return (
            f"<ClassificationResult "
            f"doc={self.document_id} "
            f"class={self.predicted_class} "
            f"conf={self.confidence:.2f}>"
        )
