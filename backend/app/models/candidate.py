"""
ARA OCR — Candidate ORM Model

A candidate may have multiple documents.
The candidate-document relationship is the foundation
of the data model (§7, §22).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Candidate(Base):
    __tablename__ = "candidates"

    candidate_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    enrollment_number: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    reserve_category: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="candidate", lazy="selectin"
    )

    def __repr__(self) -> str:
        return (
            f"<Candidate {self.candidate_id} "
            f"enrollment={self.enrollment_number} "
            f"name={self.name}>"
        )
