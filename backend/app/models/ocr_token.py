"""
ARA OCR — OCR Token ORM Model

Individual tokens/words extracted by PaddleOCR, with their
confidence scores, bounding boxes, and page numbers (§13, §22).
"""

import uuid

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class OcrToken(Base):
    __tablename__ = "ocr_tokens"

    token_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    ocr_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("ocr_results.ocr_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(String(1024), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationships
    ocr_result: Mapped["OcrResult"] = relationship(
        "OcrResult", back_populates="tokens"
    )

    def __repr__(self) -> str:
        return (
            f"<OcrToken '{self.text}' "
            f"conf={self.confidence:.2f} "
            f"page={self.page_number} "
            f"bbox=({self.x1},{self.y1},{self.x2},{self.y2})>"
        )
