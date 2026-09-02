"""
ARA OCR — Classification Pipeline API

POST /api/documents/{document_id}/classify — Run the full pipeline:
    File validation → PDF → images → preprocessing → OCR (cached) →
    language detection → LayoutXLM classification → evidence → persist

GET  /api/dashboard/stats — Dashboard aggregate statistics
GET  /api/documents/review — Documents needing human review

The pipeline enforces OCR-ONCE-REUSE-EVERYWHERE (§2, §29).
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.models.candidate import Candidate
from app.models.classification_result import ClassificationResult
from app.models.document import Document
from app.schemas.classification import (
    ClassificationResultResponse,
    ClassifyResponse,
    DashboardStats,
)
from app.schemas.document import DocumentResponse

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api", tags=["Classification"])

# ── Lazy-import heavy ML services so the API can start even without them ──
SERVICES_AVAILABLE = False
_import_error_msg = ""
try:
    from app.services.file_handler import FileHandler
    from app.services.pdf_service import PdfService
    from app.services.preprocessing import PreprocessingService
    from app.services.ocr_cache import OcrCache
    from app.services.language import LanguageService
    from app.services.classifier import classify_document, ClassificationOutput
    from app.services.evidence import EvidenceService

    SERVICES_AVAILABLE = True
except ImportError as e:
    _import_error_msg = str(e)
    logger.warning("ml_services_import_failed", error=_import_error_msg)


def _stage(name: str, status: str, error: str | None = None) -> dict[str, Any]:
    """Create a pipeline stage log entry."""
    entry: dict[str, Any] = {
        "stage": name,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        entry["error"] = error
    return entry


# ────────────────────────────────────────────────────────────────────────────
# POST /api/documents/{document_id}/classify
# ────────────────────────────────────────────────────────────────────────────


@router.post("/documents/{document_id}/classify", response_model=ClassifyResponse)
def run_classification_pipeline(
    document_id: str,
    db: Session = Depends(get_db),
) -> ClassifyResponse:
    """Execute the full classification pipeline for a document."""

    stages: list[dict[str, Any]] = []

    # ── Load document ──
    document = (
        db.query(Document).filter(Document.document_id == document_id).first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    stages.append(_stage("fetch", "COMPLETED"))

    if not SERVICES_AVAILABLE:
        document.status = "ERROR"
        db.commit()
        stages.append(
            _stage("services", "FAILED", error=f"ML services unavailable: {_import_error_msg}")
        )
        return ClassifyResponse(
            document_id=document.document_id,
            status="ERROR",
            error=f"ML services not installed: {_import_error_msg}",
            pipeline_stages=stages,
        )

    try:
        document.status = "PROCESSING"
        db.commit()

        # ── Stage 1: File validation (§10) ──
        stages.append(_stage("file_validation", "STARTED"))
        validation = FileHandler.validate_file(document.file_path)
        if validation.status != "VALID":
            raise _PipelineError(
                "file_validation",
                f"File validation failed: {validation.status} — {validation.reason}",
            )
        stages.append(_stage("file_validation", "COMPLETED"))

        # ── Stage 2: PDF / image extraction (§11) ──
        stages.append(_stage("pdf_rendering", "STARTED"))
        page_images = PdfService.extract_images(document.file_path)
        if not page_images:
            raise _PipelineError("pdf_rendering", "No pages/images could be extracted")
        stages.append(_stage("pdf_rendering", "COMPLETED"))

        # ── Stage 3: OpenCV preprocessing (§12) ──
        stages.append(_stage("preprocessing", "STARTED"))
        processed_paths, quality_metrics = PreprocessingService.process_images(
            document.document_id, page_images
        )
        if not processed_paths:
            raise _PipelineError("preprocessing", "Preprocessing produced no output images")
        stages.append(_stage("preprocessing", "COMPLETED"))

        # ── Stage 4+5: OCR with cache (§13, §14, §29) ──
        stages.append(_stage("ocr", "STARTED"))
        ocr_result = OcrCache.get_or_create_ocr(
            db=db,
            document_id=document.document_id,
            model_version=settings.OCR_MODEL_VERSION,
            image_paths=processed_paths,
        )
        if ocr_result is None:
            raise _PipelineError("ocr", "OCR processing failed")
        cache_hit = len(stages) > 0  # simplification — OcrCache logs cache hit/miss
        stages.append(_stage("ocr", "COMPLETED"))
        stages.append(_stage("ocr_cache", "COMPLETED"))

        # ── Stage 6: Language / script detection (§15) ──
        stages.append(_stage("language_detection", "STARTED"))
        token_texts = [t.text for t in ocr_result.tokens]
        language = LanguageService.detect_language(token_texts)
        # Persist language on the OCR result
        ocr_result.language = language
        db.commit()
        stages.append(_stage("language_detection", "COMPLETED"))

        # ── Stage 7: LayoutXLM classification (§16, §40) ──
        stages.append(_stage("layoutxlm_classification", "STARTED"))
        # Use the first processed image for the classifier
        image_for_classifier = processed_paths[0] if processed_paths else document.file_path
        class_output: ClassificationOutput = classify_document(
            image_path=image_for_classifier,
            ocr_tokens=ocr_result.tokens,
        )
        stages.append(_stage("layoutxlm_classification", "COMPLETED"))

        # ── Stage 8: Evidence generation (§20) ──
        stages.append(_stage("evidence_generation", "STARTED"))
        evidence = EvidenceService.generate_evidence(
            predicted_class=class_output.predicted_class,
            ocr_tokens=ocr_result.tokens,
        )
        # Add probability breakdown to evidence
        evidence["class_probabilities"] = class_output.all_probabilities
        stages.append(_stage("evidence_generation", "COMPLETED"))

        # ── Stage 9: Persist classification result (§21) ──
        stages.append(_stage("persist_result", "STARTED"))
        classification_record = ClassificationResult(
            id=str(uuid.uuid4()),
            document_id=document.document_id,
            predicted_class=class_output.predicted_class,
            confidence=class_output.confidence,
            evidence=evidence,
            model_version=settings.CLASSIFIER_MODEL_VERSION,
        )
        db.add(classification_record)

        # Update document status and type
        if class_output.predicted_class == "UNKNOWN_OUT_OF_SCOPE":
            document.status = "UNKNOWN"
        else:
            document.status = "CLASSIFIED"
        document.document_type = class_output.predicted_class
        db.commit()
        stages.append(_stage("persist_result", "COMPLETED"))

        logger.info(
            "classification_pipeline_completed",
            document_id=document.document_id,
            predicted_class=class_output.predicted_class,
            confidence=f"{class_output.confidence:.4f}",
        )

        return ClassifyResponse(
            document_id=document.document_id,
            status="SUCCESS",
            predicted_class=class_output.predicted_class,
            confidence=class_output.confidence,
            language=language,
            ocr_confidence=ocr_result.overall_confidence,
            evidence=evidence,
            model_version=settings.CLASSIFIER_MODEL_VERSION,
            pipeline_stages=stages,
        )

    except _PipelineError as pe:
        stages.append(_stage(pe.stage, "FAILED", error=pe.message))
        document.status = "ERROR"
        db.commit()
        logger.error(
            "pipeline_stage_failed",
            document_id=document.document_id,
            stage=pe.stage,
            error=pe.message,
        )
        return ClassifyResponse(
            document_id=document.document_id,
            status="ERROR",
            error=pe.message,
            pipeline_stages=stages,
        )

    except Exception as e:
        stages.append(_stage("pipeline", "FAILED", error=str(e)))
        document.status = "ERROR"
        db.commit()
        logger.error(
            "classification_pipeline_failed",
            document_id=document.document_id,
            error=str(e),
        )
        return ClassifyResponse(
            document_id=document.document_id,
            status="ERROR",
            error=str(e),
            pipeline_stages=stages,
        )


class _PipelineError(Exception):
    """Internal error for pipeline stage failures."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(message)


# ────────────────────────────────────────────────────────────────────────────
# GET /api/dashboard/stats
# ────────────────────────────────────────────────────────────────────────────


@router.get("/dashboard/stats", response_model=DashboardStats)
def get_dashboard_stats(db: Session = Depends(get_db)) -> DashboardStats:
    """Compute aggregate statistics for the dashboard."""
    total_candidates = db.query(func.count(Candidate.candidate_id)).scalar() or 0
    total_documents = db.query(func.count(Document.document_id)).scalar() or 0
    processed = (
        db.query(func.count(Document.document_id))
        .filter(Document.status != "PENDING")
        .scalar()
        or 0
    )
    classified = (
        db.query(func.count(Document.document_id))
        .filter(Document.status == "CLASSIFIED")
        .scalar()
        or 0
    )
    unknown = (
        db.query(func.count(Document.document_id))
        .filter(Document.document_type == "UNKNOWN_OUT_OF_SCOPE")
        .scalar()
        or 0
    )
    errors = (
        db.query(func.count(Document.document_id))
        .filter(Document.status == "ERROR")
        .scalar()
        or 0
    )
    pending = (
        db.query(func.count(Document.document_id))
        .filter(Document.status == "PENDING")
        .scalar()
        or 0
    )

    # Low confidence: classification exists but below threshold
    low_confidence = 0
    try:
        low_confidence = (
            db.query(func.count(ClassificationResult.id))
            .filter(
                ClassificationResult.confidence
                < settings.CLASSIFICATION_CONFIDENCE_THRESHOLD
            )
            .scalar()
            or 0
        )
    except Exception:
        pass

    return DashboardStats(
        total_candidates=total_candidates,
        total_documents=total_documents,
        processed=processed,
        classified=classified,
        unknown=unknown,
        low_confidence=low_confidence,
        errors=errors,
        pending=pending,
    )


# ────────────────────────────────────────────────────────────────────────────
# GET /api/documents/review
# ────────────────────────────────────────────────────────────────────────────


@router.get("/documents/review", response_model=list[DocumentResponse])
def get_documents_for_review(
    db: Session = Depends(get_db),
) -> list[Document]:
    """Return documents needing human review (errors, unknown, low-confidence)."""

    error_docs = db.query(Document).filter(Document.status == "ERROR").all()
    unknown_docs = (
        db.query(Document)
        .filter(Document.document_type == "UNKNOWN_OUT_OF_SCOPE")
        .all()
    )

    # Low-confidence classified documents
    low_conf_docs = []
    try:
        low_conf_docs = (
            db.query(Document)
            .join(ClassificationResult)
            .filter(
                ClassificationResult.confidence
                < settings.CLASSIFICATION_CONFIDENCE_THRESHOLD
            )
            .all()
        )
    except Exception:
        pass

    # Deduplicate by document_id
    seen = set()
    result = []
    for doc in error_docs + unknown_docs + low_conf_docs:
        if doc.document_id not in seen:
            seen.add(doc.document_id)
            result.append(doc)

    return result
