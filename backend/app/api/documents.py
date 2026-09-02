"""
ARA OCR — Document API Endpoints

CRUD operations for documents, file upload, and file serving.
"""

import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.models.candidate import Candidate
from app.models.document import Document
from app.schemas.classification import ClassificationResultResponse
from app.schemas.document import (
    DocumentResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from app.schemas.ocr import OcrResultResponse

from app.services.sync_service import sync_disk_storage

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/sync")
def sync_documents(db: Session = Depends(get_db)):
    """Synchronize candidates and documents from disk storage (data/originals/)."""
    try:
        result = sync_disk_storage(db)
        return result
    except Exception as e:
        logger.error("sync_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.post("/reset")
def reset_all_scrutiny(db: Session = Depends(get_db)):
    """
    Reset all document scrutiny and OCR cache across the entire database,
    returning all documents to PENDING status and re-syncing disk storage.
    """
    try:
        from app.models.classification_result import ClassificationResult
        from app.models.ocr_result import OcrResult
        from app.models.ocr_token import OcrToken

        # Clear classification results & OCR cache
        db.query(ClassificationResult).delete()
        db.query(OcrToken).delete()
        db.query(OcrResult).delete()

        # Reset all document statuses to PENDING and clear document_type
        db.query(Document).update({
            Document.status: "PENDING",
            Document.document_type: None,
        })
        db.commit()

        # Re-sync disk storage to ensure all files are registered
        sync_result = sync_disk_storage(db)

        total_docs = db.query(Document).count()
        logger.info("scrutiny_reset_complete", total_documents=total_docs)
        return {
            "status": "success",
            "message": f"Successfully reset all {total_docs} documents to PENDING. Ready for fresh scrutiny.",
            "total_documents": total_docs,
            "sync": sync_result,
        }
    except Exception as e:
        db.rollback()
        logger.error("reset_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")



@router.get("", response_model=list[DocumentResponse])
def get_documents(
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db),
):
    """List all documents, optionally filtered by status."""
    query = db.query(Document)
    if status:
        query = query.filter(Document.status == status)
    return query.order_by(Document.uploaded_at.desc()).all()


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db)):
    """Get a single document by ID."""
    document = (
        db.query(Document).filter(Document.document_id == document_id).first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(document_id: str, db: Session = Depends(get_db)):
    """Get the current processing status of a document."""
    document = (
        db.query(Document).filter(Document.document_id == document_id).first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(
        document_id=document.document_id,
        status=document.status,
        document_type=document.document_type,
    )


@router.get("/{document_id}/ocr", response_model=OcrResultResponse)
def get_document_ocr(document_id: str, db: Session = Depends(get_db)):
    """Get the OCR result (with tokens) for a document."""
    document = (
        db.query(Document).filter(Document.document_id == document_id).first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if not document.ocr_results:
        raise HTTPException(
            status_code=404, detail="OCR result not found for this document"
        )
    # Return the most recent OCR result
    return document.ocr_results[-1]


@router.get("/{document_id}/classification", response_model=ClassificationResultResponse)
def get_document_classification(document_id: str, db: Session = Depends(get_db)):
    """Get the classification result for a document."""
    document = (
        db.query(Document).filter(Document.document_id == document_id).first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if not document.classification_results:
        raise HTTPException(
            status_code=404,
            detail="Classification result not found for this document",
        )
    # Return the most recent classification result
    return document.classification_results[-1]


@router.post("/upload", response_model=DocumentUploadResponse)
def upload_document(
    file: UploadFile = File(...),
    candidate_id: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Upload a document file for a candidate.

    Saves the original file to data/originals/{candidate_id}/
    and creates a Document record in the database.
    """
    # Validate candidate exists
    candidate = (
        db.query(Candidate)
        .filter(Candidate.candidate_id == candidate_id)
        .first()
    )
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Validate filename safety
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Create safe directory path
    candidate_dir = Path(settings.originals_dir) / candidate.enrollment_number
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # Generate safe filename (preserve original name but add UUID prefix)
    safe_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = candidate_dir / safe_filename

    # Path traversal protection
    if not str(file_path.resolve()).startswith(
        str(settings.originals_dir.resolve())
    ):
        raise HTTPException(status_code=400, detail="Invalid file path")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error("file_upload_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Create Document record
    new_doc = Document(
        document_id=str(uuid.uuid4()),
        candidate_id=candidate_id,
        filename=file.filename,
        file_path=str(file_path),
        document_type=None,
        status="PENDING",
        uploaded_at=datetime.now(timezone.utc),
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    logger.info(
        "document_uploaded",
        document_id=new_doc.document_id,
        candidate_id=candidate_id,
        filename=file.filename,
    )

    return DocumentUploadResponse(
        document_id=new_doc.document_id,
        candidate_id=candidate_id,
        filename=new_doc.filename,
        status=new_doc.status,
        message="Document uploaded successfully",
    )


@router.get("/{document_id}/image")
def get_document_image(document_id: str, db: Session = Depends(get_db)):
    """Serve the original document image/PDF for preview."""
    document = (
        db.query(Document).filter(Document.document_id == document_id).first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = document.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    # Path traversal protection
    abs_data = os.path.abspath(str(settings.DATA_DIR))
    abs_file = os.path.abspath(file_path)
    if not abs_file.startswith(abs_data):
        raise HTTPException(status_code=400, detail="Access denied")

    return FileResponse(file_path)


@router.get("/{document_id}/processed-image")
def get_processed_image(
    document_id: str,
    page: int = Query(1, description="Page number (1-indexed)"),
    db: Session = Depends(get_db),
):
    """Serve the preprocessed image for a specific page."""
    document = (
        db.query(Document).filter(Document.document_id == document_id).first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    processed_path = (
        Path(settings.processed_dir)
        / document.document_id
        / f"page_{page}.png"
    )

    if not processed_path.exists():
        raise HTTPException(status_code=404, detail="Processed image not found")

    # Path traversal protection
    abs_data = os.path.abspath(str(settings.DATA_DIR))
    abs_file = os.path.abspath(str(processed_path))
    if not abs_file.startswith(abs_data):
        raise HTTPException(status_code=400, detail="Access denied")

    return FileResponse(str(processed_path))
