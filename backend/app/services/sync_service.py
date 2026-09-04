import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import structlog
from sqlalchemy.orm import Session

from app.config import settings
import app.models  # Ensures all ORM models are registered in registry
from app.models.candidate import Candidate
from app.models.document import Document
from app.models.ocr_result import OcrResult
from app.models.ocr_token import OcrToken
from app.models.classification_result import ClassificationResult

logger = structlog.get_logger('sync_service')

OFFICIAL_CANDIDATE_NAMES: Dict[str, str] = {
    "EN20260001": "TAWARE TANMAY SAHADEO",
    "EN20260002": "ARYAN VIJAY JADHAV",
    "EN20260003": "MAITREYEE MAHENDRA DAHIWALE",
    "EN20260004": "ATHARVA PRAMOD LONDHE",
    "EN20260005": "NIRJA BIPIN BHOSLE",
    "EN20260006": "CHAUDHARI TEJAS JITENDRA",
    "EN20260007": "RANE SOHAM SACHIN",
    "EN20260008": "GURAV SAEE VIJAY",
    "EN20260009": "MOHAMMED MUIN IMRAN SAWANT",
    "EN20260010": "PARAS SANTOSH WAGH",
    "EN20260011": "JAYESH SHRIRANG FULPAGARE",
    "EN20260012": "AVHAD SUMEET SHANTARAM",
    "EN20260013": "GHUGE RUTIK ASHOK",
    "EN20260014": "MANSHA KUMAR KITHANI",
    "EN20260015": "BHOINKAR SHARVANI SANTOSH",
    "EN20260016": "PATIL ARYAN ANANTA",
    "EN20260017": "GORE AMEYA AMOL",
    "EN20260018": "CHOPADEKAR SURABHI HARESH",
    "EN20260019": "HARSHAL DNYANESHWAR WANKHADE",
    "EN20260020": "JETHWANI URVASHI DILIP",
    "EN20260021": "KESWANI HARSHIT VIJAYKUMAR",
    "EN20260022": "NANDWANI KANAK HIRACHAND",
    "EN20260023": "CHANDWANI GRACY SATISH",
    "EN20260024": "ROHRA MUSKAN RAVI",
    "EN20260025": "ADITYA RAJENDRA NAGARE",
    "EN20260026": "AADI ANIL MOKASHI",
    "EN20260027": "CHANDWANI KARAN KAILASH",
    "EN25272514": "Rahul Sharma",
    "EN25272515": "Priya Deshmukh",
    "EN25272516": "Amit Jadhav",
    "EN25272517": "Sneha Patil",
    "EN25272518": "Vikram Kulkarni",
}

def sync_disk_storage(db: Session) -> Dict[str, Any]:
    originals_dir = Path(settings.originals_dir).resolve()
    if not originals_dir.exists():
        originals_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "success", "candidates_synced": 0, "documents_synced": 0}

    candidates_added = 0
    documents_added = 0
    documents_updated = 0
    candidates_updated = 0

    valid_disk_doc_ids = set()

    for entry in sorted(originals_dir.iterdir()):
        if not entry.is_dir():
            continue

        enrollment_no = entry.name
        candidate_name = OFFICIAL_CANDIDATE_NAMES.get(
            enrollment_no, f"Candidate {enrollment_no}"
        )

        candidate = (
            db.query(Candidate)
            .filter(Candidate.enrollment_number == enrollment_no)
            .first()
        )

        if not candidate:
            candidate = Candidate(
                candidate_id=str(uuid.uuid4()),
                enrollment_number=enrollment_no,
                name=candidate_name,
                reserve_category="OBC",
                created_at=datetime.now(timezone.utc),
            )
            db.add(candidate)
            db.commit()
            db.refresh(candidate)
            candidates_added += 1
            logger.info("synced_new_candidate", enrollment=enrollment_no, name=candidate_name)
        else:
            if enrollment_no in OFFICIAL_CANDIDATE_NAMES and candidate.name != OFFICIAL_CANDIDATE_NAMES[enrollment_no]:
                candidate.name = OFFICIAL_CANDIDATE_NAMES[enrollment_no]
                db.commit()
                candidates_updated += 1

        supported_exts = [".pdf", ".jpg", ".jpeg", ".png"]
        for file_entry in sorted(entry.iterdir()):
            if not file_entry.is_file() or file_entry.suffix.lower() not in supported_exts:
                continue

            filename = file_entry.name
            file_path = str(file_entry.resolve())

            existing_doc = (
                db.query(Document)
                .filter(
                    Document.candidate_id == candidate.candidate_id,
                    Document.filename == filename,
                )
                .first()
            )

            if not existing_doc:
                doc = Document(
                    document_id=str(uuid.uuid4()),
                    candidate_id=candidate.candidate_id,
                    filename=filename,
                    file_path=file_path,
                    document_type=None,
                    status="PENDING",
                    uploaded_at=datetime.now(timezone.utc),
                )
                db.add(doc)
                db.commit()
                valid_disk_doc_ids.add(doc.document_id)
                documents_added += 1
                logger.info(
                    "synced_new_document",
                    candidate_id=candidate.candidate_id,
                    filename=filename,
                )
            else:
                valid_disk_doc_ids.add(existing_doc.document_id)
                # Check and update file path if changed or moved
                if existing_doc.file_path != file_path:
                    existing_doc.file_path = file_path
                    db.commit()
                    documents_updated += 1
                    logger.info(
                        "updated_document_path",
                        document_id=existing_doc.document_id,
                        file_path=file_path,
                    )

    # Prune orphaned documents in DB that no longer exist on disk
    all_db_docs = db.query(Document).all()
    pruned_count = 0
    for db_doc in all_db_docs:
        if not os.path.exists(db_doc.file_path):
            # Clean up dependent relations first
            db.query(ClassificationResult).filter(ClassificationResult.document_id == db_doc.document_id).delete()
            db.query(OcrToken).filter(OcrToken.ocr_id.in_(
                db.query(OcrResult.ocr_id).filter(OcrResult.document_id == db_doc.document_id)
            )).delete(synchronize_session=False)
            db.query(OcrResult).filter(OcrResult.document_id == db_doc.document_id).delete()
            db.delete(db_doc)
            pruned_count += 1
            logger.info("pruned_missing_document", document_id=db_doc.document_id, file_path=db_doc.file_path)

    # Prune orphaned candidates who have 0 documents and no disk folder
    all_candidates = db.query(Candidate).all()
    pruned_candidates = 0
    for cand in all_candidates:
        doc_count = db.query(Document).filter(Document.candidate_id == cand.candidate_id).count()
        cand_folder = originals_dir / cand.enrollment_number
        if doc_count == 0 and not cand_folder.exists():
            db.delete(cand)
            pruned_candidates += 1
            logger.info("pruned_empty_candidate", candidate_id=cand.candidate_id, enrollment=cand.enrollment_number)

    if pruned_count > 0 or pruned_candidates > 0:
        db.commit()

    return {
        "status": "success",
        "candidates_added": candidates_added,
        "candidates_updated": candidates_updated,
        "candidates_pruned": pruned_candidates,
        "documents_added": documents_added,
        "documents_updated": documents_updated,
        "documents_pruned": pruned_count,
        "total_candidates": db.query(Candidate).count(),
        "total_documents": db.query(Document).count(),
    }
