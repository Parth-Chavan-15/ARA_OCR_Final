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
}

def sync_disk_storage(db: Session) -> Dict[str, Any]:
    originals_dir = Path(settings.originals_dir)
    if not originals_dir.exists():
        originals_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "success", "candidates_synced": 0, "documents_synced": 0}

    candidates_added = 0
    documents_added = 0
    candidates_updated = 0

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
                documents_added += 1
                logger.info(
                    "synced_new_document",
                    candidate_id=candidate.candidate_id,
                    filename=filename,
                )


    return {
        "status": "success",
        "candidates_added": candidates_added,
        "candidates_updated": candidates_updated,
        "documents_added": documents_added,
        "total_candidates": db.query(Candidate).count(),
        "total_documents": db.query(Document).count(),
    }
