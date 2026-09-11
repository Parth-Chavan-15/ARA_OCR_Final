import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import structlog
from sqlalchemy.orm import Session

from app.config import settings
import app.models  # Ensures all ORM models are registered in registry
from app.models.candidate import Candidate
from app.models.document import Document
from app.models.ocr_result import OcrResult
from app.models.ocr_token import OcrToken
from app.models.classification_result import ClassificationResult

logger = structlog.get_logger("sync_service")

OFFICIAL_INSTITUTE_NAMES: Dict[str, str] = {
    "06122": "Vivekanand Education Society's Institute of Technology (VESIT), Mumbai",
    "01105": "Grant Government Medical College & Sir J.J. Group of Hospitals, Mumbai",
    "09112": "Institute of Nursing Education, Sir J.J. Group of Hospitals, Mumbai",
    "03103": "R. A. Podar Ayurved Medical College, Worli, Mumbai",
    "15192": "College of Agricultural Engineering & Technology, Dr. BSKKV Dapoli",
    "01005": "Sir J.J. Institute of Applied Art / Fine Art, Fort, Mumbai",
    "02104": "Government Dental College & Hospital, St. George Hospital, Mumbai",
    "04101": "Government Homoeopathic Medical College, Jalgaon",
    "05101": "Mohammadia Tibbia College & Assayer Hospital (BUMS), Malegaon",
    "19183": "College of Agricultural Biotechnology, Loni",
    "16174": "College of Food Technology, Parbhani",
    "11129": "College of Agriculture, Pune",
}

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

def parse_filename_metadata(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse filename of format: {EN}_{Candidate_Name}_{DocTag}.pdf
    Returns: (enrollment_no, candidate_name, doc_tag)
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) >= 3:
        en = parts[0].strip()
        doc_tag = parts[-1].strip()
        name = " ".join(parts[1:-1]).strip()
        return en, name, doc_tag
    elif len(parts) == 2:
        return parts[0].strip(), parts[1].strip(), None
    return None, None, None

def sync_disk_storage(db: Session) -> Dict[str, Any]:
    """
    Recursively scans data/originals/ across ARA hierarchy:
      - [Stream] / [Institute_Code] / [EN]_[Name]_[DocTag].pdf
      - [Stream] / [Institute_Code] / [EN] / [Filename].pdf
      - [EN] / [Filename].pdf (legacy flat)
    """
    originals_dir = Path(settings.originals_dir).resolve()
    if not originals_dir.exists():
        originals_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "success", "candidates_synced": 0, "documents_synced": 0}

    candidates_added = 0
    documents_added = 0
    documents_updated = 0
    candidates_updated = 0
    supported_exts = {".pdf", ".jpg", ".jpeg", ".png"}

    # Map existing candidates by enrollment_number for fast lookup
    existing_candidates = {
        c.enrollment_number: c for c in db.query(Candidate).all()
    }

    # Find all supported files in originals_dir recursively
    all_disk_files = []
    for root, dirs, files in os.walk(originals_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in supported_exts:
                all_disk_files.append(p)

    valid_disk_doc_ids = set()

    for file_path_obj in all_disk_files:
        rel_path = file_path_obj.relative_to(originals_dir)
        parts = rel_path.parts

        stream = "MHT-CET Engineering"
        institute_code = "06122"
        enrollment_no = None
        candidate_name = None

        filename = file_path_obj.name
        parsed_en, parsed_name, _ = parse_filename_metadata(filename)

        # Determine hierarchy from relative path parts
        # e.g., parts = ('NEET(UG) MBBS', '01105', '256000625_PRACHI ANAND GOPLE_CVC.pdf')
        if len(parts) >= 3:
            stream = parts[0]
            institute_code = parts[1]
            if len(parts) >= 4:
                # e.g. [Stream] / [Institute] / [Candidate_EN] / [File.pdf]
                enrollment_no = parts[2]
                candidate_name = OFFICIAL_CANDIDATE_NAMES.get(enrollment_no, parsed_name or f"Candidate {enrollment_no}")
            else:
                enrollment_no = parsed_en or f"EN_{parts[1]}_{filename[:8]}"
                candidate_name = parsed_name or OFFICIAL_CANDIDATE_NAMES.get(enrollment_no, f"Candidate {enrollment_no}")
        elif len(parts) == 2:
            # Could be [Institute] / [File] or [EN] / [File]
            parent_dir = parts[0]
            if parent_dir.isdigit() and len(parent_dir) == 5:
                institute_code = parent_dir
                enrollment_no = parsed_en or f"EN_{parent_dir}_{filename[:8]}"
                candidate_name = parsed_name or f"Candidate {enrollment_no}"
            else:
                enrollment_no = parent_dir
                candidate_name = parsed_name or OFFICIAL_CANDIDATE_NAMES.get(enrollment_no, f"Candidate {enrollment_no}")
        else:
            # Single file in root of originals
            enrollment_no = parsed_en or "EN_GENERAL"
            candidate_name = parsed_name or "General Candidate"

        if not enrollment_no:
            enrollment_no = f"EN_{filename[:12]}"
        if not candidate_name:
            candidate_name = OFFICIAL_CANDIDATE_NAMES.get(enrollment_no, f"Candidate {enrollment_no}")

        institute_name = OFFICIAL_INSTITUTE_NAMES.get(
            institute_code, f"Institute {institute_code}"
        )

        # Get or create Candidate
        candidate = existing_candidates.get(enrollment_no)
        if not candidate:
            candidate = Candidate(
                candidate_id=str(uuid.uuid4()),
                enrollment_number=enrollment_no,
                name=candidate_name,
                reserve_category="OBC",
                institute_code=institute_code,
                stream=stream,
                institute_name=institute_name,
                created_at=datetime.now(timezone.utc),
            )
            db.add(candidate)
            db.commit()
            db.refresh(candidate)
            existing_candidates[enrollment_no] = candidate
            candidates_added += 1
            logger.info("synced_new_candidate", enrollment=enrollment_no, name=candidate_name, stream=stream, inst=institute_code)
        else:
            # Update fields if changed
            changed = False
            if candidate.stream != stream:
                candidate.stream = stream
                changed = True
            if candidate.institute_code != institute_code:
                candidate.institute_code = institute_code
                candidate.institute_name = institute_name
                changed = True
            if candidate_name and candidate.name.startswith("Candidate EN") and not candidate_name.startswith("Candidate EN"):
                candidate.name = candidate_name
                changed = True
            if changed:
                db.commit()
                candidates_updated += 1

        file_path_str = str(file_path_obj.resolve())

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
                file_path=file_path_str,
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
            if existing_doc.file_path != file_path_str:
                existing_doc.file_path = file_path_str
                db.commit()
                documents_updated += 1
                logger.info(
                    "updated_document_path",
                    document_id=existing_doc.document_id,
                    file_path=file_path_str,
                )

    # Prune orphaned documents in DB that no longer exist on disk
    all_db_docs = db.query(Document).all()
    pruned_count = 0
    for db_doc in all_db_docs:
        if not os.path.exists(db_doc.file_path):
            db.query(ClassificationResult).filter(ClassificationResult.document_id == db_doc.document_id).delete()
            db.query(OcrToken).filter(OcrToken.ocr_id.in_(
                db.query(OcrResult.ocr_id).filter(OcrResult.document_id == db_doc.document_id)
            )).delete(synchronize_session=False)
            db.query(OcrResult).filter(OcrResult.document_id == db_doc.document_id).delete()
            db.delete(db_doc)
            pruned_count += 1
            logger.info("pruned_missing_document", document_id=db_doc.document_id, file_path=db_doc.file_path)

    # Prune candidates who have 0 documents
    all_candidates = db.query(Candidate).all()
    pruned_candidates = 0
    for cand in all_candidates:
        doc_count = db.query(Document).filter(Document.candidate_id == cand.candidate_id).count()
        if doc_count == 0:
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
