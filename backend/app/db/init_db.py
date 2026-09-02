import os
import structlog
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.database import Base, engine
from app.models.candidate import Candidate
from app.models.document import Document
from app.models.ocr_result import OcrResult
from app.models.ocr_token import OcrToken
from app.models.classification_result import ClassificationResult
from app.config import settings

logger = structlog.get_logger(__name__)

def init_database():
    logger.info("initializing_database")
    Base.metadata.create_all(bind=engine)
    logger.info("database_initialized")

def seed_demo_data(db: Session):
    logger.info("seeding_demo_data")
    
    demo_candidates = [
        {"enrollment_number": "EN25272514", "name": "Rahul Sharma", "reserve_category": "SC"},
        {"enrollment_number": "EN25272515", "name": "Priya Deshmukh", "reserve_category": "OBC"},
        {"enrollment_number": "EN25272516", "name": "Amit Jadhav", "reserve_category": "VJNT"},
        {"enrollment_number": "EN25272517", "name": "Sneha Patil", "reserve_category": "SBC"},
        {"enrollment_number": "EN25272518", "name": "Vikram Kulkarni", "reserve_category": "OPEN"},
    ]
    
    for c_data in demo_candidates:
        existing = db.query(Candidate).filter(Candidate.enrollment_number == c_data["enrollment_number"]).first()
        if not existing:
            candidate = Candidate(
                enrollment_number=c_data["enrollment_number"],
                name=c_data["name"],
                reserve_category=c_data["reserve_category"]
            )
            db.add(candidate)
            db.commit()
            db.refresh(candidate)
            logger.info("created_demo_candidate", candidate_id=candidate.candidate_id, name=candidate.name)
        else:
            candidate = existing
            
        enrollment_dir = settings.originals_dir / candidate.enrollment_number
        if enrollment_dir.exists():
            for filename in os.listdir(enrollment_dir):
                file_path = str(enrollment_dir / filename)
                if os.path.isfile(file_path):
                    existing_doc = db.query(Document).filter(
                        Document.candidate_id == candidate.candidate_id,
                        Document.filename == filename
                    ).first()
                    
                    if not existing_doc:
                        doc = Document(
                            candidate_id=candidate.candidate_id,
                            filename=filename,
                            file_path=file_path,
                            document_type=None,
                            status="PENDING",
                            uploaded_at=datetime.utcnow()
                        )
                        db.add(doc)
                        db.commit()
                        logger.info("created_demo_document", document_id=doc.document_id, filename=filename)
                        
    logger.info("demo_data_seeded")
