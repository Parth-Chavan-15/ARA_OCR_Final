import structlog
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.candidate import Candidate
from app.models.document import Document
from app.schemas.candidate import CandidateCreate, CandidateResponse, CandidateWithDocuments
from app.schemas.document import DocumentResponse

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/candidates", tags=["Candidates"])

@router.get("", response_model=List[CandidateResponse])
def get_candidates(
    name: Optional[str] = Query(None, description="Search by name"),
    enrollment_number: Optional[str] = Query(None, description="Search by enrollment number"),
    db: Session = Depends(get_db)
):
    query = db.query(Candidate)
    if name:
        query = query.filter(Candidate.name.ilike(f"%{name}%"))
    if enrollment_number:
        query = query.filter(Candidate.enrollment_number.ilike(f"%{enrollment_number}%"))
    
    return query.all()

@router.get("/{candidate_id}", response_model=CandidateResponse)
def get_candidate(candidate_id: str, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.candidate_id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate

@router.get("/{candidate_id}/documents", response_model=List[DocumentResponse])
def get_candidate_documents(candidate_id: str, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.candidate_id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    return db.query(Document).filter(Document.candidate_id == candidate_id).all()

@router.post("", response_model=CandidateResponse)
def create_candidate(candidate_in: CandidateCreate, db: Session = Depends(get_db)):
    db_candidate = db.query(Candidate).filter(Candidate.enrollment_number == candidate_in.enrollment_number).first()
    if db_candidate:
        raise HTTPException(status_code=400, detail="Candidate with this enrollment number already exists")
    
    new_candidate = Candidate(**candidate_in.model_dump())
    db.add(new_candidate)
    db.commit()
    db.refresh(new_candidate)
    
    logger.info("created_candidate", candidate_id=new_candidate.candidate_id, enrollment=new_candidate.enrollment_number)
    return new_candidate
