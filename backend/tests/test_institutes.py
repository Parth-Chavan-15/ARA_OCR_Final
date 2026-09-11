import pytest
from app.db.database import SessionLocal
from app.api.institutes import get_institutes_summary, get_institute_candidates, export_scrutiny_csv

def test_institutes_summary():
    db = SessionLocal()
    try:
        summary = get_institutes_summary(db=db)
        assert summary["status"] == "success"
        assert summary["total_streams"] >= 5
        assert summary["total_institutes"] >= 5
        assert summary["total_candidates"] >= 100
        assert summary["total_documents"] >= 100
        assert len(summary["institutes"]) >= 5

        # Check fields in institute summary
        inst = summary["institutes"][0]
        assert "stream" in inst
        assert "institute_code" in inst
        assert "institute_name" in inst
        assert "total_candidates" in inst
        assert "total_documents" in inst
        assert "completion_rate" in inst
    finally:
        db.close()

def test_get_institute_candidates():
    db = SessionLocal()
    try:
        # Test fetching candidates for Grant Medical College (01105)
        candidates = get_institute_candidates("01105", db=db)
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        cand = candidates[0]
        assert "candidate_id" in cand
        assert "enrollment_number" in cand
        assert "name" in cand
        assert "stream" in cand
        assert "institute_code" in cand
        assert "documents" in cand
        assert len(cand["documents"]) > 0
    finally:
        db.close()

def test_export_scrutiny_csv():
    db = SessionLocal()
    try:
        res = export_scrutiny_csv(db=db)
        content = res.body.decode("utf-8")
        lines = content.splitlines()
        assert len(lines) > 50

        # Check required headers
        headers = lines[0]
        assert "Document ID" in headers
        assert "Stream / Course" in headers
        assert "Institute Code" in headers
        assert "Candidate Enrollment No" in headers
        assert "Candidate Name" in headers
        assert "Document Type" in headers
        assert "AI Confidence Score" in headers
        assert "Bearing / Decision / Application No" in headers
        assert "Certificate / VC No" in headers
        assert "Validity Decision / Status" in headers
    finally:
        db.close()
