import csv
import io
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.candidate import Candidate
from app.models.document import Document
from app.models.classification_result import ClassificationResult
from app.services.sync_service import OFFICIAL_INSTITUTE_NAMES

router = APIRouter(prefix="/api/institutes", tags=["Institutes"])

@router.get("/summary")
def get_institutes_summary(db: Session = Depends(get_db)):
    """
    Returns stream-wise and institute-wise rollup metrics for the Institute Scrutiny UI.
    """
    candidates = db.query(Candidate).all()
    documents = db.query(Document).all()
    classifications = {
        c.document_id: c for c in db.query(ClassificationResult).all()
    }

    # Map candidate to documents
    cand_doc_map = {}
    for d in documents:
        if d.candidate_id not in cand_doc_map:
            cand_doc_map[d.candidate_id] = []
        cand_doc_map[d.candidate_id].append(d)

    # Group by stream and institute_code
    grouped = {}
    for c in candidates:
        key = (c.stream or "General Stream", c.institute_code or "00000")
        if key not in grouped:
            inst_name = c.institute_name or OFFICIAL_INSTITUTE_NAMES.get(
                c.institute_code, f"Institute {c.institute_code}"
            )
            grouped[key] = {
                "stream": key[0],
                "institute_code": key[1],
                "institute_name": inst_name,
                "candidates": [],
                "documents": [],
            }
        grouped[key]["candidates"].append(c)
        c_docs = cand_doc_map.get(c.candidate_id, [])
        grouped[key]["documents"].extend(c_docs)

    institute_list = []
    stream_set = set()

    for (stream, inst_code), data in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        stream_set.add(stream)
        c_list = data["candidates"]
        d_list = data["documents"]

        total_docs = len(d_list)
        classified_count = sum(1 for d in d_list if d.document_type)
        pending_count = total_docs - classified_count
        validity_count = sum(1 for d in d_list if d.document_type == "CASTE_VALIDITY_CERTIFICATE")
        unknown_count = sum(1 for d in d_list if d.document_type == "UNKNOWN_OUT_OF_SCOPE")
        error_count = sum(1 for d in d_list if d.status == "ERROR")
        flagged_count = unknown_count + error_count

        completion_rate = round((classified_count / total_docs * 100), 1) if total_docs > 0 else 0.0

        institute_list.append({
            "stream": stream,
            "institute_code": inst_code,
            "institute_name": data["institute_name"],
            "total_candidates": len(c_list),
            "total_documents": total_docs,
            "classified_documents": classified_count,
            "pending_documents": pending_count,
            "validity_count": validity_count,
            "flagged_documents": flagged_count,
            "completion_rate": completion_rate,
        })

    return {
        "status": "success",
        "total_streams": len(stream_set),
        "total_institutes": len(institute_list),
        "total_candidates": len(candidates),
        "total_documents": len(documents),
        "streams": sorted(list(stream_set)),
        "institutes": institute_list,
    }

@router.get("/{institute_code}/candidates")
def get_institute_candidates(
    institute_code: str,
    stream: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Returns all candidates under a specific institute with their submitted documents and extracted fields.
    """
    query = db.query(Candidate).filter(Candidate.institute_code == institute_code)
    if stream:
        query = query.filter(Candidate.stream == stream)

    candidates = query.order_by(Candidate.enrollment_number).all()
    if not candidates:
        # Check if institute exists
        return []

    cand_ids = [c.candidate_id for c in candidates]
    all_docs = db.query(Document).filter(Document.candidate_id.in_(cand_ids)).all()

    # Get latest classification results
    doc_ids = [d.document_id for d in all_docs]
    class_results = {
        cr.document_id: cr for cr in db.query(ClassificationResult).filter(
            ClassificationResult.document_id.in_(doc_ids)
        ).all()
    }

    cand_docs_map = {}
    for d in all_docs:
        if d.candidate_id not in cand_docs_map:
            cand_docs_map[d.candidate_id] = []

        cr = class_results.get(d.document_id)
        cand_docs_map[d.candidate_id].append({
            "document_id": d.document_id,
            "candidate_id": d.candidate_id,
            "filename": d.filename,
            "file_path": d.file_path,
            "document_type": d.document_type,
            "status": d.status,
            "confidence": cr.confidence if cr else None,
            "raw_confidence": cr.raw_confidence if cr else None,
            "confidence_label": cr.confidence_label if cr else None,
            "extracted_fields": cr.extracted_fields if cr else None,
            "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
        })

    response = []
    for c in candidates:
        docs = cand_docs_map.get(c.candidate_id, [])
        verified_count = sum(1 for d in docs if d["document_type"])
        response.append({
            "candidate_id": c.candidate_id,
            "enrollment_number": c.enrollment_number,
            "name": c.name,
            "reserve_category": c.reserve_category or "OBC",
            "stream": c.stream,
            "institute_code": c.institute_code,
            "institute_name": c.institute_name or OFFICIAL_INSTITUTE_NAMES.get(c.institute_code, f"Institute {c.institute_code}"),
            "total_documents": len(docs),
            "verified_documents": verified_count,
            "is_fully_verified": (len(docs) > 0 and verified_count == len(docs)),
            "documents": docs,
        })

    return response

@router.get("/export-csv")
def export_scrutiny_csv(
    stream: Optional[str] = None,
    institute_code: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Generates and downloads a complete Scrutiny CSV export with all extracted fields:
    EN Number, Student Name, Institute, Stream, Category, Document Type, AI Confidence,
    Bearing/Decision No, VC No, Dated, Validity Decision, and Authority.
    """
    query = db.query(Document).join(Candidate)
    if stream:
        query = query.filter(Candidate.stream == stream)
    if institute_code:
        query = query.filter(Candidate.institute_code == institute_code)

    documents = query.order_by(Candidate.stream, Candidate.institute_code, Candidate.enrollment_number).all()

    # Fetch classifications
    doc_ids = [d.document_id for d in documents]
    class_results = {
        cr.document_id: cr for cr in db.query(ClassificationResult).filter(
            ClassificationResult.document_id.in_(doc_ids)
        ).all()
    }

    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)

    headers = [
        "Document ID",
        "Stream / Course",
        "Institute Code",
        "Institute Name",
        "Candidate Enrollment No (EN)",
        "Candidate Name",
        "Reserve Category",
        "Document Type",
        "Scrutiny Status",
        "AI Confidence Score",
        "Bearing / Decision / Application No",
        "Certificate / VC No",
        "Document Date (Dated / Received)",
        "Validity Decision / Status",
        "Caste / Minority Claim",
        "District / Committee / Authority",
        "Original Filename",
        "Uploaded Date",
    ]
    writer.writerow(headers)

    for doc in documents:
        cand = doc.candidate
        cr = class_results.get(doc.document_id)
        ext = cr.extracted_fields if cr and isinstance(cr.extracted_fields, dict) else {}

        # Safe extraction of fields
        decision_no = ext.get("decision_no") or ext.get("bearing_no") or ext.get("application_no") or ext.get("gr_no") or "—"
        cert_no = ext.get("certificate_no") or ext.get("vc_no") or ext.get("receipt_no") or ext.get("serial_no") or "—"
        dated = ext.get("issued_date") or ext.get("received_date") or ext.get("dated") or ext.get("dob") or "—"
        validity = ext.get("validity_decision") or ("VALID" if doc.document_type == "CASTE_VALIDITY_CERTIFICATE" else "PENDING")
        caste_claim = ext.get("caste_claim") or ext.get("community_mother_tongue") or ext.get("caste_religion") or cand.reserve_category or "OBC"
        authority = ext.get("committee") or ext.get("issuing_authority") or ext.get("school_college_name") or ext.get("district") or "Maharashtra Competent Authority"

        conf_str = "N/A"
        if cr:
            conf_str = cr.confidence_label or f"{cr.confidence * 100:.1f}%"
        elif doc.confidence:
            conf_str = f"{doc.confidence * 100:.1f}%"

        writer.writerow([
            doc.document_id,
            cand.stream if cand else "—",
            cand.institute_code if cand else "—",
            cand.institute_name or (OFFICIAL_INSTITUTE_NAMES.get(cand.institute_code, "") if cand else ""),
            cand.enrollment_number if cand else "—",
            cand.name if cand else "—",
            cand.reserve_category or "OBC" if cand else "OBC",
            doc.document_type or "Unclassified (Pending)",
            doc.status,
            conf_str,
            decision_no,
            cert_no,
            dated,
            validity,
            caste_claim,
            authority,
            doc.filename,
            doc.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if doc.uploaded_at else "",
        ])

    csv_data = output.getvalue()
    filename = f"ARA_Document_Scrutiny_Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
