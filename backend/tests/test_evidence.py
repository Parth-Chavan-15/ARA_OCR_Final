import pytest
from types import SimpleNamespace
from app.services.evidence import EvidenceService


def create_tokens(words):
    return [
        SimpleNamespace(text=w, page_number=1, x1=0, y1=0, x2=10, y2=10, confidence=0.9)
        for w in words
    ]


def test_caste_certificate_evidence():
    tokens = create_tokens(["This", "is", "a", "caste", "certificate"])
    evidence = EvidenceService.generate_evidence("CASTE_CERTIFICATE", tokens)
    assert evidence is not None
    assert isinstance(evidence, dict)
    assert len(evidence.get("matched_keywords", [])) > 0


def test_validity_receipt_evidence():
    tokens = create_tokens(["Payment", "receipt", "transaction", "12345"])
    evidence = EvidenceService.generate_evidence("CASTE_VALIDITY_RECEIPT", tokens)
    assert evidence is not None
    assert isinstance(evidence, dict)
    assert len(evidence.get("matched_keywords", [])) > 0


def test_proforma_evidence():
    tokens = create_tokens(["Proforma-O", "mother", "tongue", "Marathi"])
    evidence = EvidenceService.generate_evidence("PROFORMA_O", tokens)
    assert evidence is not None
    assert isinstance(evidence, dict)
    assert len(evidence.get("matched_keywords", [])) > 0


def test_leaving_certificate_evidence():
    tokens = create_tokens(["School", "leaving", "certificate", "issued"])
    evidence = EvidenceService.generate_evidence("LEAVING_CERTIFICATE", tokens)
    assert evidence is not None
    assert isinstance(evidence, dict)
    assert len(evidence.get("matched_keywords", [])) > 0


def test_unknown_evidence():
    tokens = create_tokens(["Some", "random", "document", "text"])
    evidence = EvidenceService.generate_evidence("UNKNOWN_OUT_OF_SCOPE", tokens)
    assert evidence is not None
    assert isinstance(evidence, dict)
    assert len(evidence.get("matched_keywords", [])) == 0
