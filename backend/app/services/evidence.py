import structlog
from typing import List, Dict, Any

logger = structlog.get_logger(__name__)

class EvidenceService:
    INDICATORS = {
        "CASTE_CERTIFICATE": [
            'caste certificate', 'जातीचा दाखला', 'form 6', 'form 7', 'form 8',
            'tehsildar', 'sub-divisional officer', 'caste', 'sub-caste', 'tribe'
        ],
        "CASTE_VALIDITY_CERTIFICATE": [
            'caste validity', 'scrutiny committee', 'जात वैधता', 'validity certificate',
            'district caste certificate committee', 'validity', 'scrutiny'
        ],
        "CASTE_VALIDITY_RECEIPT": [
            'receipt', 'payment receipt', 'transaction', 'application number',
            'receipt no', 'online payment', 'receipt no.'
        ],
        "PROFORMA_O": [
            'proforma', 'proforma-o', 'mother tongue', 'linguistic minority',
            'प्रपत्र', 'प्रपत्र-ओ'
        ],
        "LEAVING_CERTIFICATE": [
            'leaving certificate', 'school leaving', 'transfer certificate',
            'date of birth', 'admission', 'शाळा सोडल्याचा दाखला', 'leaving',
            'transfer', 'conduct'
        ]
    }

    @staticmethod
    def generate_evidence(predicted_class: str, ocr_tokens: List[Any]) -> Dict[str, Any]:
        if predicted_class == "UNKNOWN_OUT_OF_SCOPE" or predicted_class not in EvidenceService.INDICATORS:
            return {
                "matched_keywords": [],
                "evidence_items": [],
                "summary": "No specific evidence found for UNKNOWN_OUT_OF_SCOPE class."
            }

        keywords = EvidenceService.INDICATORS[predicted_class]
        matched = set()
        evidence_items = []

        try:
            full_text = " ".join(getattr(token, "text", "") for token in ocr_tokens).lower()

            # First, check multi-word phrase occurrences in full concatenated text
            for kw in keywords:
                if kw in full_text:
                    matched.add(kw)

            # Next, match against individual tokens for bounding boxes
            for token in ocr_tokens:
                text = getattr(token, "text", "")
                text_lower = text.lower()
                for kw in keywords:
                    if kw == text_lower or kw in text_lower or text_lower in kw:
                        matched.add(kw)
                        evidence_items.append({
                            "text": text,
                            "page_number": getattr(token, "page_number", 1),
                            "bbox": [
                                getattr(token, "x1", 0.0),
                                getattr(token, "y1", 0.0),
                                getattr(token, "x2", 0.0),
                                getattr(token, "y2", 0.0)
                            ]
                        })

            summary = f"Found {len(matched)} matching keywords for class {predicted_class}."
            return {
                "matched_keywords": sorted(list(matched)),
                "evidence_items": evidence_items,
                "summary": summary
            }
        except Exception as e:
            logger.error("evidence_generation_error", error=str(e))
            return {
                "matched_keywords": [],
                "evidence_items": [],
                "summary": "Error generating evidence."
            }
