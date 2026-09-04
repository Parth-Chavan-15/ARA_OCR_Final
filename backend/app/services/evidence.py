import structlog
from typing import List, Dict, Any
from app.services.language import LanguageService

logger = structlog.get_logger(__name__)


class EvidenceService:
    INDICATORS = {
        "CASTE_CERTIFICATE": [
            "caste certificate", "जातीचा दाखला", "जातीचे प्रमाणपत्र", "form 6", "form 7", "form 8",
            "form-6", "form-7", "form-8", "tehsildar", "tahsildar", "sub-divisional officer",
            "certificate to be produced by", "produced by other backward classes",
            "produced by scheduled castes", "सक्षम प्राधिकारी", "महाराष्ट्र शासन",
            "प्रपत्र ६", "प्रपत्र ७", "प्रपत्र ८", "उपविभागीय अधिकारी", "मागास प्रवर्ग"
        ],
        "CASTE_VALIDITY_CERTIFICATE": [
            "certificate of validity", "validity certificate", "जात वैधता", "जात वैधता प्रमाणपत्र",
            "form 15", "form-15", "प्रपत्र १५", "प्रपत्र 15", "claim is held valid", "claim is valid",
            "district caste certificate scrutiny committee", "scrutiny committee",
            "जात प्रमाणपत्र पडताळणी समिती", "पडताळणी समिती", "वैधता प्रमाणपत्र"
        ],
        "CASTE_VALIDITY_RECEIPT": [
            "receipt", "acknowledgement receipt", "application with supporting documents receipt",
            "receipt no", "receipt no.", "transaction ref", "online submitted application",
            "पावती", "पोचपावती", "अर्ज पावती", "कागदपत्रे सादर केल्याची पावती",
            "पडताळणी समिती पावती", "अर्ज क्रमांक"
        ],
        "PROFORMA_O": [
            "proforma-o", "proforma -o", "proforma o", "proforma-0", "proforma 0",
            "प्रपत्र-ओ", "प्रपत्र ओ", "minority community student", "self declaration for minority",
            "अल्पसंख्याक समुदाय विद्यार्थी", "स्वयंघोषणापत्र", "धार्मिक / भाषिक अल्पसंख्याक"
        ],
        "LEAVING_CERTIFICATE": [
            "leaving certificate", "school leaving", "college leaving", "transfer certificate",
            "lc no", "lc no.", "general register", "name of the pupil",
            "शाळा सोडल्याचा दाखला", "महाविद्यालय सोडल्याचा दाखला", "स्थानांतरण प्रमाणपत्र",
            "विद्यार्थ्याचे नाव", "जनरल रजिस्टर"
        ],
        "UNKNOWN_OUT_OF_SCOPE": [
            "non creamy layer", "non-creamy layer", "ncl certificate", "ncl no", "income certificate",
            "domicile certificate", "certificate of nationality", "admission form", "application form",
            "state common entrance test cell", "cap round", "cap allotment",
            "नॉन क्रिमीलेअर", "उत्पन्नाचा दाखला", "अधिवास प्रमाणपत्र", "राष्ट्रीयत्व प्रमाणपत्र"
        ]
    }

    @staticmethod
    def generate_evidence(predicted_class: str, ocr_tokens: List[Any]) -> Dict[str, Any]:
        if predicted_class not in EvidenceService.INDICATORS:
            return {
                "matched_keywords": [],
                "evidence_items": [],
                "summary": f"No statutory criteria defined for class {predicted_class}."
            }

        keywords = [LanguageService.normalize_text(kw).lower() for kw in EvidenceService.INDICATORS[predicted_class]]
        matched = set()
        evidence_items = []

        try:
            token_strings = [
                LanguageService.normalize_text(getattr(token, "text", "")).lower()
                for token in ocr_tokens
            ]
            full_text = " ".join(token_strings)

            # 1. Multi-word phrase occurrences in full concatenated text
            for kw in keywords:
                if kw in full_text:
                    matched.add(kw)

            # 2. Match against individual tokens for bounding boxes
            for token in ocr_tokens:
                text_raw = getattr(token, "text", "")
                text_norm = LanguageService.normalize_text(text_raw).lower()
                if not text_norm:
                    continue

                for kw in keywords:
                    if kw == text_norm or kw in text_norm or (len(text_norm) >= 4 and text_norm in kw):
                        matched.add(kw)
                        evidence_items.append({
                            "text": text_raw,
                            "page_number": getattr(token, "page_number", 1),
                            "bbox": [
                                getattr(token, "x1", 0.0),
                                getattr(token, "y1", 0.0),
                                getattr(token, "x2", 0.0),
                                getattr(token, "y2", 0.0)
                            ]
                        })
                        break

            if predicted_class == "UNKNOWN_OUT_OF_SCOPE":
                if matched:
                    summary = f"Flagged as Out-of-Scope: Identified non-target statutory markers: {', '.join(sorted(list(matched))[:3])}."
                else:
                    summary = "Flagged as Out-of-Scope: No authorized Maharashtra State CET reservation certificate criteria met."
            else:
                summary = f"Verified {len(matched)} statutory criteria for {predicted_class}."
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
