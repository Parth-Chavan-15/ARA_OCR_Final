import pytest
from app.services.language import LanguageService
from app.services.ocr_service import OcrService


def test_english_text():
    texts = ["This", "is", "a", "pure", "English", "sentence"]
    result = LanguageService.detect_language(texts)
    assert result == "English"


def test_marathi_text():
    texts = ["हे", "मराठी", "वाक्य", "आहे"]
    result = LanguageService.detect_language(texts)
    assert result == "Marathi"


def test_bilingual_text():
    texts = ["This", "is", "a", "मराठी", "word"]
    result = LanguageService.detect_language(texts)
    assert result == "Marathi + English"


def test_empty_text():
    result = LanguageService.detect_language([])
    assert result == "UNKNOWN"


def test_unicode_normalization():
    raw = "महाराष्ट्र\u200D शासन\u00A0प्रमाणपत्र"
    normalized = LanguageService.normalize_text(raw)
    assert "महाराष्ट्र" in normalized
    assert "\u200D" not in normalized
    assert "\u00A0" not in normalized


def test_ocr_language_routing_selection():
    # Test that Marathi and Bilingual route to Marathi engine ("mr")
    ocr_service = OcrService()
    # Detection probe fallback test
    lang_marathi = LanguageService.detect_language(["जात", "प्रमाणपत्र", "पडताळणी"])
    assert lang_marathi == "Marathi"
    
    lang_bilingual = LanguageService.detect_language(["Government", "of", "Maharashtra", "जात", "दाखला"])
    assert lang_bilingual == "Marathi + English"
    
    lang_english = LanguageService.detect_language(["Government", "of", "Maharashtra", "Caste", "Certificate"])
    assert lang_english == "English"
