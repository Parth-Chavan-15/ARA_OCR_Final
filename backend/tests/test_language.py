import pytest
from app.services.language import LanguageService

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
