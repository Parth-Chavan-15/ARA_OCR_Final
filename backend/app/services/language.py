import unicodedata
from typing import List
import structlog

logger = structlog.get_logger(__name__)


class LanguageService:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize Unicode text to NFC and strip zero-width characters."""
        if not text:
            return ""
        # Unicode Normalization Form C
        text = unicodedata.normalize("NFC", text)
        # Strip zero-width characters
        for char in ("\u200B", "\u200C", "\u200D", "\uFEFF", "\u00A0"):
            text = text.replace(char, " " if char == "\u00A0" else "")
        return text.strip()

    @staticmethod
    def is_devanagari(char: str) -> bool:
        """Check if character is in Devanagari or Devanagari Extended Unicode block."""
        code = ord(char)
        return (0x0900 <= code <= 0x097F) or (0xA8E0 <= code <= 0xA8FF)

    @staticmethod
    def is_latin(char: str) -> bool:
        """Check if character is standard Latin letter."""
        code = ord(char)
        return (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A)

    @staticmethod
    def detect_language(texts: List[str]) -> str:
        """
        Analyze token texts and determine if the document is English, Marathi,
        or Bilingual (Marathi + English).
        """
        devanagari_count = 0
        latin_count = 0

        for text in texts:
            norm = LanguageService.normalize_text(text)
            for char in norm:
                if LanguageService.is_devanagari(char):
                    devanagari_count += 1
                elif LanguageService.is_latin(char):
                    latin_count += 1

        total = devanagari_count + latin_count

        if total == 0:
            return "UNKNOWN"

        deva_ratio = devanagari_count / total

        if deva_ratio > 0.75:
            return "Marathi"
        elif deva_ratio < 0.15:
            return "English"
        else:
            return "Marathi + English"
