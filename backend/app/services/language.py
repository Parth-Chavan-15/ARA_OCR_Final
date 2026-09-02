import structlog
from typing import List

logger = structlog.get_logger(__name__)

class LanguageService:
    @staticmethod
    def detect_language(texts: List[str]) -> str:
        devanagari_count = 0
        latin_count = 0
        
        for text in texts:
            for char in text:
                code = ord(char)
                if 0x0900 <= code <= 0x097F:
                    devanagari_count += 1
                elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
                    latin_count += 1
                    
        total = devanagari_count + latin_count
        
        if total == 0:
            return "UNKNOWN"
            
        deva_ratio = devanagari_count / total
        
        if deva_ratio > 0.8:
            return "Marathi"
        elif deva_ratio < 0.2:
            return "English"
        else:
            return "Marathi + English"
