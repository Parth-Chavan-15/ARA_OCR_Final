import os
import mimetypes
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import structlog
import cv2
import fitz  # PyMuPDF
from app.config import settings

logger = structlog.get_logger(__name__)

@dataclass
class FileValidationResult:
    status: str
    reason: Optional[str] = None

class FileHandler:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf'}
    ALLOWED_MIMETYPES = {'image/jpeg', 'image/png', 'application/pdf'}
    
    @staticmethod
    def validate_file(file_path: str) -> FileValidationResult:
        try:
            path = Path(file_path)
            
            if not path.exists():
                return FileValidationResult(status="INVALID", reason="File does not exist")
                
            if path.stat().st_size == 0:
                return FileValidationResult(status="EMPTY", reason="File is empty")
                
            if path.stat().st_size > settings.max_file_size_bytes:
                return FileValidationResult(status="INVALID", reason=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB}MB")

            ext = path.suffix.lower()
            if ext not in FileHandler.ALLOWED_EXTENSIONS:
                return FileValidationResult(status="INVALID", reason=f"Extension {ext} not allowed")

            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type not in FileHandler.ALLOWED_MIMETYPES:
                # Fallback to checking magic bytes or letting the specific processor fail
                pass
                
            if not os.access(path, os.R_OK):
                return FileValidationResult(status="UNREADABLE", reason="File is not readable")

            if ext == '.pdf':
                return FileHandler._validate_pdf(path)
            elif ext in {'.jpg', '.jpeg', '.png'}:
                return FileHandler._validate_image(path)
                
            return FileValidationResult(status="VALID", reason="Passed initial validation")
            
        except Exception as e:
            logger.error("file_validation_error", file_path=file_path, error=str(e))
            return FileValidationResult(status="UNREADABLE", reason=f"Error validating file: {str(e)}")
            
    @staticmethod
    def _validate_pdf(path: Path) -> FileValidationResult:
        try:
            doc = fitz.open(str(path))
            if doc.page_count == 0:
                return FileValidationResult(status="INVALID", reason="PDF has no pages")
            doc.close()
            return FileValidationResult(status="VALID")
        except Exception as e:
            logger.error("pdf_validation_error", file_path=str(path), error=str(e))
            return FileValidationResult(status="UNREADABLE", reason="Corrupt or invalid PDF file")
            
    @staticmethod
    def _validate_image(path: Path) -> FileValidationResult:
        try:
            img = cv2.imread(str(path))
            if img is None:
                return FileValidationResult(status="UNREADABLE", reason="Cannot decode image")
            return FileValidationResult(status="VALID")
        except Exception as e:
            logger.error("image_validation_error", file_path=str(path), error=str(e))
            return FileValidationResult(status="UNREADABLE", reason="Corrupt or invalid image file")
