from pathlib import Path
from typing import List, Tuple
import structlog
import fitz  # PyMuPDF
import cv2
import numpy as np

logger = structlog.get_logger(__name__)

class PdfService:
    @staticmethod
    def extract_images(file_path: str, dpi: int = 300) -> List[Tuple[int, np.ndarray]]:
        """
        Extract pages as images from a PDF or return single image for image files.
        Returns: list of (page_number, image_array) tuples (1-indexed pages)
        """
        path = Path(file_path)
        images = []
        
        try:
            if path.suffix.lower() == '.pdf':
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                
                doc = fitz.open(str(path))
                for i in range(len(doc)):
                    page = doc.load_page(i)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Convert to numpy array for OpenCV
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    elif pix.n == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        
                    images.append((i + 1, img))
                doc.close()
                logger.info("pdf_extracted", file_path=str(path), pages=len(images))
            else:
                img = cv2.imread(str(path))
                if img is not None:
                    images.append((1, img))
                else:
                    logger.error("image_read_error", file_path=str(path))
                    
        except Exception as e:
            logger.error("pdf_extraction_error", file_path=str(path), error=str(e))
            
        return images
