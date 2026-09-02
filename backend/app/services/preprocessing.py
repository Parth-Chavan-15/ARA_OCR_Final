import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import structlog
from app.config import settings
import os

logger = structlog.get_logger(__name__)

class PreprocessingService:
    @staticmethod
    def process_images(document_id: str, images: List[Tuple[int, np.ndarray]]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Process images using OpenCV for OCR optimization.
        Returns list of saved processed image paths and quality metrics.
        """
        output_dir = Path(settings.processed_dir) / document_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_paths = []
        metrics = {"blur_scores": {}}
        
        for page_num, img in images:
            try:
                # Basic metrics
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                metrics["blur_scores"][page_num] = laplacian_var
                
                # Resize if too large (standardize)
                max_dim = 2500
                h, w = img.shape[:2]
                if h > max_dim or w > max_dim:
                    scale = max_dim / max(h, w)
                    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Denoising
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                contrast_enhanced = clahe.apply(denoised)

                # Unsharp Mask for sharpening
                gaussian_blur = cv2.GaussianBlur(contrast_enhanced, (0, 0), 2.0)
                sharpened = cv2.addWeighted(contrast_enhanced, 1.5, gaussian_blur, -0.5, 0)
                
                # Save processed image
                out_path = output_dir / f"page_{page_num}.png"
                cv2.imwrite(str(out_path), sharpened)
                processed_paths.append(str(out_path))
                
                logger.info("page_processed", document_id=document_id, page=page_num, blur_score=laplacian_var)
                
            except Exception as e:
                logger.error("preprocessing_error", document_id=document_id, page=page_num, error=str(e))
                
        return processed_paths, metrics
