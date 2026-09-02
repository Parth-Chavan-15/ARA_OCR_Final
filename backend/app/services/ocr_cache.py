import structlog
from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.ocr_result import OcrResult
from app.models.ocr_token import OcrToken as DbOcrToken
from app.services.ocr_service import OcrService, OcrOutput
from app.services.language import LanguageService

logger = structlog.get_logger(__name__)

class OcrCache:
    @staticmethod
    def get_cached_ocr(db: Session, document_id: str, model_version: str) -> Optional[OcrResult]:
        try:
            cached = db.query(OcrResult).filter(
                OcrResult.document_id == document_id,
                OcrResult.model_version == model_version
            ).first()
            if cached and (len(cached.tokens) > 0 or len(cached.raw_text.strip()) > 0):
                return cached
            return None
        except Exception as e:
            logger.error("cache_read_error", document_id=document_id, error=str(e))
            return None

    @staticmethod
    def get_or_create_ocr(db: Session, document_id: str, model_version: str, image_paths: List[str]) -> Optional[OcrResult]:
        cached = OcrCache.get_cached_ocr(db, document_id, model_version)
        if cached:
            logger.info("ocr_cache_hit", document_id=document_id)
            return cached

        logger.info("ocr_cache_miss", document_id=document_id)
        
        try:
            # Delete any previous empty ocr_result for this doc
            db.query(OcrResult).filter(OcrResult.document_id == document_id).delete()
            db.commit()

            ocr_svc = OcrService()
            output: OcrOutput = ocr_svc.process_images(image_paths)
            
            # Detect language
            lang_label = LanguageService.detect_language([t.text for t in output.tokens])
            output.language = lang_label
            
            # Persist to DB
            import uuid
            ocr_id = str(uuid.uuid4())
            
            db_ocr = OcrResult(
                ocr_id=ocr_id,
                document_id=document_id,
                model_version=model_version,
                language=output.language,
                raw_text=output.raw_text,
                overall_confidence=output.overall_confidence
            )
            db.add(db_ocr)
            
            for token in output.tokens:
                db_token = DbOcrToken(
                    token_id=str(uuid.uuid4()),
                    ocr_id=ocr_id,
                    page_number=token.page_number,
                    text=token.text,
                    confidence=token.confidence,
                    x1=token.bbox[0],
                    y1=token.bbox[1],
                    x2=token.bbox[2],
                    y2=token.bbox[3]
                )
                db.add(db_token)
                
            db.commit()
            db.refresh(db_ocr)
            logger.info("ocr_result_cached", document_id=document_id)
            return db_ocr
            
        except Exception as e:
            logger.error("ocr_process_and_cache_error", document_id=document_id, error=str(e))
            db.rollback()
            return None
