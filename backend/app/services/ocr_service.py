import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import fitz
import structlog

from app.config import settings
from app.services.language import LanguageService

logger = structlog.get_logger(__name__)


@dataclass
class OcrToken:
    text: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    page_number: int


@dataclass
class OcrOutput:
    tokens: List[OcrToken]
    raw_text: str
    overall_confidence: float
    language: str


class OcrService:
    _instance: Optional["OcrService"] = None
    _paddle_engines: Dict[str, Any] = {}
    _easyocr_reader: Optional[Any] = None
    _active_device: str = "cpu"
    _can_use_gpu: bool = False

    def __new__(cls) -> "OcrService":
        if cls._instance is None:
            cls._instance = super(OcrService, cls).__new__(cls)
            cls._instance._init_environment()
        return cls._instance

    @classmethod
    def _init_environment(cls) -> None:
        """Detect hardware acceleration and initialize EasyOCR probe reader."""
        try:
            import paddle

            if settings.USE_GPU and paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                try:
                    paddle.set_device(f"gpu:{settings.GPU_DEVICE_ID}")
                    cls._can_use_gpu = True
                    cls._active_device = f"gpu:{settings.GPU_DEVICE_ID}"
                except Exception as e:
                    logger.warning("paddle_gpu_device_set_failed", error=str(e))
        except Exception as e:
            logger.warning("paddle_env_init_failed", error=str(e))

        # Initialize EasyOCR probe & backup engine for Marathi + English
        try:
            import easyocr
            import torch

            torch_gpu = bool(settings.USE_GPU and torch.cuda.is_available())
            try:
                cls._easyocr_reader = easyocr.Reader(
                    ["mr", "en"],
                    gpu=torch_gpu,
                    download_enabled=True,
                )
            except Exception:
                cls._easyocr_reader = easyocr.Reader(
                    ["en"],
                    gpu=torch_gpu,
                    download_enabled=True,
                )
            logger.info("easyocr_detector_initialized", gpu=torch_gpu)
        except Exception as e:
            logger.warning("easyocr_detector_init_failed", error=str(e))

    @classmethod
    def get_paddle_engine(cls, lang: str = "en") -> Optional[Any]:
        """
        Retrieve or lazily initialize the PaddleOCR engine for the specified language.
        Maps language names ('Marathi', 'Marathi + English', 'English') to PaddleOCR lang codes.
        """
        target_lang = "mr" if lang in ("mr", "Marathi", "Marathi + English") else "en"

        if target_lang in cls._paddle_engines:
            return cls._paddle_engines[target_lang]

        try:
            from paddleocr import PaddleOCR

            device_str = "gpu" if cls._can_use_gpu else "cpu"
            engine = None
            try:
                engine = PaddleOCR(
                    lang=target_lang,
                    use_textline_orientation=False,
                    enable_mkldnn=(not cls._can_use_gpu),
                    device=device_str,
                )
            except TypeError:
                engine = PaddleOCR(
                    lang=target_lang,
                    use_angle_cls=False,
                    use_gpu=cls._can_use_gpu,
                )

            cls._paddle_engines[target_lang] = engine
            logger.info(
                "paddleocr_engine_loaded",
                lang=target_lang,
                device=cls._active_device,
                gpu_enabled=cls._can_use_gpu,
            )
            return engine

        except Exception as e:
            logger.warning("paddleocr_engine_load_failed", lang=target_lang, error=str(e))
            if target_lang != "en" and "en" in cls._paddle_engines:
                return cls._paddle_engines["en"]
            return None

    def detect_document_language(self, image_paths: List[str]) -> str:
        """
        Fast language detection probe:
        1. Checks PyMuPDF digital text if PDF is available.
        2. Probes first page image using EasyOCR.
        3. Returns 'Marathi', 'English', or 'Marathi + English'.
        """
        sample_texts: List[str] = []

        # 1. Quick digital text check if path is PDF
        for path in image_paths:
            if path.lower().endswith(".pdf") and os.path.exists(path):
                try:
                    doc = fitz.open(path)
                    if len(doc) > 0:
                        text_sample = doc[0].get_text()
                        if text_sample.strip():
                            sample_texts.extend(text_sample.split()[:100])
                    doc.close()
                    if sample_texts:
                        break
                except Exception as e:
                    logger.debug("fitz_probe_error", error=str(e))

        # 2. EasyOCR quick probe on first page image
        if not sample_texts and self._easyocr_reader and image_paths:
            probe_path = image_paths[0]
            if os.path.exists(probe_path):
                try:
                    results = self._easyocr_reader.readtext(probe_path, detail=0)
                    if results:
                        sample_texts = [str(r) for r in results[:50]]
                except Exception as e:
                    logger.warning("easyocr_probe_error", error=str(e))

        if not sample_texts:
            return "Marathi + English"

        detected_lang = LanguageService.detect_language(sample_texts)
        logger.info("document_language_probed", detected_language=detected_lang)
        return detected_lang

    @staticmethod
    def _is_extraction_degraded(page_tokens: List[OcrToken], probed_lang: str) -> bool:
        """
        Check if primary OCR extraction is missing the expected Devanagari script
        or contains low-confidence garbled tokens.
        """
        if not page_tokens:
            return True

        if probed_lang in ("Marathi", "Marathi + English"):
            devanagari_chars = sum(
                1 for t in page_tokens for c in t.text if LanguageService.is_devanagari(c)
            )
            total_chars = sum(
                1 for t in page_tokens for c in t.text if LanguageService.is_devanagari(c) or LanguageService.is_latin(c)
            )

            # If probed as Marathi/bilingual but extracted tokens have no Devanagari or low ratio
            if total_chars > 0 and (devanagari_chars / total_chars) < 0.20:
                return True

        return False

    def _extract_easyocr_tokens(self, path: str, page_num: int) -> List[OcrToken]:
        """Extract tokens for a page using EasyOCR."""
        tokens: List[OcrToken] = []
        if not self._easyocr_reader:
            return tokens

        try:
            results = self._easyocr_reader.readtext(path)
            for item in results:
                box, raw_text_item, conf = item[0], str(item[1]), float(item[2])
                text_str = LanguageService.normalize_text(raw_text_item)
                if not text_str:
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                tokens.append(
                    OcrToken(
                        text=text_str,
                        confidence=conf,
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        page_number=page_num,
                    )
                )
        except Exception as e:
            logger.warning("easyocr_extraction_error", path=path, error=str(e))

        return tokens

    def process_images(self, image_paths: List[str]) -> OcrOutput:
        """
        Execute bilingual OCR pipeline:
        1. Probe document script / language.
        2. Primary pass: Run language-routed PaddleOCR engine ('mr' or 'en').
        3. Quality verification: If Marathi script is garbled or missing, promote EasyOCR.
        4. Fallback to PyMuPDF if zero tokens extracted.
        """
        tokens: List[OcrToken] = []
        full_text: List[str] = []
        confidences: List[float] = []

        # Step 1: Detect language
        probed_lang = self.detect_document_language(image_paths)

        # Step 2: Route to appropriate PaddleOCR engine
        paddle_engine = self.get_paddle_engine(probed_lang)

        for idx, path in enumerate(image_paths):
            page_num = idx + 1
            if not os.path.exists(path):
                continue

            page_tokens: List[OcrToken] = []

            # 1. Primary: PaddleOCR
            if paddle_engine:
                try:
                    if hasattr(paddle_engine, "predict"):
                        preds = list(paddle_engine.predict(path))
                        if preds and len(preds) > 0:
                            res = preds[0]
                            texts = res.get("rec_texts", [])
                            scores = res.get("rec_scores", [])
                            boxes = res.get("rec_boxes", [])
                            for t, s, b in zip(texts, scores, boxes):
                                text_str = LanguageService.normalize_text(str(t))
                                if not text_str:
                                    continue
                                conf_flt = float(s)
                                if len(b) == 4 and not hasattr(b[0], "__len__"):
                                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                                else:
                                    xs = [p[0] for p in b]
                                    ys = [p[1] for p in b]
                                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                                page_tokens.append(
                                    OcrToken(
                                        text=text_str,
                                        confidence=conf_flt,
                                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                                        page_number=page_num,
                                    )
                                )
                    else:
                        preds = paddle_engine.ocr(path, cls=False)
                        if preds:
                            for page in preds:
                                if not page:
                                    continue
                                for line in page:
                                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                                        poly = line[0]
                                        txt_tuple = line[1]
                                        raw_t = (
                                            txt_tuple[0]
                                            if isinstance(txt_tuple, (list, tuple))
                                            else txt_tuple
                                        )
                                        text_str = LanguageService.normalize_text(str(raw_t))
                                        conf_flt = float(
                                            txt_tuple[1]
                                            if isinstance(txt_tuple, (list, tuple))
                                            and len(txt_tuple) > 1
                                            else 0.9
                                        )
                                        if not text_str:
                                            continue
                                        xs = [p[0] for p in poly]
                                        ys = [p[1] for p in poly]
                                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                                        page_tokens.append(
                                            OcrToken(
                                                text=text_str,
                                                confidence=conf_flt,
                                                bbox=[
                                                    float(x1),
                                                    float(y1),
                                                    float(x2),
                                                    float(y2),
                                                ],
                                                page_number=page_num,
                                            )
                                        )
                except Exception as e:
                    logger.warning("paddleocr_extraction_error", path=path, error=str(e))

            # 2. Quality Verification: Fall back to EasyOCR if Paddle tokens are empty or degraded
            if self._is_extraction_degraded(page_tokens, probed_lang):
                easy_tokens = self._extract_easyocr_tokens(path, page_num)
                if easy_tokens:
                    logger.info("promoted_easyocr_for_degraded_page", page=page_num, tokens_count=len(easy_tokens))
                    page_tokens = easy_tokens

            # 3. Fallback: PyMuPDF word stream if still empty
            if not page_tokens and path.lower().endswith(".pdf"):
                try:
                    doc = fitz.open(path)
                    for p_idx in range(len(doc)):
                        page = doc[p_idx]
                        words = page.get_text("words")
                        for item in words:
                            word_text = LanguageService.normalize_text(str(item[4]))
                            if word_text:
                                page_tokens.append(
                                    OcrToken(
                                        text=word_text,
                                        confidence=0.95,
                                        bbox=[
                                            float(item[0]),
                                            float(item[1]),
                                            float(item[2]),
                                            float(item[3]),
                                        ],
                                        page_number=p_idx + 1,
                                    )
                                )
                    doc.close()
                except Exception as e:
                    logger.warning("fitz_extract_error", error=str(e))

            for t in page_tokens:
                full_text.append(t.text)
                confidences.append(t.confidence)

            tokens.extend(page_tokens)

        raw_text = "\n".join(full_text)
        overall_conf = (
            sum(confidences) / len(confidences) if confidences else 0.88
        )

        token_texts = [t.text for t in tokens]
        final_language = (
            LanguageService.detect_language(token_texts)
            if token_texts
            else probed_lang
        )

        return OcrOutput(
            tokens=tokens,
            raw_text=raw_text,
            overall_confidence=overall_conf,
            language=final_language,
        )
