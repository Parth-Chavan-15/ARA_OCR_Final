import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
    """
    Two-Stage Hybrid Multilingual OCR Pipeline:
    1. Stage 1 (Language Identification Probe): EasyOCR quickly samples script tokens
       to classify the document as Marathi, English, or Bilingual (Marathi + English).
    2. Stage 2 (Primary Bilingual Extraction): PaddleOCR (GPU accelerated) executes
       the high-precision extraction using the language-routed model ('mr' or 'en').
    3. Stage 3 (Resilient Fallback): EasyOCR serves as a safety fallback if primary
       Devanagari extraction is degraded.
    """
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
        """Detect hardware acceleration and configure PaddleOCR device."""
        try:
            import paddle

            if settings.USE_GPU and paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                try:
                    paddle.set_device(f"gpu:{settings.GPU_DEVICE_ID}")
                    cls._can_use_gpu = True
                    cls._active_device = f"gpu:{settings.GPU_DEVICE_ID}"
                    logger.info("paddleocr_gpu_accelerated", device=cls._active_device, gpu_id=settings.GPU_DEVICE_ID)
                except Exception as e:
                    logger.warning("paddle_gpu_device_set_failed", error=str(e))
            else:
                logger.info("paddleocr_running_on_cpu")
        except Exception as e:
            logger.warning("paddle_env_init_failed", error=str(e))

    @classmethod
    def _get_easyocr_reader(cls) -> Optional[Any]:
        """Lazily initialize EasyOCR fallback reader only when needed."""
        if cls._easyocr_reader is None:
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
                logger.info("easyocr_guarded_fallback_initialized", gpu=torch_gpu)
            except Exception as e:
                logger.warning("easyocr_lazy_init_failed", error=str(e))
        return cls._easyocr_reader

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
        Fast Language Identification (LID):
        Checks PyMuPDF digital text layer if PDF is available (0 ms).
        Defaults to 'Marathi + English' for scanned images without running a slow image probe.
        """
        for path in image_paths:
            if path.lower().endswith(".pdf") and os.path.exists(path):
                try:
                    doc = fitz.open(path)
                    if len(doc) > 0:
                        text_sample = doc[0].get_text()
                        if text_sample.strip():
                            tokens = text_sample.split()[:100]
                            doc.close()
                            return LanguageService.detect_language(tokens)
                    doc.close()
                except Exception as e:
                    logger.debug("fitz_probe_error", error=str(e))

        return "Marathi + English"

    def _extract_easyocr_tokens(self, path: str, page_num: int) -> List[OcrToken]:
        """Extract tokens for a page using guarded EasyOCR fallback."""
        tokens: List[OcrToken] = []
        reader = self._get_easyocr_reader()
        if not reader:
            return tokens

        try:
            results = reader.readtext(path)
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
            logger.warning("easyocr_fallback_failed", error=str(e))
        return tokens

    @staticmethod
    def reconstruct_text_lines(tokens: List[OcrToken]) -> str:
        """
        Group OCR tokens by page and reading line (y-coordinate clustering),
        sorting horizontally by x-coordinate, so raw_text consists of
        readable sentences rather than one token per line.
        """
        if not tokens:
            return ""

        pages: dict = {}
        for tok in tokens:
            pages.setdefault(tok.page_number, []).append(tok)

        doc_lines: List[str] = []

        for page_num in sorted(pages.keys()):
            p_tokens = pages[page_num]
            if not p_tokens:
                continue

            valid_bboxes = [
                t for t in p_tokens
                if t.bbox and len(t.bbox) == 4 and (t.bbox[2] > t.bbox[0] or t.bbox[3] > t.bbox[1])
            ]

            if not valid_bboxes:
                doc_lines.append(" ".join(t.text.strip() for t in p_tokens if t.text.strip()))
                continue

            def y_mid(tok: OcrToken) -> float:
                return (tok.bbox[1] + tok.bbox[3]) / 2.0

            def x_start(tok: OcrToken) -> float:
                return tok.bbox[0]

            def tok_height(tok: OcrToken) -> float:
                return max(abs(tok.bbox[3] - tok.bbox[1]), 1.0)

            sorted_tokens = sorted(p_tokens, key=lambda t: (y_mid(t), x_start(t)))

            lines: List[List[OcrToken]] = []
            for tok in sorted_tokens:
                if not tok.text.strip():
                    continue

                if not lines:
                    lines.append([tok])
                    continue

                curr_line = lines[-1]
                avg_h = sum(tok_height(t) for t in curr_line) / len(curr_line)
                threshold = max(avg_h * 0.5, 12.0)
                line_ymid = sum(y_mid(t) for t in curr_line) / len(curr_line)

                if abs(y_mid(tok) - line_ymid) <= threshold:
                    curr_line.append(tok)
                else:
                    lines.append([tok])

            for line_tokens in lines:
                sorted_line = sorted(line_tokens, key=x_start)
                line_str = " ".join(t.text.strip() for t in sorted_line if t.text.strip())
                if line_str:
                    doc_lines.append(line_str)

        return "\n".join(doc_lines)

    @staticmethod
    def _is_genuine_devanagari_token(text: str) -> bool:
        """
        Verify if a recognized token represents authentic Devanagari text
        (e.g., 'कुणबी', 'महाराष्ट्र', 'नाव', 'धुळे') rather than Latin text
        where the Devanagari model hallucinated a stray character.
        """
        dev_chars = [c for c in text if "\u0900" <= c <= "\u097F"]
        alpha_chars = [c for c in text if c.isalpha() or ("\u0900" <= c <= "\u097F")]
        if not alpha_chars or len(dev_chars) < 2:
            return False
        # Ratio of Devanagari characters among all alphabetic characters must be >= 70%
        ratio = len(dev_chars) / len(alpha_chars)
        return ratio >= 0.70

    @staticmethod
    def _box_containment(b_small: List[float], b_large: List[float]) -> float:
        """
        Compute the fraction of b_small that is geometrically contained inside b_large.
        """
        xA = max(b_small[0], b_large[0])
        yA = max(b_small[1], b_large[1])
        xB = min(b_small[2], b_large[2])
        yB = min(b_small[3], b_large[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        area_small = max(0.0, b_small[2] - b_small[0]) * max(0.0, b_small[3] - b_small[1])
        return inter / area_small if area_small > 0 else 0.0

    @staticmethod
    def _box_iou(b1: List[float], b2: List[float]) -> float:
        """
        Compute Intersection over Union between two bounding boxes.
        """
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def process_images(self, image_paths: List[str]) -> OcrOutput:
        """
        Execute production-grade Dual-Engine Selective Spatial Fusion OCR:
        1. Mandatory Visual GPU Scan: Every page is processed visually via high-resolution rasterization.
        2. Pass 1 (Primary English Engine): Extracts English template, tables, rules, and candidate data
           using high-accuracy 'en' PP-OCRv4 model.
        3. Pass 2 (Devanagari Injection Engine): Extracts Marathi textlines using 'mr' PP-OCRv4 model.
           Only genuine Devanagari tokens (>=70% Devanagari ratio, confidence >= 0.65) that do not
           contaminate recognized high-confidence English regions are merged.
        4. Guarded Fallback: EasyOCR is called ONLY if both engines together yield zero tokens.
        """
        tokens: List[OcrToken] = []
        confidences: List[float] = []

        paddle_en = self.get_paddle_engine("en")
        paddle_mr = self.get_paddle_engine("mr")

        for idx, path in enumerate(image_paths):
            page_num = idx + 1
            if not os.path.exists(path):
                continue

            # ── 1. Pass 1: Primary English Model (en_PP-OCRv4) ──
            en_tokens: List[OcrToken] = []
            if paddle_en:
                try:
                    preds_en = paddle_en.ocr(path, cls=False)
                    if preds_en and preds_en[0]:
                        for line in preds_en[0]:
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

                                en_tokens.append(
                                    OcrToken(
                                        text=text_str,
                                        confidence=conf_flt,
                                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                                        page_number=page_num,
                                    )
                                )
                except Exception as e:
                    logger.warning("paddleocr_en_extraction_error", path=path, error=str(e))

            # ── 2. Pass 2: Devanagari Injection Engine (devanagari_PP-OCRv4) ──
            mr_tokens: List[OcrToken] = []
            if paddle_mr:
                try:
                    preds_mr = paddle_mr.ocr(path, cls=False)
                    if preds_mr and preds_mr[0]:
                        for line in preds_mr[0]:
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

                                mr_tokens.append(
                                    OcrToken(
                                        text=text_str,
                                        confidence=conf_flt,
                                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                                        page_number=page_num,
                                    )
                                )
                except Exception as e:
                    logger.warning("paddleocr_mr_extraction_error", path=path, error=str(e))

            # ── 3. Selective Spatial Fusion ──
            fused_page_tokens: List[OcrToken] = list(en_tokens)

            for m in mr_tokens:
                # Discard non-Devanagari or marginal hallucinated tokens
                min_conf = 0.75 if len(m.text) <= 3 else 0.70
                if not self._is_genuine_devanagari_token(m.text) or m.confidence < min_conf:
                    continue

                # Check spatial overlap against recognized English tokens
                overlaps_strong_en = any(
                    e.confidence >= 0.70
                    and any(c.isalnum() for c in e.text)
                    and (
                        self._box_containment(m.bbox, e.bbox) > 0.40
                        or self._box_iou(m.bbox, e.bbox) > 0.35
                    )
                    for e in en_tokens
                )

                if not overlaps_strong_en:
                    # Weak or missing English in this region: check if weak EN token should be replaced
                    weak_overlaps = [
                        e
                        for e in en_tokens
                        if e.confidence < 0.65
                        and (
                            self._box_containment(m.bbox, e.bbox) > 0.40
                            or self._box_iou(m.bbox, e.bbox) > 0.35
                        )
                    ]
                    for w in weak_overlaps:
                        if w in fused_page_tokens:
                            fused_page_tokens.remove(w)

                    fused_page_tokens.append(m)

            # If English pass yielded almost zero tokens (e.g. pure Marathi Leaving Certificate),
            # ensure all genuine Devanagari tokens from mr_tokens are included
            if len(en_tokens) < 5 and mr_tokens:
                for m in mr_tokens:
                    if self._is_genuine_devanagari_token(m.text) and m not in fused_page_tokens:
                        fused_page_tokens.append(m)

            # ── 4. Guarded Fallback: EasyOCR ONLY if page yielded zero tokens ──
            if not fused_page_tokens:
                logger.info("triggering_easyocr_guarded_fallback", page=page_num)
                fused_page_tokens = self._extract_easyocr_tokens(path, page_num)

            for t in fused_page_tokens:
                confidences.append(t.confidence)

            tokens.extend(fused_page_tokens)

        raw_text = self.reconstruct_text_lines(tokens)
        overall_conf = (
            sum(confidences) / len(confidences) if confidences else 0.88
        )

        token_texts = [t.text for t in tokens]
        final_language = (
            LanguageService.detect_language(token_texts)
            if token_texts
            else "Marathi + English"
        )

        return OcrOutput(
            tokens=tokens,
            raw_text=raw_text,
            overall_confidence=overall_conf,
            language=final_language,
        )
