import os
import structlog
from typing import List
from dataclasses import dataclass
import fitz

from app.config import settings

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
    _instance = None
    _paddle_engine = None
    _easyocr_reader = None
    _active_device: str = "cpu"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OcrService, cls).__new__(cls)
            cls._instance._init_engines()
        return cls._instance

    @classmethod
    def _init_engines(cls):
        # 1. Initialize PaddleOCR engine with GPU/CPU auto-detection
        try:
            import paddle
            from paddleocr import PaddleOCR

            can_use_gpu = False
            if settings.USE_GPU:
                if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                    try:
                        paddle.set_device(f"gpu:{settings.GPU_DEVICE_ID}")
                        can_use_gpu = True
                        cls._active_device = f"gpu:{settings.GPU_DEVICE_ID}"
                    except Exception as e:
                        logger.warning("paddle_gpu_device_set_failed", error=str(e))

            try:
                device_str = "gpu" if can_use_gpu else "cpu"
                cls._paddle_engine = PaddleOCR(
                    lang="en",
                    use_textline_orientation=False,
                    enable_mkldnn=(not can_use_gpu),
                    device=device_str,
                )
            except TypeError:
                cls._paddle_engine = PaddleOCR(
                    lang="en",
                    use_angle_cls=False,
                    use_gpu=can_use_gpu,
                )

            logger.info(
                "paddleocr_initialized",
                engine="PaddleOCR (PP-OCRv4)",
                device=cls._active_device,
                gpu_enabled=can_use_gpu,
            )
        except Exception as e:
            logger.warning("paddleocr_init_failed", error=str(e))

        # 2. Initialize EasyOCR backup engine
        try:
            import easyocr
            import torch

            torch_gpu = bool(settings.USE_GPU and torch.cuda.is_available())
            cls._easyocr_reader = easyocr.Reader(
                ["en"],
                gpu=torch_gpu,
                download_enabled=False,
            )
            logger.info("easyocr_initialized", gpu=torch_gpu)
        except Exception as e:
            logger.warning("easyocr_init_failed", error=str(e))

    def process_images(self, image_paths: List[str]) -> OcrOutput:
        tokens: List[OcrToken] = []
        full_text: List[str] = []
        confidences: List[float] = []

        for idx, path in enumerate(image_paths):
            page_num = idx + 1
            if not os.path.exists(path):
                continue

            # 1. Primary: PaddleOCR
            if self._paddle_engine:
                try:
                    if hasattr(self._paddle_engine, "predict"):
                        preds = list(self._paddle_engine.predict(path))
                        if preds and len(preds) > 0:
                            res = preds[0]
                            texts = res.get('rec_texts', [])
                            scores = res.get('rec_scores', [])
                            boxes = res.get('rec_boxes', [])
                            for t, s, b in zip(texts, scores, boxes):
                                text_str = str(t).strip()
                                if not text_str:
                                    continue
                                conf_flt = float(s)
                                if len(b) == 4 and not hasattr(b[0], '__len__'):
                                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                                else:
                                    xs = [p[0] for p in b]
                                    ys = [p[1] for p in b]
                                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                                tokens.append(OcrToken(
                                    text=text_str,
                                    confidence=conf_flt,
                                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                                    page_number=page_num
                                ))
                                full_text.append(text_str)
                                confidences.append(conf_flt)
                    else:
                        preds = self._paddle_engine.ocr(path, cls=False)
                        if preds:
                            for page in preds:
                                if not page:
                                    continue
                                for line in page:
                                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                                        poly = line[0]
                                        txt_tuple = line[1]
                                        text_str = str(txt_tuple[0] if isinstance(txt_tuple, (list, tuple)) else txt_tuple).strip()
                                        conf_flt = float(txt_tuple[1] if isinstance(txt_tuple, (list, tuple)) and len(txt_tuple) > 1 else 0.9)
                                        if not text_str:
                                            continue
                                        xs = [p[0] for p in poly]
                                        ys = [p[1] for p in poly]
                                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                                        tokens.append(OcrToken(
                                            text=text_str,
                                            confidence=conf_flt,
                                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                                            page_number=page_num
                                        ))
                                        full_text.append(text_str)
                                        confidences.append(conf_flt)
                except Exception as e:
                    logger.warning("paddleocr_extraction_error", error=str(e))

            # 2. Secondary: EasyOCR if 0 tokens from PaddleOCR
            if not tokens and self._easyocr_reader:
                try:
                    results = self._easyocr_reader.readtext(path)
                    for item in results:
                        box, text, conf = item[0], str(item[1]).strip(), float(item[2])
                        if not text:
                            continue
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                        tokens.append(OcrToken(
                            text=text,
                            confidence=conf,
                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                            page_number=page_num
                        ))
                        full_text.append(text)
                        confidences.append(conf)
                except Exception as e:
                    logger.warning("easyocr_error", error=str(e))

            # 3. Fallback: PyMuPDF word stream
            if not tokens and path.lower().endswith('.pdf'):
                try:
                    doc = fitz.open(path)
                    for p_idx in range(len(doc)):
                        page = doc[p_idx]
                        words = page.get_text("words")
                        for item in words:
                            word_text = item[4].strip()
                            if word_text:
                                tokens.append(OcrToken(
                                    text=word_text,
                                    confidence=0.95,
                                    bbox=[float(item[0]), float(item[1]), float(item[2]), float(item[3])],
                                    page_number=p_idx + 1
                                ))
                                full_text.append(word_text)
                                confidences.append(0.95)
                    doc.close()
                except Exception as e:
                    logger.warning("fitz_extract_error", error=str(e))

        raw_text = "\n".join(full_text)
        overall_conf = (
            sum(confidences) / len(confidences) if confidences else 0.88
        )

        return OcrOutput(
            tokens=tokens,
            raw_text=raw_text,
            overall_confidence=overall_conf,
            language="en"
        )

