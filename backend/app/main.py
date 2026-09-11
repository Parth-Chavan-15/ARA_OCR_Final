import os
import json
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.database import SessionLocal
from app.db.init_db import init_database, seed_demo_data
from app.api.candidates import router as candidates_router
from app.api.documents import router as documents_router
from app.api.classification import router as classification_router
from app.api.institutes import router as institutes_router
from app.api.ws import router as ws_router

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="ARA OCR Document Classification System",
    version=settings.PIPELINE_VERSION,
    description="Backend API layer for ARA OCR system"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list if hasattr(settings, 'cors_origins_list') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(candidates_router)
app.include_router(classification_router)
app.include_router(documents_router)
app.include_router(institutes_router)
app.include_router(ws_router)

@app.on_event("startup")
async def on_startup():
    logger.info("starting_application", version=settings.PIPELINE_VERSION)
    
    try:
        import asyncio
        from app.api.ws import set_main_loop
        set_main_loop(asyncio.get_running_loop())
    except Exception as e:
        logger.warning("failed_to_set_ws_main_loop", error=str(e))
    
    if hasattr(settings, 'ensure_directories'):
        settings.ensure_directories()
    
    init_database()
    
    db = SessionLocal()
    try:
        seed_demo_data(db)
    finally:
        db.close()
        
    originals_dir = os.path.join(settings.DATA_DIR, "originals")
    if os.path.exists(originals_dir):
        app.mount("/static/originals", StaticFiles(directory=originals_dir), name="originals")
        
    processed_dir = os.path.join(settings.DATA_DIR, "processed")
    if os.path.exists(processed_dir):
        app.mount("/static/processed", StaticFiles(directory=processed_dir), name="processed")
        

@app.get("/api/health", tags=["Health"])
def health_check():
    import time
    from sqlalchemy import text
    db_status = "disconnected"
    latency_ms = None
    start_t = time.time()
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_status = "connected"
        latency_ms = round((time.time() - start_t) * 1000, 1)
    except Exception as e:
        logger.warning("health_db_check_failed", error=str(e))

    return {
        "status": "ok" if db_status == "connected" else "degraded",
        "database": db_status,
        "database_type": "PostgreSQL",
        "latency_ms": latency_ms,
        "version": settings.PIPELINE_VERSION
    }


@app.get("/api/model-info", tags=["Model"])
def get_model_info():
    report_path = Path("training_report.json")
    if not report_path.exists():
        report_path = Path(__file__).resolve().parent.parent.parent / "training_report.json"

    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            return {
                "status": "success",
                "architecture": "LayoutLMv3 (Champion Run 4)",
                "ocr_engine": "PaddleOCR + EasyOCR (GPU)",
                "best_run_id": report.get("best_run_id", 4),
                "best_run_name": report.get("best_run_name", "Extended Cosine Schedule (LR 3.5e-5)"),
                "accuracy": report.get("best_accuracy", 88.27),
                "weighted_f1": report.get("best_weighted_f1", 87.90),
                "macro_f1": report.get("best_macro_f1", 67.75),
                "ood_f1": report.get("ood_f1", 84.71),
                "composite_score": report.get("best_composite_score", 81.85),
                "total_dataset_size": report.get("total_dataset_size", 1138),
                "train_samples": report.get("train_samples", 797),
                "test_samples": report.get("test_samples", 341),
                "benchmark_timestamp": report.get("benchmark_timestamp", "")
            }
        except Exception as e:
            logger.warning("failed_to_read_training_report", error=str(e))

    return {
        "status": "fallback",
        "architecture": "LayoutLMv3 (Champion Run 4)",
        "ocr_engine": "PaddleOCR + EasyOCR (GPU)",
        "best_run_id": 4,
        "best_run_name": "Extended Cosine Schedule (LR 3.5e-5)",
        "accuracy": 88.27,
        "weighted_f1": 87.90,
        "macro_f1": 67.75,
        "ood_f1": 84.71,
        "composite_score": 81.85,
        "total_dataset_size": 1138,
        "train_samples": 797,
        "test_samples": 341,
        "benchmark_timestamp": "2026-09-10"
    }
