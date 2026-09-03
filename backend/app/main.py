import os
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

@app.on_event("startup")
def on_startup():
    logger.info("starting_application", version=settings.PIPELINE_VERSION)
    
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
    return {
        "status": "ok",
        "version": settings.PIPELINE_VERSION
    }
