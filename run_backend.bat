@echo off
title ARA OCR - FastAPI Backend Server (GPU)
echo ======================================================================
echo Starting ARA OCR FastAPI Backend with GPU Acceleration...
echo Backend will be available at: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo ======================================================================
cd /d "%~dp0\backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
