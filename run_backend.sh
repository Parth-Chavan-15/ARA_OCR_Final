#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

echo "======================================================================"
echo "Starting ARA OCR FastAPI Backend..."
echo "Backend URL: http://localhost:8000"
echo "Swagger Docs: http://localhost:8000/docs"
echo "======================================================================"

cd "$SCRIPT_DIR/backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
