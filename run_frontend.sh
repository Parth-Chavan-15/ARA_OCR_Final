#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/frontend"

echo "======================================================================"
echo "Starting ARA OCR React Frontend..."
echo "Frontend URL: http://localhost:5173"
echo "======================================================================"

npm run dev
