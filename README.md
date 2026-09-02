# ARA OCR — Document Classification System

> **Admissions Regulating Authority (ARA)** — Local Bilingual OCR + 5-Way Broad Document Classification Prototype

A local, GUI-based document processing system that performs bilingual (English + Marathi) OCR on government documents and classifies them into 5 broad categories using a LayoutXLM multimodal classifier.

---

## Architecture

```
LOCAL DOCUMENT STORE
        ↓
FETCH DOCUMENT (on Start Classification)
        ↓
FILE VALIDATION (extension, MIME, size, integrity)
        ↓
PDF → PAGE IMAGES (PyMuPDF, 300 DPI)
        ↓
OpenCV PREPROCESSING (denoise, CLAHE, sharpen, deskew)
        ↓
PaddleOCR (English + Marathi/Devanagari)
        ↓
TEXT + TOKENS + BBOX + CONFIDENCE + PAGE
        ↓
PERSIST OCR CACHE (PostgreSQL) ← OCR ONCE, REUSE EVERYWHERE
        ↓
LANGUAGE / SCRIPT DETECTION (English / Marathi / Bilingual)
        ↓
LayoutXLM CLASSIFICATION (text + image + layout)
        ↓
5-WAY BROAD CLASSIFICATION + UNKNOWN
        ↓
CONFIDENCE / SAFE REJECTION
        ↓
EVIDENCE GENERATION
        ↓
PERSIST CLASSIFICATION RESULT → GUI
```

### Supported Document Classes

| # | Class | Description |
|---|-------|-------------|
| 1 | `CASTE_CERTIFICATE` | Caste Certificate (Form 6, 7, 8 variants) |
| 2 | `CASTE_VALIDITY_CERTIFICATE` | Certificate of Validity of Caste Certificate |
| 3 | `CASTE_VALIDITY_RECEIPT` | Online payment receipt for validity application |
| 4 | `PROFORMA_O` | Linguistic minority certificate (Proforma-O) |
| 5 | `LEAVING_CERTIFICATE` | School/College Leaving Certificate |
| 6 | `UNKNOWN_OUT_OF_SCOPE` | Any unsupported or unrecognized document |

---

## Prerequisites

- **Python** 3.10+ (tested with 3.11)
- **Node.js** 18+ and npm
- **PostgreSQL** 15+ (running on localhost:5432)
- **CUDA** (optional but recommended — RTX 4050 or similar)

---

## Installation

### 1. Clone and navigate to the project

```bash
cd ARA_OCR_Final
```

### 2. Create PostgreSQL database

Open pgAdmin or use the command line:

```sql
CREATE DATABASE ara_ocr;
```

### 3. Set up environment

```bash
copy .env.example .env
```

Edit `.env` to set your PostgreSQL password:

```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/ara_ocr
```

### 4. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

**For GPU support (recommended with RTX 4050):**

```bash
# Install CUDA-enabled PaddlePaddle (check your CUDA version first)
pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install frontend dependencies

```bash
cd frontend
npm install
```

### 6. Generate demo data

```bash
cd ..
python scripts/create_demo_data.py
```

This creates synthetic document images in `data/originals/` and `data/demo/`.

---

## Database Setup

The database schema is created automatically when the backend starts. Tables:

- `candidates` — Candidate records (enrollment_number, name, category)
- `documents` — Documents linked to candidates (file_path, status, type)
- `ocr_results` — Cached OCR output (model_version, language, raw_text, confidence)
- `ocr_tokens` — Individual OCR words (text, bbox, confidence, page_number)
- `classification_results` — Classification output (predicted_class, confidence, evidence)

---

## Model Setup

### PaddleOCR

PaddleOCR models are downloaded automatically on first use. Supports English + Marathi/Devanagari.

### LayoutXLM

The base model (`microsoft/layoutxlm-base`) is downloaded from Hugging Face on first use (~1.2 GB).

**Fine-tuning** (after providing training data):

```bash
# Place labeled training images in training_data/{CLASS_NAME}/
python scripts/train_classifier.py --data_dir training_data --epochs 10

# Fine-tuned model is saved to models/layoutxlm/
```

---

## Running

### Start the backend

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start the frontend (in a new terminal)

```bash
cd frontend
npm run dev
```

### Access the application

Open http://localhost:5173 in your browser.

---

## Demo Walkthrough

1. **Start the application** — backend on port 8000, frontend on port 5173
2. **Open Dashboard** — see aggregate statistics (total documents, classified, pending)
3. **Navigate to "Incoming Documents"** — see documents from all candidates
4. **Select a document** and click **"Start Classification"**
5. **Observe the pipeline** — watch each stage complete:
   - File Validation ✓
   - PDF Rendering ✓
   - Preprocessing ✓
   - OCR ✓
   - OCR Cache ✓
   - Language Detection ✓
   - LayoutXLM Classification ✓
   - Completed ✓
6. **Open the Result** — inspect:
   - Original document preview
   - OCR text (bilingual)
   - Detected language
   - Classification with confidence
   - Evidence (matched keywords)
7. **Navigate to "Review Queue"** — see low-confidence and UNKNOWN documents

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/dashboard/stats` | Dashboard statistics |
| GET | `/api/candidates` | List all candidates |
| GET | `/api/candidates/{id}/documents` | Candidate's documents |
| POST | `/api/candidates` | Create candidate |
| GET | `/api/documents` | List all documents |
| GET | `/api/documents/{id}` | Get document details |
| GET | `/api/documents/{id}/status` | Processing status |
| GET | `/api/documents/{id}/ocr` | OCR result with tokens |
| GET | `/api/documents/{id}/classification` | Classification result |
| POST | `/api/documents/{id}/classify` | **Run classification pipeline** |
| POST | `/api/documents/upload` | Upload a document |
| GET | `/api/documents/{id}/image` | Original document preview |
| GET | `/api/documents/{id}/processed-image` | Preprocessed image |
| GET | `/api/review` | Documents needing review |

---

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

---

## Project Structure

```
ARA_OCR_Final/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point
│   │   ├── config.py                # Environment-based config
│   │   ├── api/
│   │   │   ├── candidates.py        # Candidate endpoints
│   │   │   ├── documents.py         # Document endpoints
│   │   │   └── classification.py    # Classification pipeline + dashboard
│   │   ├── services/
│   │   │   ├── file_handler.py      # File validation
│   │   │   ├── pdf_service.py       # PDF → images (PyMuPDF)
│   │   │   ├── preprocessing.py     # OpenCV preprocessing
│   │   │   ├── ocr_service.py       # PaddleOCR (English + Marathi)
│   │   │   ├── ocr_cache.py         # OCR cache management
│   │   │   ├── language.py          # Script/language detection
│   │   │   ├── classifier.py        # LayoutXLM broad classification
│   │   │   └── evidence.py          # Evidence generation
│   │   ├── db/
│   │   │   ├── database.py          # SQLAlchemy engine
│   │   │   └── init_db.py           # Schema + seed data
│   │   ├── models/                  # SQLAlchemy ORM models
│   │   └── schemas/                 # Pydantic schemas
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/                   # Dashboard, Incoming, Result, Review
│   │   ├── components/              # Layout, Sidebar, StatusBadge
│   │   └── services/api.js          # Axios API client
│   └── package.json
├── data/
│   ├── originals/                   # Original documents (never modified)
│   ├── processed/                   # Preprocessed images
│   └── demo/                        # Synthetic demo documents
├── models/
│   └── layoutxlm/                   # Fine-tuned model weights
├── scripts/
│   ├── create_demo_data.py          # Generate synthetic documents
│   └── train_classifier.py          # Fine-tune LayoutXLM
├── .env.example
└── README.md
```

---

## Key Design Decisions

### OCR Once — Reuse Everywhere
OCR is run once per document and cached to PostgreSQL. Classification and all future downstream modules read from the cache. No second OCR pass is performed.

### Safe Failure
Every pipeline stage fails gracefully. A corrupt file produces an error status, not a crash.

### Confidence Threshold
Documents below the classification confidence threshold are automatically classified as `UNKNOWN_OUT_OF_SCOPE` rather than forcing a prediction.

### Model Versioning
Every OCR result and classification stores the model version. A model upgrade produces new results rather than silently overwriting history.

### Deferred Scope
The architecture is ready for future expansion:
- Form 6/7/8 subtype classification
- Detailed field extraction
- NCL dependency logic
- Cross-document validation
- YOLO zone detection
- Celery/Redis scaling

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI |
| Frontend | React, Tailwind CSS v4 |
| Database | PostgreSQL |
| OCR | PaddleOCR (English + Marathi) |
| Classification | LayoutXLM (microsoft/layoutxlm-base) |
| Image Processing | OpenCV |
| PDF Processing | PyMuPDF |
