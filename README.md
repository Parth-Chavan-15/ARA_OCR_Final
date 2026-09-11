# ARA OCR — Autonomous Document Scrutiny & Classification System

> **Admissions Regulating Authority (ARA), Government of Maharashtra** — Production Bilingual OCR + Multimodal Document Classification & Field Extraction System

A local, GUI-based document processing system that performs high-precision bilingual (English + Marathi/Devanagari) OCR on Maharashtra State CET admission documents, classifies them into 6 categories using a fine-tuned LayoutXLM multimodal champion model, enforces statutory legal verification gates, extracts critical fields, and maintains an immutable audit ledger.

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
DUAL-ENGINE SELECTIVE SPATIAL FUSION (GPU-Accelerated)
    ├─ Pass 1: English PP-OCRv4 (form template, rules, outward numbers)
    ├─ Pass 2: Devanagari PP-OCRv4 (Marathi stamps, names, caste categories)
    └─ Spatial Containment & Devanagari Ratio (>=70%) Purity Filter
        ↓
TEXT + TOKENS + BBOX + CONFIDENCE + RECONSTRUCTED LINES
        ↓
PERSIST OCR CACHE (PostgreSQL) ← OCR ONCE, REUSE EVERYWHERE
        ↓
LANGUAGE / SCRIPT DETECTION (English / Marathi / Bilingual)
        ↓
LayoutXLM CHAMPION MULTIMODAL CLASSIFIER (visual tokens + 2D bboxes + image)
        ↓
6-WAY BROAD CLASSIFICATION (CVC, CC, CVR, Prof-O, LC, UNKNOWN)
        ↓
STATUTORY RULE GATES & CALIBRATION (Form 15, Form 6/7/8, Unverified OOS Gate)
        ↓
EVIDENCE GENERATION (statutory keywords & token bounding boxes)
        ↓
POST-OCR KEY FIELD EXTRACTION (serial_no, gr_no, certificate_no, institution)
        ↓
REAL-TIME WEBSOCKET SCRUTINY STREAM → AUDIT LEDGER & EXECUTIVE UI
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
# Labeled training images are organized in unified_training_data/{CLASS_NAME}/
python scripts/train_classifier.py --data_dir unified_training_data --epochs 10

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
│   │   │   ├── classification.py    # Classification pipeline + dashboard
│   │   │   ├── institutes.py        # Institute hierarchy & batch scrutiny
│   │   │   └── ws.py                # WebSocket scrutiny broadcast
│   │   ├── services/
│   │   │   ├── file_handler.py      # File validation
│   │   │   ├── pdf_service.py       # PDF → images (PyMuPDF)
│   │   │   ├── preprocessing.py     # OpenCV preprocessing
│   │   │   ├── ocr_service.py       # Dual-Engine Fusion (PaddleOCR en + mr)
│   │   │   ├── ocr_cache.py         # OCR cache management
│   │   │   ├── language.py          # Script/language detection
│   │   │   ├── classifier.py        # LayoutXLM Champion Classifier & Rule Gates
│   │   │   ├── evidence.py          # Statutory evidence generation
│   │   │   ├── field_extractor.py   # Post-OCR key field extraction
│   │   │   └── sync_service.py      # Multi-tier status synchronization
│   │   ├── db/
│   │   │   ├── database.py          # SQLAlchemy engine
│   │   │   └── init_db.py           # Schema + seed data
│   │   ├── models/                  # SQLAlchemy ORM models
│   │   └── schemas/                 # Pydantic schemas
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/                   # Dashboard, Incoming, Result, Review, InstituteHierarchy
│   │   ├── components/              # Layout, Sidebar, Header, StatusBadge
│   │   └── services/api.js          # Axios API & WebSocket client
│   └── package.json
├── data/
│   ├── originals/                   # Original documents (never modified)
│   ├── processed/                   # Preprocessed images
│   └── demo/                        # Synthetic demo documents
├── unified_training_data/           # Unified 1,138-sample training dataset
│   ├── CASTE_CERTIFICATE/           # Caste certificate samples
│   ├── CASTE_VALIDITY_CERTIFICATE/  # Caste validity certificate samples
│   ├── CASTE_VALIDITY_RECEIPT/      # Scrutiny committee validity receipts
│   ├── LEAVING_CERTIFICATE/         # School/college leaving certificates
│   ├── PROFORMA_O/                  # Minority self-declarations
│   └── UNKNOWN_OUT_OF_SCOPE/        # Non-reservation, NCL, income, civil IDs
├── models/
│   └── layoutxlm/                   # Fine-tuned Champion model weights
├── scripts/
│   ├── consolidate_datasets.py      # SHA-256 zero-data-loss dataset merger
│   ├── run_experiments.py           # 4-experiment GPU benchmark runner
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

## Dual-Engine Selective Spatial Fusion OCR (GPU-Accelerated)

Real-world Maharashtra admission documents frequently blend pre-printed English legal frameworks with candidate details, names, stamps, and caste categories typed or stamped in Marathi (Devanagari):
- Running solely `lang='mr'` loaded PaddleOCR's `devanagari_PP-OCRv4_rec_infer` which severely degrades English alphanumeric text (misinterpreting outward numbers and legal headers).
- Running solely `lang='en'` omitted Devanagari text entirely.
- Relying on "digital fast-paths" bypassed physical stamps, seals, and handwriting on scanned pages.

**Our Solution**:
1. **Mandatory Per-Page GPU Optical Scan**: Every page is rasterized at 300 DPI and processed visually on the GPU (`gpu:0`).
2. **Pass 1 (Primary English Engine)**: High-accuracy English PP-OCRv4 captures all English lines, form templates, rules, and candidate data with maximum precision.
3. **Pass 2 (Devanagari Injection Engine)**: Marathi PP-OCRv4 scans the document for authentic Devanagari script.
4. **Selective Spatial Containment & Purity Filtering**:
   - Discards non-authentic Devanagari tokens with strict Unicode ratio filtering ($\ge 70\%$ Devanagari characters) and confidence thresholding ($\ge 0.70$).
   - Rejects hallucinated Devanagari tokens that geometrically overlap with established high-confidence English lines.
   - Merges genuine Marathi entries (stamps, candidate names, castes) into reconstructed reading lines.

---

## Champion LayoutXLM Benchmark Suite (1,138 Unified Samples)

Evaluated on NVIDIA GeForce RTX 4050 Laptop GPU (`cuda:0`) with FP16 automatic mixed precision across 4 experimental schedules:

| Run # | Experiment Name & Schedule | Test Acc | Weighted F1 | Macro F1 | OOD F1 | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Run 1** | Class-Weighted Fine-Tuning (`LR 2.5e-5`, Step) | 82.99% | 82.76% | 61.53% | 78.21% | Baseline |
| **Run 2** | Cosine Annealing (`LR 3.0e-5`, Cosine) | 84.16% | 83.76% | 64.24% | 76.25% | Evaluated |
| **Run 3** | Regularized Weighted (`LR 2.0e-5`, Dropout 0.15) | 86.80% | 86.26% | 65.92% | 78.82% | Evaluated |
| **Run 4** | **Extended Cosine Schedule (`LR 3.5e-5`, 8 Epochs)** | **88.27%** | **87.90%** | **67.75%** | **84.71%** | **CHAMPION DEPLOYED** |

### Per-Class Performance (Champion Run 4)
- **`CASTE_VALIDITY_CERTIFICATE`**: **98.72% F1** (100% Precision, 97.48% Recall on 119 test samples)
- **`LEAVING_CERTIFICATE`**: **85.21% F1** (82.76% Precision, 87.80% Recall on 82 test samples)
- **`PROFORMA_O`**: **83.33% F1** (85.37% Precision, 81.40% Recall on 43 test samples)
- **`UNKNOWN_OUT_OF_SCOPE`**: **84.71% F1** (81.82% Precision, 87.80% Recall on 82 test samples)
- **`CASTE_CERTIFICATE`**: **54.55% F1** (66.67% Precision, 46.15% Recall on 13 test samples)

---

## Real-World Document Evaluation (5/5 Correct — 100% Accuracy)

Tested on physical and scanned Maharashtra State CET admission documents:

| Test Case | Statutory Document Category | Model Prediction | Confidence Label | Rule Applied | Extracted Key Fields | Status |
|:---:|:---|:---|:---:|:---|:---|:---:|
| Sample 1 | Higher Secondary School Leaving Certificate (HSC) | `LEAVING_CERTIFICATE` | 98.0% (LayoutLMv3: 90.7%) | `LEAVING_CERTIFICATE` | School: D.G. Ruparel College, GR: 044246, Serial: 0859, DOB: 23/09/2006 | **PERFECT** |
| Sample 2 | ₹100 Non-Judicial Stamp Paper / Affidavit | `UNKNOWN_OUT_OF_SCOPE` | 95.0% (LayoutLMv3: 0.9%) | `UNVERIFIED_RESERVATION_OOS_GATE` | Blocked from falsely claiming caste reservation | **PERFECT** |
| Sample 3 | Educational Gap Affidavit | `UNKNOWN_OUT_OF_SCOPE` | 95.0% (LayoutLMv3: 0.1%) | `UNVERIFIED_RESERVATION_OOS_GATE` | Blocked from falsely claiming caste validity | **PERFECT** |
| Sample 4 | Form B-2 Non-Creamy Layer Certificate | `UNKNOWN_OUT_OF_SCOPE` | 98.0% (LayoutLMv3: 84.4%) | `OUT_OF_SCOPE` | Outward No: `40112389156` (real outward number), Valid Upto: `31/03/2026` | **PERFECT** |
| Sample 5 | Secondary School Leaving Certificate (SSC - Marathi) | `LEAVING_CERTIFICATE` | 98.1% (LayoutLMv3: 98.1%) | `LEAVING_CERTIFICATE` | School: Balmohan Vidyamandir, GR: 527, Serial: 0859, DOB: 23/09/2006 | **PERFECT** |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend API | Python, FastAPI, Uvicorn |
| Frontend UI | React 18, Tailwind CSS v4, Lucide Icons |
| Database | PostgreSQL 15+ (Relational Schema & OCR Spatial Cache) |
| OCR Engine | Dual-Engine Selective Spatial Fusion (PaddleOCR PP-OCRv4 English + Devanagari) |
| Multimodal Classifier | Fine-Tuned LayoutXLM Champion Model (`microsoft/layoutxlm-base`) |
| Hardware Acceleration | NVIDIA CUDA (`cuda:0`, RTX 4050 Laptop GPU), PyTorch AMP (FP16) |
| Real-Time Communication | FastAPI WebSockets + Vite HMR Proxy |
| Field Extraction | Regex Pattern Matching & Multilingual Keyword Parsing (`FieldExtractor`) |
| Image & PDF Processing | PyMuPDF (300 DPI Rendering), OpenCV (CLAHE, Denoising) |
