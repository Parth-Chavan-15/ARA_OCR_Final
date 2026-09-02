# ARA OCR — Master Implementation Prompt for Coding Agent
## Tomorrow Scope: Local Bilingual OCR + OCR Cache + Broad 5-Way Document Classification

You are the primary implementation engineer for the **Admissions Regulating Authority (ARA) OCR-Based Document Verification System** prototype.

Your task is to implement the complete **tomorrow-only scope** described below as a working, runnable, local, GUI-based application.

This is a government-oriented document-processing project involving confidential candidate documents. Treat **data locality, traceability, correctness, reproducibility, and safe failure behavior** as first-class requirements.

---

# 0. NON-NEGOTIABLE SCOPE

## Implement tomorrow

Implement the pipeline **only up to broad document classification**:

```text
Local Document Store
        ↓
Fetch Document on Start Classification
        ↓
File Validation
        ↓
PDF → page images when required
        ↓
OpenCV preprocessing
        ↓
PaddleOCR (English + Marathi)
        ↓
OCR cache persisted to PostgreSQL
        ↓
Language / script detection
        ↓
LayoutXLM broad classification
        ↓
5 top-level classes OR UNKNOWN / OUT OF SCOPE
        ↓
Evidence + confidence
        ↓
Persist classification result
        ↓
GUI result
```

## The five supported top-level classes

1. `CASTE_CERTIFICATE`
2. `CASTE_VALIDITY_CERTIFICATE`
3. `CASTE_VALIDITY_RECEIPT`
4. `PROFORMA_O`
5. `LEAVING_CERTIFICATE`

Also support:

6. `UNKNOWN / OUT_OF_SCOPE`

The classifier must **not** force an unsupported document into one of the five classes.

---

# 1. EXPLICITLY DO NOT IMPLEMENT TOMORROW

Do NOT expand tomorrow's scope into the following:

- Caste Certificate → Form 6 / Form 7 / Form 8 / other template-level classification
- Detailed form-specific field extraction for every form
- NCL dependency logic for OBC/VJNT/SBC
- Cross-document checks between LC, Caste Certificate, Validity and Proforma-O
- Missing/supplemented-field audit findings
- Fraud/duplicate/authority-verification decisions
- YOLO-based specialist zone detection
- Celery/Redis scaling
- Kubernetes deployment

The architecture/data contract must be ready for those later modules, but they must remain **deferred**.

Later modules will consume the same cached `OCR_RESULT` and `OCR_TOKEN` data. They should not need a second OCR pass unless an intentionally different OCR model/version is invoked.

---

# 2. PRIMARY ARCHITECTURE PRINCIPLE

## OCR ONCE — REUSE EVERYWHERE

The OCR result is the central reusable artifact.

Use this architecture:

```text
DOCUMENT
   ↓
FILE HANDLING
   ↓
PREPROCESSING
   ↓
PaddleOCR
   ↓
OCR_RESULT + OCR_TOKEN
   ↓
PERSIST TO DATABASE
   ↓
ALL DOWNSTREAM LOGIC READS CACHED OCR
```

Do NOT implement:

```text
OCR → classify
OCR → language detection
OCR → another classifier
OCR → another validation step
```

as separate OCR executions.

Instead:

```text
ONE OCR RESULT
      ↓
cached
      ↓
classification
language detection
future form classification
future field extraction
future validation
future cross-document checks
```

This is a fundamental architectural requirement.

---

# 3. CONFIDENTIALITY / LOCAL-FIRST REQUIREMENTS

All document processing must remain local.

## Do not use

- Cloud OCR APIs
- Gemini/ChatGPT/cloud vision APIs
- External document upload services
- External OCR endpoints
- External CDN assets for document processing
- Remote inference services

The GUI, backend, OCR models, processing pipeline and data store should operate on the local machine for tomorrow's prototype.

It is acceptable to download/install model packages/dependencies during setup, but **the actual document-processing execution must be local**.

---

# 4. TECHNOLOGY STACK — USE THIS STACK

Do not spend implementation time evaluating alternatives. Use the following stack.

## Backend

- Python
- FastAPI

## Database

- PostgreSQL

## Frontend / GUI

- React
- Tailwind CSS

If the repository already contains the existing HTML prototype, preserve its visual language/workflow while moving toward the React implementation.

## OCR

- PaddleOCR
- English + Marathi / Devanagari

PaddleOCR is the **primary OCR engine** for tomorrow.

## Image processing

- OpenCV

## Document understanding / broad classification

- LayoutXLM

Use LayoutXLM because the task is multilingual document understanding using:

```text
Text
+
Bounding boxes / layout
+
Document image context
```

## PDF processing

Use a reliable local PDF renderer/parser. Prefer a well-supported Python/local solution such as PyMuPDF where appropriate.

## No YOLO tomorrow

YOLO-based specialist region detection is explicitly deferred.

---

# 5. FIRST STEP — INSPECT THE EXISTING REPOSITORY

Before writing new code:

1. Inspect the entire repository.
2. Identify the existing frontend/prototype.
3. Identify any existing backend.
4. Identify existing datasets.
5. Identify configuration files.
6. Identify Python/Node/package dependencies.
7. Identify current database setup.
8. Preserve useful existing UI/logic instead of unnecessarily rewriting working pieces.
9. Do not delete working functionality without a concrete reason.

Create/update a clear README describing how to run the system locally.

---

# 6. RECOMMENDED PROJECT STRUCTURE

Use a modular structure similar to:

```text
ara-ocr/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── services/
│   │   │   ├── file_handler.py
│   │   │   ├── pdf_service.py
│   │   │   ├── preprocessing.py
│   │   │   ├── ocr_service.py
│   │   │   ├── ocr_cache.py
│   │   │   ├── language.py
│   │   │   ├── classifier.py
│   │   │   └── evidence.py
│   │   ├── db/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── config.py
│   ├── tests/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── App.*
│   └── package.json
│
├── data/
│   ├── originals/
│   ├── processed/
│   └── demo/
│
├── models/
│   └── layoutxlm/
│
├── scripts/
├── docker/
├── README.md
└── .env.example
```

Adapt this to the existing repository instead of creating unnecessary duplication.

---

# 7. STEP-BY-STEP IMPLEMENTATION

## Step 1 — Candidate and document records

Create a candidate-centric data model.

A candidate may have multiple documents.

Every document must have:

- `document_id`
- `candidate_id`
- `enrollment_number`
- `filename`
- `file_path`
- `status`
- `uploaded_at`

Do NOT represent each document as an unrelated record without a candidate relationship.

Example:

```text
Candidate: CAND001
Enrollment: EN25272514

Documents:
- Caste Certificate
- Caste Validity Receipt
- Leaving Certificate
- Proforma-O
- NCL
```

NCL can exist in storage even though it is out of tomorrow's five-class scope.

---

# 8. STEP 2 — Local document ingestion

Keep the original file unchanged.

Store:

- original path
- filename
- document ID
- candidate ID
- upload time
- processing state

Do NOT overwrite original files.

Use separate processing directories/paths for:

```text
original
processed
```

The GUI should display documents that are already present in the local store.

---

# 9. STEP 3 — Fetch only when Start Classification is clicked

Follow the prototype workflow.

The incoming document list should show:

- Candidate
- Enrollment number
- Document/file
- Current processing state

When the user clicks:

```text
START CLASSIFICATION
```

the backend fetches the document and begins processing.

Do not silently process documents when the application first opens.

---

# 10. STEP 4 — File validation

Before processing:

Validate:

- extension
- MIME/type
- file size
- readability
- file corruption
- PDF integrity
- image decodability

Return structured statuses rather than crashing:

```text
VALID
INVALID
EMPTY
UNREADABLE
```

The UI should show a meaningful error state.

A bad file must not crash the whole application.

---

# 11. STEP 5 — PDF handling

For PDF input:

1. Inspect the PDF.
2. Determine whether a usable text layer exists.
3. For scanned/image-based pages, render to images.
4. Preserve page numbers.
5. Keep one `document_id` for the whole PDF.
6. Create separate page-level representations internally.

For example:

```text
document_id = DOC001

page 1
page 2
page 3
```

must remain one logical document.

The OCR result must retain the page number of every token.

---

# 12. STEP 6 — OpenCV preprocessing

For each page/image, safely perform:

- orientation correction
- resize where necessary
- denoising
- contrast normalization
- deskew
- adaptive thresholding when useful
- controlled sharpening
- basic blur / image-quality estimation

Where appropriate, support perspective correction.

Preserve:

```text
original image
processed image
```

Never destroy the original.

Preprocessing should improve robustness for:

- blur
- skew
- shadows
- low contrast
- photocopy-like scans
- camera/scanned documents

Do not apply aggressive processing blindly if it damages text.

---

# 13. STEP 7 — PaddleOCR

Run **PaddleOCR once per document/pages** using English + Marathi support.

The OCR service must return and persist:

```text
full text
tokens/words
confidence
bounding box
page number
detected language/script
```

Each token should conceptually look like:

```json
{
  "text": "Certificate",
  "confidence": 0.98,
  "bbox": [100, 120, 300, 155],
  "page": 1
}
```

The OCR output is the authoritative intermediate artifact for downstream processing.

---

# 14. STEP 8 — OCR cache persistence

Persist the OCR output **before broad classification**.

Required logical records:

```text
OCR_RESULT
- ocr_id
- document_id
- model_version
- language
- raw_text
- overall_confidence
- created_at
```

```text
OCR_TOKEN
- token_id
- ocr_id
- page_number
- text
- confidence
- x1
- y1
- x2
- y2
```

The OCR model version is mandatory.

If the OCR model later changes:

```text
v1
v2
```

the system must be able to intentionally regenerate/cache a new result rather than silently overwriting historical output.

---

# 15. STEP 9 — Language / script detection

Use the OCR result to classify the document's script/language as:

```text
English
Marathi
Marathi + English
```

The raw OCR must preserve both scripts.

Do NOT transliterate Marathi away.

Example:

```text
CERTIFICATE OF VALIDITY
जात वैधता प्रमाणपत्र
```

must remain representable as bilingual content.

Store the resulting language label with the OCR result.

---

# 16. STEP 10 — Broad document classification

Use **LayoutXLM** as the broad document-understanding/classification model.

Input to the classifier:

```text
document image
+
OCR text
+
OCR bounding boxes/layout
```

Predict only:

```text
CASTE_CERTIFICATE
CASTE_VALIDITY_CERTIFICATE
CASTE_VALIDITY_RECEIPT
PROFORMA_O
LEAVING_CERTIFICATE
UNKNOWN / OUT_OF_SCOPE
```

Do NOT classify:

```text
SC
OBC
VJNT
SBC
```

as the top-level document class.

Category and document type are separate concepts.

Example:

```text
document_type = CASTE_CERTIFICATE
reserve_category = OBC
```

is correct.

---

# 17. STEP 11 — Important distinction for caste forms

For tomorrow, Form 6/Form 7/Form 8 and similar caste-certificate variants are **not separate broad classes**.

They belong to the later Caste Certificate subtype layer.

Therefore:

```text
Form 6
↓
CASTE_CERTIFICATE
```

```text
Form 7
↓
CASTE_CERTIFICATE
```

```text
Form 8
↓
CASTE_CERTIFICATE
```

The later module will distinguish:

```text
CASTE_CERTIFICATE
   ↓
Form 6 / Form 7 / Form 8 / other template
```

Do NOT misclassify a Form 8 caste certificate as:

```text
CASTE_VALIDITY_CERTIFICATE
```

just because it contains the word "caste".

This is a required robustness test.

---

# 18. STEP 12 — UNKNOWN / OUT-OF-SCOPE

Implement safe out-of-scope handling.

Examples:

- Non-Creamy Layer Certificate
- Non-Creamy Layer Receipt
- Form 6/7/8 if treated as a future subtype before subtype implementation
- other ordinary caste certificates
- income certificate
- domicile certificate
- unrelated document
- corrupt/unreadable document

The result can be:

```text
Top-level class:
UNKNOWN / OUT OF SCOPE

Reason:
Document does not match any supported broad document class.
```

The system must never blindly force an unknown document into one of the five classes.

---

# 19. STEP 13 — Confidence / safe rejection

Display classification confidence.

Example:

```text
Caste Validity Receipt
Confidence: 97.2%
```

Implement a configurable broad-classification acceptance threshold.

If:

```text
maximum class confidence < threshold
```

return:

```text
UNKNOWN / OUT_OF_SCOPE
```

rather than forcing a prediction.

Also treat strongly unsupported documents as out-of-scope.

Do not claim that a confidence number means authenticity. It is classification confidence only.

---

# 20. STEP 14 — Classification evidence

For every result, generate human-readable evidence.

Do not only show:

```text
Prediction = 97%
```

Show:

```text
Predicted Document:
Caste Validity Receipt

Confidence:
97%

Language:
Marathi + English

Evidence:
✓ DISTRICT CASTE CERTIFICATE SCRUTINY COMMITTEE
✓ Online Payment Receipt
✓ Application Number
✓ Receipt No.
✓ Transaction Reference
✓ Transaction Status
```

Evidence should be linked to OCR text and, where possible, page/bounding-box locations.

Evidence is explanatory and is **not** a final validity/fraud decision.

---

# 21. STEP 15 — Persist broad classification

Create:

```text
CLASSIFICATION_RESULT
- document_id
- predicted_class
- confidence
- evidence
- model_version
- created_at
```

The result must be linked to the exact document and exact model version.

No second OCR execution should happen during classification.

The classifier must read the cached OCR result.

---

# 22. DATABASE DESIGN

Implement at least these tables/models.

## CANDIDATE

```text
candidate_id
enrollment_number
name
reserve_category
```

## DOCUMENT

```text
document_id
candidate_id
filename
file_path
document_type
status
uploaded_at
```

`document_type` may remain null until classification.

## OCR_RESULT

```text
ocr_id
document_id
model_version
language
raw_text
overall_confidence
created_at
```

## OCR_TOKEN

```text
token_id
ocr_id
page_number
text
confidence
x1
y1
x2
y2
```

## CLASSIFICATION_RESULT

```text
document_id
predicted_class
confidence
evidence
model_version
created_at
```

Design the schema so future modules can add:

```text
template_type
extracted_fields
validation_result
cross_document_findings
```

without redesigning the candidate/document relationship.

---

# 23. GUI REQUIREMENTS

Keep the GUI similar to the existing ARA prototype.

## Dashboard

Display:

- Total candidates/documents
- Processed
- Classified
- Unknown
- Low-confidence

## Candidate Documents

Display:

- Candidate name
- Enrollment number
- Reserve category
- All linked documents

## Incoming Documents

Display:

- candidate
- enrollment number
- document
- local-store status
- `Start Classification` button

Documents must already appear as stored documents.

## AI Processing

Show the live processing stages:

```text
Fetch
↓
PDF Rendering
↓
Preprocessing
↓
OCR
↓
OCR Cache
↓
Language Detection
↓
LayoutXLM Classification
↓
Completed
```

## Result

Display:

- original document preview
- processed preview when useful
- OCR text
- detected language
- broad classification
- confidence
- evidence
- OCR bounding boxes where practical

## Review

Display:

- low-confidence documents
- UNKNOWN / OUT_OF_SCOPE documents
- original document
- OCR output
- reason for review/rejection of automatic classification

This is only a review/fallback view; detailed verification is deferred.

---

# 24. OCR / CLASSIFICATION RESULT PRESENTATION

The GUI should make it obvious that:

```text
OCR confidence
≠
Classification confidence
≠
Document authenticity
```

Display them as separate concepts.

Example:

```text
OCR confidence:            94.2%
Language:                  Marathi + English
Classification:            Proforma-O
Classification confidence: 97.1%
Status:                    Classified
```

---

# 25. TEST / DEMONSTRATION DATA

Use both:

1. supplied dataset/documents
2. synthetic/dummy documents

The goal is to demonstrate robustness; do not claim production-grade accuracy from a tiny demonstration set.

Create variations for all five target classes:

## Caste Certificate

Use variations represented by the supplied dataset, including:

- SC
- OBC
- VJNT
- SBC
- different layouts/forms
- English
- Marathi
- bilingual
- different years/numbers

Important observed variants may include:

- Form 6
- Form 7
- Form 8
- Office of D.Y. Collector format
- other caste-certificate templates

These are used as **Caste Certificate family examples**, not top-level classes.

## Caste Validity Certificate

Create:

- English
- Marathi
- bilingual
- different district/committee wording
- different years/numbers
- different certificate layouts

## Caste Validity Receipt

Create:

- different districts
- different receipt/application numbers
- English
- Marathi
- bilingual
- different transaction details
- different receipt layouts

## Proforma-O

Create:

- Sindhi examples
- other minority examples
- English
- Marathi
- bilingual
- mother tongue present
- mother tongue absent

## Leaving Certificate

Create:

- different layouts/templates
- SC/OBC/VJNT/SBC/other category examples
- Marathi
- English
- bilingual
- caste/sub-caste present
- caste/sub-caste missing
- mother tongue present
- mother tongue missing

## Out-of-scope examples

Include:

- Non-Creamy Layer Certificate
- Non-Creamy Layer Receipt
- ordinary Caste Certificate variants when testing the broad classifier boundary
- Income Certificate
- Domicile Certificate
- unrelated document

The NCL document must demonstrate safe handling rather than being incorrectly mapped to a Caste Validity class.

---

# 26. IMAGE ROBUSTNESS TESTS

Create degraded demo examples:

- blur
- rotation
- low contrast
- noise
- scan shadows
- compression

The purpose is to show that the pipeline does not depend on a single clean template.

Test:

```text
clean
blurred
rotated
shadowed
low contrast
bilingual
Marathi
English
```

---

# 27. SAFETY RULES

The implementation must obey all of these:

1. Never overwrite the original document.
2. Never discard raw OCR text.
3. Never discard OCR bounding boxes.
4. Never discard page numbers.
5. Never rerun OCR merely because a downstream module needs the OCR text.
6. Always read the cached OCR result.
7. Base64 is only an encoding representation; it is not encryption.
8. Never use reserve category as document type.
9. Never treat Form 6/Form 7/Form 8 as Caste Validity documents.
10. Never force unsupported documents into a supported class.
11. Missing data must not be hallucinated.
12. Store model versions.
13. Store OCR model version.
14. Store classification timestamp.
15. Handle corrupt/unreadable files without crashing the pipeline.
16. Keep processing local.
17. Do not use external CDN dependencies for document processing.

---

# 28. API DESIGN

Implement a clean backend API.

At minimum support endpoints conceptually equivalent to:

```text
GET  /candidates
GET  /candidates/{candidate_id}/documents
GET  /documents
GET  /documents/{document_id}

POST /documents/{document_id}/classify

GET  /documents/{document_id}/status
GET  /documents/{document_id}/ocr
GET  /documents/{document_id}/classification
```

The classification endpoint should:

1. load document
2. validate
3. render PDF if required
4. preprocess
5. use/create OCR cache
6. detect language/script
7. run LayoutXLM
8. generate evidence
9. save classification
10. return a structured result

If OCR cache already exists for the same OCR model/version, **reuse it**.

---

# 29. OCR CACHE REUSE RULE

When `/classify` runs:

```text
Does OCR_RESULT exist for this document
with the requested OCR model version?
```

### YES

```text
read cache
↓
skip OCR
↓
continue classification
```

### NO

```text
run OCR
↓
persist OCR_RESULT/OCR_TOKEN
↓
continue classification
```

This is essential.

---

# 30. MODEL VERSIONING

Store:

```text
ocr_model_version
classifier_model_version
pipeline_version
```

Example:

```text
ocr_model_version = paddleocr_mr_en_v1
classifier_model_version = layoutxlm_ara_v1
pipeline_version = 0.1.0
```

A model upgrade should produce a new cache/result rather than silently changing the meaning of old results.

---

# 31. ERROR HANDLING

Every stage should fail safely.

Examples:

```text
FILE_INVALID
FILE_EMPTY
FILE_UNREADABLE
PDF_RENDER_FAILED
PREPROCESSING_FAILED
OCR_FAILED
OCR_EMPTY
CLASSIFICATION_FAILED
LOW_CONFIDENCE
UNKNOWN_DOCUMENT
```

Do not allow one document's failure to bring down the whole GUI/backend.

---

# 32. PERFORMANCE EXPECTATIONS

Tomorrow is a local demonstration, so prioritize:

1. correctness
2. deterministic behavior
3. robust error handling
4. evidence
5. reasonable speed

Do not spend time implementing distributed scaling.

However, keep the code modular enough that Celery/Redis can later replace direct synchronous execution.

---

# 33. SECURITY EXPECTATIONS

Even in prototype form:

- keep files local
- do not log raw document bytes
- do not expose uploaded files through unrestricted paths
- validate paths to avoid traversal
- validate MIME/type
- avoid returning unnecessary sensitive data
- do not include candidate documents in frontend telemetry
- do not load external tracking/CDN resources for document processing

For future production, the architecture can add:

```text
RBAC
SSO
TLS
AES-256 at rest
audit trail
VAPT
government infrastructure deployment
```

Those are not tomorrow's implementation scope.

---

# 34. README REQUIREMENTS

Create a README that explains:

## Prerequisites

- Python version
- Node version
- PostgreSQL setup
- required local model files
- PaddleOCR dependencies
- LayoutXLM dependencies

## Installation

Provide exact commands.

## Database setup

Provide schema/migration steps.

## Model setup

Explain where local OCR/classification models are stored.

## Running

Provide exact commands for:

```text
backend
frontend
```

## Demo

Explain:

```text
1. Start application
2. Open Incoming Documents
3. Select a candidate/document
4. Click Start Classification
5. Observe pipeline
6. Open Result
7. Inspect OCR text/evidence/classification
```

---

# 35. TESTING REQUIREMENTS

Write tests for:

## File handling

- valid JPG
- valid JPEG
- valid PNG
- valid PDF
- multi-page PDF
- corrupt file
- zero-byte file
- unsupported type
- unreadable image

## OCR

- English
- Marathi
- bilingual
- empty OCR
- low confidence
- bounding boxes
- page numbers
- cache reuse

## Classification

Test:

- Caste Certificate
- Caste Validity Certificate
- Caste Validity Receipt
- Proforma-O
- Leaving Certificate
- UNKNOWN / OUT_OF_SCOPE

## Important negative tests

Verify that:

```text
Form 6 → not Caste Validity Certificate
Form 7 → not Caste Validity Certificate
Form 8 → not Caste Validity Certificate
NCL Receipt → UNKNOWN / OUT_OF_SCOPE
Income Certificate → UNKNOWN / OUT_OF_SCOPE
Domicile Certificate → UNKNOWN / OUT_OF_SCOPE
```

---

# 36. ACCEPTANCE CRITERIA

The implementation is considered complete for tomorrow only when all of these are true:

- Local GUI starts.
- Candidate-linked documents are visible.
- Multiple documents can belong to one candidate.
- JPG/JPEG/PNG/PDF inputs work locally.
- Multi-page PDFs are handled page-by-page.
- Original files are preserved.
- File errors are handled safely.
- OpenCV preprocessing runs.
- PaddleOCR performs English + Marathi OCR.
- OCR confidence is retained.
- OCR bounding boxes are retained.
- Page numbers are retained.
- Language/script is identified.
- OCR output is persisted to PostgreSQL.
- Classification consumes the cached OCR result.
- OCR is not repeated unnecessarily.
- LayoutXLM produces one of the five broad classes or UNKNOWN.
- Caste Certificate is a distinct top-level class.
- Form 6/Form 7/Form 8 do not become separate top-level classes.
- Form 6/Form 7/Form 8 are not misclassified as Caste Validity documents.
- Marathi examples work.
- bilingual examples work.
- NCL certificate/receipt can be shown as UNKNOWN/OUT_OF_SCOPE.
- Confidence is visible.
- Evidence is visible.
- Classification result is persisted.
- Model versions and timestamps are stored.
- No external OCR/cloud document processing is used.
- The system fails safely instead of crashing.

---

# 37. DEFERRED ARCHITECTURE — PREPARE FOR LATER

Do not implement these now, but ensure the current schema and service boundaries make them easy to add later:

```text
CASTE_CERTIFICATE
    ↓
Form 6 / Form 7 / Form 8 / other template classification

Detailed form-specific field extraction

NCL applicability logic:
    OBC / VJNT / SBC
        ↓
    NCL expected/present/missing

Cross-document matching:
    Admission Form
    ↕
    Caste Certificate
    ↕
    Caste Validity
    ↕
    LC
    ↕
    Proforma-O

Missing/supplemented-field findings

Fraud / duplicate / authority verification

YOLO specialist zone detection

Celery + Redis

Kubernetes
```

The critical requirement is that all of those future modules should be able to consume:

```text
OCR_RESULT
OCR_TOKEN
CLASSIFICATION_RESULT
```

without rerunning OCR unnecessarily.

---

# 38. DEVELOPMENT WORKFLOW FOR THE AGENT

Follow this order strictly:

```text
1. Inspect repository
2. Set up local environment
3. Set up PostgreSQL schema
4. Implement candidate/document storage
5. Implement file validation
6. Implement PDF/image handling
7. Implement OpenCV preprocessing
8. Implement PaddleOCR
9. Implement OCR cache
10. Implement language detection
11. Implement LayoutXLM classification
12. Implement UNKNOWN threshold
13. Implement evidence generation
14. Persist classification
15. Build GUI
16. Connect GUI to API
17. Add demo data
18. Add automated tests
19. Run end-to-end tests
20. Fix failures
21. Verify local/offline execution
22. Update README
```

Do not jump directly to the UI before the backend pipeline works.

---

# 39. ENGINEERING QUALITY REQUIREMENTS

Write clean, maintainable code.

Use:

- type hints
- Pydantic models
- structured logging
- clear exceptions
- environment-based configuration
- modular service interfaces
- meaningful names
- comments only where they add value

Avoid:

- giant single-file implementations
- hard-coded absolute paths
- hard-coded candidate-specific logic
- hard-coded class decisions based on a few keywords
- duplicated OCR calls
- external cloud dependencies
- silent exception swallowing

---

# 40. IMPORTANT CLASSIFICATION DESIGN

Do not build the classifier as:

```python
if "receipt" in text:
    return "Caste Validity Receipt"
```

The broad classifier must use the intended multimodal approach:

```text
document image
+
OCR text
+
OCR bounding boxes/layout
↓
LayoutXLM
↓
class probabilities
```

Evidence/keywords can be generated from the OCR output to explain the prediction, but they must not replace the classifier.

---

# 41. IMPORTANT UNKNOWN DESIGN

The system must be able to say:

```text
UNKNOWN / OUT OF SCOPE
```

A prediction should not be forced simply because one familiar keyword exists.

For example:

```text
"caste"
```

does not automatically mean:

```text
Caste Validity Certificate
```

The model must consider the document as a whole.

---

# 42. FINAL END-TO-END CONTRACT

The canonical flow is:

```text
LOCAL DOCUMENT STORE
        ↓
FETCH DOCUMENT
        ↓
FILE VALIDATION
        ↓
PDF → IMAGE (if required)
        ↓
OpenCV PREPROCESSING
        ↓
PaddleOCR
(Marathi + English)
        ↓
TEXT + TOKENS + BBOX + CONFIDENCE + PAGE
        ↓
PERSIST OCR CACHE
        ↓
LANGUAGE / SCRIPT DETECTION
        ↓
LayoutXLM
(TEXT + IMAGE + LAYOUT)
        ↓
BROAD DOCUMENT CLASSIFICATION
        ↓
┌──────────────────────────────────┐
│ CASTE CERTIFICATE                │
│ CASTE VALIDITY CERTIFICATE       │
│ CASTE VALIDITY RECEIPT           │
│ PROFORMA-O                       │
│ LEAVING CERTIFICATE              │
│ UNKNOWN / OUT OF SCOPE           │
└──────────────────────────────────┘
        ↓
CONFIDENCE / SAFE REJECTION
        ↓
EVIDENCE GENERATION
        ↓
PERSIST CLASSIFICATION RESULT
        ↓
GUI RESULT
```

---

# 43. FINAL RULE

The goal is not to build the largest-looking architecture.

The goal is to make the **smallest complete, robust and genuinely working vertical slice**:

```text
INPUT
→ OCR
→ CACHE
→ LANGUAGE
→ LAYOUTXLM
→ 5-WAY BROAD CLASSIFICATION
→ UNKNOWN SAFETY
→ EVIDENCE
→ GUI
→ DATABASE
```

Everything must be implemented so that a later team member can add form classification, detailed extraction, NCL logic and cross-document validation **without redesigning the OCR layer or scanning the same document again**.

Do not claim functionality that has not actually been implemented. At the end, report exactly:

- what works
- what model is actually being used
- what data is actually supported
- what remains deferred
- how to run the application
- how to reproduce the demo
