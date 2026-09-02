"""End-to-end test of the classification pipeline."""
import requests
import time

BASE = "http://localhost:8000"

# Get first document
docs = requests.get(f"{BASE}/api/documents").json()
did = docs[0]["document_id"]
fname = docs[0]["filename"]
print(f"Testing classification pipeline on: {fname} (ID: {did})")
print()

# Run classification
print("Running POST /api/documents/{id}/classify ...")
start = time.time()
r = requests.post(f"{BASE}/api/documents/{did}/classify")
elapsed = time.time() - start
print(f"Response in {elapsed:.1f}s - Status: {r.status_code}")
print()

result = r.json()
print(f"Pipeline Status: {result.get('status')}")
print(f"Predicted Class: {result.get('predicted_class')}")
print(f"Confidence: {result.get('confidence')}")
print(f"Language: {result.get('language')}")
print(f"OCR Confidence: {result.get('ocr_confidence')}")
if result.get("error"):
    print(f"Error: {result.get('error')}")
print()

print("Pipeline Stages:")
for stage in result.get("pipeline_stages", []):
    s = stage.get("stage", "")
    st = stage.get("status", "")
    err = stage.get("error", "")
    marker = "OK" if st == "COMPLETED" else "XX"
    line = f"  [{marker}] {s}: {st}"
    if err:
        line += f" -> {err}"
    print(line)

# Verify OCR result exists
print()
r2 = requests.get(f"{BASE}/api/documents/{did}/ocr")
print(f"GET /ocr: {r2.status_code}")
if r2.status_code == 200:
    ocr = r2.json()
    print(f"  Language: {ocr.get('language')}")
    print(f"  Overall Confidence: {ocr.get('overall_confidence')}")
    print(f"  Tokens: {len(ocr.get('tokens', []))}")
    raw = ocr.get("raw_text", "")[:200]
    print(f"  Raw text (first 200 chars): {raw}")

# Verify classification result persisted
r3 = requests.get(f"{BASE}/api/documents/{did}/classification")
print(f"GET /classification: {r3.status_code}")
if r3.status_code == 200:
    cls = r3.json()
    print(f"  Predicted: {cls.get('predicted_class')}")
    print(f"  Confidence: {cls.get('confidence')}")
    print(f"  Evidence keys: {list(cls.get('evidence', {}).keys())}")

# Verify document status updated
r4 = requests.get(f"{BASE}/api/documents/{did}/status")
print(f"GET /status: {r4.status_code} -> {r4.json()}")

# Check dashboard updated
r5 = requests.get(f"{BASE}/api/dashboard/stats")
print(f"Dashboard: {r5.json()}")
