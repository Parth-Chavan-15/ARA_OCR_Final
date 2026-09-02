import fitz
from pathlib import Path
from PIL import Image

data_dir = Path("training_data")
classes = [
    "CASTE_CERTIFICATE",
    "CASTE_VALIDITY_CERTIFICATE",
    "CASTE_VALIDITY_RECEIPT",
    "PROFORMA_O",
    "LEAVING_CERTIFICATE",
    "UNKNOWN_OUT_OF_SCOPE"
]

print("=" * 60)
print("INSPECTING DATASET FILES IN training_data/")
print("=" * 60)

for c in classes:
    cdir = data_dir / c
    if not cdir.exists():
        continue
    files = list(cdir.iterdir())
    print(f"\n[{c}] Total files: {len(files)}")
    
    text_count = 0
    scanned_count = 0
    for f in files:
        if f.suffix.lower() == ".pdf":
            try:
                doc = fitz.open(str(f))
                words = []
                for p in doc:
                    words.extend(p.get_text("words"))
                if len(words) > 0:
                    text_count += 1
                else:
                    scanned_count += 1
                doc.close()
            except Exception as e:
                print(f"  Error reading {f.name}: {e}")
        elif f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            scanned_count += 1

    print(f"  - PDFs with native text layers: {text_count}")
    print(f"  - Scanned PDFs / Image files: {scanned_count}")
