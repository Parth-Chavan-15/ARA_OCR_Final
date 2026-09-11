import os
import sys
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict

CLASS_DIR_MAP = {
    # Lowercase & variations
    "caste_certificate": "CASTE_CERTIFICATE",
    "caste certificate": "CASTE_CERTIFICATE",
    "caste_validity_certificate": "CASTE_VALIDITY_CERTIFICATE",
    "cvc": "CASTE_VALIDITY_CERTIFICATE",
    "caste_validity_receipt": "CASTE_VALIDITY_RECEIPT",
    "cvc reciepts": "CASTE_VALIDITY_RECEIPT",
    "cvc receipts": "CASTE_VALIDITY_RECEIPT",
    "leaving_certificate": "LEAVING_CERTIFICATE",
    "lc": "LEAVING_CERTIFICATE",
    "proforma_o": "PROFORMA_O",
    "proforma": "PROFORMA_O",
    "profoma-o": "PROFORMA_O",
    "unknown_out_of_scope": "UNKNOWN_OUT_OF_SCOPE",
    # Uppercase
    "CASTE_CERTIFICATE": "CASTE_CERTIFICATE",
    "CASTE_VALIDITY_CERTIFICATE": "CASTE_VALIDITY_CERTIFICATE",
    "CASTE_VALIDITY_RECEIPT": "CASTE_VALIDITY_RECEIPT",
    "LEAVING_CERTIFICATE": "LEAVING_CERTIFICATE",
    "PROFORMA_O": "PROFORMA_O",
    "UNKNOWN_OUT_OF_SCOPE": "UNKNOWN_OUT_OF_SCOPE",
}

VALID_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".jfif", ".webp"}

def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def consolidate():
    base_dir = Path(r"c:\Users\parth\Documents\ARA_OCR_Final")
    sources = [
        ("training_data", base_dir / "training_data"),
        ("new_training_data", base_dir / "new_training_data"),
        ("downloads_data", Path(r"C:\Users\parth\Downloads\Data (yet to be used to tune)"))
    ]
    target_dir = base_dir / "unified_training_data"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Initialize target subdirectories
    classes = [
        "CASTE_CERTIFICATE",
        "CASTE_VALIDITY_CERTIFICATE",
        "CASTE_VALIDITY_RECEIPT",
        "LEAVING_CERTIFICATE",
        "PROFORMA_O",
        "UNKNOWN_OUT_OF_SCOPE"
    ]
    for c in classes:
        (target_dir / c).mkdir(parents=True, exist_ok=True)

    seen_hashes_by_class = defaultdict(set)
    stats = {
        "scanned": defaultdict(int),
        "copied": defaultdict(int),
        "renamed_distinct": defaultdict(int),
        "skipped_exact_duplicate": defaultdict(int)
    }

    print("=" * 70)
    print("STARTING DATASET CONSOLIDATION & CRYPTOGRAPHIC DEDUPLICATION")
    print("=" * 70)

    for src_label, src_path in sources:
        if not src_path.exists():
            print(f"[WARNING] Source path does not exist: {src_path}")
            continue

        print(f"\nScanning source: [{src_label}] ({src_path})")

        for item in sorted(src_path.iterdir()):
            if not item.is_dir():
                continue

            folder_key = item.name.strip().lower()
            target_class = CLASS_DIR_MAP.get(folder_key)
            if not target_class:
                # Try direct name
                target_class = CLASS_DIR_MAP.get(item.name.strip())

            if not target_class:
                print(f"  [SKIP] Unknown category folder: {item.name}")
                continue

            class_dest = target_dir / target_class
            files = [f for f in item.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]

            for file_path in files:
                stats["scanned"][target_class] += 1
                f_hash = compute_sha256(file_path)

                # Check if exact byte-for-byte identical copy already in this class
                if f_hash in seen_hashes_by_class[target_class]:
                    stats["skipped_exact_duplicate"][target_class] += 1
                    continue

                seen_hashes_by_class[target_class].add(f_hash)

                # Determine destination filename
                dest_file = class_dest / file_path.name
                if dest_file.exists():
                    # Filename collision but DIFFERENT content hash -> preserve both by namespacing
                    new_name = f"{src_label}_{file_path.name}"
                    dest_file = class_dest / new_name
                    stats["renamed_distinct"][target_class] += 1

                shutil.copy2(file_path, dest_file)
                stats["copied"][target_class] += 1

    print("\n" + "=" * 70)
    print("CONSOLIDATION COMPLETE: SUMMARY AUDIT")
    print("=" * 70)
    print(f"{'Target Class':<28} | {'Scanned':<8} | {'Copied':<8} | {'Renamed':<8} | {'Exact Dupes':<12}")
    print("-" * 70)
    total_scanned = 0
    total_copied = 0
    total_renamed = 0
    total_dupes = 0

    for c in classes:
        sc = stats["scanned"][c]
        cp = stats["copied"][c]
        rn = stats["renamed_distinct"][c]
        dp = stats["skipped_exact_duplicate"][c]
        total_scanned += sc
        total_copied += cp
        total_renamed += rn
        total_dupes += dp
        print(f"{c:<28} | {sc:<8} | {cp:<8} | {rn:<8} | {dp:<12}")

    print("-" * 70)
    print(f"{'TOTAL':<28} | {total_scanned:<8} | {total_copied:<8} | {total_renamed:<8} | {total_dupes:<12}")
    print("=" * 70)
    print(f"\nAll distinct files consolidated into: {target_dir}")

if __name__ == "__main__":
    consolidate()
