import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ARA_Classifier_Trainer")

CLASS_NAMES = [
    "CASTE_CERTIFICATE",
    "CASTE_VALIDITY_CERTIFICATE",
    "CASTE_VALIDITY_RECEIPT",
    "PROFORMA_O",
    "LEAVING_CERTIFICATE",
    "UNKNOWN_OUT_OF_SCOPE"
]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

CLASS_DIR_MAP = {
    "caste_certificate": "CASTE_CERTIFICATE",
    "caste_validity_certificate": "CASTE_VALIDITY_CERTIFICATE",
    "caste_validity_receipt": "CASTE_VALIDITY_RECEIPT",
    "leaving_certificate": "LEAVING_CERTIFICATE",
    "proforma_o": "PROFORMA_O",
    "unknown_out_of_scope": "UNKNOWN_OUT_OF_SCOPE",
    "CASTE_CERTIFICATE": "CASTE_CERTIFICATE",
    "CASTE_VALIDITY_CERTIFICATE": "CASTE_VALIDITY_CERTIFICATE",
    "CASTE_VALIDITY_RECEIPT": "CASTE_VALIDITY_RECEIPT",
    "LEAVING_CERTIFICATE": "LEAVING_CERTIFICATE",
    "PROFORMA_O": "PROFORMA_O",
    "UNKNOWN_OUT_OF_SCOPE": "UNKNOWN_OUT_OF_SCOPE",
}


def extract_document_pages(file_path: Path) -> list[dict]:
    samples = []
    suffix = file_path.suffix.lower()
    student_id = file_path.stem.split('_')[0].split('-')[0].strip()

    if suffix == ".pdf":
        try:
            doc = fitz.open(str(file_path))
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                w, h = pix.width, pix.height

                raw_words = page.get_text("words")
                words = []
                boxes = []

                for item in raw_words:
                    word_text = str(item[4]).strip()
                    if word_text:
                        x0, y0, x1, y1 = item[:4]
                        norm_box = [
                            int(max(0, min(1000, (x0 / w) * 1000))),
                            int(max(0, min(1000, (y0 / h) * 1000))),
                            int(max(0, min(1000, (x1 / w) * 1000))),
                            int(max(0, min(1000, (y1 / h) * 1000))),
                        ]
                        words.append(word_text)
                        boxes.append(norm_box)

                if not words:
                    stem_words = file_path.stem.replace("_", " ").replace("-", " ").split()
                    words = stem_words if stem_words else ["document", "certificate"]
                    boxes = [[100, 100, 900, 300] for _ in words]

                samples.append({
                    "image": img,
                    "words": words[:512],
                    "boxes": boxes[:512],
                    "source": f"{file_path.name}_p{page_idx+1}",
                    "student_id": student_id,
                })
            doc.close()
        except Exception as e:
            logger.warning(f"Error extracting PDF {file_path}: {e}")

    elif suffix in [".jpg", ".jpeg", ".png"]:
        try:
            img = Image.open(file_path).convert("RGB")
            stem_words = file_path.stem.replace("_", " ").replace("-", " ").split()
            words = stem_words if stem_words else ["document", "certificate"]
            boxes = [[100, 100, 900, 300] for _ in words]

            samples.append({
                "image": img,
                "words": words[:512],
                "boxes": boxes[:512],
                "source": file_path.name,
                "student_id": student_id,
            })
        except Exception as e:
            logger.warning(f"Error loading image {file_path}: {e}")

    return samples


def load_dataset(primary_dir: Path, auxiliary_dirs: list[Path] = None) -> list[dict]:
    all_samples = []
    logger.info(f"Loading primary dataset from {primary_dir}...")

    for item in sorted(primary_dir.iterdir()):
        if not item.is_dir():
            continue
        cname = CLASS_DIR_MAP.get(item.name, item.name.upper())
        if cname not in CLASS_MAP:
            continue

        files = [f for f in item.iterdir() if f.is_file() and f.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png"]]
        for f in files:
            doc_samples = extract_document_pages(f)
            for s in doc_samples:
                s["label"] = CLASS_MAP[cname]
                s["class_name"] = cname
                all_samples.append(s)

    if auxiliary_dirs:
        for aux_dir in auxiliary_dirs:
            if not aux_dir.exists():
                continue
            for item in sorted(aux_dir.iterdir()):
                if not item.is_dir():
                    continue
                cname = CLASS_DIR_MAP.get(item.name, item.name.upper())
                if cname not in CLASS_MAP:
                    continue

                current_count = sum(1 for s in all_samples if s["class_name"] == cname)
                if current_count < 20:
                    files = [f for f in item.iterdir() if f.is_file() and f.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png"]]
                    for f in files:
                        doc_samples = extract_document_pages(f)
                        for s in doc_samples:
                            s["label"] = CLASS_MAP[cname]
                            s["class_name"] = cname
                            s["source"] = f"aux_{f.name}"
                            all_samples.append(s)

    class_counts = Counter(s["class_name"] for s in all_samples)
    logger.info("Loaded Dataset Summary:")
    for c in CLASS_NAMES:
        logger.info(f"  - {c:30s}: {class_counts.get(c, 0):4d} samples")
    logger.info(f"Total: {len(all_samples)} samples across {len(CLASS_NAMES)} classes.\n")

    return all_samples


def group_stratified_split(samples: list[dict], train_ratio: float = 0.7, seed: int = 42) -> tuple[list[dict], list[dict]]:
    random.seed(seed)
    np.random.seed(seed)

    class_groups = defaultdict(lambda: defaultdict(list))
    for s in samples:
        c_idx = s["label"]
        grp = s.get("student_id", s["source"])
        class_groups[c_idx][grp].append(s)

    train_samples = []
    test_samples = []

    for c_idx in sorted(class_groups.keys()):
        groups_dict = class_groups[c_idx]
        group_keys = list(groups_dict.keys())
        random.shuffle(group_keys)

        total_samples_in_class = sum(len(v) for v in groups_dict.values())
        target_train_count = max(1, int(round(total_samples_in_class * train_ratio)))

        if len(group_keys) == 1:
            single_grp_samples = groups_dict[group_keys[0]]
            if len(single_grp_samples) > 1:
                n_tr = max(1, int(len(single_grp_samples) * train_ratio))
                train_samples.extend(single_grp_samples[:n_tr])
                test_samples.extend(single_grp_samples[n_tr:])
            else:
                train_samples.extend(single_grp_samples)
                test_samples.append(single_grp_samples[0])
            continue

        curr_train = []
        curr_test = []
        accum_train_count = 0

        for grp in group_keys:
            grp_samples = groups_dict[grp]
            if accum_train_count < target_train_count or len(curr_test) == 0:
                if accum_train_count < target_train_count:
                    curr_train.extend(grp_samples)
                    accum_train_count += len(grp_samples)
                else:
                    curr_test.extend(grp_samples)
            else:
                curr_test.extend(grp_samples)

        if len(curr_test) == 0 and len(curr_train) > 1:
            popped = curr_train.pop()
            curr_test.append(popped)

        train_samples.extend(curr_train)
        test_samples.extend(curr_test)

    random.shuffle(train_samples)
    random.shuffle(test_samples)
    return train_samples, test_samples


class MultimodalDataset(Dataset):
    def __init__(self, samples, processor, max_seq_length=256):
        self.samples = samples
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["image"]
        words = sample["words"]
        boxes = sample["boxes"]

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["label"] = torch.tensor(sample["label"], dtype=torch.long)
        return item


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Compute Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB")

    from transformers import AutoTokenizer, LayoutLMv3ImageProcessor
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
    image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
    processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(CLASS_NAMES)
    )
    model.to(device)

    all_samples = load_dataset(Path(args.data_dir), [Path(p) for p in args.auxiliary_dirs])
    if not all_samples:
        logger.error("No dataset samples found.")
        return

    train_samples, test_samples = group_stratified_split(all_samples, train_ratio=args.train_split, seed=42)

    train_loader = DataLoader(
        MultimodalDataset(train_samples, processor, max_seq_length=256),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        MultimodalDataset(test_samples, processor, max_seq_length=256),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Class weights for loss
    counts = Counter(s["label"] for s in train_samples)
    weights = [min(8.0, max(1.0, float(np.sqrt(len(train_samples) / (len(CLASS_NAMES) * max(1, counts.get(i, 1))))))) for i in range(len(CLASS_NAMES))]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_macro_f1 = 0.0
    best_state = None

    logger.info(f"Starting training for {args.epochs} epochs with FP16 on {device}...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**batch)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []

        with torch.inference_mode():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("label")
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        acc = test_correct / test_total if test_total else 0.0
        f1_scores = []
        for i in range(len(CLASS_NAMES)):
            tp = sum(1 for p, t in zip(all_preds, all_labels) if p == i and t == i)
            fp = sum(1 for p, t in zip(all_preds, all_labels) if p == i and t != i)
            fn = sum(1 for p, t in zip(all_preds, all_labels) if p != i and t == i)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if sum(1 for t in all_labels if t == i) > 0:
                f1_scores.append(f1)

        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        logger.info(f"Epoch {epoch+1:2d}/{args.epochs:2d} | Train Acc: {train_correct/train_total*100:.1f}% | Test Acc: {acc*100:.1f}% | Macro F1: {macro_f1*100:.1f}%")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.load_state_dict(best_state)
        model.save_pretrained(str(out_dir))
        processor.save_pretrained(str(out_dir))
        logger.info(f"✓ Best model saved to {out_dir} (Macro F1: {best_macro_f1*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 on Document Dataset")
    parser.add_argument("--data_dir", type=str, default="new_training_data", help="Primary dataset directory")
    parser.add_argument("--auxiliary_dirs", nargs="*", default=["data/demo", "data/originals"], help="Auxiliary balance datasets")
    parser.add_argument("--output_dir", type=str, default="models/layoutxlm", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--train_split", type=float, default=0.7, help="Train split ratio")
    args = parser.parse_args()
    train(args)