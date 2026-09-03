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
from transformers import (
    AutoTokenizer,
    LayoutLMv3Config,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
    LayoutLMv3ImageProcessor,
    get_linear_schedule_with_warmup,
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
    "proforma": "PROFORMA_O",
    "proforma_o": "PROFORMA_O",
    "ncl_certificate": "UNKNOWN_OUT_OF_SCOPE",
    "unknown_out_of_scope": "UNKNOWN_OUT_OF_SCOPE",
    "CASTE_CERTIFICATE": "CASTE_CERTIFICATE",
    "CASTE_VALIDITY_CERTIFICATE": "CASTE_VALIDITY_CERTIFICATE",
    "CASTE_VALIDITY_RECEIPT": "CASTE_VALIDITY_RECEIPT",
    "LEAVING_CERTIFICATE": "LEAVING_CERTIFICATE",
    "PROFORMA_O": "PROFORMA_O",
    "UNKNOWN_OUT_OF_SCOPE": "UNKNOWN_OUT_OF_SCOPE",
}


def extract_document_pages(file_path: Path) -> list[dict]:
    """
    Extract high-res page image, text words, and 0-1000 normalized bounding boxes.
    """
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


def load_dataset(primary_dir: Path, auxiliary_dirs: list[Path] = None, max_unknown_per_category: int = 4) -> list[dict]:
    all_samples = []
    logger.info(f"Loading primary dataset from {primary_dir}...")

    for item in sorted(primary_dir.iterdir()):
        if not item.is_dir():
            continue

        if item.name == "sorted":
            # Extract balanced out-of-scope samples from sorted subdirectories
            for sub_cat in sorted(item.iterdir()):
                if sub_cat.is_dir():
                    cat_files = [f for f in sub_cat.iterdir() if f.is_file() and f.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png"]]
                    random.seed(42)
                    selected_files = random.sample(cat_files, min(len(cat_files), max_unknown_per_category))
                    for f in selected_files:
                        doc_samples = extract_document_pages(f)
                        for s in doc_samples:
                            s["label"] = CLASS_MAP["UNKNOWN_OUT_OF_SCOPE"]
                            s["class_name"] = "UNKNOWN_OUT_OF_SCOPE"
                            all_samples.append(s)
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

    # Auxiliary datasets to balance small classes (e.g. CASTE_VALIDITY_RECEIPT)
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
                if current_count < 25:
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

    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
    image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
    processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)
    
    # Configure label mapping in config
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(CLASS_NAMES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    all_samples = load_dataset(Path(args.data_dir), [Path(p) for p in args.auxiliary_dirs])
    if not all_samples:
        logger.error("No dataset samples found.")
        return

    train_samples, test_samples = group_stratified_split(all_samples, train_ratio=args.train_split, seed=42)
    logger.info(f"Split completed: Train={len(train_samples)} samples, Test={len(test_samples)} samples")

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
    weights = [min(6.0, max(1.0, float(np.sqrt(len(train_samples) / (len(CLASS_NAMES) * max(1, counts.get(i, 1))))))) for i in range(len(CLASS_NAMES))]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))

    # Differential learning rates for backbone vs newly initialized classification head
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n]

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": args.learning_rate, "weight_decay": args.weight_decay},
        {"params": classifier_params, "lr": args.learning_rate * 5.0, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_macro_f1 = 0.0
    best_state = None
    best_metrics = {}

    logger.info(f"Starting training for {args.epochs} epochs with FP16 (warmup={warmup_steps} steps, total={total_steps} steps) on {device}...")
    start_time = time.time()

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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

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
        class_details = {}
        for i, name in enumerate(CLASS_NAMES):
            tp = sum(1 for p, t in zip(all_preds, all_labels) if p == i and t == i)
            fp = sum(1 for p, t in zip(all_preds, all_labels) if p == i and t != i)
            fn = sum(1 for p, t in zip(all_preds, all_labels) if p != i and t == i)
            support = sum(1 for t in all_labels if t == i)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if support > 0:
                f1_scores.append(f1)
            class_details[name] = {"precision": prec, "recall": rec, "f1": f1, "support": support}

        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        logger.info(f"Epoch {epoch+1:2d}/{args.epochs:2d} | Train Acc: {train_correct/train_total*100:5.1f}% | Val Acc: {acc*100:5.1f}% | Macro F1: {macro_f1*100:5.1f}% | Loss: {train_loss/train_total:.4f}")

        if macro_f1 >= best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch + 1,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "class_details": class_details,
            }

    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}\nTRAINING COMPLETE in {total_time:.1f}s\n{'='*60}")
    logger.info(f"Best Validation Macro F1: {best_metrics.get('macro_f1', 0)*100:.2f}% at Epoch {best_metrics.get('epoch', 1)}")
    logger.info("Per-Class Performance on Validation Set:")
    for name, m in best_metrics.get("class_details", {}).items():
        logger.info(f"  {name:30s} | Prec: {m['precision']*100:5.1f}% | Rec: {m['recall']*100:5.1f}% | F1: {m['f1']*100:5.1f}% | Samples: {m['support']}")

    if best_state:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.load_state_dict(best_state)
        model.save_pretrained(str(out_dir))
        processor.save_pretrained(str(out_dir))
        logger.info(f"\n✓ Best model successfully saved to {out_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 on Document Dataset")
    parser.add_argument("--data_dir", type=str, default="../DATA", help="Primary dataset directory")
    parser.add_argument("--auxiliary_dirs", nargs="*", default=["data/demo", "data/originals"], help="Auxiliary balance datasets")
    parser.add_argument("--output_dir", type=str, default="models/layoutxlm", help="Output directory")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--train_split", type=float, default=0.7, help="Train split ratio")
    args = parser.parse_args()
    train(args)
