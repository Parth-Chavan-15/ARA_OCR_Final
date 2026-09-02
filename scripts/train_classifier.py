import os
import argparse
import logging
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMv2ForSequenceClassification, LayoutXLMProcessor
from torch.optim import AdamW
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "CASTE_CERTIFICATE",
    "CASTE_VALIDITY_CERTIFICATE",
    "CASTE_VALIDITY_RECEIPT",
    "PROFORMA_O",
    "LEAVING_CERTIFICATE",
    "UNKNOWN_OUT_OF_SCOPE"
]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

def extract_ocr_if_needed(img_path: Path) -> Path:
    """
    Ensure OCR JSON exists for image. If not, runs PaddleOCR and caches JSON.
    """
    ocr_path = img_path.with_suffix(".json")
    if ocr_path.exists():
        return ocr_path

    try:
        from paddleocr import PaddleOCR
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
        img_np = np.array(Image.open(img_path).convert('RGB'))
        h, w, _ = img_np.shape
        
        result = ocr_engine.ocr(str(img_path), cls=True)
        words = []
        
        if result and result[0]:
            for line in result[0]:
                box = line[0]  # [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                text = line[1][0]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                
                # Normalize box to 0-1000 for LayoutXLM
                norm_box = [
                    int(max(0, min(1000, (min(xs) / w) * 1000))),
                    int(max(0, min(1000, (min(ys) / h) * 1000))),
                    int(max(0, min(1000, (max(xs) / w) * 1000))),
                    int(max(0, min(1000, (max(ys) / h) * 1000))),
                ]
                words.append({"text": text, "box": norm_box})
                
        with open(ocr_path, "w", encoding="utf-8") as f:
            json.dump({"words": words}, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Generated OCR cache: {ocr_path}")
        return ocr_path
    except Exception as e:
        logger.warning(f"Failed OCR generation for {img_path}: {e}")
        with open(ocr_path, "w", encoding="utf-8") as f:
            json.dump({"words": [{"text": "empty", "box": [0, 0, 1000, 1000]}]}, f)
        return ocr_path

def convert_pdf_to_images(pdf_path: Path) -> list[Path]:
    """Convert a PDF to page images if PDF is provided."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        img_paths = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=200)
            out_img = pdf_path.parent / f"{pdf_path.stem}_page_{i+1}.jpg"
            if not out_img.exists():
                pix.save(str(out_img))
            img_paths.append(out_img)
        doc.close()
        return img_paths
    except Exception as e:
        logger.error(f"Error converting PDF {pdf_path}: {e}")
        return []

class DocumentClassificationDataset(Dataset):
    def __init__(self, data_dir, processor, max_seq_length=512):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.samples = []
        
        logger.info(f"Loading data from {self.data_dir}")
        for class_name in CLASS_NAMES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            # Handle PDFs first
            for pdf_path in class_dir.glob("*.pdf"):
                convert_pdf_to_images(pdf_path)

            # Gather all images
            img_extensions = ["*.jpg", "*.jpeg", "*.png"]
            class_images = []
            for ext in img_extensions:
                class_images.extend(list(class_dir.glob(ext)))
                
            for img_path in class_images:
                ocr_path = extract_ocr_if_needed(img_path)
                self.samples.append({
                    "image_path": img_path,
                    "ocr_path": ocr_path,
                    "label": CLASS_MAP[class_name]
                })
        logger.info(f"Loaded total {len(self.samples)} samples across {len(CLASS_NAMES)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        
        with open(sample["ocr_path"], "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
            
        words = []
        boxes = []
        for item in ocr_data.get("words", []):
            words.append(item["text"])
            boxes.append(item["box"])
            
        if not words:
            words = ["empty"]
            boxes = [[0, 0, 1000, 1000]]
            
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["label"] = torch.tensor(sample["label"])
        return encoding

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} (train/val split: {int(args.train_split*100)}/{int((1-args.train_split)*100)})")
    
    logger.info("Loading LayoutXLM processor and model...")
    processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
    model = LayoutLMv2ForSequenceClassification.from_pretrained(
        "microsoft/layoutxlm-base", 
        num_labels=len(CLASS_NAMES)
    )
    model.to(device)
    
    dataset = DocumentClassificationDataset(args.data_dir, processor)
    if len(dataset) == 0:
        logger.error("No training data found. Cannot proceed with training.")
        return
        
    # 70-30 train-test (validation) split by default
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Dataset split: {train_size} training samples (70%), {val_size} testing/validation samples (30%)")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    logger.info("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # Validation / Testing
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.item()
                
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["label"]).sum().item()
                total += batch["label"].size(0)
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = (correct / total) if total > 0 else 0.0
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} Summary:")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Test/Val Acc: {val_acc*100:.2f}% ({correct}/{total})")
        
        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_path = Path(args.output_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            logger.info(f"✓ Best model saved to {save_path} (Acc: {val_acc*100:.2f}%)")

    logger.info("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LayoutXLM for document classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training data (images + OCR json)")
    parser.add_argument("--output_dir", type=str, default="models/layoutxlm", help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--train_split", type=float, default=0.7, help="Train/test split ratio (default: 0.7 for 70-30 split)")
    
    args = parser.parse_args()
    train(args)
