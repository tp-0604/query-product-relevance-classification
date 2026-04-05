"""
distilbert.py
-------------
Fine-tuned DistilBERT for query-product relevance classification.
Input format: [CLS] query [SEP] product_title [SEP]
The [SEP] token explicitly marks the boundary between query and product,
allowing the model to learn cross-sequence relationships.
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilbert-base-uncased"
logger.info(f"DistilBERT using device: {DEVICE}")


# ── Dataset ────────────────────────────────────────────────────────────────────

class ESCIPairDataset(Dataset):
    """
    Encodes (query, product_title) as a sentence pair using [SEP].
    """

    def __init__(self, queries, titles, labels, tokenizer, max_len: int = 128):
        self.encodings = tokenizer(
            list(queries),
            list(titles),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ── Train ──────────────────────────────────────────────────────────────────────

def train(
    train_queries,
    train_titles,
    train_labels,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 128,
    warmup_ratio: float = 0.1,
):
    logger.info("Loading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4
    ).to(DEVICE)

    train_ds = ESCIPairDataset(train_queries, train_titles, train_labels, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"DistilBERT Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        acc = correct / total
        logger.info(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, acc={acc:.4f}")

    return model, tokenizer


# ── Predict ────────────────────────────────────────────────────────────────────

def predict(model, tokenizer, queries, titles, batch_size: int = 64, max_len: int = 128):
    model.eval()
    dummy_labels = np.zeros(len(queries), dtype=int)
    ds = ESCIPairDataset(queries, titles, dummy_labels, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())

    return np.array(all_preds)


# ── Save / Load ────────────────────────────────────────────────────────────────

def save(model, tokenizer, save_dir: str):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"DistilBERT saved to {save_dir}")


def load(save_dir: str):
    tokenizer = DistilBertTokenizerFast.from_pretrained(save_dir)
    model = DistilBertForSequenceClassification.from_pretrained(save_dir).to(DEVICE)
    return model, tokenizer
