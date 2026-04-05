"""
bilstm.py
---------
Bidirectional LSTM with pre-trained GloVe embeddings (or random init fallback).
Processes concatenated query + product title as a sequence.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"BiLSTM using device: {DEVICE}")


# ── Tokenizer ──────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    """Word-level tokenizer with vocab built from training data."""

    def __init__(self, max_vocab: int = 30000, max_len: int = 64):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, texts):
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        most_common = counter.most_common(self.max_vocab - 2)
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        logger.info(f"Vocab size: {len(self.word2idx)}")

    def encode(self, text: str):
        tokens = text.split()[: self.max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        # Pad or truncate
        ids += [0] * (self.max_len - len(ids))
        return ids


# ── Dataset ────────────────────────────────────────────────────────────────────

class ESCIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer: SimpleTokenizer):
        self.encodings = [tokenizer.encode(t) for t in texts]
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── Model ──────────────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        pretrained_embeddings=None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(pretrained_embeddings)
            )

        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))              # (B, L, E)
        out, (hn, _) = self.bilstm(emb)                   # (B, L, 2H)
        # Concat last forward and backward hidden states
        hidden = torch.cat((hn[-2], hn[-1]), dim=1)        # (B, 2H)
        hidden = self.dropout(hidden)
        return self.fc(hidden)                             # (B, C)


# ── GloVe loader ──────────────────────────────────────────────────────────────

def load_glove(glove_path: str, word2idx: dict, embed_dim: int = 100):
    """
    Load GloVe embeddings for words in vocab.
    Falls back to random if file not found.
    """
    if not os.path.exists(glove_path):
        logger.warning(f"GloVe not found at {glove_path}. Using random embeddings.")
        return None

    logger.info(f"Loading GloVe from {glove_path}...")
    embeddings = np.random.uniform(-0.1, 0.1, (len(word2idx), embed_dim))
    embeddings[0] = 0  # PAD

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in word2idx:
                embeddings[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1

    logger.info(f"GloVe: {found}/{len(word2idx)} words found.")
    return embeddings


# ── Train / Predict ───────────────────────────────────────────────────────────

def train(
    train_texts,
    train_labels,
    val_texts=None,
    val_labels=None,
    glove_path: str = None,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_vocab: int = 30000,
    max_len: int = 64,
    embed_dim: int = 100,
    hidden_dim: int = 128,
):
    tokenizer = SimpleTokenizer(max_vocab=max_vocab, max_len=max_len)
    tokenizer.build_vocab(train_texts)

    pretrained = load_glove(glove_path, tokenizer.word2idx, embed_dim) if glove_path else None

    train_ds = ESCIDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = BiLSTMClassifier(
        vocab_size=len(tokenizer.word2idx),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=4,
        pretrained_embeddings=pretrained,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"BiLSTM Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.4f}")
        scheduler.step(avg_loss)

    return model, tokenizer


def predict(model, tokenizer, texts, batch_size: int = 64):
    model.eval()
    ds = ESCIDataset(texts, np.zeros(len(texts), dtype=int), tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
    return np.array(all_preds)


def save(model, tokenizer, model_path: str, tokenizer_path: str):
    torch.save(model.state_dict(), model_path)
    import joblib
    joblib.dump(tokenizer, tokenizer_path)
    logger.info(f"BiLSTM saved to {model_path}")


def load(model_path: str, tokenizer_path: str):
    import joblib
    tokenizer = joblib.load(tokenizer_path)
    model = BiLSTMClassifier(vocab_size=len(tokenizer.word2idx)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model, tokenizer
