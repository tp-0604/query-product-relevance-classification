"""
preprocess.py
-------------
Data loading, cleaning, tokenization, stop-word removal,
normalization, and feature extraction for the ESCI dataset.
"""

import re
import string
import logging
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

LABEL_MAP = {"E": 0, "S": 1, "C": 2, "I": 3}
LABEL_NAMES = ["Exact", "Substitute", "Complement", "Irrelevant"]


def load_data(examples_path: str, products_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load and join examples + products parquet files.
    Filter to English (us locale) only.
    Optionally sample for faster experimentation.
    """
    logger.info("Loading examples parquet...")
    examples = pd.read_parquet(examples_path)

    logger.info("Loading products parquet...")
    products = pd.read_parquet(products_path)

    logger.info("Merging on product_id...")
    df = examples.merge(
        products[["product_id", "product_title"]],
        on="product_id",
        how="left"
    )

    # Filter to English only
    df = df[df["product_locale"] == "us"].reset_index(drop=True)
    logger.info(f"Total English samples: {len(df)}")

    # Drop rows with missing titles or queries
    df = df.dropna(subset=["query", "product_title", "esci_label"])
    logger.info(f"After dropping nulls: {len(df)}")

    if sample_size:
        # Stratified sample to preserve class balance
        df = (
            df.groupby("esci_label", group_keys=False)
              .apply(lambda x: x.sample(min(len(x), sample_size // 4), random_state=42))
              .reset_index(drop=True)
        )
        logger.info(f"Sampled to: {len(df)}")

    return df


def clean_text(text: str) -> str:
    """
    Lowercase, remove HTML tags, punctuation, extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # remove HTML
    text = re.sub(r"http\S+|www\S+", " ", text)    # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # remove punctuation/special chars
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


def remove_stopwords(text: str, stop_words: set) -> str:
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in stop_words)


def preprocess_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Clean query and product_title
    2. Remove stopwords
    3. Concatenate as 'text' for classical ML
    4. Keep separate 'query_clean' and 'title_clean' for deep models
    """
    logger.info("Cleaning texts...")
    stop_words = set(stopwords.words("english"))

    df = df.copy()
    df["query_clean"] = df["query"].apply(clean_text).apply(
        lambda x: remove_stopwords(x, stop_words)
    )
    df["title_clean"] = df["product_title"].apply(clean_text).apply(
        lambda x: remove_stopwords(x, stop_words)
    )

    # Combined text for classical ML (query + title)
    df["text"] = df["query_clean"] + " " + df["title_clean"]

    # Encode labels
    df["label"] = df["esci_label"].map(LABEL_MAP)

    logger.info("Preprocessing complete.")
    logger.info(f"Label distribution:\n{df['esci_label'].value_counts()}")

    return df


def get_tfidf_features(
    train_texts, test_texts,
    max_features: int = 50000,
    ngram_range: tuple = (1, 2)
):
    """
    Fit TF-IDF on train, transform train and test.
    Returns sparse matrices X_train, X_test and fitted vectorizer.
    """
    logger.info(f"Fitting TF-IDF (max_features={max_features}, ngrams={ngram_range})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,       # log normalization
        min_df=2,
        strip_accents="unicode"
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    logger.info(f"TF-IDF shape: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, vectorizer


def get_train_test_split(df: pd.DataFrame):
    """
    Use the built-in split column if present, else 80/20.
    """
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)
        logger.info(f"Using dataset split: train={len(train_df)}, test={len(test_df)}")
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
        logger.info(f"Random split: train={len(train_df)}, test={len(test_df)}")

    return train_df, test_df
