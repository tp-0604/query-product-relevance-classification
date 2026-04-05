"""
naive_bayes.py
--------------
Multinomial Naive Bayes with TF-IDF features.
Baseline classical ML model.
"""

import logging
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger(__name__)


def train(X_train, y_train, alpha: float = 0.1):
    logger.info(f"Training Naive Bayes (alpha={alpha})...")
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    logger.info("Naive Bayes training complete.")
    return model


def predict(model, X_test):
    return model.predict(X_test)


def save(model, path: str):
    joblib.dump(model, path)
    logger.info(f"Naive Bayes model saved to {path}")


def load(path: str):
    return joblib.load(path)
