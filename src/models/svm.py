"""
svm.py
------
Support Vector Machine with LinearSVC + TF-IDF features.
LinearSVC is faster than kernel SVM for high-dim sparse text data.
"""

import logging
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib

logger = logging.getLogger(__name__)


def train(X_train, y_train, C: float = 1.0, max_iter: int = 2000):
    logger.info(f"Training SVM / LinearSVC (C={C})...")
    base = LinearSVC(C=C, max_iter=max_iter, random_state=42)
    model = CalibratedClassifierCV(base, cv=3)
    model.fit(X_train, y_train)
    logger.info("SVM training complete.")
    return model


def predict(model, X_test):
    return model.predict(X_test)


def save(model, path: str):
    joblib.dump(model, path)
    logger.info(f"SVM model saved to {path}")


def load(path: str):
    return joblib.load(path)
