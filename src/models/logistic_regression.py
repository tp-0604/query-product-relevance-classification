"""
logistic_regression.py
----------------------
Logistic Regression with TF-IDF features.
Strong classical ML baseline — often competitive with deep models on short text.
"""

import logging
from sklearn.linear_model import LogisticRegression
import joblib

logger = logging.getLogger(__name__)


def train(X_train, y_train, C: float = 1.0, max_iter: int = 1000):
    logger.info(f"Training Logistic Regression (C={C}, max_iter={max_iter})...")
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return model


def predict(model, X_test):
    return model.predict(X_test)


def save(model, path: str):
    joblib.dump(model, path)
    logger.info(f"Logistic Regression model saved to {path}")


def load(path: str):
    return joblib.load(path)
