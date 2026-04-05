"""
main.py
-------
Full pipeline runner for ESCI Query-Product Relevance Classification.

Usage:
    python main.py                        # all models, full dataset
    python main.py --model nb             # single model
    python main.py --sample 20000         # limit dataset size
    python main.py --model bilstm --epochs 3
"""

import argparse
import logging
import os
import sys
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

# Paths
EXAMPLES_PATH = "data/shopping_queries_dataset_examples.parquet"
PRODUCTS_PATH = "data/shopping_queries_dataset_products.parquet"
MODELS_DIR = "outputs/models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="ESCI Query-Product Relevance Classification")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "nb", "lr", "svm", "bilstm", "distilbert"],
        help="Which model to run"
    )
    parser.add_argument("--sample", type=int, default=None, help="Sample N rows (stratified)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for BiLSTM / DistilBERT")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for deep models")
    parser.add_argument("--glove_path", type=str, default=None, help="Path to GloVe .txt file")
    parser.add_argument(
        "--distilbert_epochs", type=int, default=3, help="Epochs for DistilBERT fine-tuning"
    )
    return parser.parse_args()


def load_and_preprocess(sample_size):
    from src.preprocessing.preprocess import load_data, preprocess_texts, get_train_test_split, get_tfidf_features

    df = load_data(EXAMPLES_PATH, PRODUCTS_PATH, sample_size=sample_size)
    df = preprocess_texts(df)
    train_df, test_df = get_train_test_split(df)

    logger.info("Building TF-IDF features...")
    X_train_tfidf, X_test_tfidf, vectorizer = get_tfidf_features(
        train_df["text"], test_df["text"]
    )

    return train_df, test_df, X_train_tfidf, X_test_tfidf


# ── Model runners ──────────────────────────────────────────────────────────────

def run_naive_bayes(train_df, test_df, X_train, X_test):
    from src.models import naive_bayes
    from src.evaluation.evaluate import full_evaluation

    model = naive_bayes.train(X_train, train_df["label"].values)
    y_pred = naive_bayes.predict(model, X_test)
    naive_bayes.save(model, os.path.join(MODELS_DIR, "naive_bayes.joblib"))
    return full_evaluation(test_df["label"].values, y_pred, "Naive_Bayes")


def run_logistic_regression(train_df, test_df, X_train, X_test):
    from src.models import logistic_regression
    from src.evaluation.evaluate import full_evaluation

    model = logistic_regression.train(X_train, train_df["label"].values)
    y_pred = logistic_regression.predict(model, X_test)
    logistic_regression.save(model, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    return full_evaluation(test_df["label"].values, y_pred, "Logistic_Regression")


def run_svm(train_df, test_df, X_train, X_test):
    from src.models import svm
    from src.evaluation.evaluate import full_evaluation

    model = svm.train(X_train, train_df["label"].values)
    y_pred = svm.predict(model, X_test)
    svm.save(model, os.path.join(MODELS_DIR, "svm.joblib"))
    return full_evaluation(test_df["label"].values, y_pred, "SVM")


def run_bilstm(train_df, test_df, epochs, batch_size, glove_path):
    from src.models import bilstm
    from src.evaluation.evaluate import full_evaluation

    model, tokenizer = bilstm.train(
        train_df["text"].tolist(),
        train_df["label"].values,
        epochs=epochs,
        batch_size=batch_size,
        glove_path=glove_path,
    )
    y_pred = bilstm.predict(model, tokenizer, test_df["text"].tolist())
    bilstm.save(
        model, tokenizer,
        os.path.join(MODELS_DIR, "bilstm.pt"),
        os.path.join(MODELS_DIR, "bilstm_tokenizer.joblib"),
    )
    return full_evaluation(test_df["label"].values, y_pred, "BiLSTM")


def run_distilbert(train_df, test_df, epochs, batch_size):
    from src.models import distilbert
    from src.evaluation.evaluate import full_evaluation

    model, tokenizer = distilbert.train(
        train_df["query_clean"].tolist(),
        train_df["title_clean"].tolist(),
        train_df["label"].values,
        epochs=epochs,
        batch_size=batch_size,
    )
    y_pred = distilbert.predict(
        model, tokenizer,
        test_df["query_clean"].tolist(),
        test_df["title_clean"].tolist(),
    )
    distilbert.save(model, tokenizer, os.path.join(MODELS_DIR, "distilbert"))
    return full_evaluation(test_df["label"].values, y_pred, "DistilBERT")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger.info(f"Starting pipeline | model={args.model} | sample={args.sample}")

    logger.info("Loading and preprocessing data...")
    train_df, test_df, X_train_tfidf, X_test_tfidf = load_and_preprocess(args.sample)

    all_metrics = []
    per_class_f1 = {}

    run_all = args.model == "all"

    if run_all or args.model == "nb":
        logger.info("--- Running Naive Bayes ---")
        m = run_naive_bayes(train_df, test_df, X_train_tfidf, X_test_tfidf)
        all_metrics.append(m)
        _store_per_class(test_df, train_df, "Naive_Bayes", per_class_f1, X_test_tfidf, "nb")

    if run_all or args.model == "lr":
        logger.info("--- Running Logistic Regression ---")
        m = run_logistic_regression(train_df, test_df, X_train_tfidf, X_test_tfidf)
        all_metrics.append(m)
        _store_per_class(test_df, train_df, "Logistic_Regression", per_class_f1, X_test_tfidf, "lr")

    if run_all or args.model == "svm":
        logger.info("--- Running SVM ---")
        m = run_svm(train_df, test_df, X_train_tfidf, X_test_tfidf)
        all_metrics.append(m)
        _store_per_class(test_df, train_df, "SVM", per_class_f1, X_test_tfidf, "svm")

    if run_all or args.model == "bilstm":
        logger.info("--- Running BiLSTM ---")
        m = run_bilstm(train_df, test_df, args.epochs, args.batch_size, args.glove_path)
        all_metrics.append(m)

    if run_all or args.model == "distilbert":
        logger.info("--- Running DistilBERT ---")
        m = run_distilbert(train_df, test_df, args.distilbert_epochs, args.batch_size)
        all_metrics.append(m)

    if len(all_metrics) > 1:
        from src.evaluation.evaluate import (
            plot_comparative_analysis,
            plot_f1_heatmap,
            save_comparative_summary,
        )
        logger.info("Generating comparative analysis plots...")
        plot_comparative_analysis(all_metrics)
        if per_class_f1:
            plot_f1_heatmap(per_class_f1)
        save_comparative_summary(all_metrics)

    logger.info("Pipeline complete. All outputs saved to outputs/")


def _store_per_class(test_df, train_df, model_name, per_class_f1, X_test, model_key):
    """Helper to collect per-class F1 for heatmap."""
    from sklearn.metrics import f1_score
    import joblib

    model_path = os.path.join(MODELS_DIR, f"{model_key if model_key != 'lr' else 'logistic_regression'}.joblib")
    # Re-use saved model to get predictions
    try:
        model = joblib.load(
            os.path.join(MODELS_DIR, {
                "nb": "naive_bayes.joblib",
                "lr": "logistic_regression.joblib",
                "svm": "svm.joblib",
            }[model_key])
        )
        y_pred = model.predict(X_test)
        f1s = f1_score(test_df["label"].values, y_pred, average=None, zero_division=0)
        # Pad to 4 classes if needed
        if len(f1s) < 4:
            f1s = np.pad(f1s, (0, 4 - len(f1s)))
        per_class_f1[model_name] = f1s.tolist()
    except Exception as e:
        logger.warning(f"Could not store per-class F1 for {model_name}: {e}")


if __name__ == "__main__":
    main()
