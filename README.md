# Query-Product Relevance Classification in Amazon Search
### NLP Case Study | MTech AI/ML | Symbiosis Institute of Technology

---

## Problem Statement
Given a customer search query and a product title from Amazon, classify the relevance as:
- **E** — Exact match
- **S** — Substitute
- **C** — Complement
- **I** — Irrelevant

Dataset: [Amazon ESCI Shopping Queries Dataset](https://github.com/amazon-science/esci-data)

---

## Project Structure
```
esci-query-classification/
├── data/                          # Place .parquet files here (not committed)
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py          # Cleaning, tokenization, TF-IDF, encoding
│   ├── models/
│   │   ├── naive_bayes.py
│   │   ├── logistic_regression.py
│   │   ├── svm.py
│   │   ├── bilstm.py
│   │   └── distilbert.py
│   └── evaluation/
│       └── evaluate.py            # Metrics, confusion matrix, plots
├── outputs/
│   ├── figures/                   # All plots (pushed to GitHub)
│   └── reports/                   # Classification reports as CSV
├── notebooks/
│   └── EDA.ipynb                  # Exploratory Data Analysis
├── main.py                        # Full pipeline runner
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <your-repo-url>
cd esci-query-classification
pip install -r requirements.txt
```

### Download Dataset
Download from [Amazon ESCI GitHub](https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset) and place files in `data/`:
```
data/
├── shopping_queries_dataset_examples.parquet
├── shopping_queries_dataset_products.parquet
```

---

## Run

```bash
# Run full pipeline (all 5 models)
python main.py

# Run specific model only
python main.py --model nb          # Naive Bayes
python main.py --model lr          # Logistic Regression
python main.py --model svm         # SVM
python main.py --model bilstm      # BiLSTM
python main.py --model distilbert  # DistilBERT

# Limit dataset size (for quick testing)
python main.py --sample 10000
```

---

## Models
| # | Model | Type |
|---|-------|------|
| 1 | Naive Bayes (TF-IDF) | Classical ML |
| 2 | Logistic Regression (TF-IDF) | Classical ML |
| 3 | SVM (TF-IDF) | Classical ML |
| 4 | BiLSTM + GloVe | Deep Learning |
| 5 | DistilBERT (fine-tuned) | Transformer |

---

## Outputs
All figures and reports are saved to `outputs/` and committed to GitHub:
- `outputs/figures/confusion_matrix_<model>.png`
- `outputs/figures/comparative_analysis.png`
- `outputs/figures/f1_per_class.png`
- `outputs/reports/classification_report_<model>.csv`
- `outputs/reports/comparative_summary.csv`
