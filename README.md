

# Predictive Analytics with Synthetic Data

Privacy-friendly, end‑to‑end ML workflows using **synthetic datasets**.  
This repo demonstrates how to generate data, build baselines, tune models, and evaluate results with clear, reproducible pipelines.

##  Highlights
- Synthetic data generation for classification/regression (no real PII).
- Clean preprocessing: splitting, scaling, encoding, class balancing (SMOTE).
- Baselines + ensembles: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
- Robust evaluation: cross‑validation, hold‑out test, metrics (accuracy, precision/recall, F1, ROC‑AUC), learning curves.
- Reproducible runs with fixed random seeds and saved artifacts.

##  Project Structure
```

.
├─ data/                # auto-created; generated datasets live here
├─ notebooks/           # exploratory notebooks (EDA, modeling)
├─ src/
│  ├─ generate_data.py  # create synthetic datasets
│  ├─ preprocess.py     # transformers/pipelines
│  ├─ train.py          # training + tuning + evaluation
│  └─ utils.py          # shared helpers, metrics, plotting
├─ models/              # saved models (.joblib)
├─ reports/             # exported figures & metrics (.csv/.json)
├─ requirements.txt
└─ README.md

```

> If your repo layout differs, keep this README and adjust file paths accordingly.

##  Requirements
- Python 3.9+  
- Recommended packages:
```

numpy
pandas
scikit-learn
matplotlib
seaborn
imbalanced-learn
joblib
jupyter

````
Install with:
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

##  Quickstart

### Option A — Notebooks

Open the notebooks to explore EDA and models step by step:

```bash
jupyter notebook
```

### Option B — Scripts

1. **Generate synthetic data** (binary classification example):

```bash
python src/generate_data.py \
  --task classification \
  --n-samples 5000 \
  --n-features 20 \
  --n-informative 8 \
  --n-redundant 4 \
  --class-weight 0.6 \
  --noise 0.05 \
  --out data/synth_classification.csv \
  --seed 42
```

2. **Train models** with cross‑validation + grid search:

```bash
python src/train.py \
  --data data/synth_classification.csv \
  --target target \
  --models logreg dt rf gbdt \
  --cv 5 \
  --test-size 0.2 \
  --smote \
  --save-models models/ \
  --report reports/metrics.json
```

3. **Outputs**

* `reports/metrics.json` — CV and test metrics per model.
* `reports/plots/` — confusion matrix, ROC curves, learning curves.
* `models/*.joblib` — serialized models.

##  Configuration (optional)

You can add YAML/JSON config files for experiments (e.g., `configs/classification.yaml`) to version your hyperparameters and dataset settings.

##  Metrics (example snippet)

```json
{
  "RandomForest": {"cv_f1": 0.89, "test_f1": 0.88, "test_roc_auc": 0.93},
  "GradientBoosting": {"cv_f1": 0.90, "test_f1": 0.89, "test_roc_auc": 0.94}
}
```

##  Why Synthetic Data?

* No privacy risk while practicing ML.
* Full control over class imbalance, noise, and feature interactions.
* Great for demonstrating pipelines, tuning, and evaluation.

##  Reproducibility

* Fix seeds (`--seed` flags).
* Save models and metrics to `models/` and `reports/`.
* Record package versions with `pip freeze > requirements.txt`.

##  License

MIT — feel free to use and adapt.

##  Acknowledgements

Built with the Python data/ML stack: `pandas`, `NumPy`, `scikit-learn`, `matplotlib`, `imbalanced-learn`.

```

---
