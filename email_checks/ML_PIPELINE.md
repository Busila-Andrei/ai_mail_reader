# ML Pipeline Guide

Complete guide for running the machine learning pipeline from email extraction to model training.

## 📋 Pipeline Overview

```
1. Extract Emails → 2. Profile Data → 3. Extract Features → 4. Prepare Dataset → 5. Train Models
```

## 🚀 Quick Start

Run the complete pipeline:

```bash
# Step 1: Extract emails
python extract_emails.py

# Step 2: Profile the data
python profile_data.py

# Step 3: Extract ML features
python extract_features.py

# Step 4: Prepare dataset (train/val/test split)
python prepare_dataset.py

# Step 5: Train baseline models
python baseline_models.py
```

## 📝 Detailed Steps

### Step 1: Extract Emails

Extracts emails from Outlook and saves raw data.

```bash
python extract_emails.py
# Or with custom settings:
python extract_emails.py --count 200 --output my_data/raw
```

**Output:** `data/raw/emails_YYYYMMDD_HHMMSS.json`

---

### Step 2: Profile Data

Analyzes emails and extracts basic features.

```bash
python profile_data.py
```

**Output:** 
- `data/processed/emails_profiled_YYYYMMDD_HHMMSS.json`
- `data/processed/statistics_YYYYMMDD_HHMMSS.json`

---

### Step 3: Extract ML Features

Creates ML-ready feature vectors using:
- **TF-IDF** for text features
- **One-hot encoding** for categorical features
- **Cyclical encoding** for temporal features
- **Normalization** for numerical features

```bash
python extract_features.py
```

**Output:**
- `data/features/text_features.pkl` - TF-IDF matrix and vectorizer
- `data/features/non_text_features.parquet` - Structured features
- `data/features/feature_info.json` - Feature metadata

**Configuration:**
Edit `config.json` → `feature_extraction` section:
- `text_features`: TF-IDF settings (max_features, ngrams, etc.)
- `structural_features`: Categorical encoding, normalization
- `temporal_features`: Cyclical encoding settings
- `metadata_features`: Metadata normalization

---

### Step 4: Prepare Dataset

Creates train/validation/test splits, balances classes, and exports to multiple formats.

```bash
python prepare_dataset.py
```

**Output:**
- `data/datasets/X_train.npz`, `y_train.npy` - Training set
- `data/datasets/X_val.npz`, `y_val.npy` - Validation set
- `data/datasets/X_test.npz`, `y_test.npy` - Test set
- `data/datasets/dataset_metadata.json` - Dataset info

**Features:**
- Automatic train/val/test split (default: 70/10/20)
- Class balancing using SMOTE (optional)
- Multiple export formats (NPZ, Parquet, NumPy arrays)

**Configuration:**
Edit `config.json` → `dataset` section:
- `labels`: Label creation strategy (importance, unread, flagged, etc.)
- `split`: Train/val/test proportions
- `balance`: Class balancing method (SMOTE, undersampling, or none)
- `export`: Export formats

---

### Step 5: Train Baseline Models

Trains baseline models for:
- **Classification**: Spam, category, priority prediction
- **Clustering**: Email groups, topic discovery
- **NLP**: Topic modeling (LDA, NMF)
- **Anomaly Detection**: Unusual email patterns

```bash
# Train all models
python baseline_models.py

# Train specific task
python baseline_models.py --task classification
python baseline_models.py --task clustering
python baseline_models.py --task nlp
python baseline_models.py --task anomaly
```

**Output:**
- `data/models/classification/` - Classification models
- `data/models/clustering/` - Clustering models
- `data/models/nlp/` - NLP models
- `data/models/anomaly/` - Anomaly detection models
- `data/models/model_results.json` - Performance metrics

**Models Trained:**

**Classification:**
- Logistic Regression
- Random Forest
- SVM (if dataset < 10k samples)

**Clustering:**
- K-Means
- DBSCAN

**NLP:**
- LDA Topic Modeling
- NMF Topic Modeling

**Anomaly Detection:**
- Isolation Forest

---

## 🔧 Configuration

All settings are in `config.json`. Key sections:

### Feature Extraction

```json
"feature_extraction": {
  "text_features": {
    "tfidf_max_features": 1000,  // Number of TF-IDF features
    "use_char_ngrams": false,     // Enable character n-grams
    "use_dimensionality_reduction": false  // Use SVD reduction
  },
  "structural_features": {
    "categorical_columns": ["sender_domain"],
    "max_categories_per_column": 20,
    "normalization_method": "standard"  // or "minmax"
  }
}
```

### Dataset Preparation

```json
"dataset": {
  "labels": {
    "type": "importance",  // "importance", "unread", "flagged", "custom_column"
    "threshold": 2
  },
  "split": {
    "test_size": 0.2,
    "val_size": 0.1
  },
  "balance": {
    "enabled": true,
    "method": "smote"  // "smote" or "undersample"
  }
}
```

### Models

```json
"models": {
  "clustering": {
    "n_clusters": 5
  },
  "nlp": {
    "n_topics": 5
  },
  "anomaly_detection": {
    "contamination": 0.1  // Expected anomaly rate
  }
}
```

---

## 📊 Using the Models

### Load a Trained Model

```python
import pickle
import numpy as np
from scipy.sparse import load_npz

# Load model
with open('data/models/classification/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test data
X_test = load_npz('data/datasets/X_test.npz')
y_test = np.load('data/datasets/y_test.npy')

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Load Dataset

```python
import numpy as np
from scipy.sparse import load_npz
import pandas as pd

# Load sparse matrices
X_train = load_npz('data/datasets/X_train.npz')
y_train = np.load('data/datasets/y_train.npy')

# Or load as DataFrame (if exported)
df = pd.read_parquet('data/datasets/train_set.parquet')
```

### Feature Information

```python
import json

with open('data/features/feature_info.json', 'r') as f:
    feature_info = json.load(f)

print(f"Total features: {feature_info['feature_count']}")
print(f"TF-IDF features: {feature_info['text_features']['tfidf_features']}")
print(f"Non-text features: {len(feature_info['non_text_features']['columns'])}")
```

---

## 📈 Model Results

Check model performance:

```python
import json

with open('data/models/model_results.json', 'r') as f:
    results = json.load(f)

# Classification results
print("Classification Results:")
for model_name, metrics in results['classification'].items():
    print(f"  {model_name}: Accuracy = {metrics['accuracy']:.4f}")

# Clustering results
print("\nClustering Results:")
for model_name, metrics in results['clustering'].items():
    if 'silhouette_score' in metrics:
        print(f"  {model_name}: Silhouette = {metrics['silhouette_score']:.4f}")

# Anomaly detection results
print("\nAnomaly Detection Results:")
for model_name, metrics in results['anomaly'].items():
    print(f"  {model_name}: {metrics['n_anomalies']} anomalies detected")
```

---

## 🔍 Troubleshooting

### Issue: "No module named 'sklearn'"

**Solution:** Install required packages:
```bash
pip install scikit-learn imbalanced-learn pyarrow
```

### Issue: Memory error with large datasets

**Solutions:**
1. Reduce `tfidf_max_features` in config
2. Enable `use_dimensionality_reduction` in config
3. Process smaller batches

### Issue: Class imbalance warnings

**Solution:** Enable class balancing in config:
```json
"balance": {
  "enabled": true,
  "method": "smote"
}
```

### Issue: NLP models fail

**Solution:** Ensure profiled data exists:
```bash
python profile_data.py  # Run this first
```

---

## 📚 Next Steps

After training baseline models:

1. **Evaluate Performance**: Review `model_results.json`
2. **Feature Engineering**: Add domain-specific features
3. **Hyperparameter Tuning**: Optimize model parameters
4. **Advanced Models**: Try deep learning (BERT, etc.)
5. **Deployment**: Create API or integration for predictions

---

## 📁 Directory Structure

```
email_checks/
├── data/
│   ├── raw/              # Raw extracted emails
│   ├── processed/        # Profiled emails with features
│   ├── features/         # ML-ready feature matrices
│   ├── datasets/         # Train/val/test splits
│   └── models/          # Trained models
├── extract_emails.py
├── profile_data.py
├── extract_features.py   # Step 3
├── prepare_dataset.py    # Step 4
├── baseline_models.py    # Step 5
├── ml_utils.py          # ML utility functions
└── config.json          # Configuration
```

---

## 🎯 Example Workflow

```bash
# Complete pipeline
python extract_emails.py --count 500
python profile_data.py
python extract_features.py
python prepare_dataset.py
python baseline_models.py --task all

# Check results
python -c "import json; print(json.load(open('data/models/model_results.json')))"
```

---

**You're ready to build ML models on your email data!** 🚀

