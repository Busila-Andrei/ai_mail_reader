# Getting Started: Complete Guide to Using email_checks

This guide walks you through using the entire email_checks ML pipeline from start to finish.

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Outlook** installed and configured with a signed-in profile
3. **Dependencies** installed (see Step 1 below)

---

## 🚀 Step-by-Step Instructions

### **Step 1: Install Dependencies**

First, install the required Python packages:

```bash
cd email_checks
pip install -r requirements.txt
```

**Key packages you need:**
- `pywin32` - Outlook COM interface
- `python-dateutil` - Date parsing
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning
- `imbalanced-learn` - Class balancing
- `pyarrow` - Parquet file support

**Note:** If you're on an offline machine, use:
```bash
python prepare_packages.py  # On online machine first
python bootstrap_offline.py  # On offline machine
```

---

### **Step 2: Configure Settings (Optional)**

Edit `config.json` to customize behavior:

```json
{
  "extraction": {
    "email_count": 100,        // Change number of emails to extract
    "output_directory": "data/raw",
    "source_folder": "Inbox"   // Change folder if needed
  },
  "profiling": {
    "output_directory": "data/processed"
  },
  "feature_extraction": {
    "text_features": {
      "tfidf_max_features": 1000  // Adjust TF-IDF features
    }
  },
  "dataset": {
    "labels": {
      "type": "importance"     // Change label strategy
    }
  }
}
```

**Default settings work fine!** You can skip this step for now.

---

### **Step 3: Extract Emails from Outlook**

Extract emails from your Outlook inbox:

```bash
python extract_emails.py
```

**What happens:**
- Connects to Outlook
- Extracts 100 emails (default) from your Inbox
- Saves to `data/raw/emails_YYYYMMDD_HHMMSS.json`

**Customize:**
```bash
# Extract more emails
python extract_emails.py --count 500

# Different output directory
python extract_emails.py --output my_data/raw
```

**Time:** 1-5 minutes depending on email size

**Check output:**
```bash
ls data/raw/
# Should see: emails_20250101_120000.json
```

---

### **Step 4: Profile the Data**

Analyze extracted emails and extract basic features:

```bash
python profile_data.py
```

**What happens:**
- Loads the most recent extraction
- Analyzes each email (word counts, URLs, temporal patterns, etc.)
- Generates statistics
- Saves to `data/processed/emails_profiled_*.json` and `statistics_*.json`

**Output files:**
- `emails_profiled_*.json` - Emails with extracted features
- `statistics_*.json` - Summary statistics

**Time:** Less than 1 minute

**View statistics:**
```python
import json
with open('data/processed/statistics_*.json', 'r') as f:
    stats = json.load(f)
    print(stats)
```

---

### **Step 5: Extract ML Features**

Create ML-ready feature vectors:

```bash
python extract_features.py
```

**What happens:**
- Creates TF-IDF vectors from email text
- One-hot encodes categorical features (sender domain, etc.)
- Applies cyclical encoding to temporal features (hour, day)
- Normalizes numerical features
- Saves feature matrices

**Output files:**
- `data/features/text_features.pkl` - TF-IDF matrix and vectorizer
- `data/features/non_text_features.parquet` - Structured features
- `data/features/feature_info.json` - Feature metadata

**Time:** 1-3 minutes

**Check feature info:**
```python
import json
with open('data/features/feature_info.json', 'r') as f:
    info = json.load(f)
    print(f"Total features: {info['feature_count']}")
    print(f"TF-IDF features: {info['text_features']['tfidf_features']}")
```

---

### **Step 6: Prepare Dataset**

Create train/validation/test splits:

```bash
python prepare_dataset.py
```

**What happens:**
- Combines text and non-text features
- Splits data: 70% train, 10% validation, 20% test
- Balances classes using SMOTE (if enabled)
- Exports to multiple formats

**Output files:**
- `data/datasets/X_train.npz`, `y_train.npy` - Training set
- `data/datasets/X_val.npz`, `y_val.npy` - Validation set
- `data/datasets/X_test.npz`, `y_test.npy` - Test set
- `data/datasets/dataset_metadata.json` - Dataset info

**Time:** Less than 1 minute

**Check dataset:**
```python
import numpy as np
from scipy.sparse import load_npz

X_train = load_npz('data/datasets/X_train.npz')
y_train = np.load('data/datasets/y_train.npy')

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Label distribution: {np.bincount(y_train)}")
```

---

### **Step 7: Train Baseline Models**

Train ML models:

```bash
python baseline_models.py
```

**What happens:**
- Trains classification models (Logistic Regression, Random Forest, SVM)
- Trains clustering models (K-Means, DBSCAN)
- Trains NLP models (LDA, NMF topic modeling)
- Trains anomaly detection (Isolation Forest)
- Saves all models and performance metrics

**Output files:**
- `data/models/classification/` - Classification models
- `data/models/clustering/` - Clustering models
- `data/models/nlp/` - NLP models
- `data/models/anomaly/` - Anomaly detection models
- `data/models/model_results.json` - Performance metrics

**Time:** 2-10 minutes depending on dataset size

**Train specific models:**
```bash
# Only classification
python baseline_models.py --task classification

# Only clustering
python baseline_models.py --task clustering

# Only NLP
python baseline_models.py --task nlp

# Only anomaly detection
python baseline_models.py --task anomaly
```

**View results:**
```python
import json
with open('data/models/model_results.json', 'r') as f:
    results = json.load(f)
    
# Classification results
for model, metrics in results['classification'].items():
    print(f"{model}: Accuracy = {metrics['accuracy']:.4f}")
```

---

## 📊 Complete Pipeline Example

Run everything in sequence:

```bash
# Step 1: Extract emails
python extract_emails.py --count 100

# Step 2: Profile data
python profile_data.py

# Step 3: Extract ML features
python extract_features.py

# Step 4: Prepare dataset
python prepare_dataset.py

# Step 5: Train models
python baseline_models.py --task all
```

**Total time:** ~5-15 minutes for 100 emails

---

## 🔍 Using Your Trained Models

### Load and Use a Classification Model

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

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Check accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy:.4f}")
```

### Load and Use a Clustering Model

```python
import pickle
import numpy as np
from scipy.sparse import load_npz

# Load model
with open('data/models/clustering/kmeans.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data
X_train = load_npz('data/datasets/X_train.npz')

# Convert to dense (clustering needs dense matrices)
X_dense = X_train.toarray()

# Predict clusters
clusters = model.predict(X_dense)
print(f"Found {len(set(clusters))} clusters")
print(f"Cluster sizes: {np.bincount(clusters)}")
```

### Load and Use NLP Models

```python
import pickle

# Load LDA model
with open('data/models/nlp/lda.pkl', 'rb') as f:
    lda = pickle.load(f)

# Load vectorizer
with open('data/models/nlp/lda_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Process new text
text = "Your email text here"
doc_term = vectorizer.transform([text])
topics = lda.transform(doc_term)

# Get top topics
top_topic = topics[0].argmax()
print(f"Top topic: {top_topic}")
```

---

## 📁 Understanding Your Data Structure

After running all steps, your directory will look like:

```
email_checks/
├── data/
│   ├── raw/                    # Step 3 output
│   │   └── emails_*.json      # Raw extracted emails
│   │
│   ├── processed/              # Step 4 output
│   │   ├── emails_profiled_*.json    # Emails with features
│   │   └── statistics_*.json          # Summary stats
│   │
│   ├── features/               # Step 5 output
│   │   ├── text_features.pkl          # TF-IDF matrix
│   │   ├── non_text_features.parquet  # Structured features
│   │   └── feature_info.json           # Feature metadata
│   │
│   ├── datasets/                # Step 6 output
│   │   ├── X_train.npz, y_train.npy    # Training set
│   │   ├── X_val.npz, y_val.npy        # Validation set
│   │   ├── X_test.npz, y_test.npy      # Test set
│   │   └── dataset_metadata.json       # Dataset info
│   │
│   └── models/                 # Step 7 output
│       ├── classification/              # Classification models
│       ├── clustering/                  # Clustering models
│       ├── nlp/                         # NLP models
│       ├── anomaly/                     # Anomaly models
│       └── model_results.json           # Performance metrics
│
├── extract_emails.py           # Step 3 script
├── profile_data.py             # Step 4 script
├── extract_features.py         # Step 5 script
├── prepare_dataset.py          # Step 6 script
├── baseline_models.py         # Step 7 script
└── config.json                 # Configuration
```

---

## 🎯 Common Use Cases

### Use Case 1: Email Classification (Spam/Not Spam)

1. **Label your emails** (manually or use existing metadata):
   ```python
   # In prepare_dataset.py config, change labels:
   "labels": {
     "type": "custom_column",
     "column_name": "your_spam_column",
     "threshold": 1
   }
   ```

2. **Train models:**
   ```bash
   python prepare_dataset.py
   python baseline_models.py --task classification
   ```

3. **Use the model:**
   ```python
   # Load and predict
   model = pickle.load(open('data/models/classification/random_forest.pkl', 'rb'))
   prediction = model.predict(new_email_features)
   ```

### Use Case 2: Email Clustering (Find Email Groups)

1. **Train clustering:**
   ```bash
   python baseline_models.py --task clustering
   ```

2. **Analyze clusters:**
   ```python
   # Load model and data
   kmeans = pickle.load(open('data/models/clustering/kmeans.pkl', 'rb'))
   X = load_npz('data/datasets/X_train.npz').toarray()
   
   # Get clusters
   clusters = kmeans.predict(X)
   
   # See which emails belong to which cluster
   for i, cluster_id in enumerate(clusters):
       print(f"Email {i} → Cluster {cluster_id}")
   ```

### Use Case 3: Topic Modeling (Discover Topics)

1. **Train NLP models:**
   ```bash
   python baseline_models.py --task nlp
   ```

2. **View topics:**
   ```python
   import json
   with open('data/models/model_results.json', 'r') as f:
       results = json.load(f)
   
   # View LDA topics
   for topic in results['nlp']['lda']['topics']:
       print(f"Topic {topic['topic_id']}: {', '.join(topic['top_words'][:5])}")
   ```

### Use Case 4: Anomaly Detection (Find Unusual Emails)

1. **Train anomaly detection:**
   ```bash
   python baseline_models.py --task anomaly
   ```

2. **Find anomalies:**
   ```python
   iso_forest = pickle.load(open('data/models/anomaly/isolation_forest.pkl', 'rb'))
   X = load_npz('data/datasets/X_train.npz').toarray()
   
   predictions = iso_forest.predict(X)
   anomalies = X[predictions == -1]
   print(f"Found {len(anomalies)} anomalous emails")
   ```

---

## 🔧 Troubleshooting

### Issue: "No module named 'sklearn'"

**Solution:**
```bash
pip install scikit-learn imbalanced-learn pyarrow
```

### Issue: "Outlook not found" or COM errors

**Solution:**
- Make sure Outlook is installed
- Make sure you're signed into Outlook
- Try running `python check_outlook_com.py` first to test

### Issue: Memory errors with large datasets

**Solution:**
1. Reduce email count: `python extract_emails.py --count 50`
2. Reduce TF-IDF features in `config.json`:
   ```json
   "text_features": {
     "tfidf_max_features": 500
   }
   ```

### Issue: No emails extracted

**Solution:**
- Check Outlook is connected
- Verify you have emails in your Inbox
- Try `python check_outlook_com.py` to test Outlook access

### Issue: Class imbalance warnings

**Solution:**
- Enable class balancing in `config.json`:
   ```json
   "balance": {
     "enabled": true,
     "method": "smote"
   }
   ```

---

## 📚 Next Steps After Training

1. **Evaluate Performance:**
   - Check `data/models/model_results.json`
   - Compare model accuracies
   - Identify best model for your task

2. **Improve Models:**
   - Tune hyperparameters
   - Try different feature combinations
   - Add more training data

3. **Deploy Models:**
   - Create API endpoints
   - Integrate into applications
   - Set up automated predictions

4. **Advanced Techniques:**
   - Try deep learning (BERT, etc.)
   - Experiment with embeddings
   - Add domain-specific features

---

## 🎓 Quick Reference

```bash
# Complete pipeline
python extract_emails.py
python profile_data.py
python extract_features.py
python prepare_dataset.py
python baseline_models.py

# Check results
python -c "import json; print(json.load(open('data/models/model_results.json')))"
```

---

## 📖 Documentation Files

- **`README_EXTRACTION.md`** - Detailed extraction guide
- **`ML_PIPELINE.md`** - Complete ML pipeline documentation
- **`ML_NEXT_STEPS.md`** - Advanced ML usage examples
- **`GETTING_STARTED.md`** - This file (step-by-step guide)

---

**You're ready to start! Begin with Step 1 and work through the pipeline.** 🚀

