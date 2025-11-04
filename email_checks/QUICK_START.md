# Quick Start Guide - email_checks

## 🚀 Run Everything in 5 Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract emails
python extract_emails.py

# 3. Profile data
python profile_data.py

# 4. Extract ML features
python extract_features.py

# 5. Prepare dataset & train models
python prepare_dataset.py
python baseline_models.py
```

**That's it!** Your models are in `data/models/`

---

## 📋 What Each Script Does

| Script | Purpose | Output |
|--------|---------|--------|
| `extract_emails.py` | Extract emails from Outlook | `data/raw/emails_*.json` |
| `profile_data.py` | Analyze emails, extract features | `data/processed/emails_profiled_*.json` |
| `extract_features.py` | Create ML feature vectors | `data/features/` |
| `prepare_dataset.py` | Split into train/val/test | `data/datasets/` |
| `baseline_models.py` | Train ML models | `data/models/` |

---

## 🎯 Common Tasks

### Extract More Emails
```bash
python extract_emails.py --count 500
```

### Train Only Classification
```bash
python baseline_models.py --task classification
```

### View Model Results
```python
import json
results = json.load(open('data/models/model_results.json'))
print(results['classification'])
```

### Use a Trained Model
```python
import pickle
model = pickle.load(open('data/models/classification/random_forest.pkl', 'rb'))
prediction = model.predict(new_features)
```

---

## 📁 Output Structure

```
data/
├── raw/          # Raw emails (JSON)
├── processed/    # Profiled emails (JSON)
├── features/     # Feature matrices (PKL, Parquet)
├── datasets/     # Train/val/test splits (NPZ, NumPy)
└── models/       # Trained models (PKL)
```

---

## 🔧 Configuration

Edit `config.json` to customize:
- Email count
- Feature extraction settings
- Model parameters

**See `GETTING_STARTED.md` for detailed instructions!**

