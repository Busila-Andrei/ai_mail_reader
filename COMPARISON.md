# Comparison: `email_checks` vs `ace_tool`

## Overview

Both tools read emails from Outlook, but they serve **completely different purposes**:

- **`ace_tool`**: Domain-specific business tool for IBM ACE deployment management
- **`email_checks`**: Generic ML pipeline for email analysis and machine learning

---

## 🎯 Purpose & Logic

### `ace_tool` - IBM ACE Deployment Tool

**Purpose:**
- Extract deployment information from emails related to IBM ACE (App Connect Enterprise)
- Validate extracted information against ACE REST APIs
- Process deployment requests and track application deployments
- Generate reports and overviews for ACE applications

**Logic Flow:**
```
1. Read emails from Outlook (filtered by action keywords like "deploy", "deployati", "aplicati")
2. Extract structured data (execution_group, application, bar_file, cfg_name, path)
   - Uses rule-based extraction (regex patterns)
   - Falls back to LLM (AI) if rules fail
   - Uses local memory/RAG for context
3. Validate against ACE REST API
   - Check if execution groups exist
   - Verify applications exist in execution groups
   - Validate BAR files and CFG files
4. Cross-reference with SSH-discovered .cfg files
5. Generate HTML overview tables
6. Store results in CSV/JSON
```

**Key Features:**
- ✅ ACE REST API integration
- ✅ SSH/remote file discovery
- ✅ AI/LLM fallback for extraction
- ✅ Local memory (RAG) for context
- ✅ HTML overview generation
- ✅ Validation against ACE systems
- ✅ Tracks processed emails by EntryID

**Domain:** IBM ACE (App Connect Enterprise) deployment management

---

### `email_checks` - ML Pipeline for Email Analysis

**Purpose:**
- Extract emails from Outlook for machine learning analysis
- Build feature vectors for ML models
- Train classification, clustering, NLP, and anomaly detection models
- General-purpose email analysis pipeline

**Logic Flow:**
```
1. Extract emails from Outlook (configurable count, folder)
2. Profile emails (extract basic features: word counts, URLs, temporal patterns)
3. Extract ML features:
   - Text features: TF-IDF vectorization
   - Structural features: One-hot encoding for categorical
   - Temporal features: Cyclical encoding (sin/cos)
   - Metadata features: Normalized counts, ratios
4. Prepare dataset:
   - Train/validation/test split
   - Class balancing (SMOTE)
   - Export to multiple formats (NPZ, Parquet, CSV, NumPy)
5. Train baseline models:
   - Classification (Logistic Regression, Random Forest, SVM)
   - Clustering (K-Means, DBSCAN)
   - NLP (LDA, NMF topic modeling)
   - Anomaly Detection (Isolation Forest)
```

**Key Features:**
- ✅ Complete ML pipeline (extraction → features → models)
- ✅ Multiple feature extraction methods (TF-IDF, embeddings ready)
- ✅ Dataset preparation with splits and balancing
- ✅ Multiple baseline models
- ✅ Configurable via JSON
- ✅ Export to various formats for ML

**Domain:** General email analysis and machine learning

---

## 📊 Side-by-Side Comparison

| Aspect | `ace_tool` | `email_checks` |
|--------|-----------|----------------|
| **Primary Goal** | Extract & validate ACE deployment info | Build ML models on emails |
| **Email Filtering** | Action keywords (deploy, aplicati, etc.) | All emails (configurable) |
| **Data Extraction** | Structured fields (EG, App, BAR, CFG) | Full email content + features |
| **External APIs** | ACE REST API, SSH | None (standalone) |
| **AI/LLM** | Yes (Ollama/OpenAI for extraction fallback) | No (pure ML) |
| **Output Format** | CSV, JSON, HTML reports | JSON, Parquet, NPZ, NumPy arrays |
| **Validation** | Against ACE systems | None (ML models) |
| **Memory/RAG** | Yes (local memory for context) | No |
| **ML Models** | No | Yes (classification, clustering, NLP, anomaly) |
| **Feature Engineering** | No | Yes (TF-IDF, one-hot, cyclical encoding) |
| **Dataset Splits** | No | Yes (train/val/test) |
| **Class Balancing** | No | Yes (SMOTE) |
| **Configuration** | Environment variables (.env) | JSON config file |
| **Domain Specific** | IBM ACE deployment | Generic emails |

---

## 🔄 Logic Differences

### `ace_tool` Logic

```python
# Simplified flow
1. Read Outlook emails (filtered by action patterns)
2. Extract structured data:
   - execution_group
   - application  
   - bar_file
   - cfg_name
   - path
3. Validate against ACE REST API
4. Cross-check with CFG files (local/SSH)
5. Generate reports
```

**Focus:** Information extraction and validation for business operations

### `email_checks` Logic

```python
# Simplified flow
1. Extract emails from Outlook
2. Profile emails (extract basic features)
3. Create ML feature vectors:
   - Text: TF-IDF
   - Structural: One-hot encoding
   - Temporal: Cyclical encoding
   - Metadata: Normalized features
4. Prepare dataset (split, balance)
5. Train ML models
```

**Focus:** Machine learning and pattern discovery

---

## 🎯 Destination & Use Cases

### `ace_tool` - Use Cases

1. **Deployment Management**
   - Track deployment requests from emails
   - Validate deployment information
   - Generate deployment reports

2. **ACE Operations**
   - Monitor application deployments
   - Validate configuration files
   - Track execution groups and applications

3. **Business Process**
   - Automate deployment request processing
   - Generate HTML overviews for stakeholders
   - Maintain deployment history

**Destination:** Production deployment management system, ACE operations dashboard

---

### `email_checks` - Use Cases

1. **Email Classification**
   - Spam detection
   - Category classification
   - Priority prediction

2. **Pattern Discovery**
   - Email clustering (topic discovery)
   - Anomaly detection (unusual emails)
   - Temporal pattern analysis

3. **Research & Analysis**
   - Email corpus analysis
   - Feature engineering experiments
   - ML model development

**Destination:** ML research, email analysis systems, automated email processing

---

## 🔗 Integration Possibilities

### Can They Work Together?

**Yes, but for different purposes:**

1. **`ace_tool`** could use `email_checks` ML models:
   - Use classification to filter relevant deployment emails
   - Use anomaly detection to flag unusual deployment requests
   - Use clustering to group similar deployment patterns

2. **`email_checks`** could use `ace_tool` data:
   - Train models on labeled ACE deployment emails
   - Use `ace_tool` extraction results as training labels
   - Analyze patterns in ACE deployment emails

### Example Integration:

```python
# Use email_checks to classify emails first
from email_checks.baseline_models import load_classification_model

model = load_classification_model('data/models/classification/random_forest.pkl')
is_deployment_email = model.predict(email_features)

# Then use ace_tool for detailed extraction
if is_deployment_email:
    extracted = ace_tool.extraction.extract_one(subject, body)
    validated = ace_tool.validator.validate_against_ace(ace_client, extracted)
```

---

## 📁 Code Structure

### `ace_tool/`
```
ace_tool/
├── __init__.py          # Module exports
├── config.py            # Environment variables (.env)
├── utils.py             # Text/HTML helpers
├── ace_client.py        # ACE REST API client
├── ace_helpers.py        # ACE inference helpers
├── extraction.py         # Rule-based + LLM extraction
├── validator.py          # ACE validation
├── cfg_discovery.py      # CFG file discovery (local/SSH)
├── ai_utils.py          # AI/LLM helpers
├── outlook_reader.py     # Outlook email reading
└── overview.py          # HTML overview generation
```

### `email_checks/`
```
email_checks/
├── extract_emails.py     # Email extraction
├── profile_data.py       # Basic feature extraction
├── extract_features.py   # ML feature extraction
├── prepare_dataset.py   # Dataset preparation
├── baseline_models.py   # ML model training
├── ml_utils.py           # ML utility functions
├── config_loader.py      # Config loader
├── config.json           # Configuration
└── data/                 # Output directories
    ├── raw/              # Raw emails
    ├── processed/        # Profiled emails
    ├── features/          # Feature matrices
    ├── datasets/          # Train/val/test splits
    └── models/            # Trained models
```

---

## 🎯 Summary

| Question | `ace_tool` | `email_checks` |
|----------|-----------|----------------|
| **What is it?** | Business tool for ACE deployment management | ML pipeline for email analysis |
| **What does it do?** | Extracts & validates deployment info from emails | Builds ML models to analyze emails |
| **Who uses it?** | ACE administrators, deployment teams | Data scientists, ML engineers |
| **Output?** | CSV/JSON reports, HTML overviews | Trained ML models, feature matrices |
| **Destination?** | Production deployment system | ML research, analysis systems |

---

**Bottom Line:**
- **`ace_tool`** = Business tool for a specific domain (IBM ACE)
- **`email_checks`** = Generic ML pipeline for email analysis

They complement each other but serve different purposes!

