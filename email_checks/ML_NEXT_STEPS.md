# What You Have & ML Next Steps

## 📦 What You Currently Have

### **Your Email Extraction System**

You have a complete email data extraction and analysis pipeline:

1. **`extract_emails.py`** - Extracts emails from Outlook
   - Connects to Outlook via COM interface
   - Extracts 100 emails (configurable) with full details
   - Saves raw data to JSON files

2. **`profile_data.py`** - Analyzes extracted emails
   - Processes the JSON files
   - Extracts features (word counts, URLs, temporal patterns, etc.)
   - Generates statistics and feature-rich datasets

3. **`config.json`** - Configuration file
   - All settings in one place
   - Easy to modify without changing code

### **What Data You'll Get**

After running the scripts, you'll have:

**Raw Data** (`data/raw/emails_*.json`):
- Full email content (subject, body, HTML)
- Sender, recipients, dates
- Attachments info
- Categories, flags, importance
- Headers, conversation IDs

**Profiled Data** (`data/processed/emails_profiled_*.json`):
- All raw data PLUS extracted features:
  - Text features: word count, sentence count, character count
  - Content features: URLs, email addresses, phone numbers
  - Structural features: attachment count, recipient count, HTML ratio
  - Temporal features: hour of day, day of week, weekend/weekday
  - Metadata: sender domain, categories, importance flags

**Statistics** (`data/processed/statistics_*.json`):
- Summary statistics across all emails
- Top sender domains
- Attachment patterns
- Temporal patterns

---

## 🚀 What To Do Next

### **Step 1: Extract Your Emails** (First Time Setup)

```bash
# Make sure you're in the email_checks directory
cd email_checks

# Run the extraction (will extract 100 emails from your Inbox)
python extract_emails.py
```

**What happens:**
- Connects to Outlook
- Extracts 100 emails from your Inbox (newest first)
- Saves to `data/raw/emails_YYYYMMDD_HHMMSS.json`

**Time:** Usually 1-5 minutes depending on email size

---

### **Step 2: Profile the Data**

```bash
# Analyze the extracted emails and generate features
python profile_data.py
```

**What happens:**
- Loads the most recent extraction
- Analyzes each email and extracts features
- Generates statistics
- Saves to `data/processed/emails_profiled_*.json` and `statistics_*.json`

**Time:** Usually less than 1 minute

---

### **Step 3: Load Data for ML**

Now you have structured data ready for machine learning. Here's how to use it:

#### **Option A: Load Profiled Data (Recommended)**

```python
import json
import pandas as pd

# Load the profiled data
with open('data/processed/emails_profiled_YYYYMMDD_HHMMSS.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get emails with features
emails = data['emails']

# Extract just the features (for ML)
features_list = []
for email in emails:
    features = email['features']
    # Add some basic email info if needed
    features['subject'] = email.get('subject', '')
    features['sender_domain'] = email.get('sender_email', '').split('@')[-1] if '@' in email.get('sender_email', '') else ''
    features_list.append(features)

# Convert to pandas DataFrame
df = pd.DataFrame(features_list)

# Now you have a DataFrame ready for ML!
print(df.head())
print(df.columns)  # See all available features
```

#### **Option B: Load Raw Data (If you need full content)**

```python
import json
import pandas as pd

with open('data/raw/emails_YYYYMMDD_HHMMSS.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

emails = data['emails']

# Convert to DataFrame
df = pd.DataFrame(emails)

# Access full email content
print(df[['subject', 'body_plain', 'sender_email']].head())
```

---

### **Step 4: Prepare Features for ML Models**

Now you can create features for different ML tasks:

#### **For Text Classification (Spam, Category, etc.)**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Use email subjects and bodies as text features
texts = df['body_plain'].fillna('') + ' ' + df['subject'].fillna('')

# Create TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = vectorizer.fit_transform(texts)

# Combine with other features
X_other = df[['word_count', 'attachment_count', 'recipient_count', 
              'hour_of_day', 'day_of_week', 'importance']].fillna(0).values

# Combine features (if using sparse matrices, use hstack)
from scipy.sparse import hstack
X_combined = hstack([X_text, X_other])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, labels, test_size=0.2, random_state=42
)
```

#### **For Clustering (Find Email Groups)**

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for clustering
feature_cols = ['word_count', 'attachment_count', 'recipient_count',
                'hour_of_day', 'day_of_week', 'url_count']

X = df[feature_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['cluster'] = clusters
```

#### **For Anomaly Detection (Find Unusual Emails)**

```python
from sklearn.ensemble import IsolationForest

# Use multiple features
X = df[['word_count', 'attachment_count', 'url_count', 
        'hour_of_day', 'importance']].fillna(0)

# Detect anomalies
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X)

df['is_anomaly'] = anomalies == -1
```

---

### **Step 5: Common ML Tasks You Can Do**

#### **1. Email Classification**
- **Spam Detection**: Label spam vs. not spam
- **Category Classification**: Classify emails into categories (work, personal, newsletters, etc.)
- **Priority Prediction**: Predict if email is important based on features

#### **2. Email Clustering**
- **Topic Discovery**: Group similar emails together
- **Sender Patterns**: Find patterns in sender behavior
- **Time-based Clusters**: Group emails by temporal patterns

#### **3. Text Analysis**
- **Sentiment Analysis**: Analyze email tone/sentiment
- **Topic Modeling**: Find main topics in your emails (LDA, NMF)
- **Summarization**: Generate summaries of emails

#### **4. Predictive Analytics**
- **Response Time Prediction**: Predict how long until you respond
- **Email Volume Forecasting**: Predict future email volume
- **Attachment Prediction**: Predict if email will have attachments

---

## 📋 Checklist: What You Need

### **Already Have (from your requirements.txt):**
- ✅ `pandas` - Data manipulation
- ✅ `numpy` - Numerical operations
- ✅ `python-dateutil` - Date parsing
- ✅ `beautifulsoup4` - HTML parsing (if needed)

### **Need to Install for ML:**
```bash
pip install scikit-learn  # For ML models
pip install nltk          # For NLP preprocessing (optional)
pip install matplotlib    # For visualization (optional)
pip install seaborn       # For visualization (optional)
```

---

## 🎯 Quick Start Example

Here's a complete example to get you started:

```python
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load profiled data
with open('data/processed/emails_profiled_YYYYMMDD_HHMMSS.json', 'r') as f:
    data = json.load(f)

emails = data['emails']

# 2. Extract features
features = []
labels = []  # You'll need to create labels based on your goal

for email in emails:
    feat = email['features']
    features.append({
        'word_count': feat['word_count'],
        'attachment_count': feat['attachment_count'],
        'url_count': feat['url_count'],
        'hour_of_day': feat.get('hour_of_day', 12),
        'day_of_week': feat.get('day_of_week', 0),
        'importance': email.get('importance', 1),
        'sender_domain': feat.get('sender_domain', '')
    })
    
    # Example label: High importance = 1, else 0
    labels.append(1 if email.get('importance', 1) == 2 else 0)

# 3. Create DataFrame
df = pd.DataFrame(features)

# 4. Prepare features (encode categorical, handle missing)
df = pd.get_dummies(df, columns=['sender_domain'], prefix='domain')
X = df.fillna(0)

# 5. Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# 7. Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Features:")
print(feature_importance.head(10))
```

---

## 🔄 Iterative Process

1. **Extract more emails** if 100 isn't enough:
   ```bash
   python extract_emails.py --count 500
   ```

2. **Experiment with features**: Try different combinations
3. **Label your data**: Create labels for supervised learning
4. **Train models**: Try different algorithms
5. **Evaluate and iterate**: Improve based on results

---

## 💡 Tips

- **Start small**: 100 emails is good for testing, but you may need more for production ML
- **Feature engineering**: The profiled data gives you a good start, but you may want to create domain-specific features
- **Labeling**: For supervised learning, you'll need to label your emails (manually or using existing metadata like categories/flags)
- **Balancing**: Check if your classes are balanced (e.g., spam vs. not spam)
- **Validation**: Always use train/test splits and cross-validation

---

## 📚 Next Resources

- **scikit-learn documentation**: https://scikit-learn.org/
- **pandas documentation**: https://pandas.pydata.org/
- **Text classification tutorials**: Search for "email classification sklearn"
- **Feature engineering**: Learn about creating domain-specific features

---

**You're ready to start! Run `extract_emails.py` and `profile_data.py`, then start experimenting with the data!** 🚀

