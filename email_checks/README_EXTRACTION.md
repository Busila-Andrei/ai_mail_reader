# Email Extraction & Analysis for ML

This directory contains scripts for extracting emails from Outlook and analyzing their structure for machine learning purposes.

## Configuration

All settings are managed through `config.json`. The configuration file includes:

- **Extraction settings**: Email count, output directory, sorting, included fields
- **Profiling settings**: Input patterns, output directory
- **Outlook connection**: Profile name, folder settings
- **Processing options**: Encoding, JSON formatting, error handling

Command-line arguments override config values when provided.

### Example `config.json`:
```json
{
  "extraction": {
    "email_count": 100,
    "output_directory": "data/raw",
    "source_folder": "Inbox",
    "sort_by": "ReceivedTime",
    "sort_descending": true
  },
  "profiling": {
    "input_pattern": "data/raw/emails_*.json",
    "output_directory": "data/processed"
  },
  "outlook": {
    "profile_name": null,
    "folder_id": 6
  }
}
```

See `config.json` for all available options.

## Scripts

### `check_outlook_com.py`
Simple test script to verify Outlook COM access. Lists top 5 email subjects.

### `extract_emails.py`
Main extraction script that extracts comprehensive email data from Outlook.

**Usage:**
```bash
# Extract emails using config.json settings
python extract_emails.py

# Override config with command-line arguments
python extract_emails.py --count 200
python extract_emails.py --count 100 --output data/raw

# Use custom config file
python extract_emails.py --config my_config.json
```

**Output:**
- Saves to `data/raw/emails_YYYYMMDD_HHMMSS.json`
- Includes metadata, full email data, and extraction statistics

**Extracted Fields:**
- Basic: Subject, Sender, Recipients, Date/Time
- Content: Body (plain text & HTML), Word count
- Attachments: Filename, Size, Content type, Count
- Metadata: Categories, Flags, Importance, Headers
- Structure: Conversation ID, Reply chains

### `profile_data.py`
Analyzes extracted email data and generates statistics and features.

**Usage:**
```bash
# Profile most recent extraction (uses config.json)
python profile_data.py

# Override config with command-line arguments
python profile_data.py --input data/raw/emails_20250101_120000.json
python profile_data.py --output data/processed

# Use custom config file
python profile_data.py --config my_config.json
```

**Output:**
- `data/processed/emails_profiled_YYYYMMDD_HHMMSS.json` - Emails with extracted features
- `data/processed/statistics_YYYYMMDD_HHMMSS.json` - Summary statistics

**Generated Features:**
- Text: Word count, sentence count, character count
- Content: URLs, email addresses, phone numbers
- Structural: Attachment count, recipient count, HTML ratio
- Temporal: Hour of day, day of week, weekend/weekday
- Metadata: Sender domain, categories, importance, flags

## Workflow

1. **Extract emails:**
   ```bash
   python extract_emails.py --count 100
   ```

2. **Profile the data:**
   ```bash
   python profile_data.py
   ```

3. **Review statistics:**
   - Check `data/processed/statistics_*.json` for overview
   - Check `data/processed/emails_profiled_*.json` for detailed features

## Directory Structure

```
email_checks/
├── extract_emails.py          # Main extraction script
├── profile_data.py            # Data profiling script
├── check_outlook_com.py       # Simple test script
├── config.json                # Configuration file (edit this!)
├── config_loader.py           # Configuration loader utility
├── config.example.json        # Example configuration template
├── data/
│   ├── raw/                   # Raw extracted emails (JSON)
│   └── processed/             # Profiled data with features
└── requirements.txt           # Python dependencies
```

## Next Steps for ML

After extraction and profiling, you can:

1. **Load profiled data:**
   ```python
   import json
   with open('data/processed/emails_profiled_*.json') as f:
       data = json.load(f)
   emails = data['emails']
   features = [e['features'] for e in emails]
   ```

2. **Convert to ML-ready format:**
   - Use pandas for DataFrame creation
   - Extract feature vectors for scikit-learn
   - Create feature matrices (TF-IDF, embeddings, etc.)

3. **Additional dependencies for ML:**
   - `pandas` - Data manipulation
   - `numpy` - Numerical operations
   - `scikit-learn` - Feature extraction and ML models
   - `beautifulsoup4` or `html2text` - HTML parsing
   - `nltk` or `spacy` - NLP preprocessing

## Notes

- Requires Outlook to be installed and configured with a signed-in profile
- Windows-only (uses COM interface)
- Emails are sorted by received time (newest first)
- Large emails with many attachments may take longer to process
- JSON files can grow large; consider archiving old extractions

