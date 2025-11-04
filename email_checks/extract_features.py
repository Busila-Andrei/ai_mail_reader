from __future__ import annotations

"""
Extract ML-ready features from profiled email data.
Creates feature vectors using TF-IDF, embeddings, one-hot encoding, and cyclical encoding.

Usage:
    python extract_features.py [--input data/processed/emails_profiled_*.json] [--output data/features/]
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from config_loader import load_config, get_config_value
from ml_utils import cyclical_encode_time, normalize_features, create_ratio_features, prepare_categorical_features


def extract_text_features(texts: list[str], config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract text features using TF-IDF and optionally embeddings.
    
    Args:
        texts: List of text strings (email bodies/subjects)
        config: Configuration dictionary
        
    Returns:
        Dictionary with text feature matrices and vectorizers
    """
    text_config = config.get('text_features', {})
    
    # TF-IDF features
    tfidf_max_features = text_config.get('tfidf_max_features', 1000)
    tfidf_min_df = text_config.get('tfidf_min_df', 2)
    tfidf_max_df = text_config.get('tfidf_max_df', 0.95)
    
    tfidf_vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        min_df=tfidf_min_df,
        max_df=tfidf_max_df,
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        lowercase=True,
        strip_accents='unicode'
    )
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    result = {
        'tfidf_matrix': tfidf_matrix,
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_feature_names': tfidf_vectorizer.get_feature_names_out().tolist()
    }
    
    # Character n-grams (optional)
    if text_config.get('use_char_ngrams', False):
        char_vectorizer = CountVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=text_config.get('char_ngram_features', 500)
        )
        char_matrix = char_vectorizer.fit_transform(texts)
        result['char_ngram_matrix'] = char_matrix
        result['char_ngram_vectorizer'] = char_vectorizer
    
    # Dimensionality reduction (optional)
    if text_config.get('use_dimensionality_reduction', False):
        n_components = text_config.get('svd_components', 100)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)
        result['tfidf_reduced'] = tfidf_reduced
        result['svd_transformer'] = svd
    
    return result


def extract_structural_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Extract structural features (one-hot encoding, counts, ratios).
    
    Args:
        df: DataFrame with email features
        config: Configuration dictionary
        
    Returns:
        DataFrame with structural features
    """
    df = df.copy()
    structural_config = config.get('structural_features', {})
    
    # One-hot encode categorical features
    categorical_cols = structural_config.get('categorical_columns', ['sender_domain'])
    max_categories = structural_config.get('max_categories_per_column', 20)
    min_frequency = structural_config.get('min_category_frequency', 2)
    
    # Prepare categorical features
    df, cat_mappings = prepare_categorical_features(df, categorical_cols, max_categories, min_frequency)
    
    # One-hot encode
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_')
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
    
    # Create ratio features
    df = create_ratio_features(df)
    
    # Normalize count features
    count_cols = structural_config.get('normalize_columns', [
        'word_count', 'attachment_count', 'recipient_count', 
        'url_count', 'email_count_in_body', 'phone_count'
    ])
    
    normalization_method = structural_config.get('normalization_method', 'standard')
    df = normalize_features(df, count_cols, method=normalization_method)
    
    return df, cat_mappings


def extract_temporal_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Extract temporal features with cyclical encoding.
    
    Args:
        df: DataFrame with temporal features
        config: Configuration dictionary
        
    Returns:
        DataFrame with cyclical temporal features
    """
    df = df.copy()
    temporal_config = config.get('temporal_features', {})
    
    if temporal_config.get('use_cyclical_encoding', True):
        # Hour of day (24-hour cycle)
        if 'hour_of_day' in df.columns:
            df = cyclical_encode_time(df, 'hour_of_day', 24, 'hour')
        
        # Day of week (7-day cycle)
        if 'day_of_week' in df.columns:
            df = cyclical_encode_time(df, 'day_of_week', 7, 'day')
        
        # Month (12-month cycle)
        if 'month' in df.columns:
            df = cyclical_encode_time(df, 'month', 12, 'month')
    
    return df


def extract_metadata_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Extract and normalize metadata features.
    
    Args:
        df: DataFrame with metadata
        config: Configuration dictionary
        
    Returns:
        DataFrame with normalized metadata features
    """
    df = df.copy()
    metadata_config = config.get('metadata_features', {})
    
    # Normalize metadata counts and ratios
    metadata_cols = metadata_config.get('normalize_columns', [
        'importance', 'category_count', 'html_ratio'
    ])
    
    normalization_method = metadata_config.get('normalization_method', 'standard')
    df = normalize_features(df, metadata_cols, method=normalization_method)
    
    # Binary flags
    if 'is_unread' in df.columns:
        df['is_unread'] = df['is_unread'].astype(int)
    if 'is_flagged' in df.columns:
        df['is_flagged'] = df['is_flagged'].astype(int)
    if 'has_attachments' in df.columns:
        df['has_attachments'] = df['has_attachments'].astype(int)
    if 'has_html' in df.columns:
        df['has_html'] = df['has_html'].astype(int)
    
    return df


def extract_all_features(input_file: str, output_dir: str, config: dict[str, Any]) -> None:
    """
    Extract all features from profiled email data.
    
    Args:
        input_file: Path to profiled emails JSON file
        output_dir: Output directory for feature files
        config: Configuration dictionary
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading profiled email data from {input_file}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    emails = data.get('emails', [])
    if not emails:
        print("No emails found in file.")
        return
    
    print(f"Processing {len(emails)} emails...")
    
    # Extract features from each email
    features_list = []
    texts_subject = []
    texts_body = []
    
    for email in emails:
        feat = email.get('features', {})
        email_data = {
            **feat,
            'subject': email.get('subject', ''),
            'sender_email': email.get('sender_email', ''),
            'importance': email.get('importance', 1),
            'unread': email.get('unread', False),
            'flag_status': email.get('flag_status', 0),
            'body_plain': email.get('body_plain', ''),
            'body_html': email.get('body_html', ''),
            'body_length': email.get('body_length', 0),
            'body_html_length': email.get('body_html_length', 0),
        }
        features_list.append(email_data)
        texts_subject.append(email_data.get('subject', ''))
        texts_body.append(email_data.get('body_plain', ''))
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    # Extract text features
    print("Extracting text features...")
    combined_texts = [f"{subj} {body}" for subj, body in zip(texts_subject, texts_body)]
    text_features = extract_text_features(combined_texts, config)
    
    # Extract structural features
    print("Extracting structural features...")
    df_structural, cat_mappings = extract_structural_features(df, config)
    
    # Extract temporal features
    print("Extracting temporal features...")
    df_temporal = extract_temporal_features(df_structural, config)
    
    # Extract metadata features
    print("Extracting metadata features...")
    df_metadata = extract_metadata_features(df_temporal, config)
    
    # Save feature matrices
    print("Saving feature matrices...")
    
    # Save text features
    with open(output_path / 'text_features.pkl', 'wb') as f:
        pickle.dump({
            'tfidf_matrix': text_features['tfidf_matrix'],
            'tfidf_vectorizer': text_features['tfidf_vectorizer'],
            'tfidf_feature_names': text_features['tfidf_feature_names']
        }, f)
    
    if 'char_ngram_matrix' in text_features:
        with open(output_path / 'char_ngram_features.pkl', 'wb') as f:
            pickle.dump({
                'char_ngram_matrix': text_features['char_ngram_matrix'],
                'char_ngram_vectorizer': text_features['char_ngram_vectorizer']
            }, f)
    
    if 'tfidf_reduced' in text_features:
        np.save(output_path / 'tfidf_reduced.npy', text_features['tfidf_reduced'])
    
    # Save structural/temporal/metadata features as DataFrame
    df_metadata.to_parquet(output_path / 'non_text_features.parquet', index=False)
    df_metadata.to_csv(output_path / 'non_text_features.csv', index=False)
    
    # Save feature info
    feature_info = {
        'total_samples': len(emails),
        'text_features': {
            'tfidf_shape': text_features['tfidf_matrix'].shape,
            'tfidf_features': len(text_features['tfidf_feature_names'])
        },
        'non_text_features': {
            'columns': df_metadata.columns.tolist(),
            'shape': df_metadata.shape
        },
        'categorical_mappings': cat_mappings
    }
    
    with open(output_path / 'feature_info.json', 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Feature extraction complete!")
    print(f"✓ Text features (TF-IDF): {text_features['tfidf_matrix'].shape}")
    print(f"✓ Non-text features: {df_metadata.shape}")
    print(f"✓ Saved to: {output_path}")
    print(f"\nFiles created:")
    print(f"  - text_features.pkl (TF-IDF matrix and vectorizer)")
    print(f"  - non_text_features.parquet (structured features)")
    print(f"  - non_text_features.csv (structured features)")
    print(f"  - feature_info.json (metadata)")


def main() -> None:
    """Main entry point."""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Extract ML-ready features from profiled emails')
    parser.add_argument('--input', type=str, default=None,
                       help='Input file pattern (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config file (default: config.json)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Get input pattern from config or args
    input_pattern = args.input if args.input is not None else get_config_value(
        config, 'feature_extraction', 'input_pattern', 
        default='data/processed/emails_profiled_*.json'
    )
    output_directory = args.output if args.output is not None else get_config_value(
        config, 'feature_extraction', 'output_directory', 
        default='data/features'
    )
    
    # Find matching files
    files = glob.glob(input_pattern)
    if not files:
        print(f"No files found matching: {input_pattern}")
        return
    
    # Use the most recent file
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    latest_file = files[0]
    
    extract_all_features(latest_file, output_directory, config)


if __name__ == "__main__":
    main()

