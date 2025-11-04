from __future__ import annotations

"""
ML utility functions for email feature extraction and preprocessing.
"""

import numpy as np
import pandas as pd
from typing import Any


def cyclical_encode_time(df: pd.DataFrame, time_col: str, period: int, prefix: str = None) -> pd.DataFrame:
    """
    Encode time features cyclically (sin/cos transformation).
    
    Args:
        df: DataFrame with time column
        time_col: Name of time column (e.g., 'hour_of_day', 'day_of_week')
        period: Period of the cycle (e.g., 24 for hours, 7 for days)
        prefix: Prefix for new columns (defaults to time_col)
        
    Returns:
        DataFrame with added sin/cos columns
    """
    if prefix is None:
        prefix = time_col
    
    values = df[time_col].fillna(0).values
    df[f'{prefix}_sin'] = np.sin(2 * np.pi * values / period)
    df[f'{prefix}_cos'] = np.cos(2 * np.pi * values / period)
    
    return df


def normalize_features(df: pd.DataFrame, columns: list[str], method: str = 'standard') -> pd.DataFrame:
    """
    Normalize numerical features.
    
    Args:
        df: DataFrame with features
        columns: List of column names to normalize
        method: 'standard' (z-score) or 'minmax' (0-1 scaling)
        
    Returns:
        DataFrame with normalized columns
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'standard':
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0
    
    return df


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio features from existing counts.
    
    Args:
        df: DataFrame with count features
        
    Returns:
        DataFrame with added ratio features
    """
    df = df.copy()
    
    # Word density (words per character)
    if 'word_count' in df.columns and 'character_count' in df.columns:
        df['word_density'] = df['word_count'] / (df['character_count'] + 1)
    
    # URL/email ratio
    if 'url_count' in df.columns and 'word_count' in df.columns:
        df['url_ratio'] = df['url_count'] / (df['word_count'] + 1)
    
    if 'email_count_in_body' in df.columns and 'word_count' in df.columns:
        df['email_ratio'] = df['email_count_in_body'] / (df['word_count'] + 1)
    
    # Attachment ratio
    if 'attachment_count' in df.columns:
        df['has_attachment'] = (df['attachment_count'] > 0).astype(int)
    
    # HTML ratio (already exists but ensure it's there)
    if 'html_ratio' not in df.columns and 'body_html_length' in df.columns and 'body_length' in df.columns:
        df['html_ratio'] = df['body_html_length'] / (df['body_length'] + 1)
    
    return df


def prepare_categorical_features(df: pd.DataFrame, categorical_cols: list[str], 
                                max_categories: int = 20, min_frequency: int = 2) -> tuple[pd.DataFrame, dict]:
    """
    Prepare categorical features for one-hot encoding.
    
    Args:
        df: DataFrame with categorical columns
        categorical_cols: List of categorical column names
        max_categories: Maximum categories to keep per column
        min_frequency: Minimum frequency to keep a category
        
    Returns:
        Tuple of (DataFrame with prepared categories, mapping dict)
    """
    df = df.copy()
    mappings = {}
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        # Count frequencies
        value_counts = df[col].value_counts()
        
        # Keep top categories by frequency
        top_categories = value_counts[value_counts >= min_frequency].head(max_categories).index.tolist()
        
        # Replace rare categories with 'other'
        df[col] = df[col].apply(lambda x: x if x in top_categories else 'other')
        
        mappings[col] = {
            'categories': top_categories + ['other'],
            'value_counts': value_counts.to_dict()
        }
    
    return df, mappings

