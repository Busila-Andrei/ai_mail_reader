from __future__ import annotations

"""
Prepare ML datasets from extracted features.
Creates train/validation/test splits, balances classes, and exports to various formats.

Usage:
    python prepare_dataset.py [--input data/features/] [--output data/datasets/]
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack, save_npz, load_npz
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from config_loader import load_config, get_config_value


def load_features(features_dir: str) -> tuple[Any, pd.DataFrame]:
    """
    Load extracted features.
    
    Args:
        features_dir: Directory containing feature files
        
    Returns:
        Tuple of (text_feature_matrix, non_text_dataframe)
    """
    features_path = Path(features_dir)
    
    # Load text features
    with open(features_path / 'text_features.pkl', 'rb') as f:
        text_data = pickle.load(f)
        tfidf_matrix = text_data['tfidf_matrix']
    
    # Load non-text features
    non_text_df = pd.read_parquet(features_path / 'non_text_features.parquet')
    
    return tfidf_matrix, non_text_df


def create_labels(df: pd.DataFrame, label_config: dict[str, Any]) -> np.ndarray:
    """
    Create labels based on configuration.
    
    Args:
        df: DataFrame with email data
        label_config: Label configuration
        
    Returns:
        Array of labels
    """
    label_type = label_config.get('type', 'importance')
    
    if label_type == 'importance':
        # High importance = 1, else 0
        labels = (df['importance'] == 2).astype(int).values
    elif label_type == 'unread':
        # Unread = 1, read = 0
        labels = df['is_unread'].astype(int).values
    elif label_type == 'flagged':
        # Flagged = 1, not flagged = 0
        labels = (df['flag_status'] > 0).astype(int).values
    elif label_type == 'has_attachment':
        # Has attachment = 1, no attachment = 0
        labels = df['has_attachments'].astype(int).values
    elif label_type == 'custom_column':
        # Use a custom column
        col = label_config.get('column_name', 'importance')
        threshold = label_config.get('threshold', 2)
        labels = (df[col] >= threshold).astype(int).values
    else:
        # Default: binary classification on importance
        labels = (df['importance'] == 2).astype(int).values
    
    return labels


def combine_features(tfidf_matrix: Any, non_text_df: pd.DataFrame, 
                     config: dict[str, Any]) -> tuple[Any, list[str]]:
    """
    Combine text and non-text features.
    
    Args:
        tfidf_matrix: Sparse matrix of text features
        non_text_df: DataFrame of non-text features
        config: Configuration dictionary
        
    Returns:
        Tuple of (combined_feature_matrix, feature_names)
    """
    feature_config = config.get('dataset', {})
    
    # Select which non-text features to include
    included_features = feature_config.get('include_features', [
        'word_count', 'attachment_count', 'recipient_count',
        'url_count', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'importance', 'is_unread', 'is_flagged', 'has_attachments'
    ])
    
    # Filter to only include features that exist
    available_features = [f for f in included_features if f in non_text_df.columns]
    non_text_selected = non_text_df[available_features].fillna(0)
    
    # Convert to sparse matrix
    from scipy.sparse import csr_matrix
    non_text_matrix = csr_matrix(non_text_selected.values)
    
    # Combine
    combined_matrix = hstack([tfidf_matrix, non_text_matrix])
    
    # Feature names
    feature_names = []
    if hasattr(tfidf_matrix, 'shape'):
        feature_names.extend([f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    feature_names.extend([f'nontext_{f}' for f in available_features])
    
    return combined_matrix, feature_names


def split_dataset(X: Any, y: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        X: Feature matrix
        y: Labels
        config: Configuration dictionary
        
    Returns:
        Dictionary with train/val/test splits
    """
    split_config = config.get('dataset', {}).get('split', {})
    
    test_size = split_config.get('test_size', 0.2)
    val_size = split_config.get('val_size', 0.1)
    random_state = split_config.get('random_state', 42)
    
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for test split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_train_val
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


def balance_classes(X: Any, y: np.ndarray, config: dict[str, Any]) -> tuple[Any, np.ndarray]:
    """
    Balance classes using oversampling/undersampling if needed.
    
    Args:
        X: Feature matrix
        y: Labels
        config: Configuration dictionary
        
    Returns:
        Tuple of (balanced_X, balanced_y)
    """
    balance_config = config.get('dataset', {}).get('balance', {})
    
    if not balance_config.get('enabled', False):
        return X, y
    
    balance_method = balance_config.get('method', 'smote')
    
    if balance_method == 'smote':
        # SMOTE oversampling
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced
    
    elif balance_method == 'undersample':
        # Random undersampling
        undersampler = RandomUnderSampler(random_state=42, sampling_strategy='auto')
        X_balanced, y_balanced = undersampler.fit_resample(X, y)
        return X_balanced, y_balanced
    
    else:
        return X, y


def export_dataset(splits: dict[str, Any], output_dir: str, 
                   feature_names: list[str], config: dict[str, Any]) -> None:
    """
    Export dataset to various formats.
    
    Args:
        splits: Dictionary with train/val/test splits
        output_dir: Output directory
        feature_names: List of feature names
        config: Configuration dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    export_config = config.get('dataset', {}).get('export', {})
    
    # Export formats
    formats = export_config.get('formats', ['npz', 'parquet', 'npy'])
    
    print("Exporting datasets...")
    
    for split_name in ['train', 'val', 'test']:
        X_key = f'X_{split_name}'
        y_key = f'y_{split_name}'
        
        if X_key not in splits or y_key not in splits:
            continue
        
        X = splits[X_key]
        y = splits[y_key]
        
        print(f"  Exporting {split_name} set ({X.shape[0]} samples)...")
        
        # NPZ format (sparse matrices)
        if 'npz' in formats:
            save_npz(output_path / f'X_{split_name}.npz', X)
            np.save(output_path / f'y_{split_name}.npy', y)
        
        # Parquet format (if dense)
        if 'parquet' in formats and hasattr(X, 'toarray'):
            X_dense = X.toarray()
            df = pd.DataFrame(X_dense, columns=feature_names[:X_dense.shape[1]])
            df['label'] = y
            df.to_parquet(output_path / f'{split_name}_set.parquet', index=False)
        
        # CSV format (if dense and small)
        if 'csv' in formats and hasattr(X, 'toarray') and X.shape[0] < 10000:
            X_dense = X.toarray()
            df = pd.DataFrame(X_dense, columns=feature_names[:X_dense.shape[1]])
            df['label'] = y
            df.to_csv(output_path / f'{split_name}_set.csv', index=False)
        
        # NumPy arrays
        if 'npy' in formats:
            if hasattr(X, 'toarray'):
                np.save(output_path / f'X_{split_name}_dense.npy', X.toarray())
            else:
                np.save(output_path / f'X_{split_name}_dense.npy', X)
    
    # Save metadata
    metadata = {
        'splits': {
            'train': {'samples': len(splits['y_train']), 'positive': int(splits['y_train'].sum())},
            'val': {'samples': len(splits['y_val']), 'positive': int(splits['y_val'].sum())},
            'test': {'samples': len(splits['y_test']), 'positive': int(splits['y_test'].sum())}
        },
        'feature_count': len(feature_names),
        'feature_names': feature_names[:100] if len(feature_names) > 100 else feature_names
    }
    
    with open(output_path / 'dataset_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Dataset exported to: {output_path}")


def prepare_dataset(input_dir: str, output_dir: str, config: dict[str, Any]) -> None:
    """
    Prepare complete ML dataset.
    
    Args:
        input_dir: Directory with extracted features
        output_dir: Output directory for datasets
        config: Configuration dictionary
    """
    print(f"Loading features from {input_dir}...")
    tfidf_matrix, non_text_df = load_features(input_dir)
    
    print(f"Text features shape: {tfidf_matrix.shape}")
    print(f"Non-text features shape: {non_text_df.shape}")
    
    # Create labels
    print("Creating labels...")
    label_config = config.get('dataset', {}).get('labels', {})
    labels = create_labels(non_text_df, label_config)
    
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Combine features
    print("Combining features...")
    X_combined, feature_names = combine_features(tfidf_matrix, non_text_df, config)
    
    print(f"Combined features shape: {X_combined.shape}")
    
    # Balance classes if needed
    print("Balancing classes...")
    X_balanced, y_balanced = balance_classes(X_combined, labels, config)
    
    # Split dataset
    print("Splitting dataset...")
    splits = split_dataset(X_balanced, y_balanced, config)
    
    # Export
    export_dataset(splits, output_dir, feature_names, config)
    
    print("\n✓ Dataset preparation complete!")


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare ML dataset from extracted features')
    parser.add_argument('--input', type=str, default=None,
                       help='Input features directory (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config file (default: config.json)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    input_directory = args.input if args.input is not None else get_config_value(
        config, 'dataset', 'input_directory', default='data/features'
    )
    output_directory = args.output if args.output is not None else get_config_value(
        config, 'dataset', 'output_directory', default='data/datasets'
    )
    
    prepare_dataset(input_directory, output_directory, config)


if __name__ == "__main__":
    main()

