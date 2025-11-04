from __future__ import annotations

"""
Baseline ML models for email analysis.
Includes classification, clustering, NLP, and anomaly detection models.

Usage:
    python baseline_models.py [--task classification] [--input data/datasets/]
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    silhouette_score, adjusted_rand_score
)
from sklearn.model_selection import cross_val_score

from config_loader import load_config, get_config_value


def load_dataset(data_dir: str, split: str = 'train') -> tuple[Any, np.ndarray]:
    """Load dataset split."""
    data_path = Path(data_dir)
    
    X = load_npz(data_path / f'X_{split}.npz')
    y = np.load(data_path / f'y_{split}.npy')
    
    return X, y


def train_classification_models(X_train: Any, y_train: np.ndarray,
                                X_val: Any, y_val: np.ndarray,
                                config: dict[str, Any]) -> dict[str, Any]:
    """
    Train baseline classification models.
    
    Tasks: Spam detection, category classification, priority prediction
    """
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*60)
    
    models = {}
    results = {}
    
    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr
    
    y_pred_lr = lr.predict(X_val)
    results['logistic_regression'] = {
        'accuracy': accuracy_score(y_val, y_pred_lr),
        'classification_report': classification_report(y_val, y_pred_lr, output_dict=True)
    }
    print(f"  Accuracy: {results['logistic_regression']['accuracy']:.4f}")
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    y_pred_rf = rf.predict(X_val)
    results['random_forest'] = {
        'accuracy': accuracy_score(y_val, y_pred_rf),
        'classification_report': classification_report(y_val, y_pred_rf, output_dict=True),
        'feature_importance': rf.feature_importances_.tolist()[:20]  # Top 20
    }
    print(f"  Accuracy: {results['random_forest']['accuracy']:.4f}")
    
    # SVM (on smaller subset if too large)
    if X_train.shape[0] < 10000:
        print("\nTraining SVM...")
        svm = SVC(kernel='linear', random_state=42, probability=True)
        svm.fit(X_train, y_train)
        models['svm'] = svm
        
        y_pred_svm = svm.predict(X_val)
        results['svm'] = {
            'accuracy': accuracy_score(y_val, y_pred_svm),
            'classification_report': classification_report(y_val, y_pred_svm, output_dict=True)
        }
        print(f"  Accuracy: {results['svm']['accuracy']:.4f}")
    else:
        print("\nSkipping SVM (dataset too large)")
    
    return models, results


def train_clustering_models(X_train: Any, config: dict[str, Any]) -> dict[str, Any]:
    """
    Train clustering models.
    
    Tasks: Email groups, topic discovery
    """
    print("\n" + "="*60)
    print("TRAINING CLUSTERING MODELS")
    print("="*60)
    
    models = {}
    results = {}
    
    # Convert to dense if needed (for clustering)
    if hasattr(X_train, 'toarray'):
        X_dense = X_train.toarray()
    else:
        X_dense = X_train
    
    # K-Means
    print("\nTraining K-Means...")
    n_clusters = config.get('clustering', {}).get('n_clusters', 5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_dense)
    models['kmeans'] = kmeans
    
    labels_kmeans = kmeans.labels_
    silhouette = silhouette_score(X_dense, labels_kmeans)
    results['kmeans'] = {
        'n_clusters': n_clusters,
        'silhouette_score': float(silhouette),
        'cluster_sizes': [int(np.sum(labels_kmeans == i)) for i in range(n_clusters)]
    }
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Cluster sizes: {results['kmeans']['cluster_sizes']}")
    
    # DBSCAN
    print("\nTraining DBSCAN...")
    eps = config.get('clustering', {}).get('dbscan_eps', 0.5)
    min_samples = config.get('clustering', {}).get('dbscan_min_samples', 5)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    dbscan.fit(X_dense)
    models['dbscan'] = dbscan
    
    labels_dbscan = dbscan.labels_
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = int(np.sum(labels_dbscan == -1))
    
    results['dbscan'] = {
        'n_clusters': n_clusters_dbscan,
        'n_noise': n_noise,
        'n_samples': len(labels_dbscan)
    }
    print(f"  Number of clusters: {n_clusters_dbscan}")
    print(f"  Noise points: {n_noise}")
    
    return models, results


def train_nlp_models(texts: list[str], config: dict[str, Any]) -> dict[str, Any]:
    """
    Train NLP models.
    
    Tasks: Sentiment analysis, topic modeling, summarization
    """
    print("\n" + "="*60)
    print("TRAINING NLP MODELS")
    print("="*60)
    
    models = {}
    results = {}
    
    # Topic Modeling with LDA
    print("\nTraining LDA Topic Model...")
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Vectorize texts
    vectorizer = CountVectorizer(max_features=1000, stop_words='english', min_df=2)
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    n_topics = config.get('nlp', {}).get('n_topics', 5)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda.fit(doc_term_matrix)
    models['lda'] = lda
    models['lda_vectorizer'] = vectorizer
    
    # Get topic-word distributions
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_id': topic_idx,
            'top_words': top_words,
            'word_weights': topic[top_words_idx].tolist()
        })
    
    results['lda'] = {
        'n_topics': n_topics,
        'topics': topics
    }
    print(f"  Topics extracted: {n_topics}")
    for topic in topics[:3]:
        print(f"    Topic {topic['topic_id']}: {', '.join(topic['top_words'][:5])}")
    
    # NMF Topic Modeling
    print("\nTraining NMF Topic Model...")
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
    nmf.fit(doc_term_matrix)
    models['nmf'] = nmf
    
    nmf_topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        nmf_topics.append({
            'topic_id': topic_idx,
            'top_words': top_words
        })
    
    results['nmf'] = {
        'n_topics': n_topics,
        'topics': nmf_topics
    }
    print(f"  Topics extracted: {n_topics}")
    
    return models, results


def train_anomaly_detection_models(X_train: Any, config: dict[str, Any]) -> dict[str, Any]:
    """
    Train anomaly detection models.
    
    Tasks: Find unusual email patterns
    """
    print("\n" + "="*60)
    print("TRAINING ANOMALY DETECTION MODELS")
    print("="*60)
    
    models = {}
    results = {}
    
    # Convert to dense if needed
    if hasattr(X_train, 'toarray'):
        X_dense = X_train.toarray()
    else:
        X_dense = X_train
    
    # Isolation Forest
    print("\nTraining Isolation Forest...")
    contamination = config.get('anomaly_detection', {}).get('contamination', 0.1)
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    iso_forest.fit(X_dense)
    models['isolation_forest'] = iso_forest
    
    # Predict anomalies
    predictions = iso_forest.predict(X_dense)
    anomaly_scores = iso_forest.score_samples(X_dense)
    n_anomalies = int(np.sum(predictions == -1))
    
    results['isolation_forest'] = {
        'n_anomalies': n_anomalies,
        'n_total': len(predictions),
        'anomaly_rate': float(n_anomalies / len(predictions)),
        'mean_anomaly_score': float(np.mean(anomaly_scores))
    }
    print(f"  Anomalies detected: {n_anomalies} / {len(predictions)} ({n_anomalies/len(predictions)*100:.2f}%)")
    
    return models, results


def train_baseline_models(dataset_dir: str, output_dir: str, task: str, config: dict[str, Any]) -> None:
    """
    Train baseline models for specified task.
    
    Args:
        dataset_dir: Directory with prepared datasets
        output_dir: Output directory for models
        task: Task type ('classification', 'clustering', 'nlp', 'anomaly', 'all')
        config: Configuration dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_models = {}
    all_results = {}
    
    if task in ['classification', 'all']:
        print("\nLoading classification dataset...")
        X_train, y_train = load_dataset(dataset_dir, 'train')
        X_val, y_val = load_dataset(dataset_dir, 'val')
        
        models, results = train_classification_models(X_train, y_train, X_val, y_val, config)
        all_models['classification'] = models
        all_results['classification'] = results
    
    if task in ['clustering', 'all']:
        print("\nLoading clustering dataset...")
        X_train, _ = load_dataset(dataset_dir, 'train')
        
        models, results = train_clustering_models(X_train, config)
        all_models['clustering'] = models
        all_results['clustering'] = results
    
    if task in ['nlp', 'all']:
        print("\nLoading text data for NLP...")
        # Load original profiled data for texts
        processed_dir = Path(dataset_dir).parent.parent / 'processed'
        profiled_files = list(processed_dir.glob('emails_profiled_*.json'))
        if profiled_files:
            with open(profiled_files[-1], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = [email.get('body_plain', '') + ' ' + email.get('subject', '') 
                    for email in data.get('emails', [])]
            
            if texts:
                models, results = train_nlp_models(texts, config)
                all_models['nlp'] = models
                all_results['nlp'] = results
            else:
                print("  No text data found, skipping NLP models")
        else:
            print("  No profiled data found, skipping NLP models")
    
    if task in ['anomaly', 'all']:
        print("\nLoading anomaly detection dataset...")
        X_train, _ = load_dataset(dataset_dir, 'train')
        
        models, results = train_anomaly_detection_models(X_train, config)
        all_models['anomaly'] = models
        all_results['anomaly'] = results
    
    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    for task_name, models_dict in all_models.items():
        task_dir = output_path / task_name
        task_dir.mkdir(exist_ok=True)
        
        for model_name, model in models_dict.items():
            model_file = task_dir / f'{model_name}.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved {task_name}/{model_name}.pkl")
    
    # Save results
    results_file = output_path / 'model_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Models saved to: {output_path}")
    print(f"✓ Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    for task_name, results_dict in all_results.items():
        print(f"\n{task_name.upper()}:")
        for model_name, metrics in results_dict.items():
            if 'accuracy' in metrics:
                print(f"  {model_name}: Accuracy = {metrics['accuracy']:.4f}")
            elif 'silhouette_score' in metrics:
                print(f"  {model_name}: Silhouette = {metrics['silhouette_score']:.4f}")
            elif 'n_anomalies' in metrics:
                print(f"  {model_name}: Anomalies = {metrics['n_anomalies']} ({metrics['anomaly_rate']*100:.2f}%)")


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline ML models')
    parser.add_argument('--task', type=str, default='all',
                       choices=['classification', 'clustering', 'nlp', 'anomaly', 'all'],
                       help='Task to train models for (default: all)')
    parser.add_argument('--input', type=str, default=None,
                       help='Input dataset directory (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config file (default: config.json)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    input_directory = args.input if args.input is not None else get_config_value(
        config, 'models', 'input_directory', default='data/datasets'
    )
    output_directory = args.output if args.output is not None else get_config_value(
        config, 'models', 'output_directory', default='data/models'
    )
    
    train_baseline_models(input_directory, output_directory, args.task, config)


if __name__ == "__main__":
    main()

