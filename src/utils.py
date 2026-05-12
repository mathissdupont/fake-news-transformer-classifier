"""Utility functions for reproducibility, seeding, and common operations."""
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Note: TensorFlow/Torch seeding not included unless those libraries are used


def save_metrics_csv(y_test, preds, model_name: str, classifier_name: str, output_path: str) -> pd.DataFrame:
    """Compute and save classification metrics to CSV.
    
    Args:
        y_test: true labels (list or array)
        preds: predicted labels (list or array)
        model_name: e.g., 'tfidf' or 'embedding'
        classifier_name: e.g., 'LogisticRegression', 'SVM'
        output_path: path to save CSV
    
    Returns:
        DataFrame with metrics
    """
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', pos_label='real')
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=['fake','real']).ravel()
    
    df = pd.DataFrame([{
        'Model': model_name,
        'Classifier': classifier_name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'True_Negatives': int(tn),
        'False_Positives': int(fp),
        'False_Negatives': int(fn),
        'True_Positives': int(tp)
    }])
    
    df.to_csv(output_path, index=False)
    return df


def load_split_files(proc_dir: str):
    """Load train and test splits from processed directory."""
    train = pd.read_csv(os.path.join(proc_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(proc_dir, 'test.csv'))
    return train, test
