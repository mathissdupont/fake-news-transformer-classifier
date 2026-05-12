"""Extract sentence embeddings, train classifiers (LR and SVM), and save metrics/models.

Usage:
    python src/train_embeddings.py
"""
import os
import pandas as pd
import joblib
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(ROOT, 'results')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    train = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'))
    return train['text'].tolist(), train['label'].tolist(), test['text'].tolist(), test['label'].tolist()


def embed_texts(model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    X_train_texts, y_train, X_test_texts, y_test = load_data()
    model = SentenceTransformer(model_name)
    Xtr = model.encode(X_train_texts, show_progress_bar=True, convert_to_numpy=True)
    Xte = model.encode(X_test_texts, show_progress_bar=True, convert_to_numpy=True)
    return Xtr, y_train, Xte, y_test, model


def fit_and_eval(Xtr, y_train, Xte, y_test, name_prefix='embedding'):
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(Xtr, y_train)
    preds = lr.predict(Xte)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', pos_label='real')
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=['fake','real']).ravel()
    df_lr = pd.DataFrame([{
        'Model': name_prefix,
        'Classifier': 'LogisticRegression',
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'True_Negatives': int(tn),
        'False_Positives': int(fp),
        'False_Negatives': int(fn),
        'True_Positives': int(tp)
    }])
    df_lr.to_csv(os.path.join(RESULTS_DIR, f'{name_prefix}_lr_metrics.csv'), index=False)
    joblib.dump(lr, os.path.join(MODELS_DIR, f'{name_prefix}_lr.pkl'))

    # SVM
    svm = SVC(kernel='linear', probability=True)
    svm.fit(Xtr, y_train)
    preds_svm = svm.predict(Xte)
    acc2 = accuracy_score(y_test, preds_svm)
    precision2, recall2, f12, _ = precision_recall_fscore_support(y_test, preds_svm, average='binary', pos_label='real')
    tn2, fp2, fn2, tp2 = confusion_matrix(y_test, preds_svm, labels=['fake','real']).ravel()
    df_svm = pd.DataFrame([{
        'Model': name_prefix,
        'Classifier': 'SVM',
        'Accuracy': acc2,
        'Precision': precision2,
        'Recall': recall2,
        'F1-Score': f12,
        'True_Negatives': int(tn2),
        'False_Positives': int(fp2),
        'False_Negatives': int(fn2),
        'True_Positives': int(tp2)
    }])
    df_svm.to_csv(os.path.join(RESULTS_DIR, f'{name_prefix}_svm_metrics.csv'), index=False)
    joblib.dump(svm, os.path.join(MODELS_DIR, f'{name_prefix}_svm.pkl'))


def main():
    print('Computing embeddings and training classifiers. This may download model weights (~100MB).')
    Xtr, y_train, Xte, y_test, model = embed_texts()
    fit_and_eval(Xtr, y_train, Xte, y_test, name_prefix='embedding')
    # save embedding model reference
    joblib.dump({'model_name': 'paraphrase-multilingual-MiniLM-L12-v2'}, os.path.join(MODELS_DIR, 'embedding_model_info.pkl'))


if __name__ == '__main__':
    main()
"""Train embedding-based models for fake news detection."""
