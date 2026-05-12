"""Train TF-IDF baseline and save metrics, model, and confusion matrix.

Usage: run from project root venv activated:
    python src/train_tfidf.py
"""
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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


def train_and_eval():
    X_train, y_train, X_test, y_test = load_data()
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)

    clf = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xte)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', pos_label='real')

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=['fake','real']).ravel()

    metrics_df = pd.DataFrame([{
        'Model': 'tfidf',
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

    metrics_path = os.path.join(RESULTS_DIR, 'tfidf_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print('Saved metrics to', metrics_path)

    # save model and vectorizer
    joblib.dump(clf, os.path.join(MODELS_DIR, 'tfidf_model.pkl'))
    joblib.dump(vect, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    print('Saved model & vectorizer to', MODELS_DIR)

    # confusion matrix plot
    cm = np.array([[tn, fp],[fn, tp]])
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['fake','real'], yticklabels=['fake','real'], ylabel='True label', xlabel='Predicted label', title='Confusion Matrix TF-IDF')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='white' if val>cm.max()/2 else 'black')
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'confusion_matrix_tfidf.png')
    fig.savefig(fig_path)
    print('Saved confusion matrix to', fig_path)


if __name__ == '__main__':
    train_and_eval()
"""Train a TF-IDF baseline model."""
