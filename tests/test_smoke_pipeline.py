import os
import pandas as pd

from src.utils import save_metrics_csv


def test_save_metrics_csv_creates_file(tmp_path):
    y_test = ['real', 'fake', 'real', 'fake']
    preds = ['real', 'fake', 'fake', 'fake']
    out = tmp_path / 'metrics.csv'
    df = save_metrics_csv(y_test, preds, 'tfidf', 'LogisticRegression', str(out))
    assert out.exists()
    loaded = pd.read_csv(out)
    assert 'Accuracy' in loaded.columns
    assert loaded.shape[0] == 1
