"""
Generate a small CSV of random test-set examples for manual label QC.
Saves `data/processed/sample_label_check.csv` with 20 examples.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, 'data', 'processed')

test_path = os.path.join(PROC_DIR, 'test.csv')
out_path = os.path.join(PROC_DIR, 'sample_label_check.csv')

if not os.path.exists(test_path):
    print('Test CSV not found at', test_path)
    raise SystemExit(1)

df = pd.read_csv(test_path)
sample = df.sample(n=20, random_state=42)
sample.to_csv(out_path, index=False)
print('Sample saved to', out_path)
