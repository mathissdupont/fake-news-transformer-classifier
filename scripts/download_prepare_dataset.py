"""
Download and prepare the `isakulaksiz/turkish-fake-news-detection` dataset.
Saves raw CSV to `data/raw/` and processed train/test CSVs to `data/processed/`.
"""
import os
import sys

try:
    from datasets import load_dataset
except Exception as e:
    print("Missing package 'datasets'. Install with: pip install datasets")
    raise

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
PROC_DIR = os.path.join(ROOT, 'data', 'processed')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

print('Loading dataset from Hugging Face...')
ds = load_dataset("isakulaksiz/turkish-fake-news-detection")

# The dataset may have a single split named 'train' or others
split_name = list(ds.keys())[0]
df = ds[split_name].to_pandas()

raw_path = os.path.join(RAW_DIR, 'turkish_fake_news_raw.csv')
df.to_csv(raw_path, index=False)
print(f'Raw dataset saved to: {raw_path}')

print('Columns:', df.columns.tolist())
print('Shape:', df.shape)

# Heuristic to produce a text column
if 'description' in df.columns and df['description'].notna().sum() > 0:
    df['text'] = df['title'].fillna('') + '\n' + df['description'].fillna('')
else:
    df['text'] = df['title'].fillna('')

# Heuristic to produce a binary label column with values 'fake'/'real'
label_col = None
if 'status' in df.columns:
    # Assume 1 -> real, 0 -> fake
    df['label'] = df['status'].map({1: 'real', 0: 'fake'})
    label_col = 'status'
elif 'label' in df.columns:
    df['label'] = df['label'].astype(str)
    # try to normalize common encodings
    df['label'] = df['label'].replace({'1':'real','0':'fake','true':'real','false':'fake', 'True':'real','False':'fake'})
    label_col = 'label'
elif 'Resource' in df.columns:
    # look for strings containing 'True' or 'False'
    df['label'] = df['Resource'].astype(str).apply(lambda x: 'real' if 'True' in x or 'true' in x or 'TeyitOrg-True' in x else 'fake')
    label_col = 'Resource'
else:
    print('No known label column found. Please inspect the raw CSV.')

print('Label distribution:')
if 'label' in df.columns:
    print(df['label'].value_counts(dropna=False))
else:
    print('label column not created')

# Drop rows without text or label
df = df[df['text'].str.strip().astype(bool)]
if 'label' in df.columns:
    df = df[df['label'].notna()]

print('After dropping missing text/label rows:', df.shape)

# Save processed full CSV
proc_full = os.path.join(PROC_DIR, 'turkish_fake_news_processed_full.csv')
df[['text','label']].to_csv(proc_full, index=False)
print(f'Processed full CSV saved to: {proc_full}')

# Stratified train/test split
if 'label' in df.columns:
    train, test = train_test_split(df[['text','label']], test_size=0.2, random_state=42, stratify=df['label'])
    train_path = os.path.join(PROC_DIR, 'train.csv')
    test_path = os.path.join(PROC_DIR, 'test.csv')
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print('Train shape:', train.shape, 'Test shape:', test.shape)
    print('Train label distribution:')
    print(train['label'].value_counts())
    print('Test label distribution:')
    print(test['label'].value_counts())
else:
    print('Skipping split because label column missing.')

print('\nDone.')
