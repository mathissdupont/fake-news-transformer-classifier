"""
Preprocessing utilities for the project.
Provides functions to load the HF dataset, normalize text and labels,
and save processed train/test CSVs.
"""
import os
from typing import Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
PROC_DIR = os.path.join(ROOT, 'data', 'processed')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)


def load_from_hf(name: str = "isakulaksiz/turkish-fake-news-detection") -> pd.DataFrame:
    ds = load_dataset(name)
    split = list(ds.keys())[0]
    df = ds[split].to_pandas()
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # create text
    if 'description' in df.columns and df['description'].notna().sum() > 0:
        df['text'] = df['title'].fillna('') + '\n' + df['description'].fillna('')
    else:
        df['text'] = df['title'].fillna('')

    # normalize label
    if 'status' in df.columns:
        df['label'] = df['status'].map({1: 'real', 0: 'fake'})
    elif 'label' in df.columns:
        df['label'] = df['label'].astype(str).replace({'1':'real','0':'fake','true':'real','false':'fake','True':'real','False':'fake'})
    elif 'Resource' in df.columns:
        df['label'] = df['Resource'].astype(str).apply(lambda x: 'real' if 'True' in x or 'true' in x or 'TeyitOrg-True' in x else 'fake')
    else:
        raise ValueError('No known label column found')

    # drop missing
    df = df[df['text'].str.strip().astype(bool)]
    df = df[df['label'].notna()]
    return df[['text', 'label']]


def save_raw_and_processed(df: pd.DataFrame, raw_name: str = 'turkish_fake_news_raw.csv') -> None:
    raw_path = os.path.join(RAW_DIR, raw_name)
    df.to_csv(raw_path, index=False)
    proc_full = os.path.join(PROC_DIR, 'turkish_fake_news_processed_full.csv')
    df[['text','label']].to_csv(proc_full, index=False)


def stratified_split_and_save(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[str, str]:
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    train_path = os.path.join(PROC_DIR, 'train.csv')
    test_path = os.path.join(PROC_DIR, 'test.csv')
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def main():
    print('Loading from Hugging Face...')
    df = load_from_hf()
    print('Preparing dataframe...')
    prepared = prepare_dataframe(df)
    print('Saving raw and processed...')
    save_raw_and_processed(prepared)
    print('Splitting...')
    train_path, test_path = stratified_split_and_save(prepared)
    print('Saved:', train_path, test_path)


if __name__ == '__main__':
    main()
"""Dataset cleaning and preprocessing helpers."""
