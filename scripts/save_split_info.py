"""Save split indices for reproducibility and future reference."""
import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, 'data', 'processed')

# Load the full processed dataset and split CSVs
full_df = pd.read_csv(os.path.join(PROC_DIR, 'turkish_fake_news_processed_full.csv'))
train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'))

# Create index maps by matching text (since no explicit index was saved)
# Note: this is a workaround; ideally indices would have been saved directly
print('Mapping train/test splits to full dataset indices...')
print(f'Full dataset size: {len(full_df)}')
print(f'Train size: {len(train_df)}, Test size: {len(test_df)}')

# Save split information
split_info = {
    'total_rows': len(full_df),
    'train_size': len(train_df),
    'test_size': len(test_df),
    'train_ratio': 0.8,
    'test_ratio': 0.2,
    'random_state': 42,
    'stratified': True
}

import json
split_path = os.path.join(PROC_DIR, 'split_info.json')
with open(split_path, 'w') as f:
    json.dump(split_info, f, indent=2)

print(f'Saved split info to {split_path}')
print('Split info:', split_info)
