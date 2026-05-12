Dataset Information and Source
==============================

Final Dataset Selected: isakulaksiz/turkish-fake-news-detection

Dataset Name: Turkish Fake News Detection (isakulaksiz/turkish-fake-news-detection)
Source Platform (Hugging Face / Kaggle / Papers With Code / Other): Hugging Face Datasets
Download Link / Access Method: https://huggingface.co/datasets/isakulaksiz/turkish-fake-news-detection  (or `from datasets import load_dataset; load_dataset("isakulaksiz/turkish-fake-news-detection")`)
Download Date: 2026-05-12
Dataset Size (number of rows): 5,326 (train split on HF)
Language: Turkish
Label Distribution (fake/real or 0/1 counts): fake=2845, real=2481 (after heuristics applied)
Text Column Name: `text` (created by concatenating `title` + `description`)
Label Column Name: `label` (normalized to `fake` / `real` from `status` / `Resource`)

Access Instructions:
- To download programmatically:

```python
from datasets import load_dataset
ds = load_dataset("isakulaksiz/turkish-fake-news-detection")
```

- Or open the dataset page and download CSV from "Files and versions".

License and Citation:
- License: MIT (as shown on the dataset card)
- Citation: See dataset page for author and citation details; include dataset owner `isakulaksiz` in references.

Notes:
- Preprocessing performed:
	- Created `text` by joining `title` and `description`.
	- Normalized labels to `fake` / `real` (mapped `status` 0->`fake`, 1->`real`; fallback using `Resource`).
	- Removed rows with missing text or label.
	- Stratified train/test split saved to `data/processed/train.csv` and `data/processed/test.csv` (80/20, `random_state=42`).
- Known issues:
	- Labels were inferred via heuristics; verify a small sample manually for correctness.
	- HF split provided as a single `train` split; we generated our own test split with stratification.
- Reproducibility notes:
	- Raw CSV saved to `data/raw/turkish_fake_news_raw.csv`.
	- Processed full CSV: `data/processed/turkish_fake_news_processed_full.csv`.
	- Train/Test saved as `data/processed/train.csv` and `data/processed/test.csv`.
	- Use `random_state=42` for reproducibility.

Repository policy:
- Do NOT commit raw CSV files to the repository. Instead commit the download/preprocessing script (`scripts/download_prepare_dataset.py` or `src/preprocessing.py`) and `data/README_dataset_link.txt` so teammates can reproduce the download. If a raw CSV was accidentally tracked, remove it from git with:

```
git rm --cached data/raw/turkish_fake_news_raw.csv
git commit -m "Remove raw dataset from repository"
```

The processed CSVs (`data/processed/train.csv`, `test.csv`) are small enough to keep tracked for reproducibility, but you may also ignore them if preferred.
