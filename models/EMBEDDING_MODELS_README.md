Embedding models and artifacts

The embedding classifier models are not committed to the repo due to size.
To reproduce embeddings and classifiers locally:

1. Ensure `data/processed/train.csv` and `test.csv` exist.
2. Run the embedding extraction and training script:

```bash
python src/train_embeddings.py --model-name paraphrase-multilingual-MiniLM-L12-v2
```

3. Expected artifacts (place in `models/`):
- `models/embedding_lr.pkl`
- `models/embedding_svm.pkl`
- `models/embedding_metadata.json`

Precomputed embeddings are available under `data/processed/embeddings_*.npy` in this repo.
