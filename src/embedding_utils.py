"""Embedding utilities with caching support."""
import os
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer


def load_or_compute_embeddings(
    texts, 
    model_name='paraphrase-multilingual-MiniLM-L12-v2',
    cache_path=None,
    show_progress=True
):
    """Load embeddings from cache or compute them.
    
    Args:
        texts: list of text strings to embed
        model_name: sentence-transformers model name
        cache_path: optional path to save/load embeddings (e.g., 'embeddings_train.npy')
        show_progress: show progress bar during encoding
    
    Returns:
        embeddings array (n_samples, embedding_dim)
    """
    # Try to load from cache first
    if cache_path and os.path.exists(cache_path):
        print(f'Loading embeddings from cache: {cache_path}')
        embeddings = np.load(cache_path)
        return embeddings
    
    # Compute embeddings
    print(f'Loading model {model_name}...')
    model = SentenceTransformer(model_name)
    print(f'Computing embeddings for {len(texts)} texts...')
    embeddings = model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
    
    # Save to cache if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.save(cache_path, embeddings)
        print(f'Saved embeddings to cache: {cache_path}')
    
    return embeddings


def save_embedding_metadata(model_name: str, cache_dir: str = 'models'):
    """Save embedding model metadata for reproducibility."""
    metadata = {
        'model_name': model_name,
        'embedding_dim': 384  # for paraphrase-multilingual-MiniLM-L12-v2
    }
    os.makedirs(cache_dir, exist_ok=True)
    joblib.dump(metadata, os.path.join(cache_dir, 'embedding_metadata.pkl'))
    print(f'Saved embedding metadata to {cache_dir}/embedding_metadata.pkl')
