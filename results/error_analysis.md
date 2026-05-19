# Error Analysis

This note summarizes the model errors using the saved test-set metrics. All counts come from the files in `results/`.

## Test Set

- Total test samples: 1066
- Labels: `fake`, `real`
- Positive class for precision/recall/F1: `real`

## Model Comparison

| Model | Classifier | Accuracy | F1-Score | False Positives | False Negatives |
|-------|------------|----------|----------|-----------------|-----------------|
| TF-IDF | Logistic Regression | 0.9381 | 0.9317 | 19 | 47 |
| Embedding | Logistic Regression | 0.9043 | 0.8984 | 56 | 46 |
| Embedding | SVM | 0.9109 | 0.9043 | 47 | 48 |

## Observations

- TF-IDF + Logistic Regression is the strongest model in this run.
- The TF-IDF model has the fewest false positives, meaning it is more conservative when predicting `real` for fake news.
- Embedding Logistic Regression has recall close to TF-IDF, but more false positives.
- Embedding SVM improves over embedding Logistic Regression, but still remains below TF-IDF on accuracy and F1.
- The results suggest that lexical patterns in this dataset are highly informative, so the simpler TF-IDF baseline performs better than the selected sentence embedding setup.

## Report Note

The final report should not claim that transformer embeddings are always worse. The correct interpretation is narrower: on this dataset, with `paraphrase-multilingual-MiniLM-L12-v2` embeddings and LR/SVM classifiers, TF-IDF + Logistic Regression achieved the best measured result.
