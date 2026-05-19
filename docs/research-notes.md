# Research Notes

This file documents the research and background reading done by the team before and during the project.

## Shared Research Goals

- [x] Understand fake news detection as a binary text classification problem
- [x] Compare TF-IDF baselines with transformer-based embeddings
- [x] Confirm that the chosen dataset is reproducible and suitable for the deadline
- [x] Collect at least 10 relevant academic or technical sources for the final bibliography
- [x] Write down candidate datasets, candidate models, and candidate evaluation metrics before implementation

## Person 1: Data and Preprocessing Research

### Papers and Resources Read

| Title | Authors/Source | Year | Link/DOI | Key Takeaway |
|-------|----------------|------|----------|--------------|
| Fake News Detection on Social Media: A Data Mining Perspective | Shu, Sliva, Wang, Tang, Liu | 2017 | https://doi.org/10.1145/3137597.3137600 | Frames fake news detection as a data mining problem and discusses datasets, features, and evaluation challenges. |
| Turkish Fake News Detection dataset card | isakulaksiz / Hugging Face | 2024 | https://huggingface.co/datasets/isakulaksiz/turkish-fake-news-detection | Turkish dataset with binary fake/real style labels and usable news text fields. |
| scikit-learn TF-IDF documentation | scikit-learn | 2026 | https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html | TF-IDF converts text into weighted sparse lexical features. |

### Dataset Candidates Evaluated

| Dataset Name | Source | Text Quality | Label Quality | Sample Size | Decision |
|--------------|--------|--------------|---------------|-------------|----------|
| isakulaksiz/turkish-fake-news-detection | Hugging Face | Turkish title + description fields | Binary mapping possible | 5326 rows | Selected |
| WELFake | Kaggle / public mirrors | English full text | Binary fake/real | Larger than project needs | Backup only because the project preferred Turkish data |
| Generic Kaggle fake news datasets | Kaggle | Mostly English | Usually binary | Varies | Not selected due to language mismatch or weaker dataset documentation |

### Preprocessing Best Practices Found

- Keep a reproducible train/test split with `random_state=42`.
- Use stratification so fake/real ratios remain similar across train and test.
- Save processed `train.csv`, `test.csv`, and `split_info.json` instead of relying on manual local state.
- Record label mapping explicitly because dataset column names and label meanings may differ across sources.

## Person 2: Embedding and Model Research

### Papers and Resources Read

| Title | Authors | Year | Link/DOI | Key Takeaway |
|-------|---------|------|----------|--------------|
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Devlin, Chang, Lee, Toutanova | 2019 | https://doi.org/10.18653/v1/N19-1423 | BERT introduced bidirectional transformer pretraining for downstream NLP tasks. |
| Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Reimers, Gurevych | 2019 | https://arxiv.org/abs/1908.10084 | SBERT creates efficient sentence embeddings suitable for downstream classifiers and similarity tasks. |
| BERTurk model family | dbmdz / MDZ Digital Library Team | 2020 | https://huggingface.co/dbmdz/bert-base-turkish-cased | Turkish-specific BERT is a possible future improvement if fine-tuning is added. |
| Sentence Transformers documentation | UKP / Hugging Face | 2026 | https://www.sbert.net/ | Provides practical embedding models and encoding APIs. |

### Embedding Models Considered

| Model Name | Source | Language Support | Dimension | Pros | Cons |
|------------|--------|------------------|-----------|------|------|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Sentence Transformers | Multilingual, including Turkish | 384 | Lightweight, fast enough locally, easy to cache | Generic multilingual model, not fake-news-specific |
| `dbmdz/bert-base-turkish-cased` | Hugging Face | Turkish | 768 | Turkish-specific language model | Requires fine-tuning or custom pooling setup |
| TF-IDF features | scikit-learn | Language independent token features | Sparse vocabulary size | Strong, simple baseline | Less semantic than transformer embeddings |

### Classifier Comparison Literature

- Logistic Regression is a strong linear baseline for sparse TF-IDF and dense embedding features.
- Linear SVM is commonly used for text classification because it handles high-dimensional decision boundaries well.
- Random Forest was kept optional because it can be slower and is not always stronger on sparse or dense text representations.

## Person 3: Report Writing and Evaluation Research

### Report Format Examples

| Source | Type | Key Format Elements |
|--------|------|---------------------|
| CENG454 final project PDF | Course document | Max 8 pages, method/results/discussion, real measured results |
| IEEE citation style | Formatting reference | Numbered citations in order of appearance |
| scikit-learn metrics documentation | Technical reference | Accuracy, precision, recall, F1, confusion matrix definitions |

### Evaluation Metrics for Binary Classification

- Accuracy: total correct predictions divided by all predictions; useful as a broad summary when class balance is acceptable.
- Precision: among examples predicted as `real`, how many were actually `real`.
- Recall: among actual `real` examples, how many the model recovered.
- F1-Score: harmonic mean of precision and recall; useful when precision/recall trade-offs matter.
- Confusion Matrix: shows true/false predictions per class; useful for explaining whether the model makes more false positive or false negative errors.

### IEEE Citation Format Examples

- Journal: K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, "Fake news detection on social media: A data mining perspective," *ACM SIGKDD Explorations Newsletter*, vol. 19, no. 1, pp. 22-36, 2017.
- Conference: J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171-4186.
- Dataset: isakulaksiz, "Turkish Fake News Detection," Hugging Face Datasets. [Online]. Available: https://huggingface.co/datasets/isakulaksiz/turkish-fake-news-detection

## Shared Bibliography

1. K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, "Fake news detection on social media: A data mining perspective," *ACM SIGKDD Explorations Newsletter*, vol. 19, no. 1, pp. 22-36, 2017, doi: 10.1145/3137597.3137600.
2. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. EMNLP-IJCNLP*, 2019.
3. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171-4186, doi: 10.18653/v1/N19-1423.
4. T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient estimation of word representations in vector space," arXiv:1301.3781, 2013.
5. A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems*, 2017.
6. isakulaksiz, "Turkish Fake News Detection," Hugging Face Datasets. [Online]. Available: https://huggingface.co/datasets/isakulaksiz/turkish-fake-news-detection
7. scikit-learn developers, "TfidfVectorizer," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
8. scikit-learn developers, "LogisticRegression," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
9. scikit-learn developers, "Support Vector Machines," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/svm.html
10. Hugging Face, "Transformers documentation." [Online]. Available: https://huggingface.co/docs/transformers
11. UKP Lab, "Sentence Transformers documentation." [Online]. Available: https://www.sbert.net/
12. dbmdz, "bert-base-turkish-cased," Hugging Face Models. [Online]. Available: https://huggingface.co/dbmdz/bert-base-turkish-cased

## Search Keywords Used

- fake news detection transformer embeddings
- Sentence-BERT text classification
- Turkish fake news detection dataset
- multilingual fake news classification
- TF-IDF logistic regression text classification
- misinformation detection survey
- news classification BERT Turkish
- WELFake dataset fake news
- Turkish NLP BERTurk classification
- text classification evaluation F1 confusion matrix

## Notes on Findings

### TF-IDF vs Embeddings Trade-offs

TF-IDF is simpler and more interpretable because features map directly to lexical patterns. In this project it achieved the best measured result: Accuracy 0.9381 and F1 0.9317. The selected embedding setup was useful as a semantic comparison, but it did not outperform TF-IDF on this dataset.

### Dataset Language Implications

The selected dataset is Turkish, so multilingual or Turkish-specific models are preferable to English-only models. The current project uses a multilingual MiniLM Sentence-BERT model. A future extension could fine-tune a Turkish BERT model directly on the dataset.

### Model Runtime and Memory Constraints

Embedding extraction is slower than TF-IDF training and requires downloading model weights. To make reruns practical, embeddings are cached as `data/processed/embeddings_train.npy` and `data/processed/embeddings_test.npy`.

## Final Notes

The final report should emphasize that all reported metrics come from saved result files. The strongest model in the current experiment is TF-IDF + Logistic Regression, followed by embedding SVM and embedding Logistic Regression.
