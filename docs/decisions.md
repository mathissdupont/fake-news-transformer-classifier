# Project Decisions

## Working Assumptions

- Use Python for the implementation.
- Treat the task as an experimental machine learning project, not a full production system.
- Compare a TF-IDF baseline with transformer-based embeddings.
- Keep all results reproducible with fixed random seeds.

## Open Choices

- Turkish dataset or English backup dataset
- Final classifier set for embeddings
- Whether to add Random Forest as an optional comparison

## Collaboration Standard

- The project is split into three mostly independent work packages: data/baseline, embeddings/models, and evaluation/reporting.
- Every AI-assisted action must be logged in `docs/ai-usage.md` immediately after it happens.
- Any AI-influenced project decision should also be noted here if it changes the plan or deliverables.

## Dataset Selection

**Status:** [DECIDED - Selected by Person 1 on 2026-05-12]

**Candidate sources (per project requirements):**
- Hugging Face Datasets: https://huggingface.co/datasets
- Kaggle Datasets: https://www.kaggle.com/datasets
- Google Dataset Search: https://datasetsearch.research.google.com
- Papers With Code: https://paperswithcode.com/datasets
- TÜİK (for Turkish social/economic data): https://data.tuik.gov.tr

**Final choice:** isakulaksiz/turkish-fake-news-detection
**Dataset name:** Turkish Fake News Detection (isakulaksiz/turkish-fake-news-detection)
**Download link:** https://huggingface.co/datasets/isakulaksiz/turkish-fake-news-detection
**Selection date:** 2026-05-12
**Reason for choice:** Turkish-language dataset, sufficient size (5,326 rows), MIT license, CSV format, contains full text fields (`title` and `description`). Matches project priority of preferring Turkish datasets.

## Shared Parameters

**Status:** [LOCKED - Dataset selected]

**Random state for reproducibility:** `random_state=42` (locked)
 - Suggested default: `random_state=42`

**Train/Test split ratio:** 80/20 with stratification (implemented and saved)

**Label mapping:** (inferred during preprocessing)
-- Text column name: `text` (created by concatenating `title` + `description`)
-- Label column name: `label` (normalized)
-- Fake class value(s): mapped from `status` 0 or `Resource` indicating False → `fake`
-- Real class value(s): mapped from `status` 1 or `Resource` indicating True → `real`
-- Standardized labels: `fake` and `real`

## Model Choices

**Status:** [PENDING - To be decided by Person 2]

**Embedding model:** [FILL IN]
- Model name: [e.g., sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2]
- Reasoning: [explain why this model for this dataset language]

**Classifiers for embeddings:** [FILL IN]
- Always test: Logistic Regression, SVM
- Optional: Random Forest if time permits
